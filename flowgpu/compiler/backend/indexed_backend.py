"""Shape-aware expression evaluation for one-output indexed fusion regions.

Each DSL node retains its declared dtype boundary. Remapping composes logical
indices; only external leaves have pointers. The device evaluator streams one
bounded reduction tile at a time. The CPU evaluator uses independent scalar
loops, without importing Triton or reproducing its tile schedule.
"""
from flowgpu.compiler.shared import dimension_symbols
from flowgpu.compiler.backend.backends import (Argument, Names, identifier, product,
                                       group_operands, group_results,
                                       scalar_operands, tensor_operands,
                                       unique_formal)
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind
from flowgpu.compiler.optimization.indexed_fusion import REDUCTIONS
from flowgpu.compiler.backend.platform_configs import emitted_index_dtype
from flowgpu.compiler.backend.primitives import offset_for, tile_offset
from flowgpu.compiler.backend import cse



def _parts(graph, group):
    return group_operands(graph, group), group_results(graph, group), {n.output: n for n in group.members}


def _split_source_width(graph, node, part_width) -> str:
    """Source last extent of a split, symbolically if necessary.

    Split's output last extent is the part width; the number of parts is the
    number of sibling split nodes on the same input (2 for a half split, 4 for
    the fp4-style word split).  Keeping the width expression symbolic is
    required by the Task160 split-view ABI, whose half width is a symbol.
    """
    siblings = [candidate for candidate in graph.nodes
                if candidate.op == "split" and candidate.inputs == node.inputs]
    parts = max((getattr(candidate, "part", 0) or 0) for candidate in siblings) + 1 if siblings else 2
    width = str(part_width)
    return str(int(width) * parts) if width.isdigit() else f"({width}) * {parts}"


def triton_kernel(graph, group, memory, name, config, *, autotune_fold=False):
    if group.cross_domain:
        from flowgpu.compiler.backend.cross_domain_backend import triton_kernel as cross_kernel
        return cross_kernel(graph, group, memory, name, config)
    if group.reduction_domain:
        from flowgpu.compiler.backend.reduction_backend import triton_kernel as reduction_kernel
        return reduction_kernel(graph, group, memory, name, config, autotune_fold=autotune_fold)
    from flowgpu.compiler.backend.triton_backend import (_input_layouts, _view_offset, _make_kernel,
                                        _pointwise_expression, DTYPE_TO_TRITON, POINTWISE_EXPRESSION,
                                        IndexBound, resolve_alias, storage_of)
    from flowgpu.compiler.backend.value_layer import NAMES as VALUE_LAYER
    operands, results, produced = _parts(graph, group)
    assert len(results) == 1
    result = results[0]
    names = Names(graph)
    # A scalar operand is a host number, not a leaf to address: it contributes
    # one plain formal and no pointer, no layout, and the expression tree uses
    # the formal where a loaded tile would go.
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    reduction = next((n for n in group.members if n.op in REDUCTIONS), None)
    # --- optional split-half pair ABI ---------------------------------------
    # The caller contracts that its two external inputs are the contiguous
    # halves of ONE row-major `[M, 2N]` buffer and passes that buffer for both,
    # so the declared last axis `N` is its half. Then one base pointer plus a
    # constexpr half offset addresses both views:
    #     gate = idx + (idx // N) * N          up = gate + N
    # and the second pointer, its four layout constexprs and the second
    # div/mod address chain disappear from the kernel. The contract is the
    # caller's (the adapter built exactly this buffer with `reshape(-1, 2N)`),
    # never inferred from tensor metadata, and the arithmetic is untouched.
    half_pair = None
    if config.split_half_inputs:
        if (reduction is not None or len(tensors) != 2
                or len(graph.values[tensors[0]].shape) != 2
                or graph.values[tensors[0]].shape != graph.values[tensors[1]].shape):
            raise CompileError(
                DiagnosticKind.INTERNAL,
                'split_half_inputs requires exactly two same-shaped rank-2 '
                'external tensor inputs and no reduction in the region',
                pipeline=graph.pipeline,
            )
        half_pair = (tensors[0], tensors[1])
    pointer_tensors = [half_pair[0]] if half_pair is not None else tensors
    pointer_values = list(pointer_tensors) + list(results)
    ptrs = {v: names.fresh(identifier(v) + '_ptr') for v in pointer_values}
    pairs = [(ptrs[v], Argument('tensor', v)) for v in pointer_values]
    dims = []
    for v in tensors + tuple(produced):
        for d in graph.values[v].shape:
            if not str(d).isdigit() and d not in dims:
                dims.append(d)
    block = names.fresh('FG_BLOCK')
    used_formals = {str(d) for d in dims} | {block}
    used_formals.update(parameter for parameter, _ in pairs)
    scalar_formals = {}
    for v in scalars:
        formal = unique_formal(identifier(v), used_formals)
        names.reserve(formal)
        scalar_formals[v] = formal
        pairs.append((formal, Argument('scalar', v)))
    for d in dims:
        pairs.append((f'{d}: tl.constexpr', Argument('value', str(d))))
    # --- split-view contiguity contract ------------------------------------
    # A split whose source is an external tensor is lowered as a remapped index
    # over that source. When the source is a *caller-provided* buffer the
    # compact form is valid only for a contiguous buffer, so the lowering
    # requests the contiguity contract instead of describing the layout with
    # four constexprs: no per-axis shape/stride metadata, no `if not
    # contiguous` address chain, and the host refuses a non-contiguous argument
    # (fail closed -- see the guard in triton_backend._host). This is decided
    # per split source, never for a specific program, and it composes with any
    # split axis count. A source produced inside the region is computed inline
    # and needs no contract. Off by default because `split` never reaches an
    # indexed region unless `split_views` opted it in.
    contracted = tuple(
        value for value in tensors
        if storage_of(memory, resolve_alias(graph.aliases, value)) == 'input'
        and any(node.op == 'split' and node.inputs
                and node.inputs[0] == value for node in group.members)
    )
    described = [value for value in tensors if value not in contracted]
    layouts = ({} if half_pair is not None
               else _input_layouts(graph, memory, described, names, pairs))
    pairs.append((f'{block}: tl.constexpr', Argument('value', str(config.default_block_size))))
    idx, valid = names.fresh('idx'), names.fresh('valid')
    count = product(graph.values[result].shape)
    output_shape = graph.values[result].shape
    row_broadcasts = {n.output for n in group.members if n.op == 'broadcast_like'
                      and len(output_shape) == 2 and graph.values[n.output].shape == output_shape
                      and graph.values[n.inputs[0]].shape == [output_shape[0], '1']}
    row_tiled = bool(config.broadcast_row_tiling and reduction is None and row_broadcasts
                     and not any(n.op in {'reshape', 'move_first_last', 'gather_rows', 'gather_row_offset'}
                                 for n in group.members))
    # --- per-program 2-D tile (the measured axis `vector_tile_2d`) ----------
    # The flat form below walks one contiguous run of a FLAT index space and
    # decodes every strided input view with `idx // dim1 * stride0 + idx % dim1
    # * stride1`. That expression is a division/modulo, so the backend cannot
    # prove the run contiguous even when it happens to lie inside one row: on a
    # target that vectorises provable contiguous runs that is the expensive
    # shape, and the platform record says the tile shape is what moves it
    # (Task144 华为 0.07x -> 0.17x, seven launch parameters neutral). This form
    # gives each program `tile_rows` rows of `tile_cols` contiguous elements and
    # an explicit affine address, so the contiguous axis is provable. Only a
    # rank-2 result has the row/column split it needs; every other rank keeps
    # the flat form, and a platform whose record says no 2-D tile keeps it too.
    vector_tile = (config.vector_tile_2d and reduction is None and not row_tiled
                   and len(output_shape) == 2
                   and config.tile_rows * config.tile_cols > 1)
    if half_pair is not None and (vector_tile or row_tiled):
        raise CompileError(
            DiagnosticKind.INTERNAL,
            'split_half_inputs composes with the flat one-dimensional index '
            'domain only; the 2-D tile and row-tiled mappings address their '
            'own coordinates',
            pipeline=graph.pipeline,
        )
    row_coordinates = None
    tile_coordinates = None
    reduction_tile = None
    # --- 32-bit program-index policy ---------------------------------------
    # Every address this kernel reads or writes is derived from this one
    # index, so 64-bit index arithmetic is a per-element multiply/add cost.
    # 昆仑芯/华为 measure it as a steep one (Task160: 1.49x / 0.78x while the
    # CUDA-class chips read 4.11x with the same kernel). The dtype is a
    # platform policy, and it is only honoured with an address bound the
    # emitted host enforces before launch: the largest flat index plus the
    # masked tail lanes, then each external leaf's real strided extent (a
    # column slice reaches about twice its element count, so its logical size
    # is not the bound). No bound -- paired-view loads read an adjacent lane
    # -- means no int32.
    # The reduction form addresses `idx * width + ...` inside its row, so its
    # proof counts the whole reduction input, not just the output rows.
    index_padding = max(config.default_block_size, config.indexed_rows or 16,
                        config.reduction_block_limit or 128) - 1
    index_proof = None
    if half_pair is not None:
        # The largest derived address is the end of the caller's full buffer,
        # about twice the logical element count, so the bound doubles.
        index_proof = IndexBound(count=f'2 * ({count})', padding=index_padding,
                                 leaves=())
    elif not config.paired_view_loads:
        if vector_tile:
            # The tile form derives `row * N + col` from two program ids. Its
            # last row and column block may run past the logical extent before
            # the mask excludes the lanes: at most `tile_rows - 1` rows and
            # `tile_cols - 1` columns beyond `M * N`. That largest derivable
            # index replaces the flat form's `count + BLOCK - 1` padding, and it
            # also bounds the per-axis coordinates (each is smaller than the
            # index for N >= 1), so no lane can wrap below its own mask.
            index_proof = IndexBound(
                count=(f"({count}) + {config.tile_rows - 1} * ({output_shape[1]})"
                       f" + {config.tile_cols - 1}"),
                padding=0,
                leaves=tuple(layouts),
            )
        else:
            index_proof = IndexBound(
                count=(product(graph.values[reduction.inputs[0]].shape)
                       if reduction is not None else count),
                padding=index_padding,
                leaves=tuple(layouts),
            )
    index_dtype = emitted_index_dtype(config, proven=index_proof is not None)
    index_type = 'tl.int32' if index_dtype == 'int32' else 'tl.int64'
    if reduction:
        width = graph.values[reduction.inputs[0]].shape[-1]
        limit = config.reduction_block_limit or 128
        column_limit = min(limit, config.indexed_columns or limit)
        reduction_tile = names.fresh('FG_REDUCE_TILE')
        pairs.append((f'{reduction_tile}: tl.constexpr',
                      Argument('value', f'min({column_limit}, triton.next_power_of_2({width}))')))
        rows = names.fresh('FG_REDUCE_ROWS')
        # Bound the 2D tile and vectorize adjacent output elements. Reduction
        # coordinates occupy axis 0 so row-major leaves can coalesce along
        # output rows, instead of launching one program for every scalar.
        row_limit = config.indexed_rows or 16
        rows_expr = f'min({row_limit}, max(1, {limit} // min({column_limit}, triton.next_power_of_2({width}))))'
        pairs.append((f'{rows}: tl.constexpr', Argument('value', rows_expr)))
        lines = [f'    {idx} = tl.program_id(0).to({index_type}) * {rows} + tl.arange(0, {rows})',
                 f'    {valid} = {idx} < ({count})']
        grid = f'(triton.cdiv({count}, {rows_expr}),)'
        grid_axes = (f'triton.cdiv({count}, {rows_expr})',)
    elif row_tiled:
        height, width = output_shape
        tile_row, tile_col, row_valid = names.fresh('tile_row'), names.fresh('tile_col'), names.fresh('row_valid')
        block_expr = f'min({config.default_block_size}, triton.next_power_of_2({width}))'
        pairs[-1] = (f'{block}: tl.constexpr', Argument('value', block_expr))
        row_tiles = f'tl.cdiv({width}, {block})'
        lines = [f'    {tile_row} = tl.program_id(0).to({index_type}) // ({row_tiles})',
                 f'    {tile_col} = (tl.program_id(0).to({index_type}) % ({row_tiles})) * {block} + tl.arange(0, {block})',
                 f'    {idx} = {tile_row} * {width} + {tile_col}',
                 f'    {row_valid} = {tile_row} < {height}',
                 f'    {valid} = {row_valid} & ({tile_col} < {width})']
        grid = f'({height} * triton.cdiv({width}, {block_expr}),)'
        grid_axes = (f'{height} * triton.cdiv({width}, {block_expr})',)
        row_coordinates = [tile_row, tile_col]
    elif vector_tile:
        height, width = output_shape
        row_block = names.fresh('row_block')
        column_block = names.fresh('column_block')
        rows = names.fresh('rows')
        cols = names.fresh('cols')
        tile_rows = names.fresh('FG_BLOCK_M')
        tile_cols = names.fresh('FG_BLOCK_N')
        # The block constexpr is the last formal the flat path needs; the tile
        # replaces it with its two shape formals rather than adding a dead one.
        pairs[-1] = (f'{tile_rows}: tl.constexpr',
                     Argument('value', str(config.tile_rows)))
        pairs.append((f'{tile_cols}: tl.constexpr',
                      Argument('value', str(config.tile_cols))))
        lines = [
            f'    {row_block} = tl.program_id(0)',
            f'    {column_block} = tl.program_id(1)',
            f'    {rows} = {row_block} * {tile_rows} + tl.arange(0, {tile_rows})',
            f'    {cols} = {column_block} * {tile_cols} + tl.arange(0, {tile_cols})',
            f'    {idx} = {rows}[:, None] * {width} + {cols}[None, :]',
            f'    {valid} = ({rows}[:, None] < {height}) & ({cols}[None, :] < {width})',
        ]
        grid = (f'(triton.cdiv({height}, {config.tile_rows}), '
                f'triton.cdiv({width}, {config.tile_cols}))')
        grid_axes = (f'triton.cdiv({height}, {config.tile_rows})',
                     f'triton.cdiv({width}, {config.tile_cols})')
        tile_coordinates = (f'{rows}[:, None]', f'{cols}[None, :]')
    else:
        lines = [f'    {idx} = tl.program_id(0).to({index_type}) * {block} + tl.arange(0, {block})',
                 f'    {valid} = {idx} < ({count})']
        grid = f'(triton.cdiv({count}, {config.default_block_size}),)'
        grid_axes = (f'triton.cdiv({count}, {config.default_block_size})',)
    if half_pair is not None:
        # Both views are addressed off this one base: gate is `half_address`,
        # up is `half_address + N` -- a constant element difference for every
        # lane, because the caller's rows are `2N` wide and contiguous.
        half_width = str(graph.values[half_pair[0]].shape[-1])
        half_base = names.fresh('half_address')
        lines.append(f'    {half_base} = {idx} + ({idx} // ({half_width})) * ({half_width})')
    else:
        half_base = None
    # --- device-side knobs: mask elision and streaming hints ----------------
    # `idx < numel` is true for every lane exactly when the grid covers the
    # element count with no tail (`numel % block == 0`). The flag below is a
    # `tl.constexpr`, so Triton compiles one predicate-free binary per shape
    # while the general case keeps today's masked kernel: that is the second,
    # mask-free kernel a device measurement wants, with one emitter and one
    # launch entry point. `evict_first` is a pure hint (see PlatformConfig),
    # spelled with the Triton 3.3.1 keyword the vendored FlagGems uses.
    mask_elision = bool(config.mask_elision and reduction is None and not row_tiled
                        and not vector_tile and not config.paired_view_loads)
    tail_flag = None
    if mask_elision:
        tail_flag = names.fresh('MASK_TAIL')
        pairs.append((f'{tail_flag}: tl.constexpr',
                      Argument('value',
                               f'({count}) % {config.default_block_size} != 0')))
    hint = ", eviction_policy='evict_first'" if config.streaming_hints else ''
    from flowgpu.compiler.backend.paired_views import PairedViewLoads
    paired = PairedViewLoads(graph, tensors, result, layouts, ptrs, names, pairs,
                            enabled=config.paired_view_loads and reduction is None)
    cache = {}
    canonical_cache = {}
    computed = {}

    # Program-uniform scalars: one device load per program, emitted at the top
    # level of the body and used as a scalar from then on. A lowering that
    # instead re-derived the value per lane would make the write a gather and
    # the address non-affine; there is one way to emit it and this is it.
    program_scalar_values = {}
    program_scalar_bounds = []
    for node in group.members:
        spec = getattr(node.attributes, 'program_scalar', None)
        if spec is None:
            continue
        from flowgpu.compiler.backend.program_scalar import emit_load
        variable, bound = emit_load(node, graph, names, pairs, layouts, lines,
                                    grid_count=grid_axes)
        program_scalar_values[node.output] = variable
        program_scalar_bounds.append(bound)

    def emit(expr, hint, indent):
        var = names.fresh(hint)
        lines.append(f'{indent}{var} = {expr}')
        return var

    def emit_load(masked, unmasked, hint_name, indent='    '):
        """One external load, split on the constexpr tail flag when enabled."""
        variable = names.fresh(hint_name)
        if tail_flag is not None:
            lines.extend([f'{indent}if {tail_flag}:',
                          f'{indent}    {variable} = {masked}',
                          f'{indent}else:',
                          f'{indent}    {variable} = {unmasked}'])
        else:
            lines.append(f'{indent}{variable} = {masked}')
        return variable

    # In-kernel common subexpression elimination.
    #
    # A fused region can contain two nodes that compute the same expression
    # through different intermediate names -- an encoder written with helpers
    # that recompute `a = max(|v|, tiny)` at each use is the motivating case.
    # Both nodes emit identical device arithmetic, so the second one is a
    # register-level duplicate, not a memory boundary: hash-consing them inside
    # one kernel removes the recomputation *without* introducing a buffer (which
    # is what a graph-level merge would do, and what the fusion partitioner
    # deliberately avoids).
    def cse_eligible(node):
        # `attributes` is always an OperationAttributes object (its fields are
        # default for a pure elementwise node), so it is part of the canonical
        # key instead of being required to be None.  The eligibility set lives
        # in cse.py so the reference lowering uses the same one.
        return node is not None and cse.eligible(node.op)

    def canonical(v, at, mask, indent):
        key = ('canonical', v, at, mask, indent)
        if key in canonical_cache:
            return canonical_cache[key]
        node = produced.get(v)
        if not cse_eligible(node):
            answer = ('value', v, at, mask, indent)
        else:
            answer = (node.op,
                      tuple(canonical(arg, at, mask, indent) for arg in node.inputs),
                      graph.values[v].dtype, node.part, repr(node.attributes))
        canonical_cache[key] = answer
        return answer

    def element(v, at, mask, indent='    '):
        if v in scalar_formals:
            # A scalar is one number: it has no address, no mask and no load.
            return scalar_formals[v]
        if v in program_scalar_values:
            # One value per program; the one load was already emitted.
            return program_scalar_values[v]
        key = v, at, mask, indent
        if key in cache:
            return cache[key]
        node = produced.get(v)
        dtype = DTYPE_TO_TRITON[graph.values[v].dtype]
        if node is None:
            if half_pair is not None and v in half_pair:
                if at != idx or mask != valid or indent != '    ':
                    raise CompileError(
                        DiagnosticKind.INTERNAL,
                        'split_half_inputs requires an identity output index '
                        f'domain for its paired inputs; {v} is loaded at a '
                        'remapped index',
                        pipeline=graph.pipeline,
                    )
                combined_key = ('half', v)
                if combined_key in cache:
                    return cache[combined_key]
                offset = '' if v == half_pair[0] else f' + {half_width}'
                expression = f'tl.load({ptrs[half_pair[0]]} + {half_base}{offset}'
                if mask == valid and indent == '    ':
                    answer = emit_load(expression + f', mask={mask}, other=0{hint})',
                                       expression + f'{hint})',
                                       identifier(v) + '_load', indent)
                else:
                    answer = emit(expression + f', mask={mask}, other=0{hint})',
                                  identifier(v) + '_load', indent)
                cache[combined_key] = answer
                return answer
            if at == idx and mask == valid and indent == '    ':
                combined = paired.load(v, at, mask, lines, indent=indent,
                                       coordinates=row_coordinates)
                if combined is not None:
                    cache[key] = combined
                    return combined
            address_lines = []
            direct = at == idx and mask == valid and graph.values[v].shape == output_shape
            coords = row_coordinates if row_tiled and direct else (
                tile_coordinates if vector_tile and direct else None)
            address = _view_offset(v, at, layouts, names, address_lines, coordinates=coords,
                                   linearize=config.linear_view_indexing)
            lines.extend(indent + line[4:] for line in address_lines)
            expression = f'tl.load({ptrs[v]} + {address}'
            if mask == valid and indent == '    ':
                answer = emit_load(expression + f', mask={mask}, other=0{hint})',
                                   expression + f'{hint})',
                                   identifier(v) + '_load', indent)
            else:
                answer = emit(expression + f', mask={mask}, other=0{hint})',
                              identifier(v) + '_load', indent)
        else:
            op = node.op
            x = node.inputs[0]
            cse_key = canonical(v, at, mask, indent) if cse_eligible(node) else None
            if cse_key is not None and cse_key in computed:
                cache[key] = computed[cse_key]
                return computed[cse_key]
            if op in {'ones_like', 'zeros_like', 'neg_inf_like'}:
                constant = {'ones_like': '1', 'zeros_like': '0', 'neg_inf_like': "float('-inf')"}[op]
                expr = f'tl.full((), {constant}, {dtype})'
            elif op in {'reshape', 'cast_like', 'cast_to', 'to_fp32'}:
                expr = element(x, at, mask, indent)
            elif op == 'concat':
                # Last-axis concat as a view: the output column selects the
                # first operand inside its width, otherwise the second.
                left, right = node.inputs
                left_width = graph.values[left].shape[-1]
                right_width = graph.values[right].shape[-1]
                total = graph.values[v].shape[-1]
                # concat(split(x)[0], split(x)[1]) is exactly x.  Folding this
                # identity avoids a nested split->concat load composition that
                # KunlunXin miscomputes even though either op alone is correct.
                left_node, right_node = produced.get(left), produced.get(right)
                if (left_node is not None and right_node is not None
                        and left_node.op == 'split' and right_node.op == 'split'
                        and left_node.inputs == right_node.inputs
                        and left_node.part == 0 and right_node.part == 1
                        and graph.values[left].shape == graph.values[right].shape
                        and int(graph.values[left].shape[-1]) * 2 == int(total)):
                    expr = element(left_node.inputs[0], at, mask, indent)
                else:
                    row, col = names.fresh('row'), names.fresh('col')
                    lines.append(f'{indent}{row} = ({at}) // ({total})')
                    lines.append(f'{indent}{col} = ({at}) % ({total})')
                    left_mask = f'({mask}) & ({col} < {left_width})'
                    right_mask = f'({mask}) & ({col} >= {left_width})'
                    # Keep both operand addresses non-negative even on masked-off
                    # lanes: some backends do not honour a load mask for a negative
                    # address, and the concat's right operand is negative for half
                    # of every row.  Clamping the column changes only lanes that the
                    # branch mask excludes.
                    int_left, int_right = int(left_width), int(right_width)
                    left_source = (f'({row})*({left_width})'
                                   f'+tl.minimum({col}, {max(int_left - 1, 0)})')
                    right_source = (f'({row})*({right_width})'
                                    f'+tl.maximum(({col})-({left_width}), 0)')
                    expr = (f'tl.where({col} < {left_width}, '
                            f'{element(left, left_source, left_mask, indent)}, '
                            f'{element(right, right_source, right_mask, indent)})')
            elif op == 'interleave':
                # Last-axis interleave: output column parity selects the left
                # or right operand; both read row*width + column//2.
                left, right = node.inputs
                width = graph.values[left].shape[-1]
                total = graph.values[v].shape[-1]
                row, col = names.fresh('row'), names.fresh('col')
                lines.append(f'{indent}{row} = ({at}) // ({total})')
                lines.append(f'{indent}{col} = ({at}) % ({total})')
                even = f'({col} % 2 == 0)'
                left_mask = f'({mask}) & {even}'
                right_mask = f'({mask}) & (~({even}))'
                source = f'({row})*({width})+({col})//2'
                expr = (f'tl.where({even}, {element(left, source, left_mask, indent)}, '
                        f'{element(right, source, right_mask, indent)})')
            elif op == 'transpose':
                # Rank-2 transpose as a view: output [W,H] element (w,h) reads
                # input [H,W] at (h,w).
                left = node.inputs[0]
                height, width = graph.values[left].shape
                source = f'(({at}) % {height}) * {width} + ({at}) // {height}'
                expr = element(left, source, mask, indent)
            elif op == 'split':
                # A last-axis split is a view, not a copy: the source is the
                # same flat row-major space with twice the row width, so the
                # part's element `i` is source logical element
                #   i + (i // part_width) * part_width + part * part_width.
                # Only reachable when `split_views` opted the split into this
                # region (indexed_fusion merges it); the default path leaves
                # split as a layout anchor and never enters this branch.
                part_width = str(graph.values[v].shape[-1])
                source_width = _split_source_width(graph, node, part_width)
                logical = emit(
                    f'(({at}) // ({part_width})) * ({source_width})'
                    f' + (({at}) % ({part_width})) + {node.part} * ({part_width})',
                    'split_index', indent)
                expr = element(x, logical, mask, indent)
            elif op == 'move_first_last':
                old = graph.values[x].shape
                mapped = emit(f'(({at}) % {old[0]}) * ({product(old[1:])}) + ({at}) // {old[0]}', 'moved_index', indent)
                expr = element(x, mapped, mask, indent)
            elif op == 'broadcast_like':
                if row_tiled and v in row_broadcasts and at == idx and mask == valid and indent == '    ':
                    expr = element(x, tile_row, row_valid, indent)
                else:
                    mapped = emit(tile_offset(offset_for(graph.values[x].shape, graph.values[v].shape, at), at), 'broadcast_index', indent)
                    expr = element(x, mapped, mask, indent)
            elif op == 'gather_row_offset':
                column = element(node.inputs[1], at, mask, indent)
                total = product(graph.values[x].shape)
                logical = emit(f'({at}).to(tl.int64) * {graph.values[x].shape[1]} + {column}.to(tl.int64)', 'gather_index', indent)
                lines.append(f'{indent}{logical} = tl.where({logical} < 0, {logical} + ({total}), {logical})')
                bounds = emit(f'({mask}) & ({logical} >= 0) & ({logical} < ({total}))', 'gather_valid', indent)
                gathered = element(x, logical, bounds, indent)
                # A masked leaf load is insufficient for a fused source:
                # exp(0), ones_like, etc. could otherwise turn OOB into nonzero.
                expr = f'tl.where({bounds}, {gathered}, 0)'
            elif op == 'gather_rows':
                width = product(graph.values[x].shape[1:])
                row = emit(f'({at}) // ({width})', 'output_row', indent)
                selected = element(node.inputs[1], row, mask, indent)
                selected = emit(f'{selected}.to(tl.int64)', 'selected_row', indent)
                height = graph.values[x].shape[0]
                lines.append(f'{indent}{selected} = tl.where({selected} < 0, {selected} + {height}, {selected})')
                bounds = emit(f'({mask}) & ({selected} >= 0) & ({selected} < {height})', 'row_valid', indent)
                logical = emit(f'{selected} * ({width}) + ({at}) % ({width})', 'row_address', indent)
                gathered = element(x, logical, bounds, indent)
                expr = f'tl.where({bounds}, {gathered}, 0)'
            elif op in REDUCTIONS:
                width = graph.values[x].shape[-1]
                tile = reduction_tile
                col, step = names.fresh('col'), names.fresh('step')
                answer = names.fresh('reduced')
                init = '0.0' if op == 'sum_last' else "float('-inf')"
                # KunlunXin's Triton legaliser miscompiles an axis-0 reduction
                # over a 2-D tile.  Its platform profile sets indexed_rows=1 so
                # this path can keep the reduction tile rank-1; expand the
                # single output row explicitly and reduce the flattened tile.
                rank1_reduction = getattr(config, "indexed_rows", None) == 1
                if rank1_reduction:
                    # One output row per program: keep the reduction tile as
                    # [1, TILE] and reduce the LAST axis.  KunlunXin's legaliser
                    # rejects/miscompiles an axis-0 reduction over a 2-D tile;
                    # a [1, TILE] tile with axis=1 avoids that shape entirely
                    # (and avoids a reshape, which has its own backend caveats).
                    lines.extend([f'{indent}{col} = tl.arange(0, {tile})[None, :]',
                                  f'{indent}{answer} = tl.full(({rows},), {init}, tl.float32)',
                                  f'{indent}for {step} in range(tl.cdiv({width}, {tile})):'])
                    inner = indent + '    '
                    logical = emit(f'({at})[:, None] * {width} + {step} * {tile} + {col}', 'reduce_index', inner)
                    column_valid = emit(f'({mask})[:, None] & ({step} * {tile} + {col} < {width})', 'reduce_valid', inner)
                    value = element(x, logical, column_valid, inner)
                    value = emit(f'tl.where({column_valid}, {value}.to(tl.float32), {init})', 'reduce_tile', inner)
                    if op == 'sum_last':
                        lines.append(f'{inner}{answer} = {answer} + tl.sum({value}, 1)')
                    else:
                        maximum = emit(f'tl.max({value}, 1)', 'tile_max', inner)
                        nan = (f'({answer} != {answer}) | '
                               f'(tl.sum(({value} != {value}).to(tl.int32), 1) > 0)')
                        lines.append(f'{inner}{answer} = tl.where({nan}, float("nan"), '
                                     f'tl.maximum({answer}, {maximum}))')
                else:
                    lines.extend([f'{indent}{col} = tl.arange(0, {tile})[:, None]',
                                  f'{indent}{answer} = tl.full(({rows},), {init}, tl.float32)',
                                  f'{indent}for {step} in range(tl.cdiv({width}, {tile})):'])
                    inner = indent + '    '
                    logical = emit(f'({at})[None, :] * {width} + {step} * {tile} + {col}', 'reduce_index', inner)
                    column_valid = emit(f'({mask})[None, :] & ({step} * {tile} + {col} < {width})', 'reduce_valid', inner)
                    value = element(x, logical, column_valid, inner)
                    value = emit(f'tl.where({column_valid}, {value}.to(tl.float32), {init})', 'reduce_tile', inner)
                    if op == 'sum_last':
                        lines.append(f'{inner}{answer} = {answer} + tl.sum({value}, 0)')
                    else:
                        maximum = emit(f'tl.max({value}, 0)', 'tile_max', inner)
                        nan = (f'({answer} != {answer}) | '
                               f'(tl.sum(({value} != {value}).to(tl.int32), 0) > 0)')
                        lines.append(f'{inner}{answer} = tl.where({nan}, float("nan"), '
                                     f'tl.maximum({answer}, {maximum}))')
                cache[key] = answer
                return answer
            else:
                if op == 'add_broadcast':
                    raise CompileError(
                        DiagnosticKind.OPERATOR,
                        "add_broadcast is emitted inside a fused scan epilogue "
                        "only; this region was lowered as a plain pointwise "
                        "kernel, where the row operand has no address mapping",
                        pipeline=graph.pipeline)
                args = [element(arg, at, mask, indent) for arg in node.inputs]
                if op in POINTWISE_EXPRESSION:
                    expr = _pointwise_expression(node, args, graph.values[v].dtype)
                elif op == 'where':
                    expr = f'tl.where({args[0]}, {args[1]}, {args[2]})'
                elif op == 'div':
                    expr = f'{args[0]}.to(tl.float32) / {args[1]}.to(tl.float32)'
                elif op == 'isfinite':
                    expr = f'({args[0]} == {args[0]}) & (tl.abs({args[0]}) != float("inf"))'
                elif op in VALUE_LAYER:
                    from flowgpu.compiler.backend.value_layer import triton_expression
                    expr = triton_expression(op, tuple(args), sigmoid_impl=config.sigmoid_impl)
                else:
                    expr = f'tl.{op}({args[0]}.to(tl.float32))'
            if op == 'split':
                # A split view keeps the source's dtype exactly: an identity
                # `.to(dtype)` here would be a no-op that still makes otherwise
                # identical kernels differ per input dtype, which blocks the
                # ABI dedupe (9 kernels instead of 3 in the Task160 package).
                answer = names.fresh(identifier(v) + '_value')
                lines.append(f'{indent}{answer} = {expr}')
            elif graph.values[v].dtype == 'bool':
                # A comparison already yields i1.  The `.to(tl.int1)` wrap is a
                # no-op on CUDA but an explicit i1 cast is a construct the Ascend
                # compiler's ConvertLinalgIRToBinary step is willing to reject, so
                # the redundant cast is not emitted at all.
                answer = emit(expr, identifier(v) + '_value', indent)
            else:
                # Fusion removes a buffer, not a rounding boundary.
                answer = emit(f'({expr}).to({dtype})', identifier(v) + '_value', indent)
            if cse_key is not None:
                computed[cse_key] = answer
        cache[key] = answer
        return answer

    answer = element(result, idx, valid)
    store_masked = f'tl.store({ptrs[result]} + {idx}, {answer}, mask={valid}{hint})'
    if tail_flag is not None:
        lines.extend([f'    if {tail_flag}:',
                      f'        {store_masked}',
                      '    else:',
                      f'        tl.store({ptrs[result]} + {idx}, {answer}{hint})'])
    else:
        lines.append('    ' + store_masked)
    if vector_tile:
        # Two launch axes, so the cap and the stride loop are the 2-D pair. The
        # product cap keeps the exact tile grid while it fits under the target's
        # `coreDim` ceiling, which is what makes the tile *shape* the only
        # variable against the flat kernel (same programs, same elements each).
        from flowgpu.compiler.backend.triton_backend import (capped_grid_2d_product,
                                                    grid_limited_body_2d)
        lines, grid = grid_limited_body_2d(
            lines, grid, config,
            rows_of_blocks=f'tl.cdiv({output_shape[0]}, {tile_rows})',
            columns_of_blocks=f'tl.cdiv({output_shape[1]}, {tile_cols})',
            row=row_block, column=column_block,
            host=(f'triton.cdiv({output_shape[0]}, {config.tile_rows})',
                  f'triton.cdiv({output_shape[1]}, {config.tile_cols})'),
            cap=capped_grid_2d_product)
    elif not reduction and not row_tiled:
        # A target that refuses a grid beyond its hardware extent (Ascend
        # coreDim 65535) gets the same one-tile loop as the pointwise emitter.
        from flowgpu.compiler.backend.triton_backend import grid_limited_body
        lines, grid = grid_limited_body(lines, grid, config, count=count, numel=count,
                                        block=block, grid_block=str(config.default_block_size),
                                        tile=names.fresh('tile'))
    elif reduction:
        # A reduction whose tile leaves one row per program has a grid equal to
        # the row count, which Ascend cannot launch above 65535. Stride rows.
        from flowgpu.compiler.backend.triton_backend import grid_limited_body
        lines, grid = grid_limited_body(lines, grid, config, count=count, numel=count,
                                        block=rows, grid_block=rows_expr,
                                        tile=names.fresh('tile'))
    kernel = _make_kernel(name, 'primitive', group, pairs, lines, grid, graph,
                          contiguity_contracts=contracted,
                         launch_options=(('num_warps', config.num_warps), ('enable_fp_fusion', False)),
                         index_dtype=index_dtype, index_bound=index_proof)
    if program_scalar_bounds:
        from dataclasses import replace as _replace
        kernel = _replace(kernel,
                          program_scalar_bounds=tuple(program_scalar_bounds))
    if autotune_fold and reduction:
        from flowgpu.compiler.backend.autotune import decorate_fold
        kernel = decorate_fold(kernel, config, 'indexed_reduction', (reduction_tile,rows),
                               (str(width), count))
        # The fixed grid used a launch-time expression, not the ROWS parameter.
        from dataclasses import replace
        from flowgpu.compiler.backend.triton_backend import grid_ceiling, capped_grid
        limit=grid_ceiling(config)
        folded=f"triton.cdiv({count}, META[{rows!r}])"
        kernel = replace(kernel,
            grid=f"lambda META: {capped_grid(folded, limit) if limit is not None else f'({folded},)'}")
    return kernel


def reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _kernel, _dimension_parameters, _rounded, _expression, POINTWISE_EXPRESSION
    from flowgpu.compiler.backend.value_layer import NAMES as VALUE_LAYER
    operands, results, produced = _parts(graph, group)
    names = Names(graph)
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    variables = {v: names.fresh(identifier(v) + '_data') for v in tensors + results}
    functions = {v: names.fresh(identifier(v) + '_element') for v in produced}
    pairs = [(variables[v], Argument('tensor', v)) for v in tensors + results]
    used_formals = ({parameter for parameter, _ in pairs}
                    | dimension_symbols(graph))
    scalar_formals = {}
    for v in scalars:
        formal = unique_formal(identifier(v), used_formals)
        names.reserve(formal)
        scalar_formals[v] = formal
        pairs.append((formal, Argument('scalar', v)))
    _dimension_parameters(names, pairs, [d for v in tensors + tuple(produced) for d in graph.values[v].shape])
    i, j = names.fresh('i'), names.fresh('j')
    lines = []
    memo = None
    if group.reduction_domain:
        memo = names.fresh('memoized')
        lines.append(f'    from functools import lru_cache as {memo}')

    def at(v, offset=i):
        if v in scalar_formals:
            # The number itself, not a list element.
            return scalar_formals[v]
        return f'{functions[v]}({offset})' if v in produced else f'{variables[v]}[{offset}]'

    from flowgpu.compiler.backend import cse
    aliases = {}
    for node in group.members:
        v, x, op = node.output, node.inputs[0], node.op
        # A value whose expression is identical to an earlier one shares that
        # earlier element function; the two backends then compute the same set of
        # values, and the reference path keeps mirroring the device path.
        duplicate = None
        if cse.ENABLED:
            values = [at(v) for v in node.inputs]
            cse_key = cse.key(op, values, graph.values[v].dtype, node.part,
                              node.attributes)
            if cse_key in aliases:
                functions[v] = aliases[cse_key]
                continue
            aliases[cse_key] = functions[v]
        if memo:
            lines.append(f'    @{memo}(maxsize=None)')
        lines.append(f'    def {functions[v]}({i}):')
        a = at(x)
        if op == 'interleave':
            left, right = node.inputs
            width = graph.values[left].shape[-1]
            total = graph.values[v].shape[-1]
            col = f'({i} % {total})'
            row = f'({i} // {total})'
            source = f'{row} * {width} + {col} // 2'
            expr = f'({at(left, source)} if {col} % 2 == 0 else {at(right, source)})'
        elif op == 'transpose':
            left = node.inputs[0]
            height, width = graph.values[left].shape
            source = f'({i} % {height}) * {width} + {i} // {height}'
            expr = at(left, source)
        elif op == 'concat':
            left, right = node.inputs
            left_width = graph.values[left].shape[-1]
            right_width = graph.values[right].shape[-1]
            total = graph.values[v].shape[-1]
            col = f'({i} % {total})'
            row = f'({i} // {total})'
            expr = f'({at(left, f"{row} * {left_width} + {col}")} if {col} < {left_width} else {at(right, f"{row} * {right_width} + ({col} - {left_width})")})'
        elif op == 'split':
            part_width = str(graph.values[v].shape[-1])
            source_width = _split_source_width(graph, node, part_width)
            logical = (f'(({i}) // ({part_width})) * ({source_width})'
                       f' + (({i}) % ({part_width})) + {node.part} * ({part_width})')
            expr = at(x, logical)
        elif op in {'reshape', 'cast_like', 'cast_to', 'to_fp32'}:
            expr = a
        elif op == 'move_first_last':
            old = graph.values[x].shape
            expr = at(x, f'({i} % {old[0]}) * ({product(old[1:])}) + {i} // {old[0]}')
        elif op == 'broadcast_like':
            expr = at(x, offset_for(graph.values[x].shape, graph.values[v].shape, i))
        elif op == 'gather_row_offset':
            logical = names.fresh('logical'); total = product(graph.values[x].shape)
            lines.extend([f'        {logical} = {i} * {graph.values[x].shape[1]} + {at(node.inputs[1])}',
                          f'        if {logical} < 0: {logical} += {total}'])
            expr = f'({at(x, logical)} if 0 <= {logical} < {total} else 0)'
        elif op == 'gather_rows':
            width = product(graph.values[x].shape[1:])
            height = graph.values[x].shape[0]
            selected = names.fresh('selected_row')
            lines.extend([f'        {selected} = {at(node.inputs[1], f"{i} // ({width})")}',
                          f'        if {selected} < 0: {selected} += {height}'])
            expr = f'({at(x, f"{selected} * ({width}) + {i} % ({width})")} if 0 <= {selected} < {height} else 0)'
        elif op in REDUCTIONS:
            width = graph.values[x].shape[-1]
            expr = f'sum({at(x, f"{i} * {width} + {j}")} for {j} in range({width}))'
            if op == 'max_last':
                vals = names.fresh('row_values')
                lines.append(f'        {vals} = [{at(x, f"{i} * {width} + {j}")} for {j} in range({width})]')
                expr = f'(math.nan if any(math.isnan(v) for v in {vals}) else max({vals}))'
        elif op in {'ones_like', 'zeros_like', 'neg_inf_like'}:
            expr = {'ones_like': '1', 'zeros_like': '0', 'neg_inf_like': '-math.inf'}[op]
        elif op == 'add_broadcast':
            raise CompileError(
                DiagnosticKind.OPERATOR,
                "add_broadcast is emitted inside a fused scan epilogue only",
                pipeline=graph.pipeline)
        elif op in POINTWISE_EXPRESSION:
            expr = _expression(node, [at(v) for v in node.inputs])
        elif op == 'where':
            expr = f'({at(node.inputs[1])} if {a} else {at(node.inputs[2])})'
        elif op == 'isfinite':
            expr = f'math.isfinite({a})'
        elif op == 'div':
            expr = f'ieee_div({a}, {at(node.inputs[1])})'
        elif op in {'log', 'log2'}:
            expr = f'(math.{op}({a}) if {a} > 0 else (-math.inf if {a} == 0 else math.nan))'
        elif op in VALUE_LAYER:
            from flowgpu.compiler.backend.value_layer import reference_expression
            expr = reference_expression(op, tuple(at(v) for v in node.inputs))
        else:
            expr = f'safe_exp({a}, base2={op == "exp2"})'
        lines.append('        return ' + _rounded(graph, v, expr))
    for result in results:
        lines.extend([f'    for {i} in range({product(graph.values[result].shape)}):',
                      f'        {variables[result]}[{i}] = {at(result)}'])
    return _kernel(name, 'primitive', group, operands, results, pairs, lines)
