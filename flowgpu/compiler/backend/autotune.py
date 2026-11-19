"""Platform-scoped Triton autotune candidates.

This is deliberately a small, explicit policy table. A platform is only
allowed to autotune when its candidate space is published; the table
is not a claim that any candidate is optimal there. The CUDA list is the
A800 PoC space. Kunlunxin is conservatively pruned by the observed uni_sram
symptom (from reductions, not GEMM): no tile dimension above 64, no BLOCK_K above 32, and no 3-stage
pipeline. Other platforms refuse autotune rather than inheriting guesses.
"""

CUDA_CONFIGS = (
    (16, 64, 32, 2, 2),
    (32, 64, 32, 2, 2),
    (64, 64, 32, 2, 4),
    (64, 64, 32, 3, 4),
    (64, 128, 32, 2, 4),
    (64, 128, 32, 3, 8),
    (128, 64, 32, 2, 4),
    (128, 128, 32, 2, 8),
    (128, 128, 32, 3, 8),
    (128, 256, 32, 3, 8),
    (256, 128, 32, 3, 8),
    (128, 128, 64, 3, 8),
)

KUNLUNXIN_CONFIGS = (
    (16, 32, 32, 2, 2),
    (32, 32, 32, 2, 2),
    (64, 64, 32, 2, 4),
)


def prune_configs(platform, configs):
    """Resource-oriented search-space guard, not a compile-safe guarantee."""
    if platform == "kunlunxin":
        return tuple(c for c in configs if c[0] <= 64 and c[1] <= 64 and c[2] <= 32 and c[3] <= 2)
    return tuple(configs)


def configs_for(platform):
    if platform == "cuda":
        return CUDA_CONFIGS
    if platform == "kunlunxin":
        return prune_configs(platform, KUNLUNXIN_CONFIGS)
    return ()


def autotune_reason(platform):
    if platform == "cuda":
        return "A800 PoC candidate space"
    if platform == "kunlunxin":
        return "conservative uni_sram-pruned space; GEMM compilation and performance unverified"
    return "no compile-safe measured candidate space published"


def render_configs(platform):
    configs = configs_for(platform)
    if not configs:
        raise ValueError(
            f"--autotune is unavailable for platform '{platform}': "
            f"{autotune_reason(platform)}"
        )
    lines = ["FLOWGPU_AUTOTUNE_CONFIGS = ["]
    for m, n, k, stages, warps in configs:
        lines.append(
            "    triton.Config({"
            f"'BLOCK_M': {m}, 'BLOCK_N': {n}, 'BLOCK_K': {k}"
            f"}}, num_stages={stages}, num_warps={warps}),"
        )
    lines.append("]")
    return "\n".join(lines)


def fold_configs(config, family):
    """Explicit CUDA reduction search space; no GEMM speedup is assumed.

    Product bounds are logical lane budgets, not occupancy/SRAM predictions.
    Other targets retain their fixed emitters until device legality is known.
    """
    if config.name != 'cuda':
        raise ValueError('fold autotune currently requires CUDA device evidence')
    limit = config.reduction_block_limit or 4096
    if family == 'rowwise':
        candidates = ((32,1,1), (64,1,2), (256,1,4), (256,3,4),
                      (1024,1,4), (1024,3,4), (4096,1,4),
                      (4096,2,4), (4096,3,4), (4096,3,8))
        return tuple(c for c in candidates if c[0] <= limit)
    if family == 'indexed_reduction':
        candidates = ((32,1,1,1), (64,4,1,4), (128,4,1,4),
                      (256,4,1,4), (512,4,1,4), (1024,4,1,4),
                      (64,16,1,4), (256,8,1,4), (256,8,3,4), (1024,4,3,4))
        columns = min(limit, config.indexed_columns or limit)
        rows = config.indexed_rows or 16
        return tuple(c for c in candidates if c[0] <= columns and c[1] <= rows
                     and c[0]*c[1] <= limit)
    raise ValueError(f'unknown fold autotune family: {family}')


def cache_options_source():
    """Negotiate disk timing-cache support without changing the tuning search.

    Triton 3.3.1 has an in-process autotuner but no ``cache_results`` keyword.
    Newer runtimes accept it. Omitting an unsupported optional keyword keeps
    autotuning available; it does not imply persistent timings on that runtime.
    """
    return '''\nimport inspect as _fg_inspect
_FG_AUTOTUNE_CACHE_KWARGS = (
    {'cache_results': True}
    if 'cache_results' in _fg_inspect.signature(triton.autotune).parameters
    else {}
)
'''


def decorate_fold(kernel, config, family, knobs, shape_expressions):
    """Let the autotuner own constexpr knobs, grid and launch resources.

    Move tuned parameters to the end of the *signature*: simply deleting
    positional launch arguments would shift later layout/dimension arguments.
    Every non-tuned value/layout argument participates in the shape cache key.
    """
    from dataclasses import replace
    import ast
    from flowgpu.compiler.backend.backends import Argument
    candidates = fold_configs(config, family)
    if not candidates:
        raise ValueError('no fold autotune configurations fit the explicit budget')
    pairs = list(zip(kernel.parameters, kernel.launch))
    names = {p.split(':')[0].strip() for p in kernel.parameters}
    def fresh(hint):
        name = hint
        while name in names: name += '_'
        names.add(name)
        return name
    tuned = [(p,a) for p,a in pairs if p.split(':')[0].strip() in knobs]
    kept = [(p,a) for p,a in pairs if p.split(':')[0].strip() not in knobs]
    if len(tuned) != len(knobs):
        raise ValueError('autotune knob does not identify one kernel parameter')
    for i, expression in enumerate(shape_expressions):
        name = fresh('FG_TUNE_SHAPE'+str(i))
        kept.append((name+': tl.constexpr', Argument('value', expression)))
    device = fresh('FG_TUNE_DEVICE')
    kept.append((device+': tl.constexpr', Argument('device_key','')))
    keys = [p.split(':')[0].strip() for p,a in kept if a.kind != 'tensor']
    table = ['(']
    for candidate in candidates:
        values, stages, warps = candidate[:-2], candidate[-2], candidate[-1]
        params = dict(zip(knobs, values))
        table.append(f'    triton.Config({params!r}, num_stages={stages}, num_warps={warps}),')
    table.append(')')
    decorator = '@triton.autotune(configs='+ '\n'.join(table)+f', key={keys!r}, **_FG_AUTOTUNE_CACHE_KWARGS)\n@triton.jit'
    signature = f'def {kernel.name}('+', '.join(p for p,a in kept+tuned)+'):'
    source = kernel.source.replace(kernel.signature, signature).replace('@triton.jit',decorator,1)
    grid = kernel.grid
    # Replace knob identifiers in the grid through AST, never substring swaps.
    class Meta(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in knobs:
                return ast.Subscript(value=ast.Name(id='META',ctx=ast.Load()),
                                     slice=ast.Constant(node.id),ctx=ast.Load())
            return node
    grid = ast.unparse(ast.fix_missing_locations(Meta().visit(ast.parse(grid,mode='eval'))))
    return replace(kernel, source=source, parameters=tuple(p for p,a in kept+tuned),
                   launch=tuple(a for p,a in kept), grid='lambda META: '+grid,
                   launch_options=tuple((k,v) for k,v in kernel.launch_options
                                        if k not in {'num_warps','num_stages'}),
                   autotune_family=family)
