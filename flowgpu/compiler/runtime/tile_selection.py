"""Choose a kernel geometry by measuring, because no geometry wins on every shape.

The emitted tile is a launch-time ``tl.constexpr``, so one compiled module can be
called at several geometries. What a caller gets from ``run_lse`` is whichever
geometry the platform config resolves to, and that resolution is a poor default
for this operator: measured on an H100 PCIe, the recompute strategy at its default
caps walks the reduction in ``cdiv(R, 128)`` steps and reaches 40 us where a
tuned tile reaches 8.8 us, and 154 us on a long-R shape where ``materialize``
reaches 29.5 us.

Which geometry is best depends on the shape, and the reason is arithmetic rather
than noise: the width is ``min(dcap, budget // row_tile, next_power_of_2(D))``, so
the same candidate produces ``cdiv(1024, 16) = 64`` programs at ``D=1024`` and
``cdiv(64, 16) = 4`` at ``D=64``. Candidates differ mainly in how finely they cut
``D``, so the best cut is a function of ``D``. The measured winner across five
shapes was ``d16`` once and ``d8`` four times, and on a long-R shape the plain
``materialize`` path beat every recompute tile.

So the space keeps both strategies: a measured selection that can only search
recompute tiles would miss the shapes where the other strategy is right.

This module holds the part that needs no device: what to try, and how to turn a
chosen geometry back into options. Measuring lives with the caller, because a
timing needs tensors.
"""
import ast
import re

from flowgpu.compiler.backend.platform_configs import resolve_platform

# resolve_platform's untuned caps, and they are not symmetric: 128 rows by 32
# columns. They are named here because the emitted expression is
# min(dcap, budget // row_cap, next_power_of_2(D)) and the roles are easy to swap.
DEFAULT_ROW_CAP = 128
DEFAULT_COLUMN_CAP = 32
DEFAULT_LIMIT = 4096

MATERIALIZE = 'materialize'

# Every arm here has been fastest on at least one measured shape. ``materialize``
# is the out-of-box program: passing no cross-domain strategy at all emits
# byte-identical source, so ``default`` is accepted as an alias and names the same
# arm rather than a second one. It is in the space because a recompute-only search
# would miss the shape where it wins.
SPACE = ('materialize',
         'recompute', 'recompute-r4096-d8-budget32768',
         'recompute-r4096-d16-budget65536', 'recompute-r2048-d16-budget65536')


def _pow2(value):
    return 1 << (int(value) - 1).bit_length() if value > 1 else 1


def tile_from_caps(columns, rows, budget, warps, row_extent, column_extent):
    """The (row_tile, column_tile, warps) a configuration emits for these extents.

    Mirrors the launch expression emitted by flowgpu/compiler/backend/cross_domain_backend.py.
    test/test_tile_selection.py reads these values back out of the emitted source
    and compares, so this cannot drift from the emitter unnoticed.

    The extents are named rather than positional on purpose: which axis is reduced
    is a property of the program, not of argument order, and guessing it from
    ``args[0]`` was wrong for every pipeline whose first input is not the reduced
    one.
    """
    row_cap = columns if columns is not None else DEFAULT_ROW_CAP
    limit = budget if budget is not None else DEFAULT_LIMIT
    column_cap = rows if rows is not None else DEFAULT_COLUMN_CAP
    row_tile = min(row_cap, _pow2(row_extent))
    column_tile = min(column_cap, max(1, limit // row_tile), _pow2(column_extent))
    return row_tile, column_tile, warps if warps is not None else 4


def tile_from_config(columns, rows, budget, warps, shape):
    """Convenience form for a shape whose first axis is reduced and last is wide."""
    r, b, h, d = shape
    return tile_from_caps(columns, rows, budget, warps, r, d)


def dimension_bindings(code, pipeline):
    """{dimension: (parameter, axis)} as the emitted host function binds them."""
    tree = ast.parse(code)
    host = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'run_' + pipeline), None)
    if host is None:
        raise ValueError('no host function run_%s in the emitted source' % pipeline)
    params = [a.arg for a in host.args.args]
    bindings = {}
    for node in ast.walk(host):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute)
                and value.value.attr == 'shape' and isinstance(value.value.value, ast.Name)
                and isinstance(value.slice, ast.Constant)
                and isinstance(value.slice.value, int)):
            bindings[node.targets[0].id] = (value.value.value.id, value.slice.value)
    return bindings, params


def dimension_values(code, pipeline, args):
    """{dimension: extent} for these tensors, read from the emitted bindings.

    Nothing is assumed about argument order: the emitted source says which
    parameter and axis each dimension comes from, and that is what is evaluated.
    """
    bindings, params = dimension_bindings(code, pipeline)
    if len(params) != len(args):
        raise ValueError('the emitted host wants %d tensors but %d were given'
                         % (len(params), len(args)))
    position = {name: i for i, name in enumerate(params)}
    values = {}
    for dim, (parameter, axis) in bindings.items():
        values[dim] = int(args[position[parameter]].shape[axis])
    return values


def launched_kernel(code):
    """The one kernel this module launches, or a refusal to guess.

    Selection is per kernel, so a module that launches several cannot be read
    without being told which; a module that launches none has nothing to select.
    """
    names = []
    for line in code.splitlines():
        match = re.match(r'\s*(\w+)\[', line)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    names = [n for n in names if n not in ('tl', 'triton')]
    if len(names) != 1:
        raise ValueError('expected one launched kernel, found %d: %s'
                         % (len(names), ', '.join(names) or 'none'))
    return names[0]


def tile_dimensions(code, kernel=None):
    """(reduced dimension, wide dimension) named by the emitted tile expressions.

    Taken from the two positional tile arguments only: the grid expression also
    mentions the wide dimension, and reading the whole line would attribute the
    reduced axis to it.

    The kernel is discovered rather than named, so this applies to any pipeline
    with a single launched kernel instead of only to the operator it was written
    for.
    """
    if kernel is None:
        kernel = launched_kernel(code)
    line = next((l for l in code.splitlines() if kernel + '[' in l), None)
    if line is None:
        raise ValueError('no %s launch in the emitted source' % kernel)
    arguments = split_arguments(line[line.index('](') + 2:line.rindex(')')])
    positional = [p for p in arguments
                  if not p.startswith('num_warps=') and not p.startswith('enable_fp_fusion=')]
    row_expr, column_expr = positional[-2], positional[-1]
    row = re.findall(r'next_power_of_2\((\w+)\)', row_expr)
    column = re.findall(r'next_power_of_2\((\w+)\)', column_expr)
    if not row or not column:
        raise ValueError('the tile expressions do not name their dimensions')
    if row[0] == column[-1]:
        # Under materialize the column cap is applied without a wide-dimension
        # term, so the only dimension named is the reduced one. That is the
        # structural difference between a cross-domain tile and a plain one, and
        # it is what makes it possible to say "nothing to select" without a
        # strategy table.
        raise ValueError('no cross-domain tile in this program: both tile caps name %r'
                         % row[0])
    return row[0], column[-1]


def tile_options(tile):
    """Options whose emitted geometry is exactly ``tile``.

    The row cap is ``min(rcap, next_power_of_2(R))`` and the width is
    ``min(dcap, budget // row_cap, next_power_of_2(D))``, so naming the geometry as
    caps and setting the budget to their product reproduces it; ``dcap`` covers
    the case where ``rcap`` exceeds ``next_power_of_2(R)``.
    """
    row_tile, column_tile, warps = tile
    return {'cross_domain_strategy': 'recompute',
            'platform_config': resolve_platform(
                indexed_columns=row_tile, indexed_rows=column_tile,
                reduction_block_limit=row_tile * column_tile, num_warps=warps)}


def _recompute_table():
    from flowgpu.compiler.optimization.strategy import candidates
    return {spec.name: spec for spec in candidates(recompute=True, large_tiles=True,
                                                   wide_columns=True)}


def space_for(shape, names=SPACE):
    """Convenience form: the space for a shape reduced on its first axis."""
    return space_for_extents(shape[0], shape[3], names=names)


def space_for_extents(row_extent, column_extent, names=SPACE):
    """[(label, options, geometry)] for these extents, with duplicate work removed.

    ``geometry`` is None for arms whose program structure is not a tile of the
    recompute kernel; those are measured through their own module.
    """
    table = _recompute_table()
    found = []
    for name in names:
        if name in ('default', MATERIALIZE):
            found.append((MATERIALIZE, {'cross_domain_strategy': MATERIALIZE}, None))
        elif name == 'recompute':
            found.append(('recompute', {'cross_domain_strategy': 'recompute'},
                          tile_from_caps(None, None, None, None, row_extent, column_extent)))
        elif name in table:
            spec = table[name]
            geometry = tile_from_caps(spec.columns, spec.rows, spec.budget, spec.warps,
                                      row_extent, column_extent)
            found.append((name, spec.options(), geometry))
        else:
            raise ValueError('unknown candidate: ' + name)
    # Two names can name the same geometry -- retile-ing 2048 rows of D into a
    # 512-row cap makes r2048-d16 and r4096-d16 identical. Measuring one of them
    # twice would present the difference between two timings of the same work as
    # a difference between two configurations.
    deduped, seen = [], set()
    for label, options, geometry in found:
        identity = geometry if geometry is not None else ('opaque', label)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append((label, options, geometry))
    return deduped


def options_for_arm(label, shape=None, extents=None):
    """Recover an arm's options from its name alone.

    A recorded selection stores the arm's name, not its options: the mapping from
    name to configuration lives in code, so storing it again in a cache file would
    create a second source of truth that could disagree with this one.
    """
    space = (space_for_extents(extents[0], extents[1]) if extents is not None
             else space_for(shape))
    for found, options, _ in space:
        if found == label:
            return dict(options)
    raise KeyError('no such arm for this program: %s' % label)


def split_arguments(text):
    """Split a call argument list on top-level commas."""
    parts, depth, current = [], 0, ''
    for char in text:
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        if char == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


class _NextPowerOfTwo:
    @staticmethod
    def next_power_of_2(value):
        return 1 << (int(value) - 1).bit_length() if value > 1 else 1


def emitted_tiles(code, values, kernel='lse_k0'):
    """(row_tile, column_tile, num_warps) as the emitted launch computes them.

    Evaluated from the emitted text rather than recomputed, because the question
    is what the emitter produced, not what it should have. A measured choice that
    the emitter renders as a different geometry is the failure this whole area
    exists to catch, so callers verify emission instead of assuming it.
    """
    line = next(l for l in code.splitlines() if kernel + '[' in l)
    arguments = line[line.index('](') + 2:line.rindex(')')]
    parts = split_arguments(arguments)
    warps = int(next(p for p in parts if p.startswith('num_warps=')).split('=')[1])
    positional = [p for p in parts
                  if not p.startswith('num_warps=') and not p.startswith('enable_fp_fusion=')]
    if not isinstance(values, dict):
        r, b, h, d = values
        values = dict(R=r, B=b, H=h, D=d)
    namespace = dict(min=min, max=max, triton=_NextPowerOfTwo,
                     **{k: int(v) for k, v in values.items()})
    return (eval(positional[-2], namespace),      # noqa: S307 - emitter-generated arithmetic
            eval(positional[-1], namespace),      # noqa: S307
            warps)


def assert_emitted_tiles(code, values, expected, kernel='lse_k0'):
    """Fail when the launch the emitter wrote is not the configuration requested.

    A candidate's options can be dropped without any error: an emitter that does
    not understand them, or one where the caps are overridden further down, still
    produces a working kernel at some other geometry. Two measurement campaigns in
    this repository were invalidated by exactly that, and in both cases the only
    visible symptom was a table of times that said the candidates were slower.

    The check reads the emitted text rather than trusting the request, so it holds
    whatever the emitter does with the numbers.
    """
    emitted = emitted_tiles(code, values, kernel=kernel)
    if tuple(emitted) != tuple(expected):
        raise ValueError('requested tile %r but the emitter produced %r'
                         % (tuple(expected), tuple(emitted)))
    return emitted
