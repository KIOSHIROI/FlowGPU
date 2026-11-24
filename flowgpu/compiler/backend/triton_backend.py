"""Backend lowering to Triton (Design.md 13.8).

13.8 says "v0.1 首先考虑 FlowGPU → Triton" and stops there: there is no kernel
specification in the document, so this module has to make one. What it does
with what the earlier stages decided:

* **One Triton kernel per fusion group** (13.6). That is what running 13.6
  first buys: a value the fusion kept in registers becomes an expression in
  the generated kernel and never a pointer, so 12.1's `Add -> RMSNorm` emits
  one function that computes `x + residual` and normalizes the result without
  `y` ever reaching memory. A group's pointwise operators are applied on the
  side of the anchor the placement chose: a prologue region to a tile as it is
  loaded, an epilogue region to the accumulator or the result.
* **Buffers come from the 13.7 plan.** A value with a slot becomes a
  `torch.empty` in the host function, named after its slot, so two values that
  share a slot share that one allocation and the reuse is visible as the same
  variable. A materialized input is a kernel parameter, and an eliminated
  value has no allocation at all.
* **Symbolic dimensions (8.3) become runtime arguments.** `bf16[N,D]` is not
  known here, so `N` and `D` are passed at launch and everything derived from
  them -- strides, grid, masks -- is computed from those values. This is how a
  shape reaches a Triton kernel, and it is why the generated host function
  reads the dimensions off its input tensors.
* **Pointwise and rowwise inputs may be strided views.** Runtime stride metadata
  maps logical row-major coordinates to storage; compiler-owned outputs remain
  contiguous. Other native kernel templates still require contiguous inputs.

This is the third per-operator table in the compiler, after the type rule and
the pointwise fact, and the first one that is real work. Registering `relu`
cost one line because nothing downstream had to know what `relu` *means*; a
backend is where "what it means" is finally written down, once per operator,
and where an operator without a lowering cannot be compiled at all. So the
tables below classify every operator in the registry, and the ones that cannot
be lowered say why:

* `rope(x, cos, sin)` uses split-half partner indexing with masked loads.
  Rotation tables are explicit inputs. It is not a pointwise identity.


Nothing here imports or runs Triton. The compiler emits source text; running
it needs a GPU and a Triton install, and neither is needed to compile. The
tests check that the text is valid Python and that its structure is what the
analyses decided -- not that the kernels compute the right numbers, which
nothing on this machine can check.
"""

import ast
from dataclasses import dataclass, replace
from math import isqrt

from flowgpu.compiler.core.ast import InputDecl, OutputDecl
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind
from flowgpu.compiler.core.operators import OPERATORS
from flowgpu.compiler.analysis.kernel_schedule import analyze_schedule
from flowgpu.compiler.backend.backends import (
    REFUSED,
    library_dispatch,
    Argument,
    Names,
    RMS_EPS,
    Refusal,
    SkippedKernel,
    allocation,
    comment as value_comment,
    group_operands,
    group_results,
    identifier,
    is_symbol,
    kernel_kind,
    product,
    refusal_for,
    scalar_operands,
    storage_of,
    tensor_operands,
    unique_formal,
)
from flowgpu.compiler.shared import dimension_symbols
from flowgpu.compiler.ir.dataflow import resolve_alias
from flowgpu.compiler.optimization.fusion import is_operator
from flowgpu.compiler.backend.platform_configs import (MAX_INDEX32, PlatformConfig,
                                               resolve_platform, wider_profile_hint)
from flowgpu.compiler.backend.custom_ops import render_custom_ops


# v0.1 decisions. The document fixes none of these, so each is a named
# constant rather than a number buried in a template.
BLOCK = 1024
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
# RMS_EPS is the shared language contract, declared in backends.py.

def _accumulator(name: str) -> str:
    """The Triton dtype a reduction accumulates in, as 21.6 declared it.

    This widening was written out as tl.float32 in four places. It was correct
    for the three reductions that exist, and would have stayed correct by luck
    rather than by contract: changing a declaration would not have changed the
    generated code, so the declaration was documentation. It is read from the
    registry now, which is what makes it a contract -- and a reduction that
    declares no accumulator is a compiler bug rather than a program's problem,
    so it raises instead of inventing a dtype.
    """
    spec = OPERATORS.get(name)

    if spec is None or spec.acc is None:
        raise AssertionError(
            f"{name} is lowered as a reduction but declares no accumulator: "
            f"21.6 requires one, and this backend will not invent it"
        )

    return DTYPE_TO_TRITON[spec.acc]

#: Phase S1: scalar inputs the caller chose to specialise as compile-time
#: literals.  ``lower()`` sets this before emitting kernels and the scalar
#: binding loops consult it instead of adding a formal.  The host signature is
#: filtered by the same mapping, so a specialised scalar disappears from the
#: launch ABI; the reference backend deliberately keeps its scalar input (it is
#: an independent executable specification, not the performance artifact).
_CONSTEXPR_LITERALS: dict[str, str] = {}


def _constexpr_literal(value: str) -> str | None:
    return _CONSTEXPR_LITERALS.get(value)




DTYPE_TO_TRITON = {
    "fp16": "tl.float16",
    "bf16": "tl.bfloat16",
    "fp32": "tl.float32",
    "fp8e4m3": "tl.float8e4nv",
    "fp8e5m2": "tl.float8e5",
    "int8": "tl.int8",
    "uint8": "tl.uint8",
    "int32": "tl.int32",
    "int64": "tl.int64",
    "bool": "tl.int1",
}

DTYPE_TO_TORCH = {
    "fp16": "torch.float16",
    "bf16": "torch.bfloat16",
    "fp32": "torch.float32",
    "fp8e4m3": "torch.float8_e4m3fn",
    "fp8e5m2": "torch.float8_e5m2",
    "int8": "torch.int8",
    "uint8": "torch.uint8",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "bool": "torch.bool",
}

FLOAT_DTYPES = frozenset({"fp16", "bf16", "fp32"})

# Pointwise operators are one expression over the tiles already in registers.
# `{a}` and `{b}` are the operand variables and `{zero}` is the dtype's zero.
POINTWISE_EXPRESSION = {
    "add": "({a} + {b})",
    # the operand mapping (row index for {b}) happens at load time in each
    # emitter; the arithmetic itself is an add
    "add_broadcast": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "relu": "tl.maximum({a}, {zero})",
    "silu": "({a} * tl.sigmoid({a}))",
    # torch.round is round-half-to-even, so `floor(x + 0.5)` is wrong at ties.
    "round": ("(tl.floor({a}) + tl.where(({a} - tl.floor({a})) > 0.5, 1.0, "
              "tl.where((({a} - tl.floor({a})) == 0.5) & "
              "((tl.floor({a}) - 2.0 * tl.floor(tl.floor({a}) * 0.5)) >= 1.0), 1.0, 0.0)))"),
}

# Operators whose expression cannot be evaluated on a bf16 tile, because the
# Triton function it uses is fp32-only. This is a fact about the device, not
# about the operator, so it lives here rather than in `flowgpu.compiler.backends`.
WIDER_ON_DEVICE = frozenset({"silu"})

# Reductions along the last axis lower to one program per row. A platform
# without a measured reduction ceiling keeps the v0.1 single-tile form
# (triton.next_power_of_2(D) as a launch-time constexpr). Both platforms
# with a measured ceiling -- cuda 4096, kunlunxin 128 -- lower a symbolic or
# over-budget row to a chunked two-sweep reduction instead. Which operators
# those are is not Triton's decision, so it lives in `flowgpu.compiler.backends`
# with the rest of the shared classification.


@dataclass(frozen=True)
class IndexBound:
    """A host-enforceable bound on every address a 32-bit index can derive.

    ``count`` is a Python expression in the names the emitted host already
    binds (the program's own dimension symbols, e.g. ``M * N``) for the
    largest *unmasked* element index of the kernel's iteration space.
    ``padding`` is the most lanes one launch adds beyond ``count``: the flat
    index is ``program_id * BLOCK + arange(BLOCK)``, so the last program's
    masked-off lanes reach ``count + BLOCK - 1``. The index must not wrap even
    there, because a wrapped negative lane would satisfy ``index < count`` and
    become an unmasked access to the wrong element.

    ``leaves`` names the tensors whose addresses are derived from that index.
    The host bounds each of them by its *runtime* shape and strides, so a
    strided view is bounded by its real address extent -- Task160's ``x[:, :n]``
    view reaches about twice its element count -- never by its logical size.

    A lowering constructs this object only when every address it emits is an
    affine function of the program index with shape/stride metadata the host
    also has. A data-dependent gather address gets an explicit int64 widening
    instead and never takes part in this proof.
    """

    count: str
    padding: int = 0
    leaves: tuple = ()

    def __post_init__(self):
        if not isinstance(self.count, str) or not self.count.strip():
            raise ValueError("index bound needs a nonempty count expression")
        if type(self.padding) is not int or self.padding < 0:
            raise ValueError("index bound padding must be a nonnegative int")
        if not isinstance(self.leaves, tuple):
            raise ValueError("index bound leaves must be a tuple of values")


@dataclass(frozen=True)
class TritonKernel:
    name: str
    kind: str
    ops: tuple
    operands: tuple
    results: tuple
    parameters: tuple
    launch: tuple
    grid: str
    source: str
    launch_options: tuple = ()
    #: Per-kernel matmul tile override ``(BLOCK_M, BLOCK_N, BLOCK_K)``. The
    #: source still spells the constexpr names; the host substitutes these
    #: numbers in the grid and launch, so a small-M matmul can use a smaller
    #: tile without changing the module-wide GEMM policy for large GEMMs.
    tile: tuple | None = None
    autotune_family: str = ''
    #: The program-index dtype this lowering *certified*. 'int32' only with
    #: ``index_bound`` set: never emit a 32-bit program index whose address
    #: extent the host does not check before launching. Lowerings that never
    #: hard-coded an index dtype (the pointwise/rowwise emitters) leave the
    #: default and keep Triton's natural 32-bit ``program_id`` arithmetic.
    index_dtype: str = 'int64'
    index_bound: IndexBound | None = None
    #: Names of external tensor inputs this kernel reads under the split-view
    #: contiguity contract: their runtime layout is NOT described by layout
    #: constexprs, so the host must refuse a non-contiguous argument. Empty for
    #: every default kernel, which is why the default host text is unchanged.
    contiguity_contracts: tuple = ()
    #: Host-side bounds records for every program-uniform scalar the kernel
    #: reads (`backend/program_scalar.host_guards`). Each record names the 1-D
    #: integer table and the index expression, so the emitted host can reject an
    #: out-of-bounds index before any allocation or launch. Empty for every
    #: default kernel, which keeps every existing host byte-identical.
    program_scalar_bounds: tuple = ()
    #: A host-side condition that decides whether this launch happens at all.
    #: `store_at` uses it for the copy pass: when the runs provably tile the
    #: destination the copy is dead, and the decision is pure shape metadata
    #: (`dst_elements != rows * lane_count`), never a device read. None for
    #: every existing kernel, so no existing host text changes.
    launch_guard: object = None
    #: The fail-closed host contract of one `store_at` call
    #: (`backend/store_at_backend.StoreAtContract`). Attached to the write
    #: kernel; `_host` emits the bounds/overlap checks before any allocation or
    #: launch. None for every kernel that is not `store_at`.
    store_at_contract: object = None

    @property
    def signature(self) -> str:
        return next(
            line
            for line in self.source.splitlines()
            if line.startswith("def ")
        )


@dataclass(frozen=True)
class TritonProgram:
    """One pipeline's kernels plus the host code that launches them."""

    pipeline: str
    kernels: tuple
    refusals: tuple
    skipped: tuple
    host: str
    gpu_backend: str = "triton"
    attention_impl: str = "math"
    platform_config: PlatformConfig | None = None
    custom_ops: str = ""
    autotune: bool = False
    autotune_fold: bool = False
    kernel_abi_audit: tuple = ()

    @property
    def complete(self) -> bool:
        """False when a kernel the fusion asked for could not be emitted.

        A skipped kernel does not make the program incomplete: nothing reads
        its result, so nothing is missing from it.
        """
        return not self.refusals

    def messages(self) -> tuple:
        return tuple(refusal.message() for refusal in self.refusals)


def _zero_for(dtype: str) -> str:
    return "0.0" if dtype in FLOAT_DTYPES else "0"


class _Registers:
    """The variable each value in a kernel is bound to.

    Every value the group computes gets one variable, in the order the group's
    operators are evaluated, and every line is commented with the FlowGPU value
    it computes -- which is what makes the generated code readable as the
    fusion report's result rather than as opaque Triton.
    """

    def __init__(self, graph, names):
        self.graph = graph
        self.names = names
        self.lines = []
        self.variables = {}

    def emit(self, expression: str, comment: str) -> str:
        name = self.names.numbered("t")
        self.lines.append(f"    {name} = {expression}    # {comment}")

        return name

    def bind(self, value: str, expression: str, comment: str) -> str:
        name = self.emit(expression, comment)
        self.variables[value] = name

        return name

    def dtype_of(self, value: str) -> str:
        return self.graph.values[value].dtype


_comment = value_comment


def _pointwise_expression(node, operands, dtype) -> str:
    template = POINTWISE_EXPRESSION[node.op]

    if node.op not in WIDER_ON_DEVICE:
        return template.format(
            **dict(zip("ab", operands)), zero=_zero_for(dtype)
        )

    # `tl.sigmoid` and `tl.exp` take fp32 operands, so an expression built from
    # them cannot run on the tile's own dtype: a bf16 `silu` does not compile
    # at all ("t1 = (t0 * tl.sigmoid(t0))"), which is what the first run on a
    # device said. So the value is computed in fp32 and rounded once at the
    # end -- the same shape as the reference, which has a double between the
    # same two boundaries.
    widened = [f"{operand}.to(tl.float32)" for operand in operands]
    body = template.format(**dict(zip("ab", widened)), zero=_zero_for("fp32"))

    return f"({body}).to({DTYPE_TO_TRITON[dtype]})"



def _pointer(value: str) -> str:
    return f"{identifier(value)}_ptr"


def _load(graph, value, offsets, mask) -> str:
    return (
        f"tl.load({_pointer(value)} + {offsets}, mask={mask}, "
        f"other={_zero_for(graph.values[value].dtype)})"
    )


import builtins
import os



#: Strict emission gate.  When on, `_make_kernel` refuses to return a kernel that
#: references a name Triton would not bind (an assignment inside a `for` body does
#: not escape it).
#:
#: On by default as of the 2026-10-22 sweep.  The backlog the gate found was 20
#: kernels across 7 test modules, and it was not "pre-existing latent bugs of the
#: same class" in the loose sense: the gate found three structural defects, all of
#: which emit *syntactically valid* Python whose names are bound in the wrong
#: scope, and every one of them was a wrong kernel rather than a false positive:
#:
#:   1. the capped 2-D matmul wrapped its preamble and then appended the k-loop
#:      body and the store at direct-grid indentation, so the accumulator was
#:      bound inside the stride loops and read (and stored) after them;
#:   2. a folded rotation chain never bound the interior `move_first_last` value,
#:      so a reduction whose operand is a rotation fell back to a pointer read of
#:      a value the fusion had proven needs no buffer;
#:   3. the final sweep of the reduction emitter classified a per-element value as
#:      a per-row "tail" whenever its chain reached a carrier address, hoisting the
#:      recomputation and the store out of the column loop.
#:
#: `FLOWGPU_STRICT_EMISSION=0` turns it off for a bisect only; nothing should ship
#: with it off.  See docs/LANGUAGE_PRIMITIVE_AUDIT.md.
STRICT_EMISSION = os.environ.get('FLOWGPU_STRICT_EMISSION', '1') not in {'0', 'false', 'False', ''}

#: Module-level names every emitted Triton module binds.  The kernel text alone
#: does not carry the module preamble, so they are supplied to the check.
_EMITTED_PREAMBLE = ("tl", "triton", "torch", "math", "struct")


def _undefined_names(source: str, preamble=()):
    """Names a kernel body loads but cannot see, under *Triton's* scoping rules.

    Emitting a kernel that references an undefined variable is worse than
    refusing to emit one: the failure happens on the device, far from the pass
    that caused it, and reads like a user error.  The first offender was a
    full-tile producer fused into a reduction group: its address variable was
    bound inside the sweep loop and used after it.

    Python would call that variable bound (a loop body binds for the whole
    function), so the check follows Triton instead: an assignment inside a `for`
    or `while` body does not escape it.  Parameters, function-top-level
    assignments, the module preamble and Python builtins are visible everywhere.
    """
    import ast

    loops = (ast.For, ast.AsyncFor, ast.While)
    problems = []
    for function in [n for n in ast.walk(ast.parse(source))
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        top = {argument.arg for argument in
               (function.args.args + function.args.kwonlyargs + function.args.posonlyargs)}
        inner = set()

        def collect(node, in_loop):
            for child in ast.iter_child_nodes(node):
                child_in_loop = in_loop or isinstance(child, loops)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    (inner if child_in_loop else top).add(child.id)
                elif isinstance(child, ast.arg):
                    top.add(child.arg)
                collect(child, child_in_loop)

        def verify(node, in_loop):
            for child in ast.iter_child_nodes(node):
                child_in_loop = in_loop or isinstance(child, loops)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    visible = top | set(preamble) | set(dir(builtins))
                    if child_in_loop:
                        visible = visible | inner
                    if child.id not in visible:
                        problems.append(child.id)
                verify(child, child_in_loop)

        collect(function, False)
        verify(function, False)
    return sorted(set(problems))


def _kernel_grid(kernel) -> str:
    """The kernel's grid with a per-kernel matmul tile substituted."""
    grid = kernel.grid
    tile = getattr(kernel, "tile", None)
    if tile:
        for block, value in zip(("BLOCK_M", "BLOCK_N", "BLOCK_K"), tile):
            grid = grid.replace(block, str(value))
    return grid


def _make_kernel(name, kind, group, pairs, lines, grid, graph, *, launch_options=(),
                 index_dtype='int64', index_bound=None,
                 contiguity_contracts=()) -> TritonKernel:
    """Assemble a kernel from its parameter pairs, body lines and grid.

    ``index_dtype='int32'`` is only accepted together with an ``IndexBound``:
    the emitted host enforces that bound before launch, and a 32-bit index
    without one is a compiler bug rather than a tuning choice.
    """
    names = [parameter.split(':')[0].strip() for parameter, _ in pairs]
    dimensions = dimension_symbols(graph)
    misnamed = sorted(
        argument.text for parameter, argument in pairs
        if argument.kind == "scalar" and argument.text in dimensions
        and parameter.split(':')[0].strip() == argument.text)
    if misnamed:
        # A rank-0 operand named like a dimension symbol keeps the dimension's
        # name unless the emitter reserved the dimension namespace first; the
        # kernel then uses the scalar value as a shape (review 2026-10-26).
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"{kind} kernel '{name}' gives scalar operand(s) {misnamed} the "
            f"same formal name as a dimension symbol",
            pipeline=graph.pipeline,
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        # Two formals with one name is a kernel that does not parse; the
        # dimension/scalar namespace collision produced exactly that before the
        # emitters started reserving dimension symbols first (review 2026-10-26).
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"{kind} kernel '{name}' repeats formal name(s) {duplicates}",
            pipeline=graph.pipeline,
        )
    if index_dtype not in {'int32', 'int64'}:
        raise ValueError(f"unknown index dtype: {index_dtype}")
    if index_dtype == 'int32' and index_bound is None:
        raise ValueError(f"{kind} kernel '{name}' cannot emit a 32-bit index "
                         f"without an enforced address bound")
    if index_dtype == 'int64':
        index_bound = None
    # A scalar input is a host number: only the pointwise and primitive
    # lowerings know how to pass one. Any other emitter that reaches this point
    # with a rank-0 value as a tensor pointer would emit a kernel whose first
    # instruction derefs a float, so it is a compiler refusal rather than
    # generated text.
    for parameter, argument in pairs:
        if argument.kind != "tensor":
            continue
        if graph.values[argument.text].shape:
            continue
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"{kind} kernel '{name}' would pass input "
            f"'{argument.text}' as a tensor pointer, but it is a scalar; "
            f"scalar inputs are supported by the pointwise and primitive "
            f"lowerings only",
            pipeline=graph.pipeline,
        )
    # A widened expression still has a declared fp16/bf16 boundary.  Preserve it
    # even when a downstream add/sub could contract the multiply into an FMA:
    # WIDER_ON_DEVICE names exactly the operators that widen to fp32 and round
    # back, so the guard follows the property instead of one operator's name.
    if any(node.op in WIDER_ON_DEVICE for node in group.members):
        if not any(key == "enable_fp_fusion" for key, _ in launch_options):
            launch_options += (("enable_fp_fusion", False),)
    ops = " -> ".join(node.op for node in group.members)
    fused = ", ".join(group.fused_values)
    header = f"# {ops}"

    if fused:
        header += f"   ({fused} fused by 13.6: no buffer, 13.7)"

    parameters = ", ".join(parameter for parameter, _ in pairs)

    source = (
        f"{header}\n"
        "@triton.jit\n"
        f"def {name}({parameters}):\n"
        + "\n".join(lines)
        + "\n"
    )

    undefined = _undefined_names(source, _EMITTED_PREAMBLE) if STRICT_EMISSION else []
    if undefined:
        # Refuse rather than emit: a kernel that references an unbound name
        # fails on the device as a NameError inside `tl.program_id`-era scoping,
        # far from the pass that produced it.  The first offender was a
        # full-tile producer fused into a reduction group, whose address
        # variable was bound inside the sweep loop and used after it.
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"{kind} kernel '{name}' would reference undefined name(s) "
            f"{', '.join(undefined)}; the lowering that built it is wrong",
            pipeline=graph.pipeline,
        )

    return TritonKernel(
        name=name,
        kind=kind,
        ops=tuple(node.op for node in group.members),
        operands=group_operands(graph, group),
        results=group_results(graph, group),
        parameters=tuple(parameter for parameter, _ in pairs),
        launch=tuple(argument for _, argument in pairs),
        grid=grid,
        source=source,
        launch_options=launch_options,
        index_dtype=index_dtype,
        index_bound=index_bound,
        contiguity_contracts=tuple(contiguity_contracts),
    )


def _tensor_layouts(graph, values, names, pairs):
    """Append shape/stride metadata to the kernel ABI for specified tensors."""
    layouts = {}
    for value in values:
        contiguous = names.fresh(f"{identifier(value)}_contiguous")
        pairs.append((f"{contiguous}: tl.constexpr", Argument("contiguous", value)))
        dimensions, strides = [], []
        for axis in range(len(graph.values[value].shape)):
            dimension = names.fresh(f"{identifier(value)}_dim{axis}")
            stride = names.fresh(f"{identifier(value)}_stride{axis}")
            pairs.extend([(f"{dimension}: tl.constexpr", Argument("shape", f"{value}:{axis}")),
                          (f"{stride}: tl.constexpr", Argument("stride", f"{value}:{axis}"))])
            dimensions.append(dimension)
            strides.append(stride)
        layouts[value] = (contiguous, dimensions, strides)
    return layouts


def _input_layouts(graph, memory, operands, names, pairs):
    """External views; default compiler-owned slots stay contiguous."""
    values=[v for v in operands if storage_of(memory,resolve_alias(graph.aliases,v))=='input']
    return _tensor_layouts(graph,values,names,pairs)


def _view_offset(value, logical_offset, layouts, names, lines, *, coordinates=None, linearize=False):
    if value not in layouts:
        return logical_offset
    contiguous, dimensions, strides = layouts[value]
    address = names.fresh(f"{identifier(value)}_address")
    terms = []
    if coordinates is not None and len(coordinates) != len(strides):
        raise ValueError('input coordinate rank does not match stride metadata')
    for axis, stride in enumerate(strides):
        if coordinates is not None:
            coordinate = coordinates[axis]
        else:
            remaining = " * ".join(dimensions[axis + 1:]) or "1"
            coordinate = f"(({logical_offset}) // ({remaining}))"
            if axis:
                coordinate += f" % {dimensions[axis]}"
        # A stride of one is written as the bare coordinate. Spelling it as
        # "() * 1" hides the contiguity from Triton's axis analysis, which then
        # emits 16-bit scalar loads instead of 128-bit vector loads (confirmed in
        # PTX and SASS on the LSE kernel). Dropping the multiply costs nothing
        # and leaves the expression the analysis can prove.
        if str(stride).strip().strip("()") == "1":
            terms.append(f"({coordinate})")
        else:
            terms.append(f"({coordinate}) * {stride}")
    lines.extend([f"    {address} = {logical_offset}",
                  f"    if not {contiguous}:"])
    if linearize and coordinates is None and strides:
        # s[i] == dim[i+1]*s[i+1] telescopes the row-major coordinate sum
        # to logical_offset*s[-1]. Interleaved views satisfy this relation;
        # split halves, row gaps and arbitrary gathers retain the old map.
        checks = [f"({strides[i]} == {dimensions[i+1]} * {strides[i+1]})"
                  for i in range(len(strides)-1)]
        if checks:
            condition=checks[0]
            for check in checks[1:]:condition=f"({condition} and {check})"
            lines.extend([f"        if {condition}:",
                          f"            {address} = ({logical_offset}) * {strides[-1]}",
                          "        else:",
                          f"            {address} = {' + '.join(terms)}"])
        else:
            lines.append(f"        {address} = ({logical_offset}) * {strides[-1]}")
    else:
        lines.append(f"        {address} = {' + '.join(terms)}")
    return address


def grid_ceiling(config):
    """The largest grid this target accepts, or None when it accepts any.

    An explicit loop mapping caps at the vector-core count; a launch path that
    refuses a grid beyond its hardware extent caps at ``max_grid_dim``. Both are
    reasons to walk several tiles per program.
    """
    limits = [limit for limit in (config.grid_limit, config.max_grid_dim)
              if limit is not None]
    return min(limits) if limits else None


def capped_grid(tiles, limit):
    """A launch expression that never exceeds ``limit``.

    Tensor and dimension names may shadow Python's built-in min, so the cap is a
    conditional rather than a call.
    """
    return f"({tiles} if {tiles} < {limit} else {limit},)"


def grid_limited_body(body, grid, config, *, count, numel, block, grid_block, tile,
                      bound=None, tiles=None):
    """Wrap a flat one-tile body in a grid-stride loop when the launch is capped.

    Two independent reasons to walk several tiles per program: an explicit loop
    mapping (grid = vector cores), or a launch path that refuses a grid beyond
    its hardware extent. Ascend rejects ``coreDim`` above 65535 with
    ``Invalid_Argument(EE1003)``, recorded in the official Task155 receipt of
    2026-10-18. In both cases the tile index replaces ``program_id`` and the
    stride is the number of programs actually launched, so masks, addressing
    and arithmetic are unchanged; the only difference is how many tiles each
    program owns.

    Every ``tl.program_id(0)`` in the body is rewritten to the loop tile rather
    than the first line being replaced, so an optional compile-time index-width
    guard above the index definition survives the wrap.

    ``block`` is the kernel-side constexpr name, ``grid_block`` the literal the
    host divides by. They differ whenever the kernel keeps a symbolic ``BLOCK``.
    ``bound`` and ``tiles`` override the loop end and the launched count for a
    kernel whose tile count is not ``cdiv(numel, block)`` -- an L2-swizzled
    matmul counts ``cdiv(rows)*cdiv(columns)`` blocks instead.
    """
    limit = grid_ceiling(config)
    if limit is None:
        return body, grid
    body = [line.replace("tl.program_id(0)", tile) for line in body]
    end = bound if bound is not None else f"tl.cdiv({count}, {block})"
    body = [f"    for {tile} in range(tl.program_id(0), {end}, tl.num_programs(0)):"] + [
        "    " + line for line in body
    ]
    launched = tiles if tiles is not None else f"triton.cdiv({numel}, {grid_block})"
    return body, capped_grid(launched, limit)


def capped_dim(expression, limit):
    """One launch dimension that never exceeds ``limit``."""
    return f"{expression} if {expression} < {limit} else {limit}"


def capped_grid_2d(rows, columns, limit):
    """A two-dimensional launch whose program *count* never exceeds ``limit``.

    Ascend's ``coreDim`` ceiling bounds the number of programs a launch asks
    for, not the extent of either axis. The official Task 144 evaluation of
    2026-10-19 was rejected with ``Invalid_Argument(EE1003): coreDim 78336``
    for a grid ``(b*c, t_out, h_out)`` whose axes were each small, so capping
    each axis independently does not bound the launch: two axes capped at
    ``limit`` can ask for ``limit ** 2`` programs.

    Each axis is therefore capped at ``floor(sqrt(limit))``, which bounds the
    product without dividing by a count read at launch time. The alternative --
    capping the second axis at ``limit // rows_launched`` -- needs the row
    expression repeated inside its own cap and turns a degenerate zero-tile
    launch into a ``ZeroDivisionError``; a constant cannot.
    """
    axis = isqrt(limit)
    if axis * axis > limit:
        raise ValueError("the per-axis cap must not exceed the ceiling")
    return f"({capped_dim(rows, axis)}, {capped_dim(columns, axis)})"


def capped_grid_2d_product(rows, columns, limit):
    """A two-dimensional launch whose program *count* never exceeds ``limit``.

    ``coreDim`` bounds the number of programs a launch asks for, i.e. the
    *product* of the launched grid extents -- the official Task144 receipt
    rejected ``coreDim 78336`` for a grid ``(b*c, t_out, h_out)`` whose axes
    were each small. ``capped_grid_2d`` bounds that product by capping every
    axis at ``floor(sqrt(limit))``, which is right for a launch whose exact
    grid is unknown, but it also shrinks a grid that already fits: a
    4096 x 8 tile grid would launch 255 x 8 programs even though 32768 is under
    the ceiling. For a tile-*shape* experiment that is the wrong trade: the
    program count per element is the quantity being held fixed.

    So this cap keeps the exact tile grid whenever the product fits, and only
    shrinks when it does not: the row axis takes at most ``limit`` and the
    column axis takes what is left. The kernel-side bargain is the same
    two-axis stride loop ``grid_limited_body_2d`` emits, whose stride is
    ``tl.num_programs`` on each axis, so the launched axes together visit every
    tile exactly once. Every factor is at least one -- the emitted host rejects
    a non-positive extent before a launch -- so the integer division cannot be
    by zero.
    """
    launched_rows = f"({rows} if {rows} < {limit} else {limit})"
    remaining = f"({limit} // {launched_rows})"
    return f"({launched_rows}, {columns} if {columns} < {remaining} else {remaining})"


def grid_limited_body_2d(body, grid, config, *, rows_of_blocks, columns_of_blocks,
                         row, column, host=None, cap=capped_grid_2d):
    """Wrap a two-dimensional launch in a two-axis stride loop when capped.

    A matmul or transpose grid is ``(cdiv(rows, BLOCK_M), cdiv(columns, BLOCK_N))``.
    Ascend rejects ``coreDim`` above 65535 on either axis, so each axis is capped
    independently; the strides are the programs actually launched on that axis,
    which keeps every (row block, column block) pair visited exactly once. The
    body's ``row``/``column`` assignments are dropped because the loop headers
    replace them.

    ``cap`` is the launch-expression capper. It must bound the number of
    programs the two axes ask for; the bounding rule decides between
    ``capped_grid_2d`` (per-axis ceiling, the conservative default) and
    ``capped_grid_2d_product`` (exact grid when it fits, for an experiment that
    must not change the program count).
    """
    limit = grid_ceiling(config)
    if limit is None:
        return body, grid
    headers = {f"    {row} = tl.program_id(0)", f"    {column} = tl.program_id(1)"}
    body = [line for line in body if line not in headers]
    body = [
        f"    for {row} in range(tl.program_id(0), {rows_of_blocks}, tl.num_programs(0)):",
        f"        for {column} in range(tl.program_id(1), {columns_of_blocks}, tl.num_programs(1)):",
    ] + ["        " + line for line in body]
    host_rows, host_columns = host if host is not None else (
        rows_of_blocks.replace("tl.cdiv", "triton.cdiv"),
        columns_of_blocks.replace("tl.cdiv", "triton.cdiv"))
    return body, cap(host_rows, host_columns, limit)


def row_limited_body(body, grid, config, *, count, row, grid_count=None):
    """Wrap a one-row-per-program body in a row-stride loop when the launch is capped.

    The row-wise reduction primitive gives every program exactly one row, so its
    grid is the row count itself. That is the one grid that can exceed a hardware
    extent with no tile size to shrink: a ``[65536, 4096]`` RMSNorm input asks for
    65536 programs no matter how small the reduction tile is, and Ascend rejects
    ``coreDim`` above 65535 with ``Invalid_Argument(EE1003)`` -- the same error
    the official Task155 receipt recorded for the flat pointwise path. Batch-10
    Tasks 147-150 are all row-wise RMSNorm variants, so this is not hypothetical.

    ``body`` may assign ``row`` from ``program_id``; the capped form drops that
    assignment because the loop header replaces it. Striding by the launched
    program count keeps every row visited exactly once, so masks, addressing and
    the per-row scalar accumulator are unchanged.
    """
    limit = grid_ceiling(config)
    if limit is None:
        return body, grid
    assignment = f"    {row} = tl.program_id(0)"
    body = [line for line in body if line != assignment]
    body = [f"    for {row} in range(tl.program_id(0), {count}, tl.num_programs(0)):"] + [
        "    " + line for line in body
    ]
    return body, capped_grid(count if grid_count is None else grid_count, limit)


def _pointwise_kernel(graph, group, memory, name, platform_config=None) -> TritonKernel:
    """A kernel that reads operands and writes its result, elementwise.

    The tensor is one flat index space, which is exactly what makes a
    pointwise operator attachable to any neighbour: its iteration space
    already matches, so its loop can be glued to the neighbour's.

    A rank-0 input is the one operand that is not a tile: the caller passes one
    number, so it contributes a plain formal (`a`, not `a_ptr`), no layout
    constexprs -- there are no strides to describe -- and its "load" is the
    formal itself. That is the whole point of the scalar declaration: no
    per-call device tensor, no dead stride metadata.
    """
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    config = resolve_platform() if platform_config is None else platform_config
    result = results[0]
    numel = product(graph.values[result].shape)

    names = Names(graph)
    offsets = names.fresh("offsets")
    mask = names.fresh("mask")
    count = names.reserve("numel")

    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    block = "BLOCK" if config.default_block_size == BLOCK else str(config.default_block_size)

    # Pointers first, then scalars, then the constexpr/launch metadata. The
    # host emits its launch arguments from this same ordered list, so the two
    # sides cannot disagree about positions.
    pairs = [
        (_pointer(value), Argument("tensor", value)) for value in tensors
    ]
    pairs += [
        (_pointer(value), Argument("tensor", value)) for value in results
    ]
    # Names the fixed ABI already uses. A program may call an input `numel` or
    # `BLOCK`, and a scalar formal may not repeat a pointer formal (`x_ptr` is
    # a legal input name), so the collision is broken explicitly.
    used_formals = ({count, "BLOCK"} | dimension_symbols(graph))
    used_formals.update(parameter.split(":")[0] for parameter, _ in pairs)
    scalar_formals = {}
    for value in scalars:
        literal = _constexpr_literal(value)
        if literal is not None:
            scalar_formals[value] = literal
            continue
        formal = unique_formal(identifier(value), used_formals)
        names.reserve(formal)
        scalar_formals[value] = formal
        pairs.append((formal, Argument("scalar", value)))

    pairs.append((count, Argument("value", numel)))
    pairs.append(("BLOCK: tl.constexpr", Argument("value", block)))

    layouts = _input_layouts(graph, memory, tensors, names, pairs)
    registers = _Registers(graph, names)
    # A scalar needs no instruction to read: binding its formal in the register
    # map makes the expression walker use it exactly where it uses a loaded
    # tile, so the operator template is unchanged for both operand kinds.
    #
    # The binding stays the bare formal: `mul(x, a)` must emit `x * a`, with no
    # wrapping op, and a test pins that.  Operators whose template calls a tensor
    # method on its operands (`div`, registered pointwise=False) therefore cannot
    # use this path; they lower through the primitive emitter, which does wrap a
    # rank-0 operand as a 0-d tile.
    registers.variables.update(scalar_formals)
    from flowgpu.compiler.backend.paired_views import PairedViewLoads
    paired = PairedViewLoads(graph, tensors, result, layouts,
                            {v: _pointer(v) for v in tensors}, names, pairs,
                            enabled=config.paired_view_loads)
    # `tl.program_id` is already int32, so this path never carried the 64-bit
    # index arithmetic the platform `index_dtype` policy replaces. It is left
    # exactly as emitted: widening it back to int64 for cuda (whose policy is
    # int64) would perturb the CUDA-class targets that currently score 4-11x.
    lines = [
        f"    {offsets} = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        f"    {mask} = {offsets} < {count}",
    ]

    def load(value):
        combined = paired.load(value, offsets, mask, registers.lines)
        if combined is not None:
            return registers.bind(value, combined, _comment(value, group))
        address = _view_offset(value, offsets, layouts, names, registers.lines,
                               linearize=config.linear_view_indexing)
        return registers.bind(
            value,
            _load(graph, value, address, mask),
            _comment(value, group),
        )

    program_scalar_bounds = []
    subexpressions = {}
    for node in group.members:
        spec = getattr(node.attributes, "program_scalar", None)
        if spec is not None:
            # One scalar load per program, emitted once at the top level of the
            # kernel and used as a register from then on. It is deliberately not
            # re-derived per lane: that would be a gather, and the write address
            # would stop being scalar_base + affine(arange).
            from flowgpu.compiler.backend.program_scalar import emit_load
            variable, bound = emit_load(
                node, graph, names, pairs, layouts, registers.lines,
                grid_count=f"triton.cdiv({numel}, {block})",
            )
            registers.variables[node.output] = variable
            program_scalar_bounds.append(bound)
            continue
        values = [
            registers.variables[value]
            if value in registers.variables
            else load(value)
            for value in node.inputs
        ]
        # In-kernel common subexpression elimination: a fused region may compute
        # the same expression twice through different value names (a helper
        # recomputed at each use). Operand *registers* are part of the key, so
        # two structurally identical nodes with identical operands collapse to
        # one `tN = ...` line; this is register reuse inside one kernel, not a
        # materialised buffer (the partitioner still decides buffers).
        from flowgpu.compiler.backend import cse
        cse_key = (node.op, tuple(values), registers.dtype_of(node.output),
                   node.part, repr(node.attributes)) if cse.ENABLED else None
        if cse_key is not None and cse_key in subexpressions:
            registers.variables[node.output] = subexpressions[cse_key]
            continue
        bound = registers.bind(
            node.output,
            _pointwise_expression(node, values, registers.dtype_of(node.output)),
            _comment(node.output, group),
        )
        if cse_key is not None:
            subexpressions[cse_key] = bound

    registers.lines.append(
        f"    tl.store({_pointer(result)} + {offsets}, "
        f"{registers.variables[result]}, mask={mask})"
    )

    body = lines + registers.lines
    grid = f"(triton.cdiv({numel}, {block}),)"
    body, grid = grid_limited_body(body, grid, config, count=count, numel=numel,
                                   block="BLOCK", grid_block=block,
                                   tile=names.fresh("tile"))
    # Preserve the default source/launch while exposing explicit tuning options.
    options = (("num_warps", config.num_warps),) if config.num_warps != 4 or config.name != "cuda" else ()

    kernel = _make_kernel(
        name,
        "pointwise",
        group,
        pairs,
        body,
        grid,
        graph,
        launch_options=options,
    )
    if program_scalar_bounds:
        from dataclasses import replace as _replace
        kernel = _replace(kernel,
                          program_scalar_bounds=tuple(program_scalar_bounds))
    return kernel


def _next_power_of_two(value: int) -> int:
    """The smallest power of two >= value, matching the launch-time helper."""
    return 1 << max(0, value - 1).bit_length()


def _laned(offsets_name, scalar_expression):
    """Make a reduction-derived scalar lane-shaped before vector arithmetic.

    The seven competition Triton forks are not uniform: several reject
    arithmetic between a 0-d reduction result and a vector (the FlagOS batch-11
    kernels had to materialise the scale and reload it lane-shaped).  Emitting
    ``offsets * 0.0 + scalar`` keeps the expression vector-shaped with one
    extra ALU op and compiles on every backend tested.
    """
    return f"({offsets_name} * 0.0 + ({scalar_expression}))"


def _rowwise_kernel(graph, group, memory, name, platform_config=None, *, autotune_fold=False) -> TritonKernel:
    """One program per row, reducing along the last axis.

    A platform without a measured reduction ceiling keeps the v0.1 form: the
    row is one tile of ``triton.next_power_of_2(D)`` lanes. Both platforms
    with a measured ``reduction_block_limit`` -- cuda 4096 (A800 timing: the
    single-tile form collapses ~10x from D=32768) and kunlunxin 128 (uni_sram
    OutOfResources) -- lower a symbolic row, or one wider than the limit, to
    a two-sweep chunked form: sweep one folds the reduction across chunks
    (softmax rescales a running maximum, rms_norm accumulates squares), sweep
    two re-reads each chunk and applies the normalization, rounding
    boundaries and the fused epilogue. Any row length is then correct on
    that target, and the chunk width respects its measured tile budget.
    """
    config = resolve_platform() if platform_config is None else platform_config
    from flowgpu.compiler.ir.lowering_ir import analyze_group
    structured = analyze_group(graph, group, config, 0)
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]
    # A trailing rotation chain rooted at the anchor output only readdresses
    # where the sweep stores: fold it into the store address instead of
    # materializing each rotation (13.6 store rotation).
    from flowgpu.compiler.backend.backends import store_rotation_chain
    rotation_chain = store_rotation_chain(graph, group)
    result = rotation_chain[-1].output if rotation_chain else results[0]
    # The sweep runs over the ANCHOR'S INPUT rows: the reduction axis is
    # the input's last axis, and the store-rotation inverse maps readdress
    # its linear indices into the (rotated) result. Deriving rows from the
    # result instead sweeps the wrong axis whenever a chain folds (the A800
    # caught softmax reducing over D instead of K). Without a folded chain
    # the input and result shapes agree, so this matches the old form.
    input_shape = graph.values[anchor.inputs[0]].shape
    row = input_shape[-1]
    leading = input_shape[:-1]

    names = Names(graph)
    index = names.fresh("row")
    offsets = names.fresh("offsets")
    mask = names.fresh("mask")
    base = names.fresh("base")
    row_name = names.reserve(str(row)) if is_symbol(row) else str(row)

    # A platform without a measured reduction ceiling (limit None) keeps the
    # v0.1 single-tile form; no ceiling is invented for it. A platform with a
    # budget lowers symbolic or over-budget rows to the chunked form.
    autoreduce = getattr(structured.body, "body", ()) if structured.kind == "reduction" else ()
    reduction_spec = autoreduce[0] if autoreduce else None
    single_tile = config.reduction_block_limit is None or (
        not is_symbol(row)
        and _next_power_of_two(int(row)) <= config.reduction_block_limit
    )
    if reduction_spec is not None:
        single_tile = reduction_spec.stages == ("single",)
    if autotune_fold:
        # A smaller selected ROW must stream the entire axis, including literals.
        single_tile = False

    # Rank-0 operands are host numbers, not tiles: they become plain kernel
    # arguments bound straight into the register map, exactly as the pointwise
    # path does.  A reduction whose epilogue divides by a scalar (dynamic_quant's
    # scale) otherwise has no way to reach one.
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    pairs = [
        (_pointer(value), Argument("tensor", value)) for value in tensors
    ]
    pairs += [
        (_pointer(value), Argument("tensor", value)) for value in results
    ]
    scalar_formals = {}
    used_formals = ({"ROW"} | dimension_symbols(graph)
                    | {parameter.split(":")[0] for parameter, _ in pairs})
    for value in scalars:
        literal = _constexpr_literal(value)
        if literal is not None:
            scalar_formals[value] = literal
            continue
        formal = unique_formal(identifier(value), used_formals)
        names.reserve(formal)
        scalar_formals[value] = formal
        pairs.append((formal, Argument("scalar", value)))
    if is_symbol(row):
        pairs.append((row_name, Argument("value", str(row))))
    if single_tile:
        pairs.append(
            (
                "ROW: tl.constexpr",
                Argument("value", f"triton.next_power_of_2({row})"),
            )
        )
    else:
        limit = config.reduction_block_limit
        if is_symbol(row):
            # DSL identifiers may shadow builtins, so the capped width is a
            # conditional expression rather than min().
            width = f"triton.next_power_of_2({row})"
            capped = f"{width} if {width} <= {limit} else {limit}"
        else:
            # A literal row on this path always exceeds the limit.
            capped = str(limit)
        pairs.append(
            (
                "ROW: tl.constexpr",
                Argument("value", capped),
            )
        )

    # The direct row launch needs only the width inside the kernel. A capped
    # launch also needs the total row count as its loop bound: leading DSL
    # dimensions are host expressions, not implicitly bound kernel variables.
    row_count = product(leading) if leading else '1'
    kernel_row_count = row_count
    if grid_ceiling(config) is not None:
        kernel_row_count = names.fresh('row_count')
        pairs.append((kernel_row_count, Argument('value', row_count)))
    layouts = _input_layouts(graph, memory, tensors, names, pairs)
    # A folded rotation chain composes `_moved(first, rest)` over its links'
    # source shapes, naming dimensions the kernel's own indexing never
    # mentions; they must join the parameter list or the kernel references
    # free names and fails to compile on the device (caught on the A800).
    chain_dims = set()
    for node in group.members:
        if node.op != 'move_first_last':
            continue
        chain_dims.update(str(d) for d in graph.values[node.inputs[0]].shape)
    for dimension in sorted(chain_dims):
        if is_symbol(dimension) and all(str(name).split(':')[0] != dimension
                                        for name, _ in pairs):
            pairs.append((names.reserve(dimension), Argument("value", dimension)))
    registers = _Registers(graph, names)
    registers.variables.update(scalar_formals)

    def offsets_of(value) -> str:
        # A row-length operand such as rms_norm's weight is indexed by the
        # position within the row, not by the position in the tensor.
        if len(graph.values[value].shape) == 1:
            return offsets

        return f"{base} + {offsets}"

    def load(value, into=None):
        target = registers if into is None else into
        address = _view_offset(value, offsets_of(value), layouts, names, target.lines)
        return target.bind(
            value,
            _load(graph, value, address, mask),
            _comment(value, group),
        )

    if single_tile:
        lines = [
            f"    {index} = tl.program_id(0)",
            f"    {offsets} = tl.arange(0, ROW)",
            f"    {mask} = {offsets} < {row_name}",
            f"    {base} = {index} * {row_name}",
        ]

        def register_load(value, address_override=None):
            """A prologue value as a tile expression, fusing index ops.

            A prologue ``move_first_last`` is addressing, not elementwise
            math: the load map composes into the address chain (13.6), so
            the element never materializes. A pure elementwise link
            (``to_fp32``, ``cast_like``, pointwise math) inside the chain is
            recomputed at the one mapped address, exactly as any 13.7
            prologue is -- ``address_override`` threads that mapped address
            down the chain.
            """
            node = next((n for n in group.prologue if n.output == value), None)
            if value in registers.variables:
                return registers.variables[value]
            if node is not None and node.op == "move_first_last":
                source = graph.values[node.inputs[0]]
                position = address_override or offsets_of(value)
                mapped = registers.emit(
                    f"(({position}) % {source.shape[0]}) * "
                    f"({product(source.shape[1:])}) + "
                    f"(({position}) // {source.shape[0]})",
                    "moved index",
                )
                address = _view_offset(
                    node.inputs[0], mapped, layouts, names, registers.lines
                )
                # A rotated value is addressing, so it has no materialized
                # buffer -- but it can still be *read*: a reduction anchor whose
                # operand is a rotation reads the rotated value directly.  The
                # chain composes the load map into one address, so the interior
                # of the chain is the same tile the final load produces: bind it
                # to that value.  Without this the interior stayed unbound and
                # the anchor fell back to a pointer read of a value the fusion
                # had proven needs no buffer, which is a `NameError` on the
                # device for a name no formal ever bound.
                mapped_value = register_load(node.inputs[0], address)
                return registers.bind(
                    node.output, mapped_value, _comment(node.output, group)
                )
            if node is not None:
                values = [
                    registers.variables[operand]
                    if operand in registers.variables
                    else register_load(operand, address_override)
                    for operand in node.inputs
                ]
                if node.op in {"cast_like", "cast_to", "to_fp32"}:
                    # A cast inside the chain is the declared-dtype
                    # boundary itself: rounding the loaded operand is the
                    # whole computation.
                    expression = values[0]
                    dtype = DTYPE_TO_TRITON[graph.values[node.output].dtype]
                    if f".to({dtype})" not in expression:
                        expression = f"({expression}).to({dtype})"
                    return registers.bind(
                        node.output, expression, _comment(node.output, group)
                    )
                return registers.bind(
                    node.output,
                    _pointwise_expression(
                        node, values, registers.dtype_of(node.output)
                    ),
                    _comment(node.output, group),
                )
            if address_override is not None:
                address = _view_offset(
                    value, address_override, layouts, names, registers.lines
                )
                return registers.bind(
                    value,
                    _load(graph, value, address, mask),
                    _comment(value, group),
                )
            return load(value)

        for node in group.prologue:
            if node.output not in registers.variables:
                register_load(node.output)

        arguments = [
            registers.variables[value]
            if value in registers.variables
            else load(value)
            for value in anchor.inputs
        ]

        if anchor.op in {"rms_norm", "rmsnorm_hf"}:
            accumulator = _accumulator(anchor.op)
            normalized = registers.emit(
                f"{arguments[0]}.to({accumulator})",
                "rms_norm: normalize in the declared acc",
            )
            squared = registers.emit(
                f"tl.sum({normalized} * {normalized}, axis=0) / {row_name}",
                "rms_norm: mean of squares",
            )
            scaled = registers.emit(
                f"{normalized} * " + _laned(offsets, f"tl.rsqrt({squared} + {RMS_EPS})"),
                "rms_norm: rsqrt(mean + eps)",
            )
            weight = registers.emit(
                f"{arguments[1]}.to({accumulator})",
                "rms_norm: weight, widened too",
            )
            if anchor.op == "rmsnorm_hf":
                input_dtype = DTYPE_TO_TRITON[graph.values[anchor.inputs[0]].dtype]
                scaled = registers.emit(
                    f"{scaled}.to({input_dtype}).to({accumulator})",
                    "rmsnorm_hf: round normalized activation before weight",
                )
            expression = f"({scaled} * {weight})"
        else:
            # Masked lanes must contribute nothing to the sum, so the mask is
            # applied to the exponentials instead of being left to the load's
            # `other` value.
            accumulator = _accumulator(anchor.op)
            normalized = registers.emit(
                f"{arguments[0]}.to({accumulator})",
                "softmax: in the declared acc",
            )
            largest = registers.emit(
                f"tl.max(tl.where({mask}, {normalized}, -float('inf')), axis=0)",
                "softmax: row maximum excludes padding"
            )
            exponentials = registers.emit(
                f"tl.where({mask}, tl.exp({normalized} - "
                + _laned(offsets, largest) + "), 0.0)",
                "softmax: exp, padding contributes nothing",
            )
            total = registers.emit(
                f"tl.sum({exponentials}, axis=0)", "softmax: row sum"
            )
            expression = f"({exponentials} / {total})"

        # A fusion removes global storage, not the DSL value's dtype boundary.
        # Normalize/softmax in fp32, then round before any bf16/fp16 epilogue.
        anchor_dtype = graph.values[anchor.output].dtype
        if anchor_dtype != "fp32":
            expression = f"{expression}.to({DTYPE_TO_TRITON[anchor_dtype]})"

        value = registers.bind(
            anchor.output, expression, _comment(anchor.output, group)
        )

        from flowgpu.compiler.backend import cse
        sharing = cse.Sharing()
        for node in group.epilogue:
            if node in rotation_chain:
                continue
            values = [
                registers.variables[operand]
                if operand in registers.variables
                else load(operand)
                for operand in node.inputs
            ]
            previous = sharing.previous(node, values, registers.dtype_of(node.output))
            if previous is not None:
                registers.variables[node.output] = previous
                continue
            value = registers.bind(
                node.output,
                _pointwise_expression(node, values, registers.dtype_of(node.output)),
                _comment(node.output, group),
            )
            sharing.record(node, values, registers.dtype_of(node.output), value)

        dtype = graph.values[result].dtype
        if dtype != "fp32":
            value = registers.emit(
                f"{value}.to({DTYPE_TO_TRITON[dtype]})", f"{result}: result dtype"
            )

        store_index = f"{base} + {offsets}"
        # The load map of one rotation sends an index of its OUTPUT to an
        # index of its input. A store needs the inverse: each link's inverse
        # has the closed form (j % rest)*d0 + j//rest over that link's source
        # shape, applied in chain order from the anchor outward.
        for node in rotation_chain:
            source = graph.values[node.inputs[0]]
            first = source.shape[0]
            rest = product(source.shape[1:])
            store_index = registers.emit(
                f"(({store_index}) % ({rest})) * ({first})"
                f" + (({store_index}) // ({rest}))",
                "store rotation",
            )
        registers.lines.append(
            f"    tl.store({_pointer(result)} + {store_index}, {value}, "
            f"mask={mask})"
        )

        # One program per row is the row grid itself, with no tile to shrink;
        # Ascend refuses coreDim above 65535, so stride the rows instead.
        body, grid = row_limited_body(lines + registers.lines, f"({row_count},)", config,
                                      count=kernel_row_count, row=index, grid_count=row_count)
        return _make_kernel(
            name,
            "rowwise",
            group,
            pairs,
            body,
            grid,
            graph,
            # Preserve declared rounding boundaries when a weight multiplication
            # feeds an add/sub epilogue; automatic FMA can otherwise bypass them.
            launch_options=(("enable_fp_fusion", False),) if anchor.op in {"rms_norm", "rmsnorm_hf"} else (),
        )

    # ---- Chunked two-sweep form (symbolic row or row beyond the limit) ----
    accumulator = _accumulator(anchor.op)
    chunk = names.fresh("chunk")
    # Contract: the chunked form below is one program sweeping the whole row,
    # not several programs owning parts of it. A split would make the row
    # maximum and the normalising denominator incomplete in every program and
    # would need an explicit merge stage; classify() refuses it rather than
    # letting the emitter produce a kernel that silently reads partial
    # statistics.
    from flowgpu.compiler.analysis.reduction_coverage import classify, require_legal
    require_legal(classify(programs_along_domain=1))
    sweep_count = f"tl.cdiv({row_name}, ROW)"

    def sweep(inputs) -> _Registers:
        """Emit one chunk's loads and fused prologue.

        Prologue arithmetic is pure, so recomputing it in the second sweep
        reproduces the same values without any intermediate buffer; that is
        what keeps a fused add -> softmax correct across sweeps.
        """
        sweep_registers = _Registers(graph, names)
        # A rank-0 operand is a host number bound as a register; the chunked
        # sweeps are separate register maps, so they need the same binding or
        # an epilogue that divides by a scalar references an undefined pointer
        # (review P2, 2026-10-26).
        sweep_registers.variables.update(scalar_formals)
        sweep_registers.lines.append(f"    {offsets} = {chunk} * ROW + tl.arange(0, ROW)")
        sweep_registers.lines.append(f"    {mask} = {offsets} < {row_name}")

        def sweep_load(value, address_override=None):
            if address_override is not None:
                address = _view_offset(
                    value, address_override, layouts, names,
                    sweep_registers.lines
                )
                return sweep_registers.bind(
                    value,
                    _load(graph, value, address, mask),
                    _comment(value, group),
                )
            return load(value, into=sweep_registers)

        def sweep_value(value, address_override=None):
            """One prologue element's tile, threading the chain's maps.

            A rotation remaps the address; a cast or pointwise op
            recomputes at the one mapped address (13.7's prologue
            recompute, extended to sit inside a rotation chain).
            """
            node = next(
                (n for n in group.prologue if n.output == value), None
            )
            if node is not None and node.op == "move_first_last":
                source = graph.values[node.inputs[0]]
                position = address_override or offsets_of(value)
                mapped = sweep_registers.emit(
                    f"(({position}) % ({source.shape[0]})) * "
                    f"({product(source.shape[1:])}) + "
                    f"(({position}) // ({source.shape[0]}))",
                    "moved index",
                )
                value = sweep_value(node.inputs[0], mapped)
                return sweep_registers.bind(
                    node.output,
                    value,
                    _comment(node.output, group),
                )
            if node is not None:
                values = [
                    sweep_value(operand, address_override)
                    if address_override is not None
                    else sweep_registers.variables[operand]
                    if operand in sweep_registers.variables
                    else sweep_value(operand)
                    for operand in node.inputs
                ]
                if node.op in {"cast_like", "cast_to", "to_fp32"}:
                    dtype = DTYPE_TO_TRITON[graph.values[node.output].dtype]
                    expression = values[0]
                    if f".to({dtype})" not in expression:
                        expression = f"({expression}).to({dtype})"
                else:
                    expression = _pointwise_expression(
                        node, values, sweep_registers.dtype_of(node.output)
                    )
                return sweep_registers.bind(
                    node.output,
                    expression,
                    _comment(node.output, group),
                )
            return sweep_load(value, address_override)

        # Outputs first: walking a chain's tail binds its inner links, so
        # the head never emits a plain load that the chain supersedes.
        for node in reversed(group.prologue):
            if node.output not in sweep_registers.variables:
                sweep_value(node.output)
        for value in inputs:
            if value not in sweep_registers.variables:
                sweep_load(value)
        return sweep_registers

    if anchor.op in {"rms_norm", "rmsnorm_hf"}:
        total = names.fresh("sum_of_squares")
        header = [
            f"    {index} = tl.program_id(0)",
            f"    {base} = {index} * {row_name}",
            f"    {total} = tl.zeros([], dtype={accumulator})",
            f"    for {chunk} in range(0, {sweep_count}):",
        ]
        first = sweep(anchor.inputs[:1])
        normalized = first.emit(
            f"{first.variables[anchor.inputs[0]]}.to({accumulator})",
            "rms_norm: normalize in the declared acc",
        )
        first.lines.append(
            f"    {total} += tl.sum({normalized} * {normalized}, axis=0)"
            "    # rms_norm: fold this chunk's squares"
        )
        mean = names.fresh("mean_of_squares")
        between = [
            f"    {mean} = {total} / {row_name}    # rms_norm: mean of squares"
        ]
        second = sweep(anchor.inputs)
        scaled = second.emit(
            f"{second.variables[anchor.inputs[0]]}.to({accumulator}) * "
            + _laned(offsets, f"tl.rsqrt({mean} + {RMS_EPS})"),
            "rms_norm: rsqrt(mean + eps)",
        )
        weight = second.emit(
            f"{second.variables[anchor.inputs[1]]}.to({accumulator})",
            "rms_norm: weight, widened too",
        )
        if anchor.op == "rmsnorm_hf":
            input_dtype = DTYPE_TO_TRITON[graph.values[anchor.inputs[0]].dtype]
            scaled = second.emit(
                f"{scaled}.to({input_dtype}).to({accumulator})",
                "rmsnorm_hf: round normalized activation before weight",
            )
        expression = f"({scaled} * {weight})"
    else:
        running_max = names.fresh("running_max")
        running_sum = names.fresh("running_sum")
        header = [
            f"    {index} = tl.program_id(0)",
            f"    {base} = {index} * {row_name}",
            f"    {running_max} = tl.full([], -float('inf'), {accumulator})",
            f"    {running_sum} = tl.zeros([], dtype={accumulator})",
            f"    for {chunk} in range(0, {sweep_count}):",
        ]
        first = sweep(anchor.inputs[:1])
        normalized = first.emit(
            f"{first.variables[anchor.inputs[0]]}.to({accumulator})",
            "softmax: in the declared acc",
        )
        # Masked lanes must contribute nothing: the chunk maximum reads them
        # as -inf and the running sum adds nothing for them.
        chunk_max = first.emit(
            f"tl.max(tl.where({mask}, {normalized}, -float('inf')), axis=0)",
            "softmax: chunk maximum excludes padding",
        )
        new_max = first.emit(
            f"tl.where({chunk_max} > {running_max}, {chunk_max}, {running_max})",
            "softmax: running maximum",
        )
        first.lines.append(
            f"    {running_sum} = {running_sum} * tl.exp({running_max} - {new_max})"
            f" + tl.sum(tl.where({mask}, tl.exp({normalized} - {new_max}), 0.0), axis=0)"
            "    # softmax: rescale the running sum to the new maximum"
        )
        first.lines.append(f"    {running_max} = {new_max}")
        between = []
        second = sweep(anchor.inputs)
        exponentials = second.emit(
            f"tl.where({mask}, tl.exp({second.variables[anchor.inputs[0]]}.to({accumulator}) - "
            + _laned(offsets, running_max) + "), 0.0)",
            "softmax: exp, padding contributes nothing",
        )
        expression = f"({exponentials} / {running_sum})"

    anchor_dtype = graph.values[anchor.output].dtype
    if anchor_dtype != "fp32":
        expression = f"{expression}.to({DTYPE_TO_TRITON[anchor_dtype]})"

    value = second.bind(anchor.output, expression, _comment(anchor.output, group))

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        if node in rotation_chain:
            continue
        values = [
            second.variables[operand]
            if operand in second.variables
            else load(operand, into=second)
            for operand in node.inputs
        ]
        previous = sharing.previous(node, values, second.dtype_of(node.output))
        if previous is not None:
            second.variables[node.output] = previous
            continue
        if node.op in {"cast_like", "cast_to", "to_fp32"}:
            dtype = DTYPE_TO_TRITON[graph.values[node.output].dtype]
            expression = values[0]
            if f".to({dtype})" not in expression:
                expression = f"({expression}).to({dtype})"
        else:
            expression = _pointwise_expression(
                node, values, second.dtype_of(node.output)
            )
        value = second.bind(
            node.output,
            expression,
            _comment(node.output, group),
        )
        sharing.record(node, values, second.dtype_of(node.output), value)

    dtype = graph.values[result].dtype
    if dtype != "fp32":
        value = second.emit(
            f"{value}.to({DTYPE_TO_TRITON[dtype]})", f"{result}: result dtype"
        )

    store_index = f"{base} + {offsets}"
    # The load map of one rotation sends an index of its OUTPUT to an
    # index of its input. A store needs the inverse: each link's inverse
    # has the closed form (j % rest)*d0 + j//rest over that link's source
    # shape, applied in chain order from the anchor outward. The same
    # composition the single-tile path applies; here it lives inside the
    # sweep because `offsets` is per-chunk.
    for node in rotation_chain:
        source = graph.values[node.inputs[0]]
        first_dim = source.shape[0]
        rest = product(source.shape[1:])
        store_index = second.emit(
            f"(({store_index}) % ({rest})) * ({first_dim})"
            f" + (({store_index}) // ({rest}))",
            "store rotation",
        )
    second.lines.append(
        f"    tl.store({_pointer(result)} + {store_index}, {value}, "
        f"mask={mask})"
    )

    body = (
        header
        + [f"    {line}" for line in first.lines]
        + between
        + [f"    for {chunk} in range(0, {sweep_count}):"]
        + [f"    {line}" for line in second.lines]
    )
    body, grid = row_limited_body(body, f"({row_count},)", config,
                                  count=kernel_row_count, row=index, grid_count=row_count)
    kernel = _make_kernel(
        name,
        "rowwise",
        group,
        pairs,
        body,
        grid,
        graph,
        launch_options=(("enable_fp_fusion", False),) if anchor.op in {"rms_norm", "rmsnorm_hf"} else (),
    )
    if autotune_fold:
        from flowgpu.compiler.backend.autotune import decorate_fold
        kernel = decorate_fold(kernel, config, 'rowwise', ('ROW',),
                               (str(row), product(leading) if leading else '1'))
    return kernel


def _matmul_kernel(graph, group, memory, name, platform: PlatformConfig | None = None, *, autotune=False) -> TritonKernel:
    """A blocked matmul, accumulating in fp32.

    Supports direct 2D grid and 1D L2 Cache Group Swizzling (group_m > 1).
    """
    platform = platform or PlatformConfig("cuda")
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]

    left, right = anchor.inputs
    rows = graph.values[left].shape[0]
    depth = graph.values[left].shape[1]
    columns = graph.values[right].shape[1]

    # Rank-0 operands are host numbers, not tiles: they become plain kernel
    # arguments bound straight into the register map, exactly as the pointwise
    # path does.  A reduction whose epilogue divides by a scalar (dynamic_quant's
    # scale) otherwise has no way to reach one.
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    pairs = [
        (_pointer(value), Argument("tensor", value)) for value in tensors
    ]
    pairs += [
        (_pointer(value), Argument("tensor", value)) for value in results
    ]
    scalar_formals = {}
    used_formals = ({"ROW"} | dimension_symbols(graph)
                    | {parameter.split(":")[0] for parameter, _ in pairs})
    # `names` must exist before the scalar formals are reserved: a rank-0 operand
    # reaches this kernel only when an epilogue carrying one fused into the
    # matmul, and that path reserved a formal against an unbound `names`.
    names = Names(graph)
    for value in scalars:
        literal = _constexpr_literal(value)
        if literal is not None:
            scalar_formals[value] = literal
            continue
        formal = unique_formal(identifier(value), used_formals)
        names.reserve(formal)
        scalar_formals[value] = formal
        pairs.append((formal, Argument("scalar", value)))
    for dimension in (rows, depth, columns):
        if is_symbol(dimension) and not any(
            parameter == str(dimension) for parameter, _ in pairs
        ):
            pairs.append((str(dimension), Argument("value", str(dimension))))
    original_dims = (rows, depth, columns)
    names = Names(graph)
    tune_keys = []
    if autotune:
        rows, depth, columns = [names.fresh(hint) for hint in ("FG_M", "FG_K", "FG_N")]
        dimension_keys = [rows, depth, columns]
        pairs += [(f"{key}: tl.constexpr", Argument("value", str(dim)))
                  for key, dim in zip(dimension_keys, original_dims)]
        device_key = names.fresh("FG_DEVICE_KEY")
        tune_keys = dimension_keys + [device_key]
        pairs.append((f"{device_key}: tl.constexpr", Argument("device_key", "")))
    pid_m = names.fresh("pid_m")
    pid_n = names.fresh("pid_n")
    offs_m = names.fresh("offs_m")
    offs_n = names.fresh("offs_n")
    offs_k = names.fresh("offs_k")
    depth_offsets = names.fresh("depth_offsets")
    depth_index = names.fresh("k")
    accumulator = names.fresh("accumulator")
    tile_of = {"a": names.fresh("a"), "b": names.fresh("b")}

    pairs.append(("BLOCK_M: tl.constexpr", Argument("value", "BLOCK_M")))
    pairs.append(("BLOCK_N: tl.constexpr", Argument("value", "BLOCK_N")))
    pairs.append(("BLOCK_K: tl.constexpr", Argument("value", "BLOCK_K")))
    if platform.group_m > 1:
        pairs.append(("GROUP_M: tl.constexpr", Argument("value", str(platform.group_m))))

    registers = _Registers(graph, names)
    # A rank-0 operand is a host number, not a pointer: binding the formal into
    # the register map lets a fused epilogue read it exactly where it reads a
    # tile, instead of falling through to `result_tile` and inventing `s_ptr`.
    registers.variables.update(scalar_formals)

    if platform.group_m > 1:
        # Grouped program order: swizzles pids to improve L2 cache hit rate on matrix B
        pid = names.fresh("pid")
        num_pid_m = names.fresh("num_pid_m")
        num_pid_n = names.fresh("num_pid_n")
        num_pid_in_group = names.fresh("num_pid_in_group")
        group_id = names.fresh("group_id")
        first_pid_m = names.fresh("first_pid_m")
        group_size_m = names.fresh("group_size_m")

        lines = [
            f"    {pid} = tl.program_id(0)",
            f"    {num_pid_m} = tl.cdiv({rows}, BLOCK_M)",
            f"    {num_pid_n} = tl.cdiv({columns}, BLOCK_N)",
            f"    {num_pid_in_group} = GROUP_M * {num_pid_n}",
            f"    {group_id} = {pid} // {num_pid_in_group}",
            f"    {first_pid_m} = {group_id} * GROUP_M",
            f"    {group_size_m} = min({num_pid_m} - {first_pid_m}, GROUP_M)",
            f"    {pid_m} = {first_pid_m} + ({pid} % {group_size_m})",
            f"    {pid_n} = ({pid} % {num_pid_in_group}) // {group_size_m}",
            f"    {offs_m} = {pid_m} * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"    {offs_n} = {pid_n} * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"    {offs_k} = tl.arange(0, BLOCK_K)",
            f"    {accumulator} = tl.zeros((BLOCK_M, BLOCK_N), dtype={_accumulator('matmul')})",
            f"    for {depth_index} in range(0, tl.cdiv({depth}, BLOCK_K)):",
            f"        {depth_offsets} = {depth_index} * BLOCK_K + {offs_k}",
        ]
    else:
        lines = [
            f"    {pid_m} = tl.program_id(0)",
            f"    {pid_n} = tl.program_id(1)",
            f"    {offs_m} = {pid_m} * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"    {offs_n} = {pid_n} * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"    {offs_k} = tl.arange(0, BLOCK_K)",
            f"    {accumulator} = tl.zeros((BLOCK_M, BLOCK_N), dtype={_accumulator('matmul')})",
            f"    for {depth_index} in range(0, tl.cdiv({depth}, BLOCK_K)):",
            f"        {depth_offsets} = {depth_index} * BLOCK_K + {offs_k}",
        ]

    def access(value, role) -> tuple:
        """The offsets, mask and tile name for one operand's shape."""
        if role == "a":
            return (
                f"{offs_m}[:, None] * {depth} + {depth_offsets}[None, :]",
                f"({offs_m}[:, None] < {rows}) & "
                f"({depth_offsets}[None, :] < {depth})",
                tile_of["a"],
            )

        return (
            f"{depth_offsets}[:, None] * {columns} + {offs_n}[None, :]",
            f"({depth_offsets}[:, None] < {depth}) & "
            f"({offs_n}[None, :] < {columns})",
            tile_of["b"],
        )

    tiles = {}
    tile_cache = {}
    prologue_nodes = {n.output: n for n in group.prologue}

    def operand_tile(value, role):
        key = (value, role)
        if key in tile_cache:
            return tile_cache[key]
        node = prologue_nodes.get(value)
        if node is not None:
            arguments = [operand_tile(v, role) for v in node.inputs]
            expression = _pointwise_expression(node, arguments, registers.dtype_of(node.output))
        else:
            offsets, mask, _ = access(value, role)
            expression = f"tl.load({_pointer(value)} + {offsets}, mask={mask}, other=0.0)"
        tile = names.fresh(identifier(value) + "_" + role + "_tile")
        lines.append(f"        {tile} = {expression}    # {_comment(value, group)}")
        tile_cache[key] = tile
        return tile

    a_tile = operand_tile(left, "a")
    b_tile = operand_tile(right, "b")

    # fp32 storage must not silently select Triton's default TF32 products.
    precision = ', input_precision="ieee"' if graph.values[left].dtype == "fp32" else ""
    lines.append(
        f"        {accumulator} = tl.dot({a_tile}, {b_tile}, "
        f"{accumulator}{precision})"
    )

    dtype = graph.values[anchor.output].dtype
    fused = (
        ", fused by 13.6: no buffer"
        if anchor.output in group.fused_values
        else ""
    )
    value = registers.bind(
        anchor.output,
        f"{accumulator}.to({DTYPE_TO_TRITON[dtype]})",
        f"{anchor.output}: fp32 accumulation, {dtype} result{fused}",
    )

    def result_tile(value) -> str:
        """An operand of an epilogue, in the index space of the result.

        `matmul(a, b)` followed by a bias add is the canonical epilogue, and
        the bias is neither an operand of the matmul nor a value the group
        computed: it is addressed like the result, which is a third access
        pattern in a kernel that already had two.
        """
        tile = names.fresh(identifier(value))
        lines.append(
            f"    {tile} = tl.load({_pointer(value)} + {offs_m}[:, None] * "
            f"{columns} + {offs_n}[None, :], mask=({offs_m}[:, None] < "
            f"{rows}) & ({offs_n}[None, :] < {columns}), other=0.0)    "
            f"# {_comment(value, group)}"
        )
        tiles[value] = tile

        return tile

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        arguments = [
            registers.variables[operand]
            if operand in registers.variables
            else tiles[operand]
            if operand in tiles
            else result_tile(operand)
            for operand in node.inputs
        ]
        previous = sharing.previous(node, arguments, registers.dtype_of(node.output))
        if previous is not None:
            registers.variables[node.output] = previous
            continue
        value = registers.bind(
            node.output,
            _pointwise_expression(node, arguments, registers.dtype_of(node.output)),
            _comment(node.output, group),
        )
        sharing.record(node, arguments, registers.dtype_of(node.output), value)

    registers.lines.append(
        f"    tl.store({_pointer(result)} + {offs_m}[:, None] * {columns} + "
        f"{offs_n}[None, :], {value}, mask=({offs_m}[:, None] < {rows}) & "
        f"({offs_n}[None, :] < {columns}))"
    )

    # The grid-stride wrap runs over the *complete* body.  Wrapping the preamble
    # and then appending the k-loop body at direct-grid indentation leaves the
    # loads, the dot and the store outside both stride loops: the kernel still
    # parses, and every name it uses is bound somewhere, but `accumulator` is
    # bound inside the loops and read after them, which Triton rejects on the
    # device as `NameError: accumulator is not defined`.  Two tests asserted the
    # presence of the loop headers and the program-count reads without asking
    # where the body landed, so this survived.
    body = lines + registers.lines
    if platform.group_m > 1:
        blocks = f"triton.cdiv({rows}, BLOCK_M) * triton.cdiv({columns}, BLOCK_N)"
        body, launch_grid = grid_limited_body(
            body, f"({blocks},)", platform, count=blocks, numel=blocks,
            block="1", grid_block="1", tile=names.fresh("tile"),
            bound=blocks.replace("triton.cdiv", "tl.cdiv"), tiles=blocks)
    else:
        body, launch_grid = grid_limited_body_2d(
            body, f"(triton.cdiv({rows}, BLOCK_M), triton.cdiv({columns}, BLOCK_N))",
            platform, rows_of_blocks=f"tl.cdiv({rows}, BLOCK_M)",
            columns_of_blocks=f"tl.cdiv({columns}, BLOCK_N)", row=pid_m, column=pid_n,
            host=(f"triton.cdiv({rows}, BLOCK_M)", f"triton.cdiv({columns}, BLOCK_N)"))

    # A platform GEMM policy may pin the measured pipeline resources; the
    # default keeps the historical `num_warps`/`num_stages` untouched so every
    # existing emitted module stays byte-identical.
    matmul_warps = (platform.matmul_num_warps or platform.num_warps)
    matmul_stages = (platform.matmul_num_stages or platform.num_stages)
    # Small-M GEMMs have too few row tiles to fill the device at the measured
    # 128-row default: 64x512 is four CTAs on a 132-SM H100 and each tile
    # wastes half its rows. A 16-row tile trades reuse for occupancy; the
    # source still spells BLOCK_M/N/K and the host substitutes these numbers,
    # so the module-wide policy for large GEMMs is unchanged.
    tile = None
    dims = (rows, depth, columns)
    if not autotune and all(str(dim).isdigit() for dim in dims):
        row_extent, depth_extent, column_extent = (int(dim) for dim in dims)
        if (platform.name == "cuda" and row_extent <= 64 and depth_extent >= 64
                and column_extent >= 128 and (platform.matmul_block_m or 0) >= 64):
            block_n = 64 if column_extent >= 64 else 32
            tile = (16, block_n, 64)
            matmul_warps, matmul_stages = 4, 3
        elif (platform.name == "cuda" and platform.matmul_block_m == 128
              and platform.matmul_block_n == 256):
            # Device policy asks for the H100 128x256 tile. Clamp each literal
            # extent to a power of two so a 64-column GEMM does not launch a
            # 256-column block that is mostly mask.
            block_m = max(16, min(128, 1 << (row_extent - 1).bit_length()))
            block_n = max(16, min(256, 1 << (column_extent - 1).bit_length()))
            tile = (block_m, block_n, platform.matmul_block_k or 64)
            matmul_warps, matmul_stages = 8, 3
    kernel = _make_kernel(
        name,
        "matmul",
        group,
        pairs,
        body,
        launch_grid,
        graph,
        launch_options=(("num_warps", matmul_warps), ("num_stages", matmul_stages)),
    )
    if tile is not None:
        kernel = replace(kernel, tile=tile)
    if autotune:
        from flowgpu.compiler.backend.autotune import render_configs
        table = f"{name}_configs"
        configs = render_configs(platform.name).replace("FLOWGPU_AUTOTUNE_CONFIGS", table)
        decorator = (f"@triton.autotune(configs={table}, key={tune_keys!r}, "
                     "**_FG_AUTOTUNE_CACHE_KWARGS)\n@triton.jit")
        grid = kernel.grid
        for key, dim in zip(dimension_keys, original_dims):
            grid = grid.replace(key, str(dim))
        for block in ("BLOCK_M", "BLOCK_N"):
            grid = grid.replace(block, f"META['{block}']")
        kept = [(param, arg) for param, arg in zip(kernel.parameters, kernel.launch)
                if param.split(":")[0] not in {"BLOCK_M", "BLOCK_N", "BLOCK_K"}]
        kernel = replace(kernel, source=configs + "\n" + kernel.source.replace("@triton.jit", decorator),
                         launch=tuple(arg for _, arg in kept), grid="lambda META: " + grid,
                         launch_options=())
    return kernel


def _transpose_gather_kernel(graph, group, memory, name, config) -> TritonKernel:
    """Read permuted source coordinates, write a contiguous output tile.

    The mapping is an experimental transfer from the competition copy kernels;
    selecting it does not grant unvalidated targets a transpose whitelist.
    """
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    result = results[0]
    rows, columns = graph.values[anchor.inputs[0]].shape
    names = Names(graph)
    offset, mask, source_offset = (names.fresh(n) for n in ("offset", "mask", "source_offset"))
    count = names.fresh("numel")
    pairs = [(_pointer(v), Argument("tensor", v)) for v in operands + results]
    for dimension in (rows, columns):
        if is_symbol(dimension) and not any(p == str(dimension) for p, _ in pairs):
            pairs.append((str(dimension), Argument("value", str(dimension))))
    pairs.extend([(count, Argument("value", product([rows, columns]))),
                  ("BLOCK: tl.constexpr", Argument("value", str(config.default_block_size)))])
    layouts = _input_layouts(graph, memory, operands, names, pairs)
    registers = _Registers(graph, names)
    lines = [f"    {offset} = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
             f"    {mask} = {offset} < {count}",
             f"    {source_offset} = ({offset} % {rows}) * {columns} + {offset} // {rows}"]

    def load(value, logical):
        address = _view_offset(value, logical, layouts, names, registers.lines)
        return registers.bind(value, _load(graph, value, address, mask), _comment(value, group))

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.prologue:
        args = [registers.variables[v] if v in registers.variables else load(v, source_offset)
                for v in node.inputs]
        previous = sharing.previous(node, args, registers.dtype_of(node.output))
        if previous is not None:
            registers.variables[node.output] = previous
            continue
        bound = registers.bind(node.output, _pointwise_expression(node, args, registers.dtype_of(node.output)),
                               _comment(node.output, group))
        sharing.record(node, args, registers.dtype_of(node.output), bound)
    source = anchor.inputs[0]
    value = registers.variables[source] if source in registers.variables else load(source, source_offset)
    registers.bind(anchor.output, value, "transpose in output index space")
    for node in group.epilogue:
        args = [registers.variables[v] if v in registers.variables else load(v, offset)
                for v in node.inputs]
        previous = sharing.previous(node, args, registers.dtype_of(node.output))
        if previous is not None:
            registers.variables[node.output] = previous
            continue
        bound = registers.bind(node.output, _pointwise_expression(node, args, registers.dtype_of(node.output)),
                               _comment(node.output, group))
        sharing.record(node, args, registers.dtype_of(node.output), bound)
    registers.lines.append(f"    tl.store({_pointer(result)} + {offset}, {registers.variables[result]}, mask={mask})")
    return _make_kernel(name, "transpose", group, pairs, lines + registers.lines,
                        f"(triton.cdiv({product([rows, columns])}, {config.default_block_size}),)", graph,
                        launch_options=(("num_warps", config.num_warps),))


def _transpose_kernel(graph, group, memory, name, platform_config=None) -> TritonKernel:
    """A 2D index permutation, 13.3's own restriction on `transpose`."""
    config = resolve_platform() if platform_config is None else platform_config
    if config.transpose_mapping == "output_gather":
        return _transpose_gather_kernel(graph, group, memory, name, config)
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]

    source = anchor.inputs[0]
    rows, columns = graph.values[source].shape
    names = Names(graph)

    # Rank-0 operands are host numbers, not tiles: they become plain kernel
    # arguments bound straight into the register map, exactly as the pointwise
    # path does.  A reduction whose epilogue divides by a scalar (dynamic_quant's
    # scale) otherwise has no way to reach one.
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    pairs = [
        (_pointer(value), Argument("tensor", value)) for value in tensors
    ]
    pairs += [
        (_pointer(value), Argument("tensor", value)) for value in results
    ]
    scalar_formals = {}
    used_formals = ({"ROW"} | dimension_symbols(graph)
                    | {parameter.split(":")[0] for parameter, _ in pairs})
    for value in scalars:
        literal = _constexpr_literal(value)
        if literal is not None:
            scalar_formals[value] = literal
            continue
        formal = unique_formal(identifier(value), used_formals)
        names.reserve(formal)
        scalar_formals[value] = formal
        pairs.append((formal, Argument("scalar", value)))
    for dimension in (rows, columns):
        if is_symbol(dimension) and not any(
            parameter == str(dimension) for parameter, _ in pairs
        ):
            pairs.append((str(dimension), Argument("value", str(dimension))))
    pid_m = names.fresh("pid_m")
    pid_n = names.fresh("pid_n")
    offs_m = names.fresh("offs_m")
    offs_n = names.fresh("offs_n")

    pairs.append(("BLOCK_M: tl.constexpr", Argument("value", "BLOCK_M")))
    pairs.append(("BLOCK_N: tl.constexpr", Argument("value", "BLOCK_N")))

    registers = _Registers(graph, names)
    registers.variables.update(scalar_formals)
    lines = [
        f"    {pid_m} = tl.program_id(0)",
        f"    {pid_n} = tl.program_id(1)",
        f"    {offs_m} = {pid_m} * BLOCK_M + tl.arange(0, BLOCK_M)",
        f"    {offs_n} = {pid_n} * BLOCK_N + tl.arange(0, BLOCK_N)",
    ]

    tiles = {}

    def load(value):
        tile = identifier(value)
        lines.append(
            f"    {tile} = tl.load({_pointer(value)} + {offs_m}[:, None] * "
            f"{columns} + {offs_n}[None, :], mask=({offs_m}[:, None] < {rows}) & "
            f"({offs_n}[None, :] < {columns}), other=0.0)    "
            f"# {_comment(value, group)}"
        )
        tiles[value] = tile

        return tile

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.prologue:
        arguments = [
            tiles[operand] if operand in tiles else load(operand)
            for operand in node.inputs
        ]
        previous = sharing.previous(node, arguments, registers.dtype_of(node.output))
        if previous is not None:
            tiles[node.output] = previous
            continue
        tile = identifier(node.output)
        lines.append(
            f"    {tile} = "
            + _pointwise_expression(
                node, arguments, registers.dtype_of(node.output)
            )
            + f"    # {_comment(node.output, group)}"
        )
        sharing.record(node, arguments, registers.dtype_of(node.output), tile)
        tiles[node.output] = tile
        tiles[node.output] = tile

    if source not in tiles:
        load(source)

    def output_tile(value) -> str:
        """An operand of an epilogue, in the index space of the result."""
        tile = names.fresh(identifier(value))
        lines.append(
            f"    {tile} = tl.load({_pointer(value)} + {offs_n}[:, None] * "
            f"{rows} + {offs_m}[None, :], mask=({offs_n}[:, None] < "
            f"{columns}) & ({offs_m}[None, :] < {rows}), other=0.0)    "
            f"# {_comment(value, group)}"
        )
        tiles[value] = tile

        return tile

    transposed = registers.emit(f"tl.trans({tiles[source]})", "transpose")
    registers.variables[anchor.output] = transposed
    value = transposed

    for node in group.epilogue:
        arguments = [
            registers.variables[operand]
            if operand in registers.variables
            else output_tile(operand)
            for operand in node.inputs
        ]
        previous = sharing.previous(node, arguments, registers.dtype_of(node.output))
        if previous is not None:
            registers.variables[node.output] = previous
            continue
        value = registers.bind(
            node.output,
            _pointwise_expression(node, arguments, registers.dtype_of(node.output)),
            _comment(node.output, group),
        )
        sharing.record(node, arguments, registers.dtype_of(node.output), value)

    registers.lines.append(
        f"    tl.store({_pointer(result)} + {offs_n}[:, None] * {rows} + "
        f"{offs_m}[None, :], {value}, mask=({offs_n}[:, None] < {columns}) & "
        f"({offs_m}[None, :] < {rows}))"
    )

    body, grid = grid_limited_body_2d(
        lines + registers.lines,
        f"(triton.cdiv({rows}, BLOCK_M), triton.cdiv({columns}, BLOCK_N))",
        config, rows_of_blocks=f"tl.cdiv({rows}, BLOCK_M)",
        columns_of_blocks=f"tl.cdiv({columns}, BLOCK_N)", row=pid_m, column=pid_n,
        host=(f"triton.cdiv({rows}, BLOCK_M)", f"triton.cdiv({columns}, BLOCK_N)"))
    return _make_kernel(
        name,
        "transpose",
        group,
        pairs,
        body,
        grid,
        graph,
    )


def _rope_kernel(graph, group, memory, name, platform_config=None) -> TritonKernel:
    """Split-half rotation with explicit, full-shape cosine and sine tables."""
    config = resolve_platform() if platform_config is None else platform_config
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    result = results[0]
    shape = graph.values[anchor.inputs[0]].shape
    names = Names(graph)
    offsets, mask, partner = (names.fresh(n) for n in ("offsets", "mask", "partner"))
    count, width = names.fresh("numel"), names.fresh("width")
    first = names.fresh("first_half")
    pairs = [(_pointer(v), Argument("tensor", v)) for v in operands + results]
    block = "BLOCK" if config.default_block_size == BLOCK else str(config.default_block_size)
    pairs += [(count, Argument("value", product(shape))),
              (width, Argument("value", str(shape[-1]))),
              ("BLOCK: tl.constexpr", Argument("value", block))]
    lines = [f"    {offsets} = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
             f"    {mask} = {offsets} < {count}",
             f"    {first} = ({offsets} % {width}) < ({width} // 2)",
             f"    {partner} = tl.where({first}, {offsets} + {width} // 2, {offsets} - {width} // 2)"]
    prologue = {node.output: node for node in group.prologue}
    cache = {}

    def element(value, index):
        key = (value, index)
        if key in cache:
            return cache[key]
        if value in prologue:
            node = prologue[value]
            args = [element(v, index) for v in node.inputs]
            expression = _pointwise_expression(node, args, graph.values[value].dtype)
        else:
            expression = _load(graph, value, index, mask)
        variable = names.numbered("t")
        lines.append(f"    {variable} = {expression}")
        cache[key] = variable
        return variable

    x, cosine, sine = anchor.inputs
    a, b = element(x, offsets), element(x, partner)
    c, d = element(cosine, offsets), element(sine, offsets)
    rotated = names.numbered("t")
    dtype = DTYPE_TO_TRITON[graph.values[anchor.output].dtype]
    lines.append(f"    {rotated} = ({a}.to(tl.float32) * {c}.to(tl.float32) + tl.where({first}, -{b}.to(tl.float32), {b}.to(tl.float32)) * {d}.to(tl.float32)).to({dtype})")
    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    computed = {anchor.output: rotated}
    for node in group.epilogue:
        args = [computed[v] if v in computed else element(v, offsets) for v in node.inputs]
        previous = sharing.previous(node, args, graph.values[node.output].dtype)
        if previous is not None:
            computed[node.output] = previous
            continue
        variable = names.numbered("t")
        lines.append(f"    {variable} = {_pointwise_expression(node, args, graph.values[node.output].dtype)}")
        computed[node.output] = variable
        sharing.record(node, args, graph.values[node.output].dtype, variable)
    lines.append(f"    tl.store({_pointer(result)} + {offsets}, {computed[result]}, mask={mask})")
    grid = f"(triton.cdiv({product(shape)}, {block}),)"
    lines, grid = grid_limited_body(lines, grid, config, count=count,
                                   numel=product(shape), block="BLOCK",
                                   grid_block=block, tile=names.fresh("tile"))
    options = (("num_warps", config.num_warps),) if config.num_warps != 4 or config.name != "cuda" else ()
    return _make_kernel(name, "rope", group, pairs, lines, grid, graph,
                        launch_options=options)






def _layout_device_kernel(graph, group, memory, name, config) -> TritonKernel:
    """A layout operator as a device copy kernel.

    The IR semantics are unchanged; only the *lowering* moves.  Every output
    element is computed from its flat output offset: ``reshape`` is an identity
    copy, ``split`` adds the part offset within the literal row, ``concat``
    selects between the two inputs by column, and ``interleave`` selects by
    parity.  The last-axis widths are literals because the frontend refuses a
    symbolic last axis for split/concat/interleave; ``reshape`` needs none.

    This is opt-in through ``PlatformConfig.layout_on_device`` so the default
    emitted bytes (host lowering) stay identical.
    """
    assert len(group.members) == 1
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    names = Names(graph)
    variables = {value: names.fresh(identifier(value)) for value in operands + results}
    pairs = [(_pointer(value), Argument("tensor", value))
             for value in operands + results]
    count = names.fresh("n")
    pairs.append((count, Argument("value", product(graph.values[anchor.output].shape))))
    # BLOCK is the platform's policy value (default_block_size), not a new
    # scheduling decision made here: the portability audit keeps every such
    # choice in the policy table.
    block = config.default_block_size
    pairs.append(("BLOCK: tl.constexpr", Argument("value", str(block))))

    out_ptr = _pointer(anchor.output)
    offsets = names.fresh("offs")
    mask = names.fresh("mask")
    lines = [
        f"    {offsets} = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        f"    {mask} = {offsets} < {count}",
    ]

    inputs = [_pointer(value) for value in anchor.inputs]
    if anchor.op == "reshape":
        value = f"tl.load({inputs[0]} + {offsets}, mask={mask}, other=0)"
    elif anchor.op == "split":
        out_width = int(graph.values[anchor.output].shape[-1])
        in_width = int(graph.values[anchor.inputs[0]].shape[-1])
        low = out_width * anchor.part
        row, column = names.fresh("row"), names.fresh("column")
        lines += [f"    {row} = {offsets} // {out_width}",
                  f"    {column} = {offsets} % {out_width}"]
        value = (f"tl.load({inputs[0]} + {row} * {in_width} + {low} + {column}, "
                 f"mask={mask}, other=0)")
    elif anchor.op == "concat":
        first = int(graph.values[anchor.inputs[0]].shape[-1])
        second = int(graph.values[anchor.inputs[1]].shape[-1])
        total = int(graph.values[anchor.output].shape[-1])
        row, column = names.fresh("row"), names.fresh("column")
        a_value, b_value = names.fresh("a_value"), names.fresh("b_value")
        lines += [f"    {row} = {offsets} // {total}",
                  f"    {column} = {offsets} % {total}",
                  f"    {a_value} = tl.load({inputs[0]} + {row} * {first} + {column}, "
                  f"mask={mask} & ({column} < {first}), other=0)",
                  f"    {b_value} = tl.load({inputs[1]} + {row} * {second} + "
                  f"({column} - {first}), mask={mask} & ({column} >= {first}), other=0)"]
        value = f"tl.where({column} < {first}, {a_value}, {b_value})"
    elif anchor.op == "interleave":
        width = int(graph.values[anchor.inputs[0]].shape[-1])
        total = int(graph.values[anchor.output].shape[-1])
        row, column = names.fresh("row"), names.fresh("column")
        a_value, b_value = names.fresh("a_value"), names.fresh("b_value")
        lines += [f"    {row} = {offsets} // {total}",
                  f"    {column} = {offsets} % {total}",
                  f"    {a_value} = tl.load({inputs[0]} + {row} * {width} + {column} // 2, "
                  f"mask={mask} & ({column} % 2 == 0), other=0)",
                  f"    {b_value} = tl.load({inputs[1]} + {row} * {width} + {column} // 2, "
                  f"mask={mask} & ({column} % 2 == 1), other=0)"]
        value = f"tl.where({column} % 2 == 0, {a_value}, {b_value})"
    else:
        raise AssertionError(f"no device layout lowering for {anchor.op!r}")
    lines.append(f"    tl.store({out_ptr} + {offsets}, {value}, mask={mask})")

    grid = f"(triton.cdiv({product(graph.values[anchor.output].shape)}, BLOCK),)"
    lines, grid = grid_limited_body(lines, grid, config,
                                    count=count,
                                    numel=product(graph.values[anchor.output].shape),
                                    block="BLOCK", grid_block=str(block),
                                    tile=names.fresh("tile"))
    return _make_kernel(name, "layout", group, pairs, lines, grid, graph)


def _layout_kernel(graph, group, memory, name) -> TritonKernel:
    """A layout operator as a host-side move into the planned slot.

    reshape/split/concat move elements between index spaces; they compute
    nothing. Emitting a device kernel for them would be a copy with extra
    steps, so the lowering is a host call -- a torch slicing copy, a cat,
    a view -- writing into the buffer 13.7 already allocated. That is the
    same boundary _library_kernel draws, minus the library.

    Two contracts the language layer defers to here, per Design.md v0.1.2:
    reshape's symbolic element conservation is checked at run time (the
    checker cannot prove N*D == 2048*N without knowing N), and the group
    is standalone by construction: fusion treats layout operators as
    opaque anchors, so no pointwise work rides along.
    """
    assert len(group.members) == 1
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    names = Names(graph)
    variables = {value: names.fresh(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    inputs = [variables[value] for value in anchor.inputs]
    out = variables[anchor.output]

    if anchor.op == "reshape":
        # The shape literal's symbolic dims are bound by the host from the
        # input tensors (8.3), so writing them as names is exactly how the
        # rowwise emitters pass lengths. The numel check is the run-time half
        # of the conservation contract: literal-only products were already
        # proven at the type layer, and a symbolic product is only decidable
        # here, with the tensors in hand.
        target = ", ".join(str(d) for d in graph.values[anchor.output].shape)
        for dimension in graph.values[anchor.output].shape:
            if is_symbol(dimension):
                # A symbol the body writes as a name is a kernel parameter,
                # passed by the host the same way lengths are (the host binds
                # it off an input tensor before the launch).
                pairs.append((str(dimension), Argument("value", str(dimension))))
        body = (
            f"    if {inputs[0]}.numel() != {out}.numel():\n"
            f"        raise ValueError('reshape does not conserve elements')\n"
            f"    {out}.copy_({inputs[0]}.reshape({target}))\n"
        )
    elif anchor.op == "split":
        # One part per group: each part is its own anchor, so this group
        # moves exactly one slice. The bounds are literals because the rule
        # refuses a symbolic last axis -- the slice is a view of the input,
        # and copy_ materializes it so the input's buffer may die after.
        src = variables[anchor.inputs[0]]
        width = int(graph.values[anchor.output].shape[-1])
        low, high = width * anchor.part, width * (anchor.part + 1)
        body = f"    {out}.copy_({src}[..., {low}:{high}])\n"
    elif anchor.op == "interleave":
        a, b = inputs
        body = (f"    {out}.copy_(torch.stack(({a}, {b}), dim=-1)"
                f".reshape({out}.shape))\n")
    else:
        a, b = inputs
        body = f"    torch.cat(({a}, {b}), dim=-1, out={out})\n"

    source = f"# host layout {anchor.op}\ndef {name}({', '.join(p for p, _ in pairs)}):\n" + body
    return TritonKernel(name, "layout", (anchor.op,), operands, results,
                        tuple(p for p, _ in pairs), tuple(arg for _, arg in pairs), "", source)


from flowgpu.compiler.backend.primitives import triton_kernel as _primitive_kernel
from flowgpu.compiler.backend.cache_store_backend import triton_kernel as _cache_store_kernel
from flowgpu.compiler.backend.cache_write_backend import cache_write_triton_kernel as _cache_write_kernel
from flowgpu.compiler.backend.cache_write_hsd_backend import cache_write_hsd_triton_kernel as _cache_write_hsd_kernel
from flowgpu.compiler.backend.cache_write_batch_backend import cache_write_batch_triton_kernel as _cache_write_batch_kernel
from flowgpu.compiler.backend.cache_write_range_backend import cache_write_range_triton_kernel as _cache_write_range_kernel
from flowgpu.compiler.backend.linear_decode_backend import triton_kernel as _linear_decode_kernel
from flowgpu.compiler.backend.pad_cat_backend import triton_kernel as _pad_cat_kernel
from flowgpu.compiler.backend.scan_backend import triton_kernel as _scan_kernel
from flowgpu.compiler.backend.patchify_backend import triton_kernel as _patchify_kernel
from flowgpu.compiler.backend.segmented_matmul_backend import triton_kernel as _segmented_kernel
from flowgpu.compiler.backend.scatter_backend import triton_kernel as _scatter_kernel
from flowgpu.compiler.backend.store_at_backend import triton_kernel as _store_at_kernel
from flowgpu.compiler.backend.segment_rank_backend import triton_kernel as _segment_rank_kernel
from flowgpu.compiler.backend.recurrence_backend import triton_kernel as _recurrence_kernel
from flowgpu.compiler.backend.topk_backend import (
    topk_triton_kernel as _topk_kernel, gather_triton_kernel as _gather_last_kernel)
from flowgpu.compiler.backend.sort_backend import sort_triton_kernel as _sort_kernel
from flowgpu.compiler.backend.mean_backend import mean_triton_kernel as _mean_kernel
from flowgpu.compiler.backend.layer_norm_backend import layer_norm_triton_kernel as _layer_norm_kernel
from flowgpu.compiler.backend.conv2d_backend import conv2d_triton_kernel as _conv2d_kernel
from flowgpu.compiler.backend.conv2d_same_backend import conv2d_same_triton_kernel as _conv2d_same_kernel
from flowgpu.compiler.backend.conv2d_stride2_backend import conv2d_stride2_triton_kernel as _conv2d_stride2_kernel
from flowgpu.compiler.backend.conv2d_stride2_backend import conv2d_stride2_same_triton_kernel as _conv2d_stride2_same_kernel
from flowgpu.compiler.backend.conv2d_ex_backend import conv2d_ex_triton_kernel as _conv2d_ex_kernel
from flowgpu.compiler.backend.attention_backend import attention_triton_kernel as _native_attention_kernel
from flowgpu.compiler.backend.reduce_with_index_backend import (
    triton_kernel as _reduce_with_index_kernel)

EMITTERS = {
    "primitive": _primitive_kernel,
    "pointwise": _pointwise_kernel,
    "rowwise": _rowwise_kernel,
    "matmul": _matmul_kernel,
    "transpose": _transpose_kernel,
    "rope": _rope_kernel,
    "layout": _layout_kernel,
    "cache_store": _cache_store_kernel,
    "cache_write": _cache_write_kernel,
    "cache_write_hsd": _cache_write_hsd_kernel,
    "cache_write_hsd_batch": _cache_write_batch_kernel,
    "cache_write_hsd_range": _cache_write_range_kernel,
    "linear_decode": _linear_decode_kernel,
    "pad_cat": _pad_cat_kernel,
    "scan": _scan_kernel,
    "patchify": _patchify_kernel,
    "segmented_matmul": _segmented_kernel,
    "scatter_by_index": _scatter_kernel,
    "store_at": _store_at_kernel,
    "topk": _topk_kernel,
    "sort": _sort_kernel,
    "mean": _mean_kernel,
    "layer_norm": _layer_norm_kernel,
    "conv2d": _conv2d_kernel,
    "conv2d_same": _conv2d_same_kernel,
    "conv2d_stride2": _conv2d_stride2_kernel,
    "conv2d_stride2_same": _conv2d_stride2_same_kernel,
    "conv2d_ex": _conv2d_ex_kernel,
    "axis_gather": _gather_last_kernel,
    "reduce_with_index": _reduce_with_index_kernel,
    "index_table": _segment_rank_kernel,
    "recurrence": _recurrence_kernel,
    "attention": _native_attention_kernel,
}


def _library_kernel(graph, group, memory, name, attention_impl="math") -> TritonKernel:
    """A visible library boundary, with no implicit fusion or dtype changes."""
    assert len(group.members) == 1
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    names = Names(graph)
    variables = {value: names.fresh(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    inputs = [variables[value] for value in anchor.inputs]
    out = variables[anchor.output]
    if anchor.op == "matmul":
        body = f"    torch.mm({inputs[0]}, {inputs[1]}, out={out})\n"
    elif anchor.op == "rms_norm":
        work = names.fresh("norm_work")
        body = (f"    {work} = {inputs[0]}.float()\n"
                f"    {out}.copy_({work} * torch.rsqrt({work}.square().mean(dim=-1, keepdim=True) + {RMS_EPS}) * {inputs[1]}.float())\n")
    elif anchor.op in {"attention", "causal_attention"} and attention_impl == "sdpa":
        q, k, v = inputs
        # Math is deliberate: automatic/fused dispatch need not preserve our
        # FP32 intermediate contract. A future fused path needs its own policy.
        causal = anchor.op == "causal_attention"
        is_4d = len(graph.values[anchor.inputs[0]].shape) == 4
        gqa = is_4d and graph.values[anchor.inputs[0]].shape[1] != graph.values[anchor.inputs[1]].shape[1]
        # causal goes as an EXPLICIT mask, not torch's is_causal flag: this
        # torch's is_causal with L != S is the offset-free triangle (probe on
        # A800: query i attends j <= i only), while the language's contract
        # is the kv-cache offset 0..i+(S-L). A flag whose meaning shifts per
        # argument shape is not a lowering; the mask is.
        options = ["dropout_p=0.0"]
        if causal:
            length, seqlen = names.fresh("attn_l"), names.fresh("attn_s")
            mask = names.fresh("attn_mask")
            options.append(f"attn_mask={mask}")
        if gqa:
            options.append("enable_gqa=True")
        if is_4d:
            operands = f"{q}.float(), {k}.float(), {v}.float()"
            suffix = ""
        else:
            operands = (f"{q}.float()[None, None], {k}.float()[None, None], "
                        f"{v}.float()[None, None]")
            suffix = "[0, 0]"
        call = (f"torch.nn.functional.scaled_dot_product_attention(\n"
                f"            {operands},\n"
                f"            {', '.join(options)}){suffix}\n")
        body = ""
        if causal:
            body += (f"    if {k}.shape[-2] < {q}.shape[-2]:\n"
                     f"        raise ValueError('causal attention requires the key length to be at least the query length')\n"
                     f"    {length}, {seqlen} = {q}.shape[-2], {k}.shape[-2]\n"
                     f"    {mask} = torch.tril(torch.ones({length}, {seqlen}, dtype=torch.bool, device={q}.device), diagonal={seqlen} - {length})\n")
        body += "    with sdpa_kernel(SDPBackend.MATH):\n" + f"        {out}.copy_({call})\n"
    elif anchor.op in {"attention", "causal_attention"}:
        q, k, v = inputs
        causal = anchor.op == "causal_attention"
        is_4d = len(graph.values[anchor.inputs[0]].shape) == 4
        q_shape = graph.values[anchor.inputs[0]].shape
        k_shape = graph.values[anchor.inputs[1]].shape
        gqa = is_4d and q_shape[1] != k_shape[1]
        scores, probs = names.fresh("attention_scores"), names.fresh("attention_probs")
        body = ""
        if causal:
            body += (f"    if {k}.shape[-2] < {q}.shape[-2]:\n"
                     f"        raise ValueError('causal attention requires the key length to be at least the query length')\n")
        if gqa:
            # The symbolic divisibility the type layer deferred: the tensors
            # are in hand, so H % G is decided here or never. The ratio is
            # computed ONCE, before either repeat: the second line used to
            # read k's head count after k was already repeated, making the
            # factor H // H == 1 and silently leaving v un-grouped -- caught
            # only by running it, on the A800, never by reading it.
            ratio = names.fresh("gqa_ratio")
            body += (f"    if {q}.shape[1] % {k}.shape[1]:\n"
                     f"        raise ValueError('GQA: the key head count must divide the query head count')\n"
                     f"    {ratio} = {q}.shape[1] // {k}.shape[1]\n"
                     f"    {k} = {k}.float().repeat_interleave({ratio}, dim=1)\n"
                     f"    {v} = {v}.float().repeat_interleave({ratio}, dim=1)\n"
                     f"    {scores} = torch.matmul({q}.float(), {k}.transpose(-1, -2)) / ({q}.shape[-1] ** 0.5)\n")
        elif is_4d:
            body += (f"    {scores} = torch.matmul({q}.float(), {k}.float().transpose(-1, -2)) / ({q}.shape[-1] ** 0.5)\n")
        else:
            body += (f"    {scores} = torch.mm({q}.float(), {k}.float().t()) / ({q}.shape[-1] ** 0.5)\n")
        if causal:
            # The causal bound, with the kv-cache offset: query i attends
            # keys 0..i+(S-L). tril keeps j <= i + (S - L), and the [L,S]
            # mask broadcasts over the batch and head axes.
            length, seqlen = names.fresh("attn_l"), names.fresh("attn_s")
            mask = names.fresh("attn_mask")
            body += (f"    {length}, {seqlen} = {q}.shape[-2], {k}.shape[-2]\n"
                     f"    {mask} = torch.ones({length}, {seqlen}, dtype=torch.bool, device={scores}.device)\n"
                     f"    {mask} = torch.tril({mask}, diagonal={seqlen} - {length})\n"
                     f"    {scores} = {scores}.masked_fill(~{mask}, float('-inf'))\n")
        body += (f"    {probs} = torch.softmax({scores}, dim=-1)\n")
        if gqa:
            body += f"    {out}.copy_(torch.matmul({probs}, {v}))\n"
        elif is_4d:
            body += f"    {out}.copy_(torch.matmul({probs}, {v}.float()))\n"
        else:
            body += f"    {out}.copy_(torch.mm({probs}, {v}.float()))\n"
    else:
        q, k, v = inputs
        scores = names.fresh("attention_scores")
        body = (f"    {scores} = torch.mm({q}.float(), {k}.float().t()) / ({q}.shape[-1] ** 0.5)\n"
                f"    {out}.copy_(torch.mm(torch.softmax({scores}, dim=-1), {v}.float()))\n")
    source = f"# explicit library {anchor.op}\ndef {name}({', '.join(p for p, _ in pairs)}):\n" + body
    return TritonKernel(name, f"library_{anchor.op}", (anchor.op,), operands, results,
                        tuple(p for p, _ in pairs), tuple(arg for _, arg in pairs), "", source)


#: Multi-result operators whose kernels can be emitted for a subset of their
#: results, because the emitter decides which stores to make.
RESULT_OPTIONAL_OPS = frozenset({"topk_last", "segment_rank",
                                 "gated_delta_recurrence",
                                 "argmax_last", "argmin_last",
                                 # the scan lowerings store each live result
                                 # separately, so either result may be dead
                                 "cumsum_totals"})


def lower(pipeline, graph, fusion, memory, *, gpu_backend="triton", library_matmul="unfused", attention_impl="math", platform_config=None, custom_ops=False, autotune=False, autotune_fold=False, concurrency="sequential", prune_kernel_abi=False, launch_cache=False, constexpr_scalars=None) -> TritonProgram:
    """Emit a Triton kernel per fusion group, plus the host launch code.

    ``launch_cache`` opts into the compiled-handle launch fast path. It is off
    by default: the platform's own code-safety check forbids module-level
    mutable containers ("Global dict/set variables can cache results across
    benchmark iterations"), and the cache is one. The option exists because the
    path is measured and admissible where that rule does not apply, not because
    a generated competition module may use it.
    """
    graph.validate()
    global _CONSTEXPR_LITERALS
    _CONSTEXPR_LITERALS = dict(constexpr_scalars or {})
    dispatch = library_dispatch(gpu_backend, library_matmul)
    kernels = []
    refusals = []
    skipped = []
    config = resolve_platform() if platform_config is None else platform_config

    for index, group in enumerate(fusion.groups):
        name = f"{pipeline.name}_k{index}"

        # 13.7 gives a value nobody reads no buffer, so a kernel computing one
        # has nowhere to store it. Dead code elimination falls out of the buffer
        # plan rather than needing a pass of its own, and it runs BEFORE the
        # refusal classification: a group whose only result is dead is not an
        # error no matter what its dtype/shape would be
        # (review 2026-10-26: dead scan+epilogue groups were refused).
        results = group_results(graph, group)
        dead = [
            value
            for value in results
            if storage_of(memory, value) is None
        ]
        if dead and len(dead) == len(results):
            skipped.append(
                SkippedKernel(
                    name=name,
                    values=tuple(dead),
                    reason="13.7 gives a value nobody reads no buffer, so "
                    "there is nowhere to write it",
                )
            )
            continue

        refusal = refusal_for(graph, group, POINTWISE_EXPRESSION, "Triton")

        if refusal is not None:
            refusals.append(refusal)
            continue

        # A multi-result operator may be asked for only some of its results:
        # a router needs `topk_last`'s indices and has no use for the values it
        # sorted. Dropping the whole kernel would leave the live result with no
        # producer, so a result-optional emitter keeps the kernel and stores only
        # what has a buffer. Everything else must refuse: silently skipping the
        # group would hand the caller an uninitialized buffer (review P0-1,
        # 2026-10-26). Only a group whose results are *all* dead is dead code.
        if dead:
            result_optional = (group.anchor is not None
                               and group.anchor.op in RESULT_OPTIONAL_OPS)
            if not result_optional:
                refusals.append(Refusal(
                    group.anchor.op if group.anchor is not None else 'pointwise',
                    tuple(results),
                    "only some results of this operator are used, and its "
                    "lowering cannot emit a subset: dropping the kernel would "
                    "leave the live result with no producer",
                    "Triton"))
                continue

        # Selective hybrid: a bare matmul group has no fused region, so
        # fusion gives it nothing and cuBLAS is the measured faster target
        # for a plain GEMM. A matmul with an attached prologue/epilogue
        # keeps its Triton kernel, because the fused single-kernel form is
        # where generated code beats the eager two-kernel sequence.
        # attention and rms_norm stay library kernels in hybrid mode either
        # way: those choices are correctness-driven (the FP32 library
        # reduction sequence the decoder comparison validated), not tuning
        # decisions.
        if dispatch.uses_library(group, pipeline=pipeline.name):
            kernels.append(_library_kernel(graph, group, memory, name, attention_impl))
            continue

        # P1 of the iteration region (docs/ITERATION_REGION.md): the carried
        # tuple (value, index) needs a merge function and a device probe before
        # it is emitted.  The reference lowering is already the numeric oracle,
        # so the program is valid FlowGPU and this is a *refusal*, not an error.
        if group.anchor is not None and group.anchor.op in {'cummin_last', 'cummax_last'}:
            refusals.append(Refusal(
                group.anchor.op, tuple(group_results(graph, group)),
                'the carried-tuple scan lowering is not emitted yet; its reference '
                'lowering is the numeric oracle (P1 step 1 of '
                'docs/ITERATION_REGION.md) and the device form needs a tuple merge '
                'function over (value, index)', 'Triton'))
            continue

        kind = kernel_kind(group.anchor)
        if group.indexed and kind in {"pointwise", "unknown"}:
            kind = "primitive"
        if kind not in config.allowed_kernel_kinds:
            refusals.append(Refusal(
                group.anchor.op if group.anchor is not None else 'pointwise', tuple(group_results(graph, group)),
                f"platform '{config.name}' currently emits {', '.join(config.allowed_kernel_kinds)} kernels only; this lowering is unvalidated"
                + wider_profile_hint(config, kind),
                "Triton",
            ))
            continue

        if config.strided_reduction_outputs and any(
                node.op in {'sum_last','max_last','softmax','rms_norm','rmsnorm_hf'}
                for node in group.members) and not group.reduction_domain:
            refusals.append(Refusal(
                group.anchor.op if group.anchor is not None else 'reduction',tuple(group_results(graph,group)),
                'strided reduction outputs require a fused reduction-domain emitter', 'Triton'))
            continue
        if config.eager_row_result_stores and any(node.op in {'sum_last','max_last'}
                for node in group.members) and not group.reduction_domain:
            refusals.append(Refusal(
                group.anchor.op if group.anchor is not None else 'reduction',tuple(group_results(graph,group)),
                'early row result stores require a fused reduction-domain emitter', 'Triton'))
            continue

        if memory.stage_plan is not None:
            from flowgpu.compiler.backend.segmented_reduction import lower_stages
            kernels.extend(lower_stages(graph,group,memory,name,config))
            continue
        emitter = EMITTERS[kind]
        if emitter is _matmul_kernel and autotune:
            kernels.append(emitter(graph, group, memory, name, config, autotune=True))
        elif emitter is _rowwise_kernel and autotune_fold:
            kernels.append(emitter(graph, group, memory, name, config, autotune_fold=True))
        elif emitter is _primitive_kernel and autotune_fold:
            kernels.append(emitter(graph, group, memory, name, config, autotune_fold=True))
        elif emitter is _layout_kernel and config.layout_on_device:
            kernels.append(_layout_device_kernel(graph, group, memory, name, config))
        elif emitter in (_pointwise_kernel, _matmul_kernel, _transpose_kernel, _rowwise_kernel, _rope_kernel, _primitive_kernel, _cache_store_kernel, _cache_write_kernel, _cache_write_hsd_kernel, _cache_write_batch_kernel, _cache_write_range_kernel, _linear_decode_kernel, _pad_cat_kernel, _scan_kernel, _patchify_kernel, _segmented_kernel, _scatter_kernel, _store_at_kernel, _topk_kernel, _sort_kernel, _mean_kernel, _layer_norm_kernel, _conv2d_kernel, _conv2d_same_kernel, _conv2d_stride2_kernel, _conv2d_stride2_same_kernel, _conv2d_ex_kernel, _native_attention_kernel, _gather_last_kernel,
                                        _segment_rank_kernel, _recurrence_kernel,
                                        _reduce_with_index_kernel):
            try:
                emitted = emitter(graph, group, memory, name, config)
            except ValueError as error:
                if kind == "attention" and "literal extents" in str(error):
                    refusals.append(Refusal(
                        group.anchor.op, tuple(group_results(graph, group)),
                        "native attention requires literal extents; use "
                        "gpu_backend='hybrid' for symbolic attention", "Triton"))
                    continue
                raise
            # A lowering may need more than one launch for one IR group: a pure
            # scatter is "copy the base, then write the addressed rows", and the
            # copy must complete before the scattered writes. The protocol is
            # therefore "a kernel or a tuple of kernels", normalised here so every
            # existing single-kernel emitter is untouched.
            kernels.extend(emitted if isinstance(emitted, tuple) else (emitted,))
        else:
            kernels.append(emitter(graph, group, memory, name))

    if config.xpu_buffer_size_limit is not None:
        from dataclasses import replace
        kernels=[replace(k,launch_options=tuple(
            (n,v) for n,v in k.launch_options if n!='buffer_size_limit')+
            (('buffer_size_limit',config.xpu_buffer_size_limit),))
            if not k.kind.startswith('library_') else k for k in kernels]
    abi_audit = []
    if prune_kernel_abi:
        from flowgpu.compiler.backend.kernel_abi import prune_unused_parameters
        rewritten = []
        for kernel in kernels:
            if kernel.kind == 'layout' or kernel.kind.startswith('library_'):
                audit = dict(skipped='host/library ABI is outside fixed Triton pruning')
            elif kernel.autotune_family or '@triton.autotune' in kernel.source:
                audit = dict(skipped='autotune owns its formal/launch contract')
            else:
                kernel, audit = prune_unused_parameters(kernel)
            rewritten.append(kernel)
            abi_audit.append(dict(kernel=kernel.name, **audit))
        kernels = rewritten
    if config.affine_write_addresses:
        # Before any kernel is handed to a caller: a target that only
        # sustains `scalar_base + affine(arange)` writes refuses a kernel
        # whose emitted store address is not that shape. Runs on the final
        # source (after ABI pruning), so the text checked is the text that
        # would be launched. Platforms without the record are untouched.
        from flowgpu.compiler.analysis.write_address import require_affine_writes
        kernels = [require_affine_writes(kernel, config, pipeline=pipeline.name)
                   for kernel in kernels]
    schedule = analyze_schedule(tuple(kernels), memory, mode=concurrency)
    return TritonProgram(
        pipeline=pipeline.name,
        autotune=autotune,
        autotune_fold=autotune_fold,
        kernel_abi_audit=tuple(abi_audit),
        kernels=tuple(kernels),
        refusals=tuple(refusals),
        skipped=tuple(skipped),
        host=_host(pipeline, graph, memory, kernels, skipped, attention_impl=attention_impl, platform_config=config, schedule=schedule, launch_cache=launch_cache, constexpr_scalars=constexpr_scalars or ()),
        gpu_backend=gpu_backend,
        attention_impl=attention_impl,
        platform_config=config,
        custom_ops=(
            _custom_ops(pipeline, graph, config)
            if custom_ops and not refusals
            else ""
        ),
    )


def _custom_ops(pipeline, graph, config) -> str:
    """The torch.library section, naming parameters exactly as the host does.

    The host renames an input when its name collides with a dimension
    symbol; the registration must call the host entry with those same
    names, so the same renaming rule is applied here. It is replicated
    rather than refactored out of `_host` because the host emission owns
    the decision, and this section is a consumer, not a second decider --
    the two must agree by construction, not by shared state.
    """
    names = Names(graph)
    symbols = {
        str(dimension)
        for value in graph.values.values()
        for dimension in value.shape
        if is_symbol(dimension)
    }
    parameters = {
        statement.name: (
            names.fresh(identifier(statement.name))
            if identifier(statement.name) in symbols | {"torch", "triton", "tl", "_flowgpu_device_key"}
            else names.reserve(identifier(statement.name))
        )
        for statement in pipeline.statements
        if isinstance(statement, InputDecl)
    }
    return render_custom_ops(
        pipeline, graph, parameters, config, DTYPE_TO_TORCH, is_symbol
    )


#: Launch argument kinds that name a tensor (a pointer at the launch site).
_TENSOR_ARGUMENT_KINDS = frozenset({"tensor", "workspace"})

#: Launch argument kinds the host evaluates to a number or flag the kernel
#: receives as a constexpr -- or to a runtime integer Triton still specialises
#: on (a matmul's dimensions are plain formals). The compiled-handle cache key
#: carries the value of every one of them, because the handle was compiled for
#: exactly those values.
_METADATA_ARGUMENT_KINDS = frozenset(
    {"value", "shape", "stride", "contiguous", "adjacent_views", "device_key"}
)

#: Alignment residue the compiled-handle cache key records per tensor argument.
#:
#: Triton 3.3's NVIDIA backend specialises a pointer argument on
#: `data_ptr() % 16 == 0` (`BaseBackend.get_arg_specialization` returns "D" for
#: an aligned tensor) and compiles a vectorised binary only for the aligned
#: case. A key of dtype plus constexprs alone would therefore reuse the aligned
#: handle for a view whose offset changed the alignment -- exactly the silent
#: wrong-bytes case this cache must never have. 128 is a superset of every
#: power-of-two alignment any measured backend uses (Triton itself checks 16;
#: the prepared-module guards in flowgpu/compiler/backend/specialization.py use
#: `alignment_mod128`), so two tensors with the same residue can never hide a
#: different specialisation, while a different residue only costs an extra
#: cache entry and one extra `warmup` (which the runtime's own per-
#: specialisation kernel cache answers without recompiling).
#:
#: This is a residue, not an address: no absolute pointer is stored and no
#: tensor is retained by the cache, so it cannot extend a buffer's lifetime.
_TENSOR_ALIGNMENT_MODULUS = 128


def _grid_arity(grid):
    """How many launch-grid dimensions an emitted grid expression has.

    The compiled handle's `__getitem__` takes exactly three grid axes, so the
    fast path pads a shorter grid with ones. A grid with more than three axes
    would be truncated by that pad-and-slice and launch the wrong program
    count, so it keeps the plain launch. Emitters only produce one- and
    two-dimensional grids today; this is the total answer, not a guess.
    """
    try:
        node = ast.parse(grid, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Tuple):
        return None

    return len(node.elts)


def _launch_groups(kernel):
    """Split a launch into (tensor, runtime-scalar, metadata) argument positions.

    The fast path rebuilds the launch argument list as
    ``tensors + runtime scalars + metadata``, which equals the original list
    only when those three groups appear in that order -- the order every
    present emitter produces (pointers, then host numbers, then constexprs).
    Returns None for an order or argument kind this emitter does not cover:
    emitting no fast path is always correct, while guessing an order would
    launch with permuted arguments.
    """
    groups = []
    for argument in kernel.launch:
        if argument.kind in _TENSOR_ARGUMENT_KINDS:
            groups.append(0)
        elif argument.kind == "scalar":
            groups.append(1)
        elif argument.kind in _METADATA_ARGUMENT_KINDS:
            groups.append(2)
        else:
            return None

    if groups != sorted(groups):
        return None

    return tuple(
        tuple(index for index, group in enumerate(groups) if group == wanted)
        for wanted in (0, 1, 2)
    )


def _tuple_source(items):
    """One Python tuple literal, including the one-element trailing comma."""
    if not items:
        return "()"
    if len(items) == 1:
        return f"({items[0]},)"

    return "(" + ", ".join(items) + ")"


def _fast_launch_lines(kernel, arguments, graph, names, device, indent):
    """Emit one launch through a cached compiled kernel handle, or None.

    `Triton`'s `JITFunction.run` rebuilds a specialisation key from every
    launch argument on every call and then looks the compiled kernel up in its
    per-device cache. On the Task160 shape that dispatch is 16.6 us of the
    31.8 us full call, while launching the already-compiled handle is 6.99 us
    for the same kernel and byte-identical output (H100, guarded;
    an earlier internal diagnosis).

    The emitted form keeps the plain launch as its single fallback, textually
    exactly as before, so a Triton build that cannot hand out a launchable
    handle behaves like the pre-cache module:

        _fg_entry = _LAUNCH_CACHE.get(kernel)          # per-kernel sentinel
        _fg_launch = None
        if _fg_entry is not False:                      # this build can try
            key = (kernel, device, tensor dtype+alignment, int scalar values,
                   every other launch argument value)
            _fg_entry = _LAUNCH_CACHE.get(key)
            if _fg_entry is None:
                try:
                    _fg_entry = kernel.warmup(*args, grid=grid3, **opts)
                    _fg_launch = _fg_entry[grid3]
                    _fg_launch(*args)                   # the launch is the proof
                    _LAUNCH_CACHE.update({key: _fg_entry})
                except Exception:
                    _fg_launch = None
                    _LAUNCH_CACHE.update({kernel: False})
            elif _fg_entry:
                _fg_launch = _fg_entry[grid3]
                _fg_launch(*args)
        if _fg_launch is None:
            kernel[grid](args, opts)

    The first launch of a specialisation does the handle launch under `try`
    instead of only building the handle: `hasattr` would only prove the
    attribute exists, and a Triton variant whose `CompiledKernel.__getitem__`
    is named the same but behaves differently must fall back on the first
    call, not fail on the first submitted one. The failure is cached per
    kernel, so a build without the API pays one exception and one dict lookup
    per call after that -- and it never rebuilds the specialisation key, which
    is the whole cost this avoids.

    Kinds, autotune grids and library/layout calls keep the plain launch.
    """
    arity = _grid_arity(kernel.grid) if kernel.grid else None
    if (kernel.kind.startswith("library_") or kernel.kind == "layout"
            or kernel.autotune_family or "@triton.autotune" in kernel.source
            or arity is None or not 1 <= arity <= 3
            or device is None or "_LAUNCH_CACHE" in names.used):
        return None

    groups = _launch_groups(kernel)
    if groups is None:
        return None

    tensors, scalars, metadata = groups
    options = [f"{key}={value}" for key, value in kernel.launch_options]
    plain = f"{kernel.name}[{_kernel_grid(kernel)}]({', '.join(arguments + options)})"

    grid = names.fresh("_fg_grid")
    grid3 = names.fresh("_fg_grid3")
    tensor_tuple = names.fresh("_fg_tensors")
    metadata_tuple = names.fresh("_fg_meta")
    key = names.fresh("_fg_key")
    entry = names.fresh("_fg_entry")
    launch = names.fresh("_fg_launch")

    tensor_expressions = [arguments[index] for index in tensors]
    scalar_expressions = [arguments[index] for index in scalars]
    # Runtime float scalars are not part of Triton's specialisation: two calls
    # that differ only in `alpha`/`limit` share one compiled handle, so they
    # are deliberately absent from the key. A runtime integer is specialised
    # on (Triton keys it on width, `% 16` and `== 1`); its value is included,
    # which is finer than the specialisation and therefore safe.
    int_scalar_expressions = [
        arguments[index] for index in scalars
        if graph.values[kernel.launch[index].text].dtype not in FLOAT_DTYPES
    ]
    # Every tensor contributes its dtype and alignment residue -- never its
    # address -- and the kernel object comes first so two structurally
    # different kernels with identical signatures cannot share an entry.
    tensor_key = [
        item
        for expression in tensor_expressions
        for item in (f"{expression}.dtype",
                     f"{expression}.data_ptr() % {_TENSOR_ALIGNMENT_MODULUS}")
    ]
    key_components = _tuple_source([
        kernel.name,
        device,
        _tuple_source(tensor_key),
        _tuple_source(int_scalar_expressions),
        metadata_tuple,
    ])
    handle_arguments = ", ".join(
        [f"*{tensor_tuple}"] + scalar_expressions + [f"*{metadata_tuple}"]
    )
    launch_options = ", ".join(options)
    warmup_options = f", {launch_options}" if launch_options else ""

    return [
        f"{indent}{entry} = _LAUNCH_CACHE.get({kernel.name})",
        f"{indent}{launch} = None",
        f"{indent}if {entry} is not False:",
        f"{indent}    {grid} = {_kernel_grid(kernel)}",
        f"{indent}    {grid3} = ({grid} + (1, 1))[:3]",
        f"{indent}    {tensor_tuple} = {_tuple_source(tensor_expressions)}",
        f"{indent}    {metadata_tuple} = {_tuple_source([arguments[index] for index in metadata])}",
        f"{indent}    {key} = {key_components}",
        f"{indent}    {entry} = _LAUNCH_CACHE.get({key})",
        f"{indent}    if {entry} is None:",
        f"{indent}        try:",
        f"{indent}            {entry} = {kernel.name}.warmup({handle_arguments}, grid={grid3}{warmup_options})",
        f"{indent}            {launch} = {entry}[{grid3}]",
        f"{indent}            {launch}({handle_arguments})",
        f"{indent}            _LAUNCH_CACHE.update({{{key}: {entry}}})",
        f"{indent}        except Exception:",
        f"{indent}            {launch} = None",
        f"{indent}            _LAUNCH_CACHE.update({{{kernel.name}: False}})",
        f"{indent}    elif {entry}:",
        f"{indent}        {launch} = {entry}[{grid3}]",
        f"{indent}        {launch}({handle_arguments})",
        f"{indent}if {launch} is None:",
        f"{indent}    {plain}",
    ]


def _flat_index_guard_lines(kernel, graph, variable, input_names, guarded) -> list:
    """Host guard for kernels whose emitted offsets are 32-bit arithmetic.

    The default emitters (pointwise, primitive, transpose, rope, matmul, scan)
    write `program_id * BLOCK + arange` and address with the result.  That is
    int32 arithmetic whatever the kernel metadata says, so an input whose real
    address span reaches 2**31 wraps to a negative index.  There is no proof
    machinery on this path (that is what the indexed/cross-domain kernels get
    via `IndexBound`), so the honest form is a per-call check that rejects the
    shape instead of computing the wrong answer.

    The span uses the tensor's *runtime* shape and stride: a non-contiguous view
    can span far more addresses than its element count suggests, and the first
    version of this guard derived the span from the declared shape and therefore
    missed exactly that case (review 2026-10-26).  All inputs share one `if`,
    each is checked once per host function, and the negative strides a view may
    carry are handled with `abs`.
    """
    if kernel.index_dtype == 'int32':
        return []      # every 32-bit kernel already has an enforced bound block
    limit = 2 ** 31 - 1
    spans = []
    for argument in kernel.launch:
        if argument.kind not in {"tensor", "workspace"}:
            continue
        value = argument.text
        if value not in input_names or value not in graph.values:
            continue
        host = variable(value)
        if host in guarded:
            continue
        guarded.add(host)
        shape = graph.values[value].shape
        if not shape:
            continue
        terms = " + ".join(
            f"({host}.shape[{axis}] - 1) * abs({host}.stride({axis}))"
            for axis in range(len(shape)))
        spans.append(f"1 + {terms}")
    if not spans:
        return []
    return [
        "    if " + f" > {limit} or ".join(spans) + f" > {limit}:",
        "        raise ValueError('an input spans more than 2**31 - 1 addresses; "
        "this kernel addresses with 32-bit arithmetic')",
    ]


def _product_of(shape) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def _index_bound_lines(kernel, graph, variable, names, indent="    "):
    """Host Python that enforces the address bound of an int32-index kernel.

    The bound uses only values the host already holds before the launch: the
    program's dimension symbols (bound from the input tensors) and each leaf
    tensor's *runtime* shape and stride metadata. A strided view is therefore
    bounded by its real address extent, not its logical element count -- for
    Task160's ``x[:, :n]`` view the largest address is about twice the element
    count, and a contiguity assumption would pass a case that must be
    rejected.

    Nothing is cached across calls: a module-level container is exactly the
    construct the platform's code-safety check rejects ("Module-level mutable
    container detected"), so the arithmetic is recomputed per call and costs a
    few hundred nanoseconds on the foreign targets only -- the int64 profiles
    emit no such block at all.

    Returns an empty list for an int64 kernel: no bound is needed because the
    index is already wide enough for any address the pointer can hold.
    """
    if kernel.index_dtype != 'int32':
        return []
    bound = kernel.index_bound
    if bound is None:
        raise ValueError(f"int32 kernel {kernel.name} has no enforced index bound")
    variable_of = variable
    expressions = []

    def add(expression):
        if expression not in expressions:
            expressions.append(expression)

    # The flat index space plus the masked tail lanes of the last program.
    add(f"({bound.count}) + {bound.padding}")
    for leaf in bound.leaves:
        shape = graph.values[leaf].shape
        if not shape:
            continue
        # The logical index itself is decomposed by these dimensions before
        # any stride scales it, so its element count must fit too.
        add(product(shape))
        terms = [
            f"({dimension} - 1) * abs({variable_of(leaf)}.stride({axis}))"
            for axis, dimension in enumerate(shape)
            if str(dimension) != "1"
        ]
        if terms:
            add(" + ".join(terms))
    if not expressions:
        raise ValueError(f"int32 kernel {kernel.name} produced an empty index bound")
    target = names.fresh("_fg_index_bound")
    lines = [
        f"{indent}# 32-bit index proof: reject before the launch when any "
        f"address this kernel derives cannot fit int32.",
        f"{indent}{target} = {expressions[0]}" if len(expressions) == 1
        else f"{indent}{target} = max({', '.join(expressions)})",
        f"{indent}if {target} > {MAX_INDEX32}:",
        # No kernel name in the message: the structural compaction that makes
        # a typed submission module small deduplicates host bodies only when
        # they are identical up to torch dtype constants, and one per-typed-
        # program name would make all nine hosts unique (module size 2x).
        f"{indent}    raise ValueError("
        f"\"int32 index bound exceeded: the largest address this kernel can "
        f"derive is %d > {MAX_INDEX32}; compile this platform with "
        f"index_dtype='int64' or use smaller extents\" % {target})",
    ]
    return lines


def _host(pipeline, graph, memory, kernels, skipped=(), *, attention_impl="math", platform_config=None, schedule=None, launch_cache=False, constexpr_scalars=()) -> str:
    """The launch code: one allocation per slot, one call per kernel.

    With ``launch_cache`` off -- the default, and the only admissible form for
    a generated competition module -- every launch is the plain
    ``kernel[grid](args, options)`` the compiler emitted before the cache
    existed. The opt-in path only changes how a launch is dispatched.
    """
    name = pipeline.name
    config = resolve_platform() if platform_config is None else platform_config
    # The per-call input-contract guard block is a policy field, not a property
    # of the language. True (the default) emits every guard; False emits none,
    # for a module whose own entry point has already validated the call and
    # whose measured host cost includes that block. `split_half_inputs`
    # forces it off: under that contract the declared half-width last axis can
    # never equal the caller's full-width tensor, so a declarative shape guard
    # would be arithmetically wrong.
    validation = config.runtime_validation and not config.split_half_inputs

    constexpr_names = set(constexpr_scalars)
    declared = [
        statement
        for statement in pipeline.statements
        if isinstance(statement, InputDecl)
        and not (statement.name in constexpr_names and statement.type.is_scalar)
    ]
    input_names = {statement.name for statement in declared}
    guarded_inputs = set()
    # A rank-0 declaration is a host number: it has no rank, shape, dtype or
    # device to interrogate with torch. Splitting the declarations here is what
    # keeps every tensor guard below from running on it -- and the launch
    # metadata (`first tensor`, the allocation device) is read from the tensor
    # side only.
    scalar_declared = [
        statement for statement in declared if statement.type.is_scalar
    ]
    tensor_declared = [
        statement for statement in declared if not statement.type.is_scalar
    ]
    outputs = [
        statement.name
        for statement in pipeline.statements
        if isinstance(statement, OutputDecl)
    ]

    # The host invents names too -- one variable per slot -- and binds the
    # dimensions the kernels are launched with, so it allocates them the same
    # way a kernel does. A program may call a value `slot0`, or name an input
    # after one of its own dimensions; the parameter yields, because the
    # dimension variables are what the launch arguments are written with.
    names = Names(graph)
    symbols = {
        str(dimension)
        for value in graph.values.values()
        for dimension in value.shape
        if is_symbol(dimension)
    }
    # An input keeps its name -- it is the program's own -- unless the program
    # also uses that name as a dimension symbol, since the dimension is what
    # the launch arguments are written with.
    # The host body itself names torch/triton/tl; an input with one of those
    # names must not shadow the module it is launched with (review P1-5.5).
    host_reserved = {'torch', 'triton', 'tl', '_flowgpu_device_key',
                     'float', 'int', 'len', 'isinstance', 'math', 'range',
                     'str', 'list', 'tuple'}
    parameters = {
        statement.name: (
            names.fresh(identifier(statement.name))
            if identifier(statement.name) in symbols | host_reserved
            else names.reserve(identifier(statement.name))
        )
        for statement in declared
    }
    slots = {
        f"slot{slot.index}": names.fresh(f"slot{slot.index}")
        for slot in memory.slots
    }
    workspace = ({s.name:names.fresh(s.name) for s in memory.stage_plan.scratch}
                 if memory.stage_plan is not None else {})

    lines = []
    # `HD` is linear_decode's derived last dimension (H * D).  It is emitted
    # after the input shape bindings below, because H and D are read from the
    # tensors there; emitting it first was an UnboundLocalError on the first
    # call (review 2026-10-26).
    input_symbols = {
        str(dimension)
        for statement in declared
        for dimension in statement.type.shape
        if is_symbol(dimension)
    }
    needs_derived_output_width = (
        'HD' in {str(dimension) for value in graph.values.values()
                 for dimension in value.shape if is_symbol(dimension)}
        and 'H' in input_symbols and 'D' in input_symbols
        and 'HD' not in input_symbols)
    cache_pipeline = any(kernel.kind in {'cache_store', 'linear_decode'} for kernel in kernels)
    device_anchor = (
        parameters[tensor_declared[0].name] if tensor_declared else None
    )
    # --- tensor input contracts ------------------------------------------
    #
    # Every property below is still checked before any allocation or launch,
    # and a rejection still names the input it is about. The *form* is chosen
    # for per-call cost: this block runs on every call of a generated host, and
    # the previous layout -- one `if` per (input, property), with
    # `tuple(x.shape) != (M, N)` allocating twice per input -- measured 6.1 us
    # of a 23.9 us host function on an H100 (Task160 diagnosis, 2026-10-21).
    #
    # Three properties are per-call rather than per-input: "is CUDA", "is
    # strided", and "same device". The first tensor is the call's device
    # anchor, so one guard on it covers the CUDA and layout contract, and a
    # second tensor's device check subsumes its own is_cuda (a CPU tensor's
    # device never equals a CUDA device). That is one guard per tensor input
    # for rank + device + layout instead of five.
    first_tensor = tensor_declared[0] if tensor_declared else None
    first_variable = parameters[first_tensor.name] if first_tensor is not None else None
    device_variable = names.fresh("device") if first_tensor is not None else None
    if first_tensor is not None and (validation or launch_cache):
        lines.append(f"    {device_variable} = {first_variable}.device")
    if validation:
        for statement in tensor_declared:
            variable = parameters[statement.name]
            rank = len(statement.type.shape)
            if statement is first_tensor:
                pointer = (
                    f"not {variable}.is_cuda" if config.device_type == "cuda"
                    else f"{variable}.device.type != 'npu'"
                )
                conditions = [f"{variable}.ndim != {rank}", pointer,
                              f"{variable}.layout != torch.strided"]
                message = (
                    f"input {statement.name} must be a {config.device_type.upper()} "
                    f"tensor with a strided layout and rank {rank}"
                )
            else:
                conditions = [f"{variable}.ndim != {rank}",
                              f"{variable}.device != {device_variable}",
                              f"{variable}.layout != torch.strided"]
                message = (
                    f"input {statement.name} must be a {config.device_type.upper()} "
                    f"tensor with a strided layout and rank {rank}, on the same "
                    f"{config.device_type.upper()} device as {first_tensor.name}"
                )
            lines += [f"    if {' or '.join(conditions)}:",
                      f"        raise ValueError({message!r})"]

    # Symbolic dimensions (8.3) are read off the tensors at run time: the
    # compiler never knows them, and nothing here specializes on them. The
    # first declaration that writes a symbol is where it is read from, which
    # is why that one occurrence needs no comparison -- it defines the name.
    bound_at = {}
    declared_shapes = [[str(dimension) for dimension in statement.type.shape]
                       for statement in tensor_declared]
    split_axis = None
    if config.split_half_inputs:
        # The pair ABI: two rank-2 inputs, the caller passes one contiguous
        # `[M, 2N]` buffer as both, and the declared last axis is its half.
        # Exactly this shape or the contract cannot be honoured.
        if len(tensor_declared) != 2 or any(len(shape) != 2 for shape in declared_shapes):
            raise ValueError(
                'split_half_inputs requires exactly two rank-2 tensor inputs, '
                f'got {[statement.name for statement in tensor_declared]}'
            )
        split_axis = len(declared_shapes[0]) - 1
        lines.append(
            "    # split-half pair contract: the caller passes one contiguous "
            "[M, 2N] buffer; the declared last axis is half of it."
        )
    for statement_index, statement in enumerate(tensor_declared):
        for axis, dimension in enumerate(declared_shapes[statement_index]):
            if not is_symbol(dimension) or dimension in bound_at:
                continue
            bound_at[dimension] = (statement_index, axis)
            source = f"{parameters[statement.name]}.shape[{axis}]"
            if split_axis is not None and statement_index == 0 and axis == split_axis:
                source += " // 2"
            lines.append(f"    {dimension} = {source}")

    # Symbolic `split` (opt-in `split_views`): the part width is not a
    # compile-time literal, it is derived at launch from the source's runtime
    # last axis -- the same division the shipped `split_half_inputs` host does.
    # The relation rides on the shape symbol's name, `N__2 == N // 2` (see
    # operator.split_part_symbol), so binding it here is what turns the part
    # width into a constexpr the kernel reads. No such symbol exists unless the
    # caller opted in, so the default host source is unchanged.
    if needs_derived_output_width:
        lines.append('    HD = H * D')

    from flowgpu.compiler.core.operators import is_derived_split_symbol
    for symbol in sorted(symbols):
        if not is_derived_split_symbol(symbol):
            continue
        base, _, parts = symbol.rpartition('__')
        if base not in bound_at:
            continue
        lines.append(f"    {symbol} = {base} // {int(parts)}")
        if validation:
            lines += [f"    if {base} % {int(parts)}:",
                      f"        raise ValueError('split requires the last dimension "
                      f"to divide into {int(parts)} equal parts')"]

    # The split-view contiguity contract is semantic, not a validation policy:
    # a kernel that reads a split source without layout constexprs computes
    # compact addresses that are only correct on a contiguous buffer, so the
    # host refuses a non-contiguous argument whether or not the optional
    # declarative guards are enabled. Fail closed, never compute wrong bytes.
    contracted = sorted({contract for kernel in kernels
                         for contract in kernel.contiguity_contracts})
    for contracted_name in contracted:
        variable = parameters[contracted_name]
        lines += [f"    if not {variable}.is_contiguous():",
                  f"        raise ValueError('input {contracted_name} must be "
                  f"contiguous for the split view contract')"]

    if validation:
        # Rank + dtype + shape, one guard per tensor input. A shape comparison is
        # a property of the declaration, not a tile: a fully literal declaration
        # compares the Size itself, an input that declares the same shape as the
        # first compares Sizes, and everything else compares the axes that matter.
        # `[N,N]` stays a real square check because the second N is not the
        # declaration that binds it.
        for statement_index, statement in enumerate(tensor_declared):
            variable = parameters[statement.name]
            shape = declared_shapes[statement_index]
            conditions = []
            if not cache_pipeline:
                conditions.append(
                    f"{variable}.dtype != {DTYPE_TO_TORCH[statement.type.dtype]}"
                )
            if shape and all(not is_symbol(dimension) for dimension in shape):
                conditions.append(f"{variable}.shape != ({', '.join(shape)},)")
            elif statement_index > 0 and shape == declared_shapes[0]:
                # The first tensor's own guard already validated its literals and
                # its repeated symbols, so equality with it is equivalent.
                conditions.append(f"{variable}.shape != {first_variable}.shape")
            else:
                for axis, dimension in enumerate(shape):
                    if is_symbol(dimension) and bound_at[dimension] == (statement_index, axis):
                        continue
                    conditions.append(f"{variable}.shape[{axis}] != {dimension}")
            if conditions:
                lines += [f"    if {' or '.join(conditions)}:",
                          f"        raise ValueError('input {statement.name} dtype or shape does not match its declaration')"]

        for statement in tensor_declared:
            variable = parameters[statement.name]
            consumers = [kernel for kernel in kernels if statement.name in kernel.operands]
            # Pointwise and rowwise kernels carry the input-layout contract
            # (contiguous flag + per-axis shape/stride constexpr metadata), so a
            # non-contiguous input is addressed through its own strides -- the
            # official fused_sigmoid_mul strided path (3D gate [B, H, R] read by
            # a 2D-flat kernel) works without a contiguous copy. Any other kernel
            # kind still has row-major indexing baked in and must refuse the
            # layout rather than silently read the wrong elements.
            unsupported = [kernel for kernel in consumers
                           if kernel.kind not in {"pointwise", "rowwise", "primitive"}
                           and kernel.kind != "cache_store"
                            and not (kernel.kind == "transpose" and config.transpose_mapping == "output_gather")]
            if unsupported:
                kinds = ", ".join(sorted({kernel.kind for kernel in unsupported}))
                lines += [f"    if not {variable}.is_contiguous():",
                          f"        raise ValueError('input {statement.name} must be contiguous for its consuming kernels ({kinds} do not carry layout metadata)')"]

    # A scalar is the caller's number, passed straight to the kernel: no
    # tensor, no device, no strides. The declaration's dtype is what the host
    # converts the number to, which is the same contract that made the old
    # adapter build a typed device tile -- minus the allocation and the copy.
    #
    # One cheap guard covers the whole scalar ABI: the failure path is the
    # only place the per-name diagnostic runs, so a value error still says
    # which input it is about, while the success path pays one branch. A
    # single scalar keeps the direct form, because a wrapper around one
    # condition is pure overhead.
    def scalar_rejection(statement):
        variable = parameters[statement.name]
        if statement.type.dtype in FLOAT_DTYPES:
            return f"isinstance({variable}, bool) or not isinstance({variable}, (int, float))"
        return f"type({variable}) is not int"

    def scalar_rejection_line(statement, indent):
        variable = parameters[statement.name]
        dtype = statement.type.dtype
        passed = ("a Python int or float, not a tensor"
                  if dtype in FLOAT_DTYPES else "a Python int, not a tensor")
        return f"{indent}raise ValueError('input {statement.name} is a scalar {dtype}; pass {passed}')"

    if validation:
        if len(scalar_declared) == 1:
            statement = scalar_declared[0]
            lines += [f"    if {scalar_rejection(statement)}:",
                      scalar_rejection_line(statement, "        ")]
        elif scalar_declared:
            rejections = [scalar_rejection(statement) for statement in scalar_declared]
            lines.append("    if (" + "\n        or ".join(rejections) + "):")
            for statement in scalar_declared:
                lines += [f"        if {scalar_rejection(statement)}:",
                          scalar_rejection_line(statement, "            ")]
    for statement in scalar_declared:
        variable = parameters[statement.name]
        conversion = "float" if statement.type.dtype in FLOAT_DTYPES else "int"
        lines.append(f"    {variable} = {conversion}({variable})")
    # Fail-closed bounds for every program-uniform scalar: the host proves the
    # largest index the launch can reach is inside the 1-D table *before* any
    # allocation or launch. This is a correctness contract, not a redundant
    # metadata guard, so it is emitted even when `runtime_validation` is off.
    from flowgpu.compiler.backend.program_scalar import host_guards
    for kernel in kernels:
        for bound in getattr(kernel, "program_scalar_bounds", ()):
            lines += host_guards((bound,), parameters)
    # `store_at`'s fail-closed caller contract: bounds and pairwise disjointness
    # of the device-table bases, checked before any allocation or launch. The
    # check reads the offsets values (one D2H + sync) unless the author used the
    # trusted spelling, whose declaration is recorded in the emitted comment.
    from flowgpu.compiler.backend.store_at_backend import host_guards as store_at_guards
    store_at_contracts = tuple(
        kernel.store_at_contract for kernel in kernels
        if getattr(kernel, "store_at_contract", None) is not None
    )
    if store_at_contracts:
        lines += store_at_guards(store_at_contracts, parameters)
    if validation:
        for symbol in sorted(symbols):
            if cache_pipeline and symbol == 'N':
                lines += [f"    if {symbol} < 0:", "        raise ValueError('tensor dimensions must be nonnegative')"]
            else:
                lines += [f"    if {symbol} <= 0:", "        raise ValueError('tensor dimensions must be positive')"]
    for node in graph.nodes:
        if node.op == "rope":
            width = str(graph.values[node.inputs[0]].shape[-1])
            lines += [f"    if {width} % 2:", "        raise ValueError('rope requires an even last dimension')"]

    library_dtypes = {graph.values[k.operands[0]].dtype for k in kernels if k.kind == "library_matmul"}
    if any(k.kind.startswith("library_") for k in kernels):
        lines += ["    if torch.is_autocast_enabled('cuda'):",
                  "        raise ValueError('hybrid library calls require CUDA autocast disabled')"]
    if "fp32" in library_dtypes:
        lines += ["    if torch.backends.cuda.matmul.allow_tf32:",
                  "        raise ValueError('hybrid fp32 GEMM requires allow_tf32=False')"]
    for dtype, flag in (("bf16", "bf16"), ("fp16", "fp16")):
        if dtype in library_dtypes:
            lines += [f"    if torch.backends.cuda.matmul.allow_{flag}_reduced_precision_reduction:",
                      f"        raise ValueError('hybrid GEMM requires allow_{flag}_reduced_precision_reduction=False')"]
    if any(k.kind == "library_attention" for k in kernels):
        lines += ["    if torch.backends.cuda.matmul.allow_tf32:",
                  "        raise ValueError('attention requires allow_tf32=False')"]
        if attention_impl == "sdpa":
            lines += ["    if torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():",
                  "        raise ValueError('attention requires FP32 SDPA math intermediates; disable reduced math SDP')"]

    # 13.7's slots become allocations, so two values that share a slot share
    # one variable. A value with no slot -- fused, or dead -- has none.
    #
    # The device is the caller's, taken from the first tensor it passes, so the
    # host never picks one: `device="cuda"` would be a decision the language
    # does not make. It used to name `x.device` -- and `x` is only the first
    # parameter when the program happens to call its first input `x`, so a
    # program starting with `input a` generated a host that raised NameError.
    # The device is the caller's, taken from the first tensor it passes.
    where = (
        f"{device_anchor}.device" if device_anchor is not None else '"cuda"'
    )

    clone_sources = {}
    alias_sources = {}
    linear_clones = {}
    if memory.stage_plan is not None:
        for condition,message in memory.stage_plan.guards:
            lines += [f'    if {condition}:',f'        raise ValueError({message!r})']
        for scratch in memory.stage_plan.scratch:
            lines.append(f'    {workspace[scratch.name]} = torch.empty(({scratch.elements},), '
                         f'dtype={DTYPE_TO_TORCH[scratch.dtype]}, device={where})    # stage-private')
    for kernel in kernels:
        if kernel.kind == 'cache_store':
            for result, source in zip(kernel.results, kernel.operands[:2]):
                clone_sources[result] = source
        elif kernel.kind == 'linear_decode':
            # The cache output is a *result*, so it gets its own buffer: the
            # input tensor is read, never written (review P1-2).  The old
            # alias made the launch pass `cache` for both cache and cache_out
            # and returned the mutated input, which is not a pure dataflow
            # module -- and the untouched rows of a clone carry the right
            # values, because the kernel writes every non-padding row.
            linear_clones[kernel.results[1]] = kernel.operands[3]
    for slot in memory.slots:
        buffer = next(
            entry for entry in memory.buffers if entry.value == slot.values[0]
        )
        shape = ", ".join(
            str(dimension) for dimension in buffer.tensor_type.shape
        )
        slot_name = slots[f'slot{slot.index}']
        if any(value in alias_sources for value in slot.values):
            continue
        linear_source = linear_clones.get(slot.values[0])
        if linear_source is not None:
            lines.append(f"    {slot_name} = {parameters[linear_source]}.clone()"
                         f"    # linear decode cache output")
            continue
        source = clone_sources.get(slot.values[0])
        if source is not None and config.device_type != 'npu':
            lines.append(f"    {slot_name} = {parameters[source]}.clone()    # clone for store_cache_4d")
        elif source is not None and config.device_type == 'npu':
            lines.append(f"    {slot_name} = {parameters[source]}    # Ascend in-place cache-store experiment")
        else:
            if any(kernel.kind == 'linear_decode' for kernel in kernels) and buffer.tensor_type.shape == ['B', 'HD']:
                lines.append(f"    {slot_name} = torch.empty(({shape}), dtype=q.dtype, device={where})    # linear decode output")
            else:
                lines.append(
                    f"    {slot_name} = torch.empty(({shape}), dtype="
                    f"{DTYPE_TO_TORCH[buffer.tensor_type.dtype]}, "
                    f"device={where})    # {', '.join(slot.values)}"
                )

    def variable(value) -> str:
        if value in alias_sources:
            return parameters[alias_sources[value]]
        where = allocation(memory, resolve_alias(graph.aliases, value))

        if where in alias_sources:
            return parameters[alias_sources[where]]
        if where in parameters:
            return parameters[where]
        if where.startswith("slot"):
            return slots[where]
        return identifier(where)

    def launch_argument(argument):
        if argument.kind == 'workspace':
            return workspace[argument.text]
        if argument.kind == "device_key":
            return f"_flowgpu_device_key({device_anchor}.device.index)"
        if argument.kind == "tensor":
            return variable(argument.text)
        if argument.kind == "scalar":
            # The host variable itself: a plain Python number, never a tensor
            # or an allocation. The kernel formal takes it positionally.
            return variable(argument.text)
        if (kernel.tile and argument.kind == "value"
                and argument.text in {"BLOCK_M", "BLOCK_N", "BLOCK_K"}):
            return str(kernel.tile[("BLOCK_M", "BLOCK_N", "BLOCK_K").index(argument.text)])
        if argument.kind == "value" and argument.text in {'page_size', 'BLOCK_SIZE'}:
            return argument.text
        if argument.kind == "contiguous":
            return f"{variable(argument.text)}.is_contiguous()"
        if argument.kind == 'adjacent_views':
            from flowgpu.compiler.backend.paired_views import adjacent_view_condition
            left, right = argument.text.split(':')
            return adjacent_view_condition(variable(left), variable(right))
        if argument.kind in {"stride", "shape"}:
            value, axis = argument.text.rsplit(":", 1)
            tensor = variable(value)
            return f"{tensor}.stride({axis})" if argument.kind == "stride" else f"{tensor}.shape[{axis}]"
        return argument.text

    cache_kernel = next((kernel for kernel in kernels if kernel.kind == 'cache_store'), None)
    if cache_kernel is not None:
        # Compare against the view tensor itself: `PS` is a bound host variable
        # only when the page axis is symbolic, and a literal declaration has
        # none -- the guard was a NameError on the first call (review P1-3,
        # 2026-10-26).
        view = variable(cache_kernel.operands[0])
        lines.append(
            f"    if page_size <= 0 or page_size != {view}.shape[1]:")
        lines.append(
            "        raise ValueError('store_cache_4d page_size must equal "
            "k_view.shape[1] and be positive')")

    overlap = schedule is not None and schedule.mode == "overlap"
    if overlap:
        lines.append("    # overlap mode uses one allocation per value and explicit CUDA events")
        for index in range(len(kernels)):
            lines.append(f"    _flowgpu_stream_{index} = torch.cuda.Stream(device={where})")
            lines.append(f"    _flowgpu_event_{index} = torch.cuda.Event()")

    for kernel_index, kernel in enumerate(kernels):
        arguments = [launch_argument(argument) for argument in kernel.launch]
        index = kernel.name.rsplit("_k", 1)[-1]
        lines.append(f"\n    # k{index}: {' -> '.join(kernel.ops)}")
        # A 32-bit program index is emitted only with this check in front of
        # the launch: the host proves from the runtime shape/stride arguments
        # that every address the kernel derives fits in signed 32 bits.
        lines.extend(_index_bound_lines(kernel, graph, variable, names))
        if config.index_guard and config.runtime_validation:
            lines.extend(_flat_index_guard_lines(kernel, graph, variable,
                                                input_names, guarded_inputs))
        # The grid is a subscript, not a call: `kernel[grid](args)`. Written
        # as a call it still parses, still looks like a launch, and raises
        # "Cannot call @triton.jit'd outside of the scope of a kernel" the
        # first time anything runs it -- which is what happened on the first
        # GPU run, and what no amount of reading the module had caught.
        indent = "        " if overlap else "    "
        launch_guard = getattr(kernel, "launch_guard", None)
        fast = (_fast_launch_lines(
            kernel, arguments, graph, names, device_variable, indent
        ) if launch_cache and not launch_guard else None)
        host_only_layout = kernel.kind == "layout" and not kernel.grid
        call = (kernel.name if kernel.kind.startswith("library_") or host_only_layout
                else f"{kernel.name}[{_kernel_grid(kernel)}]")
        options = [f"{key}={value}" for key, value in kernel.launch_options]
        launch = f"{call}({', '.join(arguments + options)})"
        if overlap:
            lines.append(f"    with torch.cuda.stream(_flowgpu_stream_{kernel_index}):")
            for dependency in schedule.dependencies[kernel_index]:
                lines.append(f"        _flowgpu_stream_{kernel_index}.wait_event(_flowgpu_event_{dependency})")
            if launch_guard:
                lines.append(f"        if {launch_guard}:")
                lines.append(f"            {launch}")
            else:
                lines.extend(fast if fast is not None else [f"        {launch}"])
            lines.append(f"        _flowgpu_event_{kernel_index}.record(_flowgpu_stream_{kernel_index})")
        elif launch_guard:
            # A metadata-only shape decision (`store_at`'s copy pass): when the
            # write kernel's runs tile the destination, the copy is dead. The
            # condition is a host shape comparison -- never a device read.
            lines.append(f"    if {launch_guard}:")
            lines.append(f"        {launch}")
        else:
            lines.extend(fast if fast is not None else [f"    {launch}"])

    if overlap:
        terminal = [index for index in range(len(kernels))
                    if not any(index in deps for deps in schedule.dependencies)]
        for index in terminal:
            lines.append(f"    torch.cuda.current_stream(device={where}).wait_event(_flowgpu_event_{index})")

    if not lines:
        lines.append("    pass")

    for entry in skipped:
        lines.append(f"\n    # {entry.message()}")

    # An output may be a second name for another value (`y = x` emits no
    # node), and a name is not what holds the data.
    returned = ", ".join(
        f'"{value}": {variable(value)}' for value in outputs
    )
    scalar_args = []
    for kernel in kernels:
        for argument in kernel.launch:
            if argument.kind == 'value' and argument.text in {'page_size', 'BLOCK_SIZE'} and argument.text not in scalar_args:
                scalar_args.append(argument.text)
    signature = ", ".join(parameters[statement.name] for statement in declared + [])
    if scalar_args:
        signature = ", ".join(filter(None, (signature, *scalar_args)))
    linear_kernel = next((kernel for kernel in kernels if kernel.kind == 'linear_decode'), None)
    if linear_kernel is not None:
        return (
            f"def {name}({signature}):\n"
            f'    """{name}: FlowGPU-generated FlagOS entry."""\n'
            + "\n".join(lines)
            + f"\n\n    return ({', '.join(variable(value) for value in outputs)})\n"
        )
    if cache_kernel is not None:
        cache_outputs = tuple(value for value in outputs if value in cache_kernel.results)
        return (
            f"def {name}({signature}):\n"
            f'    """{name}: FlowGPU-generated FlagOS entry."""\n'
            + "\n".join(lines)
            + f"\n\n    return ({', '.join(variable(value) for value in cache_outputs)})\n"
        )

    return (
        f"def run_{name}({signature}):\n"
        f'    """{name}: run the kernels of the pipeline, in order."""\n'
        + "\n".join(lines)
        + f"\n\n    return {{{returned}}}\n"
    )


def device_key_source():
    """One device/software identity helper for tuning and prepared artifacts."""
    return """
from functools import lru_cache

@lru_cache(maxsize=None)
def _flowgpu_device_key(index):
    props = torch.cuda.get_device_properties(index)
    return repr((props.name, props.total_memory, props.multi_processor_count,
                 props.major, props.minor, str(getattr(props, 'uuid', index)),
                 torch.__version__, torch.version.cuda, triton.__version__))
"""


def module(programs, *, launch_cache=None, compact_gathers=False, compact_fixed=False) -> str:
    """The whole generated file: constants, kernels, then the host code.

    The header states what the compiler did not do, because a generated file
    that looks runnable and has never been run should say so where a reader
    will see it.

    ``launch_cache`` is None by default and then follows the lowering: programs
    compiled with ``launch_cache=True`` carry the compiled-handle fast path, so
    the module preamble declares ``_LAUNCH_CACHE``; a default program has no
    fast path and gets no module-level mutable container at all. Passing the
    flag explicitly is a consistency assertion, not a way to switch the hosts
    on: a value that disagrees with the emitted hosts raises, because a module
    with fast hosts but no cache (or the reverse) is a NameError at first call.
    """
    programs = tuple(programs)
    uses_cache = any("_LAUNCH_CACHE" in program.host for program in programs)
    if launch_cache is None:
        launch_cache = uses_cache
    elif launch_cache != uses_cache:
        raise ValueError('launch_cache must match the hosts: '
                         'compile with launch_cache=%r' % uses_cache)
    if compact_fixed:
        if compact_gathers:
            raise ValueError('choose one module compaction profile')
        if any(p.autotune or p.autotune_fold or p.custom_ops or p.gpu_backend != 'triton'
               or not p.complete or any(k.kind not in {'pointwise', 'primitive'} for k in p.kernels)
               for p in programs):
            raise ValueError('fixed compaction requires complete native pointwise/primitive programs')
        from flowgpu.compiler.analysis.module_compaction import compact_source
        return compact_source(module(programs, launch_cache=launch_cache),
                              profile='fixed-native')
    if compact_gathers:
        if any(p.autotune or p.autotune_fold or p.custom_ops or p.gpu_backend != 'triton'
               or not p.complete or any(not k.ops or any(op != 'gather_rows' for op in k.ops) for k in p.kernels)
               for p in programs):
            raise ValueError('module compaction requires complete native row-gather programs')
        from flowgpu.compiler.analysis.module_compaction import compact_source
        return compact_source(module(programs, launch_cache=launch_cache))
    # A platform GEMM policy may pin the measured tile.  The constants are
    # module-level because the emitted matmul hosts and grids spell them by
    # name; keeping the old default when no policy is set is what preserves the
    # byte-identity contracts of the already-scoring chips.
    matmul_m, matmul_n, matmul_k = BLOCK_M, BLOCK_N, BLOCK_K
    for program in programs:
        config = getattr(program, 'platform_config', None)
        if config is None:
            continue
        matmul_m = config.matmul_block_m or matmul_m
        matmul_n = config.matmul_block_n or matmul_n
        matmul_k = config.matmul_block_k or matmul_k
    parts = [
        "# Generated by FlowGPU. Design.md 13.8: FlowGPU -> Triton.\n"
        "#\n"
        "# The compiler does not run this. Executing it needs a GPU and a\n"
        "# Triton install; compiling does not, and the compiler itself\n"
        "# imports neither torch nor triton.\n"
        "\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        f"BLOCK = {BLOCK}\n"
        f"BLOCK_M = {matmul_m}\n"
        f"BLOCK_N = {matmul_n}\n"
        f"BLOCK_K = {matmul_k}\n"
    ]
    if any(p.autotune or p.autotune_fold for p in programs):
        from flowgpu.compiler.backend.autotune import cache_options_source
        parts.append(cache_options_source())
        parts.append(device_key_source())
    # The opt-in compiled-handle fast path needs one module-level cache. It is
    # emitted only when a program was lowered with `launch_cache=True`: the
    # platform's code-safety check forbids module-level mutable containers in a
    # generated module ("Global dict/set variables can cache results across
    # benchmark iterations"), so a default build contains none.
    if launch_cache:
        parts.append(
            "\n# Compiled-kernel handles, keyed per specialisation. Holds no\n"
            "# tensor and no result: only launches Triton already compiled.\n"
            "_LAUNCH_CACHE = {}\n"
        )
    if any(p.gpu_backend == "hybrid" for p in programs):
        parts.insert(0, "# Hybrid backend: torch.mm library GEMM / explicit library attention + Triton elementwise kernels.\n")
    if any(p.attention_impl == "sdpa" and any(k.kind == "library_attention" for k in p.kernels) for p in programs):
        parts.append("\nfrom torch.nn.attention import sdpa_kernel, SDPBackend\n")

    for program in programs:
        if program.platform_config is not None and program.platform_config != resolve_platform():
            parts.append(f"\n# Pointwise mapping (untuned profile): {program.platform_config.as_dict()!r}\n")
        if program.platform_config is not None and program.platform_config.device_type == "npu":
            if "\nimport torch_npu\n" not in parts:
                parts.append("\nimport torch_npu\n")
        for kernel in program.kernels:
            parts.append("\n" + kernel.source)

        parts.append("\n" + program.host)
        if program.custom_ops:
            parts.append(program.custom_ops)

    return "".join(parts)
