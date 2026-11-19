"""Small structured IR for post-fusion lowering decisions.

This module deliberately sits after dataflow, fusion, and liveness.  It is a
descriptive IR: existing backends remain the source of emitted code, while this
module gives tests and future renderers an explicit representation of index
spaces, loops, and reductions.
"""

from dataclasses import dataclass


REDUCTIONS = frozenset(("sum_last", "max_last"))
LAYOUTS = frozenset(("reshape", "split", "concat"))


def _shape(shape):
    return tuple(str(d) for d in shape)


def _product(shape):
    result = 1
    symbolic = []
    for dimension in shape:
        text = str(dimension)
        if text.isdigit():
            result *= int(text)
        else:
            symbolic.append(text)
    if not symbolic:
        return str(result)
    return "*".join(([str(result)] if result != 1 else []) + symbolic)


@dataclass(frozen=True)
class IndexSpace:
    shape: tuple
    order: str = "row_major"

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape(self.shape))
        if not self.shape or any(not d for d in self.shape):
            raise ValueError("index space needs a non-empty shape")
        if self.order != "row_major":
            raise ValueError("only row_major index spaces are supported")


@dataclass(frozen=True)
class LayoutMap:
    kind: str
    source_shapes: tuple
    target_shape: tuple
    widths: tuple = ()
    part: int | None = None
    affine: tuple = ()
    tile: tuple = ()
    memory_scope: str = "global"
    contiguous: bool = True

    def __post_init__(self):
        if self.kind not in LAYOUTS:
            raise ValueError(f"unknown layout map: {self.kind}")
        object.__setattr__(self, "source_shapes", tuple(_shape(s) for s in self.source_shapes))
        object.__setattr__(self, "target_shape", _shape(self.target_shape))
        object.__setattr__(self, "widths", tuple(str(w) for w in self.widths))
        object.__setattr__(self, "affine", tuple(str(v) for v in self.affine))
        object.__setattr__(self, "tile", tuple(str(v) for v in self.tile))
        if self.memory_scope not in {"global", "shared", "local", "register"}:
            raise ValueError("invalid layout memory scope")
        if not self.target_shape or not self.source_shapes:
            raise ValueError("layout map needs source and target shapes")
        if self.kind == "split" and (self.part is None or self.part < 0):
            raise ValueError("split layout map needs a non-negative part")
        if self.kind != "split" and self.part is not None:
            raise ValueError("only split layout maps have a part")


@dataclass(frozen=True)
class LayoutOp:
    op: str
    inputs: tuple
    output: str
    index_space: IndexSpace
    mapping: LayoutMap

    def __post_init__(self):
        if self.op != self.mapping.kind:
            raise ValueError("layout op and mapping kind disagree")
        if not self.inputs or not self.output:
            raise ValueError("layout op needs inputs and an output")


@dataclass(frozen=True)
class Range:
    var: str
    start: str
    stop: str
    step: str = "1"

    def __post_init__(self):
        if not self.var or not self.stop:
            raise ValueError("loop range needs a variable and stop")
        if self.step == "0":
            raise ValueError("loop range step cannot be zero")


@dataclass(frozen=True)
class AutoReduce:
    op: str
    input: str
    output: str
    axis: int
    accumulator_dtype: str
    chunk: Range
    stages: tuple
    padding: str

    def __post_init__(self):
        if self.op not in REDUCTIONS:
            raise ValueError(f"unsupported autoreduce operator: {self.op}")
        if self.axis != -1:
            raise ValueError("autoreduce currently supports the last axis only")
        if self.accumulator_dtype not in {"fp32", "fp16", "bf16", "int32", "int64"}:
            raise ValueError(f"invalid accumulator dtype: {self.accumulator_dtype}")
        if not self.stages or any(stage not in {"single", "fold", "finalize"} for stage in self.stages):
            raise ValueError("autoreduce stages must be single, fold, or finalize")
        if len(self.stages) == 1 and self.stages[0] != "single":
            raise ValueError("a one-stage autoreduce must be single")
        if len(self.stages) == 2 and self.stages != ("fold", "finalize"):
            raise ValueError("a two-stage autoreduce must fold then finalize")


@dataclass(frozen=True)
class Loop:
    ranges: tuple
    body: tuple

    def __post_init__(self):
        if not self.ranges:
            raise ValueError("structured loop needs at least one range")
        if not self.body:
            raise ValueError("structured loop needs a body")
        if any(not isinstance(r, Range) for r in self.ranges):
            raise TypeError("loop ranges must be Range values")


@dataclass(frozen=True)
class OpaqueKernel:
    ops: tuple
    reason: str = "backend-specific lowering remains authoritative"


@dataclass(frozen=True)
class StructuredKernel:
    index: int
    kind: str
    body: object

    def __post_init__(self):
        if self.kind not in {"layout", "reduction", "opaque"}:
            raise ValueError(f"unknown structured kernel kind: {self.kind}")


def _accumulator(graph, op):
    from flowgpu.compiler.core.operators import OPERATORS
    spec = OPERATORS.get(op)
    if spec is None or spec.acc is None:
        raise ValueError(f"reduction {op} has no declared accumulator")
    return spec.acc


def _layout_ir(graph, group):
    node = group.anchor
    source_shapes = tuple(graph.values[value].shape for value in node.inputs)
    target = graph.values[node.output].shape
    widths = ()
    if node.op == "split":
        widths = (graph.values[node.output].shape[-1], graph.values[node.inputs[0]].shape[-1])
    elif node.op == "concat":
        widths = tuple(graph.values[value].shape[-1] for value in node.inputs)
    source = source_shapes[0]
    affine = ("i", "row_major(source)")
    if node.op == "reshape":
        affine = ("linear(i, target)", "linear(i, source)")
    elif node.op == "split":
        affine = ("linear(i, target)+part*width", "linear(i, source)")
    elif node.op == "concat":
        affine = ("linear(i, target)", "linear(i, source)+input_offset")
    return LayoutOp(
        node.op,
        tuple(node.inputs),
        node.output,
        IndexSpace(target),
        LayoutMap(node.op, source_shapes, target, widths, node.part,
                  affine=affine, tile=("row_major",), memory_scope="global",
                  contiguous=True),
    )


def _reduction_ir(graph, group, config):
    domain = tuple(group.reduction_domain)
    if not domain:
        reduction = next(node for node in group.members if node.op in REDUCTIONS)
        domain = tuple(graph.values[reduction.inputs[0]].shape)
    width = str(domain[-1])
    limit = config.reduction_block_limit
    chunk_stop = width if limit is None else f"min({limit}, next_power_of_2({width}))"
    stages = ("single",)
    if limit is not None and (not width.isdigit() or int(width) > limit):
        stages = ("fold", "finalize")
    reductions = tuple(
        AutoReduce(
            node.op,
            node.inputs[0],
            node.output,
            -1,
            _accumulator(graph, node.op),
            Range("column", "0", chunk_stop),
            stages,
            "identity" if node.op == "sum_last" else "-inf",
        )
        for node in group.members if node.op in REDUCTIONS
    )
    outer = Range("row", "0", _product(domain[:-1]) or "1")
    return Loop((outer,), reductions)


def analyze_group(graph, group, config, index):
    """Build a structured description without changing any compiler decision."""
    if group.anchor is not None and group.anchor.op in LAYOUTS and len(group.members) == 1:
        return StructuredKernel(index, "layout", _layout_ir(graph, group))
    if any(node.op in REDUCTIONS for node in group.members):
        return StructuredKernel(index, "reduction", _reduction_ir(graph, group, config))
    return StructuredKernel(index, "opaque", OpaqueKernel(tuple(node.op for node in group.members)))


def analyze(graph, fusion, config):
    """Return one structured kernel per existing fusion group, in order."""
    graph.validate()
    return tuple(analyze_group(graph, group, config, index) for index, group in enumerate(fusion.groups))
