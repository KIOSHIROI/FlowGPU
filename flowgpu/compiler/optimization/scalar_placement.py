"""IR-native scalar placement analysis (SPS S1).

The standalone SPS prototype showed that the placement of a
scalar is a constrained cost decision: host formal, constexpr specialisation,
program-uniform device load, or materialised tile.  S1 moves the decision into
the compiler and realises the first two placements for *scalar inputs*:

* HOST: keep the rank-0 input as a kernel formal (the existing behaviour);
* CONSTEXPR: when the caller supplies the value at compile time and the call
  regime amortises the compile, inline the literal into the kernel and remove
  the scalar from the generated host ABI.

Program-uniform and tile placements need lowering work (program-scalar IR and a
materialisation rewrite) and stay out of S1; the planner records them as
unrealised rather than pretending they are available.

The analysis is deliberately source-level: it operates on the frontend-neutral
core AST + dataflow graph, so DSL, Python eDSL and imported Triton programs all
get the same plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Placement(str, Enum):
    HOST = "host"
    CONSTEXPR = "constexpr"
    UNIFORM = "uniform"
    TILE = "tile"


@dataclass(frozen=True)
class ScalarDecision:
    name: str
    dtype: str
    placement: Placement
    value: float | int | None = None
    literal: str | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype,
                "placement": self.placement.value, "value": self.value,
                "literal": self.literal, "reason": self.reason}


@dataclass
class PlacementPlan:
    pipeline: str
    calls: int
    method: str
    decisions: tuple = field(default_factory=tuple)

    @property
    def constexpr_scalars(self) -> dict[str, str]:
        return {decision.name: decision.literal
                for decision in self.decisions
                if decision.placement is Placement.CONSTEXPR and decision.literal is not None}

    @property
    def uniform_scalars(self) -> tuple[str, ...]:
        return tuple(decision.name for decision in self.decisions
                     if decision.placement is Placement.UNIFORM)

    @property
    def host_scalars(self) -> tuple[str, ...]:
        return tuple(decision.name for decision in self.decisions
                     if decision.placement is Placement.HOST)

    def as_dict(self) -> dict:
        return {"pipeline": self.pipeline, "calls": self.calls, "method": self.method,
                "decisions": [decision.as_dict() for decision in self.decisions]}


@dataclass(frozen=True)
class PlacementCostModel:
    #: One host scalar formal per call: validation + argument binding.
    host_scalar_per_call_us: float = 0.10
    #: One compile of a specialised variant; measured compile cost is larger,
    #: but the planner only needs the crossover, and the H100 calibration in
    #: the standalone prototype used 10 ms.
    constexpr_compile_us: float = 10_000.0
    #: Code-size/instruction-cache pressure per specialised scalar (amortised).
    constexpr_code_us: float = 0.0
    #: One scalar launch argument per kernel launch; a scalar read by multiple
    #: fusion groups is passed to each of them, so the host cost is per group.
    host_arg_per_group_us: float = 0.05


def _scalar_inputs(pipeline):
    from flowgpu.compiler.core.ast import InputDecl
    return [statement for statement in pipeline.statements
            if isinstance(statement, InputDecl) and statement.type.is_scalar]


def _literal_for(value, dtype: str) -> str:
    if dtype in ("int32", "int64"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"scalar {dtype} needs a Python int, got {value!r}")
        return str(int(value))
    if dtype in ("fp32", "fp16", "bf16"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"scalar {dtype} needs a number, got {value!r}")
        return repr(float(value))
    raise ValueError(f"unsupported scalar dtype {dtype!r}")


def _groups_using(graph, fusion, name: str) -> int:
    """How many fusion groups read the scalar value ``name``."""
    if fusion is None:
        return 1
    count = 0
    for group in getattr(fusion, "groups", ()):
        if any(name in node.inputs for node in getattr(group, "members", ())):
            count += 1
    return max(1, count)


def plan_placements(pipeline, graph, *, specialization=None, calls: int = 1,
                    cost_model: PlacementCostModel | None = None,
                    method: str = "sps-ir", fusion=None) -> PlacementPlan:
    """Choose HOST or CONSTEXPR for every scalar input.

    ``specialization`` maps scalar names to compile-time values.  A value that
    is not supplied cannot become constexpr; it stays host.  The decision for
    each scalar is independent (the only shared term is the compile, which is
    paid once per specialised pipeline), so this is the SPS savings rule
    specialised to the realised placement subset.
    """
    if calls <= 0:
        raise ValueError("calls must be positive")
    if specialization is None:
        specialization = {}
    model = cost_model or PlacementCostModel()
    scalar_inputs = _scalar_inputs(pipeline)
    known = {statement.name for statement in scalar_inputs}
    unknown = set(specialization) - known
    if unknown:
        raise ValueError(f"specialization names not scalar inputs: {sorted(unknown)}")
    decisions = []
    for statement in scalar_inputs:
        name = statement.name
        dtype = statement.type.dtype
        if name not in specialization:
            decisions.append(ScalarDecision(
                name=name, dtype=dtype, placement=Placement.HOST,
                reason="no compile-time value supplied; host formal"))
            continue
        value = specialization[name]
        literal = _literal_for(value, dtype)
        # A constexpr removes the host formal every call and pays one compile.
        # The decision is per scalar under the additive SPS model; the shared
        # compile cost is charged once per specialised scalar by this S1
        # implementation (conservative: it never overstates the gain).
        groups = _groups_using(graph, fusion, name)
        saving_us = calls * (model.host_scalar_per_call_us
                             + groups * model.host_arg_per_group_us)
        cost_us = model.constexpr_compile_us + model.constexpr_code_us
        if saving_us > cost_us:
            decisions.append(ScalarDecision(
                name=name, dtype=dtype, placement=Placement.CONSTEXPR,
                value=value, literal=literal,
                reason=f"{saving_us:.1f} us saved over {calls} calls in "
                       f"{groups} group(s) > {cost_us:.0f} us compile"))
        else:
            decisions.append(ScalarDecision(
                name=name, dtype=dtype, placement=Placement.HOST, value=value,
                reason=f"specialisation not amortised: {saving_us:.1f} us saved "
                       f"in {groups} group(s) <= {cost_us:.0f} us compile"))
    # Program-uniform scalar producers already lower through the program-scalar
    # IR; S3 brings them into the same plan instead of leaving them invisible.
    if graph is not None:
        for node in getattr(graph, "nodes", ()):
            if node.op != "program_scalar":
                continue
            value = graph.values.get(node.output)
            if value is None or not value.is_scalar:
                continue
            groups = _groups_using(graph, fusion, node.output)
            decisions.append(ScalarDecision(
                name=node.output, dtype=value.dtype, placement=Placement.UNIFORM,
                reason=f"program-scalar IR: one device load per program in "
                       f"{groups} group(s), folded into a scalar base"))
    return PlacementPlan(pipeline=pipeline.name, calls=calls, method=method,
                         decisions=tuple(decisions))


def plan_for_graph(pipeline, graph, **options) -> PlacementPlan:
    """Alias with the graph argument explicit for IR-native callers."""
    return plan_placements(pipeline, graph, **options)
