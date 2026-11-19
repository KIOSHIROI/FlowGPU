"""The frontend-neutral Dataflow IR -- Design.md 7.

Phase 1a moved the graph model out of ``compiler.ir`` so a frontend can
construct it without the DSL parser; ``compiler.ir.dataflow`` re-exports these
objects for compatibility.

A program becomes a graph of one uniform node shape (7.1):

    Node { op = Add, inputs = [x, residual], output = y }

`op` is a field, not a class, so no pass ever asks what kind of node it is
holding; it reads the operator name. That is also the shape the AST already
has, which is why 13.4 is a walk rather than a translation.

An Edge (7.2) is a producer -> consumer dependency, and it carries the value
that flows along it. A Buffer (7.3) is derived from the graph by 13.7 and
lives in `liveness.py`, so nothing here mentions storage.

Values are a table rather than a class. Design.md 7 lists Value among the IR
concepts, and this is that concept: `values[name]` is its type, and its
producer and consumers are graph queries. A Value class would have to keep
the type twice (once in the table, once in the object) and each object would
point back at the nodes that point at it.
"""

from dataclasses import dataclass

from compiler.core.operators import is_known_operator
from compiler.core.types import TensorType


# The two structural operators of 13.4's figure. They are deliberately not
# in the operator registry: a user cannot write them and they have no type
# rule, because an input node is where a value enters the graph and an
# output node is where one leaves it.
INPUT = "input"
OUTPUT = "output"


@dataclass(frozen=True)
class Node:
    """One operation, in the shape 7.1 gives it.

    `output` is None only for an output node: it consumes a value and is
    where the pipeline ends, so it produces nothing. Every other node
    produces exactly one value. Unique producers additionally depend on the
    frontend's single-assignment checks and the graph invariant validator;
    this dataclass alone cannot prevent duplicate output names.

    `part` is v0.2's multi-value metadata: a split lowers to one node per
    assigned name, and the part index is which slice of the parent this
    node's value is. None for every operator that produces one value.
    """

    op: str
    inputs: list[str]
    output: str | None
    part: int | None = None
    # Static scalar arguments that are part of an operation's ABI.
    attributes: object = None


@dataclass(frozen=True)
class Edge:
    """A producer -> consumer dependency (7.2) and the value it carries."""

    producer: Node
    consumer: Node
    value: str


def resolve_alias(aliases: dict[str, str], name: str) -> str:
    """Follow aliases to the value a name actually denotes."""
    seen = set()

    while name in aliases:
        if name in seen:
            # An alias cannot point at itself: the builder only ever points
            # one at a name that already exists.
            raise AssertionError(f"alias cycle at '{name}'")
        seen.add(name)
        name = aliases[name]

    return name


@dataclass
class DataflowGraph:
    """One pipeline's dataflow, in construction order.

    `nodes` is in construction order, which is a topological order: operands
    are lowered before the node that reads them, and def-before-use holds
    throughout, so every edge points forward. 13.7's lifetimes are intervals
    in exactly this order.
    """

    pipeline: str
    values: dict[str, TensorType]
    nodes: list[Node]
    # A name that is another name for the same value (`y = x`). Resolution is
    # the IR's job, not the reader's, so every name that reaches a node is
    # already canonical.
    aliases: dict[str, str]

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check the IR contract before analyses consume this mutable graph.

        Source name/type errors have already been diagnosed by the frontend.
        These failures mean the compiler or an IR client built a broken graph.
        This checks graph structure, not operator semantics or backend equivalence.
        """
        from compiler.core.diagnostic import CompileError, DiagnosticKind

        def fail(message):
            raise CompileError(DiagnosticKind.INTERNAL,
                               "invalid dataflow graph: " + message,
                               pipeline=self.pipeline)

        for alias, target in self.aliases.items():
            if alias not in self.values or target not in self.values:
                fail(f"alias '{alias}' refers to a missing value '{target}'")
        for alias in self.aliases:
            try:
                canonical = resolve_alias(self.aliases, alias)
            except AssertionError as exc:
                fail(str(exc))
            if self.values[alias] != self.values[canonical]:
                fail(f"alias '{alias}' has a different type from '{canonical}'")

        produced = set()
        for index, node in enumerate(self.nodes):
            if node.op not in (INPUT, OUTPUT) and not is_known_operator(node.op):
                fail(f"node {index} has unknown operator '{node.op}'")
            if node.op == INPUT and node.inputs:
                fail(f"input node {index} has operands")
            if node.op == OUTPUT:
                if node.output is not None or len(node.inputs) != 1:
                    fail(f"output node {index} must consume one value and produce none")
            elif node.output is None:
                fail(f"node {index} has no result")
            for value in node.inputs:
                if value not in self.values:
                    fail(f"node {index} reads missing value '{value}'")
                if value in self.aliases:
                    fail(f"node {index} reads noncanonical alias '{value}'")
                if value not in produced:
                    fail(f"node {index} reads '{value}' before its producer")
            if node.op in {'move_first_last','transpose'}:
                from compiler.core.index_maps import permutation_of
                try:permutation_of(node,len(self.values[node.inputs[0]].shape))
                except ValueError as error:fail(str(error))
            if node.output is not None:
                value = node.output
                if value not in self.values:
                    fail(f"node {index} produces untyped value '{value}'")
                if value in self.aliases:
                    fail(f"node {index} produces alias '{value}'")
                if value in produced:
                    fail(f"value '{value}' has multiple producers")
                produced.add(value)
        missing = set(self.values) - set(self.aliases) - produced
        if missing:
            fail(f"values without producers: {', '.join(sorted(missing))}")

    def resolve(self, name: str) -> str:
        """Follow aliases to the value a name actually denotes."""
        return resolve_alias(self.aliases, name)

    def index_of(self, node: Node) -> int:
        return self.nodes.index(node)

    def producer_of(self, name: str) -> Node | None:
        target = self.resolve(name)

        for node in self.nodes:
            if node.output == target:
                return node

        return None

    def consumers_of(self, name: str) -> list[Node]:
        target = self.resolve(name)

        return [node for node in self.nodes if target in node.inputs]

    def edges(self) -> list[Edge]:
        """Every dependency, as the IR sees it: node to node."""
        self.validate()
        edges = []

        for consumer in self.nodes:
            for value in consumer.inputs:
                producer = self.producer_of(value)

                if producer is None:
                    # The builder emits a node for every value as it defines
                    # it, so a value with no producer means the graph was
                    # built wrong -- a compiler defect, not a bad program.
                    raise AssertionError(f"value '{value}' has no producer")

                edge = Edge(producer=producer, consumer=consumer, value=value)

                # A node reading the same value twice (add(x, x)) is one
                # dependency, not two.
                if edge not in edges:
                    edges.append(edge)

        return edges

    def value_edges(self) -> list[tuple[str, str]]:
        """The dependency as `value -> value`, skipping the graph's shape.

        This is the readable view: `x -> y` says that y is computed from x.
        Nodes that produce nothing (an output node) contribute no pair, and
        a diamond shows up as two pairs rather than as a path.
        """
        pairs = []

        for node in self.nodes:
            if node.output is None:
                continue

            for value in node.inputs:
                pair = (self.resolve(value), node.output)

                if pair not in pairs:
                    pairs.append(pair)

        return pairs
