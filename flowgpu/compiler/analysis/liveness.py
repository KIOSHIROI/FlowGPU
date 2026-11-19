"""Liveness and buffer analysis (Design.md 13.7).

13.7 asks for four things -- tensor lifetime, buffer lifetime,
materialization, reuse -- and 11 gives the artifact they produce:

    Buffer {
        size = ...
        lifetime = ...
        storage = ...
    }

and the six facts to derive: producer, consumer, lifetime, size, reuse
opportunity, materialization requirement. All six are derived here, and only
two of them need a decision the document does not make:

* **lifetime** is 7.3's `lifetime = [MatMul, ReLU]`: an interval from the
  earliest member of its producer kernel through the latest member of its
  consumer kernels. This conservative expansion accounts for fusion. 13.4 stores the
  nodes in a topological order, so an interval in that order is exactly that.
  A value with no consumer at all is live over its own definition, which is
  the same statement as "it is dead".

* **size** is 11's `size = ...`, which the document leaves open. Shapes hold
  symbolic dimensions (8.3), so the size is a product rather than a number:
  `bf16[N,D]` is `2*N*D` bytes. Numeric factors are folded, so `bf16[2,4]` is
  `16`.

* **storage** is 11's `storage = ...`, also left open. It records where a
  buffer lives: `input` for a tensor the caller passed in, a slot number for
  one the compiler allocates, and nothing at all for a value that never needs
  storage. Slots are what "reuse opportunity" means here -- two buffers whose
  live intervals do not overlap can share one -- and the slot's own interval
  is the buffer lifetime that 13.7 lists separately from the tensor lifetime.

  The plan assigns global tensor buffers; shared memory is internal to the
  device kernel. What it can decide is how many
  buffers are needed at once, which is the part of 13.7's goal ("fewer
  unnecessary memory traffic") that does not need a backend.

Two facts are taken from fusion (13.6) rather than recomputed: a fused value
never needs storage, and the fusion report names the kernel that keeps it.
That is 11's `materialization requirement`: fusion decides *whether* a value is
materialized, liveness decides *when* the buffers that remain are alive, so
`MatMul + ReLU` removes h's buffer and the slot allocator then reuses what is
left. The two sections run in that order for that reason.
"""

from dataclasses import dataclass

from flowgpu.compiler.core.types import TensorType
from flowgpu.compiler.ir.dataflow import INPUT, OUTPUT, DataflowGraph
from flowgpu.compiler.optimization.fusion import FusionReport
from flowgpu.compiler.analysis.precision import DTYPE_BYTES


def size_in_bytes(tensor_type: TensorType) -> str:
    """11's `size`, as a product over dtype width and dimensions.

    Symbolic dimensions cannot be multiplied out (8.3: `bf16[N,D]` has no
    numeric extent), so the result is an expression such as `2*N*D`. Factors
    that are integers are folded, so `bf16[2,4]` comes out as `16` -- the two
    forms are the same product, one of them already evaluated.
    """
    factors = [DTYPE_BYTES[tensor_type.dtype]] + list(tensor_type.shape)

    number = 1
    symbols = []

    for factor in factors:
        if str(factor).isdigit():
            number *= int(factor)
        else:
            symbols.append(str(factor))

    if not symbols:
        return str(number)

    if number == 1:
        return "*".join(symbols)

    return "*".join([str(number)] + symbols)


@dataclass(frozen=True)
class Buffer:
    """One value's storage, in the shape 11 gives a Buffer.

    `start` and `end` are node indices in the topological order 13.4 stores,
    so they are 7.3's `lifetime = [MatMul, ReLU]` for the value. `storage` is
    None when the value never needs any -- because fusion keeps it in a
    kernel, or because nothing reads it -- and `note` says which.
    """

    value: str
    tensor_type: TensorType
    start: int
    end: int
    size: str
    storage: str | None
    note: str

    @property
    def materialized(self) -> bool:
        return self.storage is not None

    @property
    def allocated(self) -> bool:
        """Compiler-allocated, as opposed to a tensor the caller passed in."""
        return self.storage is not None and self.storage != "input"


@dataclass(frozen=True)
class StorageSlot:
    """One allocation, shared by every value with the same type.

    13.7 lists tensor lifetime and buffer lifetime separately, and this is the
    second of them: the interval a piece of storage is occupied, from the
    first value that uses it to the last one that does.
    """

    index: int
    values: tuple
    start: int
    end: int


@dataclass(frozen=True)
class MemoryPlan:
    buffers: tuple
    slots: tuple
    # Stage-private allocations have a different lifetime coordinate system
    # (execution stages, not source nodes); never silently reuse tensor slots.
    stage_plan: object = None

    @property
    def materialized(self) -> tuple:
        return tuple(buffer for buffer in self.buffers if buffer.materialized)

    @property
    def eliminated(self) -> tuple:
        return tuple(
            buffer for buffer in self.buffers if not buffer.materialized
        )

    @property
    def allocated(self) -> tuple:
        return tuple(buffer for buffer in self.buffers if buffer.allocated)


def plan(graph: DataflowGraph, fusion: FusionReport, *, reuse=True,
         dependence_guard=False) -> MemoryPlan:
    """Derive buffers; disable slot reuse when kernels may overlap.

    `dependence_guard` is an **opt-in** refinement of slot reuse, off by default.
    Plain reuse trusts the liveness interval alone; the guard additionally
    refuses to share a slot between two buffers when the accesses that touch them
    cannot be shown to be disjoint. Only an *indexed* access -- a permutation,
    gather or scatter -- makes that provable, so on today's programs the guard
    and the default agree; it exists for `scatter_by_index`, where an address is
    computed and an interval argument stops working.

    It is opt-in because turning it on is a change of default, and
    `docs/COMPILER_ACCEPTANCE.md` requires a parent/candidate comparison before a
    mechanism becomes the default. Off, it changes nothing; on, it can only make
    the plan use more slots.
    """
    from flowgpu.compiler.analysis.access_shape import BLOCK, STRIDED, of_op, worst
    graph.validate()
    index_of = graph.index_of

    # Which kernel keeps each fused value.
    kernel_of = {}
    execution_span = {}
    for number, group in enumerate(fusion.groups):
        start = min(index_of(node) for node in group.members)
        end = max(index_of(node) for node in group.members)
        for node in group.members:
            execution_span[index_of(node)] = (start, end)
            if node.output is not None:
                kernel_of[node.output] = number

    buffers = []
    needs_slot = set()

    for value in graph.values:
        if value in graph.aliases:
            # Another name for a value that already has a buffer.
            continue

        producer = graph.producer_of(value)
        consumers = graph.consumers_of(value)

        if (not graph.values[value].shape and producer.op != INPUT
                and value not in fusion.fused_values):
            # A scalar is a host number or a program-uniform value: it has no
            # storage to allocate and no pointer to pass, so a rank-0 value
            # produced by an operation must be fused into the kernel that
            # consumes it (then it is a register, not a buffer). A rank-0
            # result that would need storage is a broken graph rather than a
            # bad program; the frontend refuses those.
            from flowgpu.compiler.core.diagnostic import (
                CompileError,
                DiagnosticKind,
            )
            raise CompileError(
                DiagnosticKind.INTERNAL,
                f"scalar value '{value}' has a producer; a buffer plan can "
                f"only hold tensors and caller-provided inputs",
                pipeline=graph.pipeline,
            )

        producer_index = index_of(producer)
        start = execution_span.get(producer_index, (producer_index, producer_index))[0]
        end = max((execution_span.get(index_of(consumer), (index_of(consumer), index_of(consumer)))[1]
                   for consumer in consumers), default=start)

        # An output value is handed back to the caller after every kernel has
        # run, not at the point where the `output` statement appears. So its
        # storage is live to the end of the program: reusing its slot for a
        # value defined later would return that other value instead, which is
        # a wrong answer rather than a lost optimization. Running the reference
        # backend is what showed this up -- both values came back as the same
        # list, holding the second one's numbers.
        if any(consumer.op == OUTPUT for consumer in consumers):
            end = max(end, index_of(graph.nodes[-1]))

        if value in fusion.fused_values:
            storage = None
            note = f"fused into k{kernel_of[value]}"
        elif not consumers:
            storage = None
            note = "no consumer"
        elif producer.op == INPUT:
            storage = "input"
            note = "provided by the caller"
        else:
            storage = None      # assigned by the allocator below
            note = ""
            needs_slot.add(value)

        buffers.append(
            Buffer(
                value=value,
                tensor_type=graph.values[value],
                start=start,
                end=end,
                size=size_in_bytes(graph.values[value]),
                storage=storage,
                note=note,
            )
        )

    # Slot allocation. This is linear scan over the intervals, in the same
    # order a register allocator scans them: a slot is free for a buffer when
    # the buffer that held it ended strictly before this one starts. Strictly,
    # because a buffer ending at node i is still read there while the buffer
    # starting at node i is written there.
    #
    # Reuse requires an identical type rather than a size that fits, because
    # symbolic dimensions give no ordering between two sizes: `2*N*D` and
    # `2*B*H*S*D` cannot be compared in v0.1. Fitting one buffer into another
    # is a refinement for when sizes are numbers.
    pending = sorted(
        (buffer for buffer in buffers if buffer.value in needs_slot),
        key=lambda buffer: buffer.start,
    )

    slots = {}
    free = {}
    live = []
    allocated = {}
    count = 0

    # Which buffer occupies each slot, and how each buffer's kernel addresses its
    # operands -- the two facts the guard needs.
    slot_owner = {}
    access_kind = {}

    def kind_of(value):
        if value not in access_kind:
            number = kernel_of.get(value)
            if number is None:
                access_kind[value] = BLOCK
            else:
                ops = [node.op for node in fusion.groups[number].members]
                access_kind[value] = worst([of_op(op) for op in ops]).kind
        return access_kind[value]

    for buffer in pending:
        still_live = []

        for end, key, slot in live:
            if end < buffer.start:
                free.setdefault(key, []).append(slot)
            else:
                still_live.append((end, key, slot))

        live = still_live

        key = (buffer.tensor_type.dtype, tuple(buffer.tensor_type.shape))

        if reuse and free.get(key) and dependence_guard:
            # take a free slot only if the buffer that last owned it is
            # provably disjoint from this one; both precise means both are
            # blocked/strided accesses, which an interval can order
            slot = None
            for candidate in reversed(free[key]):
                other = slot_owner.get(candidate)
                precise = (other is None
                           or kind_of(other) in (BLOCK, STRIDED)
                           and kind_of(buffer.value) in (BLOCK, STRIDED))
                if precise:
                    free[key].remove(candidate)
                    slot = candidate
                    break
            if slot is None:
                slot = count
                count += 1
        elif reuse and free.get(key):
            slot = free[key].pop()
        else:
            slot = count
            count += 1

        live.append((buffer.end, key, slot))
        allocated[buffer.value] = slot
        slot_owner[slot] = buffer.value

    buffers = [
        Buffer(
            value=buffer.value,
            tensor_type=buffer.tensor_type,
            start=buffer.start,
            end=buffer.end,
            size=buffer.size,
            storage=(
                f"slot{allocated[buffer.value]}"
                if buffer.value in allocated
                else buffer.storage
            ),
            note=buffer.note,
        )
        for buffer in buffers
    ]

    storage = {}
    for buffer in buffers:
        if not buffer.tensor_type.shape and buffer.allocated:
            # A scalar is the caller's number, never a compiler allocation:
            # `torch.empty(())` would be a scalar tensor the launch passes as a
            # pointer. The check above already refuses a produced scalar; this
            # one pins the allocator itself.
            raise AssertionError(
                f"scalar '{buffer.value}' must not receive a storage slot"
            )
        if buffer.allocated:
            storage.setdefault(int(buffer.storage[4:]), []).append(buffer)

    return MemoryPlan(
        buffers=tuple(buffers),
        slots=tuple(
            StorageSlot(
                index=slot,
                values=tuple(entry.value for entry in entries),
                start=min(entry.start for entry in entries),
                end=max(entry.end for entry in entries),
            )
            for slot, entries in sorted(storage.items())
        ),
    )
