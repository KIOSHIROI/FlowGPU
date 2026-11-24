"""DynamicBatch packs active requests and returns only active outputs."""
import pytest


def test_add_step_and_remove():
    from flowgpu.compiler.runtime.batching import DynamicBatch

    batch = DynamicBatch(3)
    assert batch.add("a", [1]) == 0
    assert batch.add("b", [2]) == 1
    outputs = batch.step(lambda rows: [sum(row) for row in rows], empty=lambda: [0])
    assert outputs == {"a": 1, "b": 2}
    batch.remove("a")
    assert batch.add("c", [3]) == 0
    assert len(batch) == 2


def test_capacity_and_duplicates_are_rejected():
    from flowgpu.compiler.runtime.batching import DynamicBatch

    batch = DynamicBatch(1)
    batch.add("a")
    with pytest.raises(MemoryError):
        batch.add("b")
    with pytest.raises(ValueError, match="already"):
        batch.add("a")


def test_step_checks_output_arity():
    from flowgpu.compiler.runtime.batching import DynamicBatch

    batch = DynamicBatch(2)
    batch.add("a", [1])
    with pytest.raises(ValueError, match="one output"):
        batch.step(lambda rows: [0], empty=lambda: [0])
