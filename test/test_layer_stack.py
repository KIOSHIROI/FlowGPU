"""LayerStack alternates output buffers and reuses one plan."""
from flowgpu.compiler.runtime.layers import LayerStack


class _FakePlan:
    def __init__(self):
        self.allocated = 0
        self.calls = []

    def allocate_buffers(self):
        self.allocated += 1
        return {"id": self.allocated}

    def run_into(self, value, layer, buffers):
        self.calls.append((value, layer, buffers))
        return {"y": value + layer}


def test_layer_stack_reuses_one_plan_and_alternates_buffers():
    plan = _FakePlan()
    stack = LayerStack(plan, [(1,), (2,), (3,)])
    assert plan.allocated == 2
    assert stack(0) == 6
    assert [entry[0] for entry in plan.calls] == [0, 1, 3]
    assert [entry[2] for entry in plan.calls] == [{"id": 1}, {"id": 2}, {"id": 1}]


def test_layer_stack_residual_is_a_host_add():
    plan = _FakePlan()
    stack = LayerStack(plan, [(1,), (2,)], residual=True)
    assert stack(0) == 4
