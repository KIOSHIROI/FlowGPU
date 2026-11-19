"""21's 显存流量, as a number.

`A = T_moved / T_floor`. For that number to be worth printing it has to get two
things right: the floor must be the program's *interface* and nothing else, and
every byte above the floor must be attributable to something an agent could go
and change. The tests here are those two claims, plus the invariant that says
the analysis has not silently stopped covering a value.
"""

import json

import pytest

from flowgpu.compiler.cli import main
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.traffic import at_unit, plan_traffic, ratio_at

FULLY_FUSED = """
pipeline fused {
    input x : bf16[N,D]
    input residual : bf16[N,D]
    input weight : bf16[D]
    y = add(x, residual)
    z = rms_norm(y, weight)
    output z
}
"""

MATERIALIZED = """
pipeline materialized {
    input r : bf16[N,D]
    input s : bf16[N,D]
    input x : bf16[N,D]
    t = mul(r, s)
    y = add(t, x)
    z = add(t, y)
    output z
}
"""

SHARED_INPUT = """
pipeline shared {
    input a : bf16[M,K]
    input b : bf16[K,N]
    input c : bf16[K,N]
    m1 = matmul(a, b)
    m2 = matmul(a, c)
    y = add(m1, m2)
    output y
}
"""


def traffic_of(source):
    result = compile_source(source).pipelines[0]
    traffic = plan_traffic(result.dataflow, result.fusion, result.memory)

    return result, traffic


def charge_of(traffic, value):
    matches = [charge for charge in traffic.charges if charge.value == value]
    assert len(matches) == 1, [charge.value for charge in traffic.charges]

    return matches[0]


def test_the_floor_is_the_interface_and_nothing_else():
    _, traffic = traffic_of(FULLY_FUSED)

    assert {entry.value for entry in traffic.floor} == {
        "x",
        "residual",
        "weight",
        "z",
    }
    # Three inputs read once, one output written once. `y` is not here: it is
    # fused into the kernel that computes `z`, so it costs nothing at all.
    assert sorted(entry.size for entry in traffic.floor) == sorted(
        ["2*N*D", "2*N*D", "2*D", "2*N*D"]
    )


def test_a_fully_fused_program_reaches_the_floor():
    _, traffic = traffic_of(FULLY_FUSED)

    assert traffic.floor_expression == traffic.moved_expression
    assert traffic.a == 1.0
    assert traffic.above_the_floor() == ()


def test_the_mlp_pays_for_the_value_between_two_matmuls():
    _, traffic = traffic_of(open("examples/mlp.fgpu").read())

    # m1 = matmul(...) -> silu -> matmul(m1, w2). `h2` is the group boundary, so
    # it is written once and read once: twice its own size, above the floor.
    h2 = charge_of(traffic, "h2")

    assert h2.count == 2
    assert h2.size == "2*B*H"
    assert "k0" in h2.reason
    assert [charge.value for charge in traffic.above_the_floor()] == ["h2"]
    assert traffic.a == 1.5


def test_a_value_read_twice_is_charged_once_per_reader():
    _, traffic = traffic_of(MATERIALIZED)

    t = charge_of(traffic, "t")

    assert t.count == 2
    # Read by two values, so no single kernel can keep it in registers: this is
    # 13.6's own single-use condition showing up as bytes.
    assert "read by 2 values" in t.reason
    assert "y" in t.reason and "z" in t.reason
    assert traffic.a == 1.5


def test_a_shared_input_is_charged_once_per_kernel():
    _, traffic = traffic_of(SHARED_INPUT)

    a = charge_of(traffic, "a")

    # `a` feeds two matmuls, and no fusion can join two matmuls -- so unlike
    # `t` above, this traffic is the program's own structure rather than a
    # decision the compiler made. A floor-only number cannot tell them apart;
    # the reason can.
    assert a.count == 2
    assert "read by 2 kernels" in a.reason
    assert traffic.a > 1.0


@pytest.mark.parametrize(
    "path",
    [
        "examples/residual_norm.fgpu",
        "examples/mlp.fgpu",
        "examples/qk_transform.fgpu",
    ],
)
def test_every_slot_value_is_paid_for(path):
    result, traffic = traffic_of(open(path).read())
    charged = {charge.value for charge in traffic.charges}

    for buffer in result.memory.buffers:
        if buffer.storage is None:
            continue

        assert buffer.value in charged, buffer.value


@pytest.mark.parametrize(
    "path",
    [
        "examples/residual_norm.fgpu",
        "examples/mlp.fgpu",
        "examples/qk_transform.fgpu",
    ],
)
def test_no_value_is_paid_for_twice(path):
    _, traffic = traffic_of(open(path).read())
    values = [charge.value for charge in traffic.charges]

    assert len(values) == len(set(values))


@pytest.mark.parametrize(
    "path",
    [
        "examples/residual_norm.fgpu",
        "examples/mlp.fgpu",
        "examples/qk_transform.fgpu",
    ],
)
def test_a_program_never_moves_less_than_its_floor(path):
    _, traffic = traffic_of(open(path).read())

    assert traffic.a >= 1.0


def test_at_unit_evaluates_a_byte_expression():
    assert at_unit("2*B*D") == 2
    assert at_unit("16") == 16
    assert at_unit("2*D + 2*N*D") == 4


def test_at_unit_refuses_something_that_is_not_a_byte_expression():
    with pytest.raises(ValueError, match="__import__\\('os'\\)\\.system\\('true'\\)"):
        at_unit("2*__import__('os').system('true')")


def test_ratio_at_a_concrete_shape_differs_from_the_unit_shape():
    _, traffic = traffic_of(open("examples/mlp.fgpu").read())

    # The unit shape is a normalisation, not the answer: with B=64, D=128, H=32
    # the intermediate is a smaller share of a bigger program.
    assert traffic.a == 1.5
    assert ratio_at(traffic, {"B": 64, "D": 128, "H": 32}) == pytest.approx(
        1.1667, abs=1e-4
    )


def test_the_same_program_always_produces_the_same_report():
    first = traffic_of(open("examples/mlp.fgpu").read())[1]
    second = traffic_of(open("examples/mlp.fgpu").read())[1]

    # Building the expressions from sets put them in hash order, which made the
    # printed report differ between runs of the same flowgpu.compiler.
    assert first.floor_expression == second.floor_expression
    assert first.moved_expression == second.moved_expression


def test_the_view_prints_the_traffic(capsys):
    main(["examples/mlp.fgpu", "--emit", "buffers"])
    printed = capsys.readouterr().out

    assert "traffic (bytes, per element)" in printed
    assert "    A       1.50   (every dimension = 1)" in printed
    assert "h2" in printed
    assert "what the boundary costs" in printed


def test_the_json_carries_the_traffic(capsys):
    main(["examples/mlp.fgpu", "--emit", "json"])
    report = json.loads(capsys.readouterr().out)
    traffic = report["pipelines"][0]["buffers"]["traffic"]

    assert traffic["A"] == 1.5
    assert traffic["floor"] == "2*B*D + 2*D*H + 2*H*D + 2*B*D"
    assert [charge["value"] for charge in traffic["charges"]] == [
        "x",
        "w1",
        "w2",
        "h2",
        "y",
    ]
    assert traffic["charges"][3]["reason"].endswith("what the boundary costs")
    assert all(charge["reason"] for charge in traffic["charges"])
