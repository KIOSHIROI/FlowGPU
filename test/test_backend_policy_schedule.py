from flowgpu.compiler.backend.backend_policy import BackendPolicy, make_backend_policy
from flowgpu.compiler.analysis.kernel_schedule import analyze_schedule
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.backend import reference


SOURCE = """
pipeline p {
    input a : bf16[N,D]
    input b : bf16[N,D]
    input c : bf16[N,D]
    x = add(a, b)
    y = mul(c, c)
    output x
    output y
}
"""


def test_compile_exposes_default_policy_and_sequential_schedule():
    result = compile_source(SOURCE).pipelines[0]
    assert isinstance(result.backend_policy, BackendPolicy)
    assert result.backend_policy.numeric_mode == "fp64"
    assert result.backend_policy.concurrency == "sequential"
    assert result.kernel_schedule.mode == "sequential"
    assert result.kernel_schedule.order == tuple(range(len(result.triton.kernels)))
    assert all(len(level) == 1 for level in result.kernel_schedule.levels)


def test_overlap_schedule_exposes_independent_level_without_changing_lowering_order():
    result = compile_source(SOURCE, concurrency="overlap").pipelines[0]
    schedule = result.kernel_schedule
    assert schedule.mode == "overlap"
    assert schedule.levels[0] == (0, 1)
    assert schedule.concurrent
    assert result.triton.host.index("p_k0[") < result.triton.host.index("p_k1[")


def test_policy_rejects_unknown_modes():
    try:
        make_backend_policy(numeric_mode="device")
    except ValueError as error:
        assert "numeric_mode" in str(error)
    else:
        raise AssertionError("unknown numeric mode accepted")


def test_reference_numeric_mode_is_named_and_legacy_alias_remains():
    source = """
    pipeline p {
        input x : fp32[N,D]
        y = add(x, x)
        output y
    }
    """
    program = compile_source(source, numeric_mode="fp32").pipelines[0].reference
    assert program.numeric_mode == "fp32"
    named = reference.run(program, numeric_mode="fp32")
    legacy = reference.run(program, emulate_fp32=True)
    assert named == legacy
