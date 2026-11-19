import pytest

from compiler.backend_policy import make_backend_policy
from compiler.compile import compile_source
from compiler.shape_constraints import divisible, equal, ge


def test_backend_policy_is_shared_and_schedule_is_exposed():
    result = compile_source(
        "pipeline p { input x : fp32[N,D] y=relu(x) output y }",
        concurrency="overlap",
        numeric_mode="fp32",
    ).pipelines[0]
    assert result.backend_policy.numeric_mode == "fp32"
    assert result.backend_policy.concurrency == "overlap"
    assert result.kernel_schedule.mode == "overlap"
    assert result.kernel_schedule.order == tuple(range(len(result.triton.kernels)))


def test_shape_constraints_validate_specialized_bindings():
    source = "pipeline p { input x : fp32[N,D] y=relu(x) output y }"
    result = compile_source(
        source,
        shape_constraints=(divisible("D", 8), ge("N", 2)),
        shape_bindings={"N": 4, "D": 16},
    )
    assert result.pipelines[0].reference.complete
    with pytest.raises(ValueError, match="shape constraint violated"):
        compile_source(
            source,
            shape_constraints=(divisible("D", 8), equal("N", 4)),
            shape_bindings={"N": 3, "D": 10},
        )


def test_shape_constraints_reject_unknown_symbols():
    with pytest.raises(ValueError, match="unknown shape symbols"):
        compile_source(
            "pipeline p { input x : fp32[N,D] y=relu(x) output y }",
            shape_constraints=(ge("Q", 1),),
        )


def test_reference_numeric_modes_are_named():
    source = "pipeline p { input x : fp32[N,D] y=softmax(x) output y }"
    fp64 = compile_source(source, numeric_mode="fp64").pipelines[0].reference
    fp32 = compile_source(source, numeric_mode="fp32").pipelines[0].reference
    assert fp64.numeric_mode == "fp64"
    assert fp32.numeric_mode == "fp32"


def test_policy_rejects_invalid_numeric_and_concurrency_modes():
    with pytest.raises(ValueError, match="numeric_mode must be one of \\('fp64', 'fp32'\\)"):
        make_backend_policy(numeric_mode="device")
    with pytest.raises(ValueError, match="concurrency must be one of \\('sequential', 'overlap'\\)"):
        make_backend_policy(concurrency="unsafe")
