"""Per-shape caching: a different shape/stride/dtype/device is a rebuild."""
import pytest

from flowgpu.compiler.runtime.shape_specialization import (
    ShapeSpecializedRunner, tensor_metadata)


class _Tensor:
    def __init__(self, shape, stride, dtype="torch.float32", device="cuda:0", ptr=0):
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self.device = device
        self._ptr = ptr

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self._ptr


def _tensor(shape, stride=None, **kwargs):
    return _Tensor(shape, tuple(stride or [1] * len(shape)), **kwargs)


def test_metadata_captures_shape_stride_dtype_device_and_alignment():
    meta = tensor_metadata(_tensor((2, 4), (4, 1), ptr=130))
    assert meta.shape == (2, 4)
    assert meta.stride == (4, 1)
    assert meta.alignment_mod128 == 2


def test_same_metadata_hits_and_changed_shape_rebuilds():
    built = []
    calls = []

    def build(*args):
        built.append(args[0].shape)
        def run(*inner):
            calls.append(inner[0].shape)
            return inner[0].shape
        return run

    runner = ShapeSpecializedRunner(build)
    first = _tensor((2, 4))
    assert runner(first) == (2, 4)
    assert runner(first) == (2, 4)
    assert built == [(2, 4)]
    assert runner.misses == 1 and runner.hits == 1

    second = _tensor((3, 4))
    assert runner(second) == (3, 4)
    assert built == [(2, 4), (3, 4)]


def test_cache_is_bounded_and_evicts_least_recently_used():
    built = []
    runner = ShapeSpecializedRunner(lambda *args: (built.append(args[0].shape) or (lambda *a: None)),
                                    max_entries=1)
    runner(_tensor((2,)))
    runner(_tensor((3,)))
    runner(_tensor((2,)))
    assert built == [(2,), (3,), (2,)]


def test_failed_build_is_not_cached():
    attempts = []

    def build(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("no device")
        return lambda *a: "ok"

    runner = ShapeSpecializedRunner(build)
    with pytest.raises(ValueError):
        runner(_tensor((2,)))
    assert runner(_tensor((2,))) == "ok"
    assert len(attempts) == 2
    assert len(runner) == 1


def test_invalid_builder_and_budget_are_rejected():
    with pytest.raises(TypeError, match="callable"):
        ShapeSpecializedRunner(None)
    with pytest.raises(ValueError, match="positive integer"):
        ShapeSpecializedRunner(lambda *a: None, max_entries=0)


def test_space_contract_filter_keeps_only_materialize_off_cuda():
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    from flowgpu.compiler.runtime.tuning_space import generate_cross_domain_space
    cuda = generate_cross_domain_space(4096, 1024, platform_config=resolve_platform("cuda"))
    other = generate_cross_domain_space(4096, 1024,
                                        platform_config=resolve_platform("kunlunxin"))
    assert any(candidate.label == "materialize" for candidate in cuda)
    assert [candidate.label for candidate in other] == ["materialize"]
