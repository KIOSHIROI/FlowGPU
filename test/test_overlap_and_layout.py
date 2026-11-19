import ast

from compiler.compile import compile_source
from compiler.triton_backend import module


def test_overlap_keeps_independent_outputs_on_independent_streams():
    source = """pipeline p {
        input a : fp32[N,D]
        input b : fp32[N,D]
        x = relu(a)
        y = relu(b)
        output x
        output y
    }"""
    pipeline = compile_source(source, concurrency="overlap").pipelines[0]
    code = module([pipeline.triton])
    ast.parse(code)
    assert len(pipeline.triton.kernels) == 2
    assert pipeline.kernel_schedule.levels == ((0, 1),)
    assert code.count("torch.cuda.Stream") == 2
    assert "wait_event" in code
    assert len(pipeline.memory.slots) >= 2


def test_layout_ir_exposes_affine_tile_and_scope_metadata():
    source = "pipeline p { input x : bf16[N,64] a,b=split(x) output a output b }"
    pipeline = compile_source(source).pipelines[0]
    mapping = pipeline.lowering_ir[0].body.mapping
    assert mapping.affine
    assert mapping.tile == ("row_major",)
    assert mapping.memory_scope == "global"
    assert mapping.contiguous is True
