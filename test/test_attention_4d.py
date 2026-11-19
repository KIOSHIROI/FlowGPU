"""The v0.3 attention surface: 4D batching, GQA, and causal masking.

Every test states which contract it holds, and the contracts are the
Design.md rows: one operator for two ranks (2D and 4D), G expressed as
k/v's head count with the H % G divisibility checked where the checker
can see it (literal) and deferred to run time where it cannot (symbolic),
and causal as a separate operator NAME because the language has no
boolean literal to write in a call -- rms_norm/rmsnorm_hf is the
precedent for two conventions, two names.
"""

import ast
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowgpu.compiler import reference as reference_backend
from flowgpu.compiler import triton_backend
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.diagnostic import CompileError, DiagnosticKind


GQA4D = """
pipeline attn {
    input q : bf16[B, H, L, D]
    input k : bf16[B, G, S, D]
    input v : bf16[B, G, S, V]
    o = attention(q, k, v)
    output o
}
"""


# ---- the type layer: 4D and GQA ----

def test_4d_mha_types_to_the_batched_headed_output():
    result = compile_source(GQA4D)
    symbols = result.pipelines[0].symbols
    assert symbols.lookup("o").type.shape == ["B", "H", "L", "V"]
    assert symbols.lookup("o").type.dtype == "bf16"


def test_the_head_counts_may_differ():
    """G < H is GQA, the whole point of the second head-count slot."""
    result = compile_source("""
    pipeline gqa {
        input q : bf16[B, 32, L, D]
        input k : bf16[B, 8, S, D]
        input v : bf16[B, 8, S, V]
        o = attention(q, k, v)
        output o
    }
    """)
    assert result.pipelines[0].symbols.lookup("o").type.shape == ["B", "32", "L", "V"]


def test_2d_keeps_the_v01_contract():
    result = compile_source("""
    pipeline a2 {
        input q : bf16[L, D]
        input k : bf16[S, D]
        input v : bf16[S, V]
        o = attention(q, k, v)
        output o
    }
    """)
    assert result.pipelines[0].symbols.lookup("o").type.shape == ["L", "V"]


def test_mixed_ranks_are_rejected():
    with pytest.raises(CompileError, match='olumn 17: attention operands must share one rank: 2D \\[L,D\\] or 4D \\[B,H,L,D\\]') as caught:
        compile_source("""
        pipeline bad {
            input q : bf16[L, D]
            input k : bf16[B, G, S, D]
            input v : bf16[S, V]
            o = attention(q, k, v)
            output o
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "one rank" in caught.value.message


def test_literal_non_dividing_heads_are_rejected_statically():
    with pytest.raises(CompileError, match='olumn 17: GQA requires the key head count 3 to divide the query head count 7') as caught:
        compile_source("""
        pipeline bad {
            input q : bf16[B, 7, L, D]
            input k : bf16[B, 3, S, D]
            input v : bf16[B, 3, S, V]
            o = attention(q, k, v)
            output o
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "divide" in caught.value.message


def test_symbolic_non_dividing_heads_defer_to_run_time():
    """H and G are names; the checker cannot divide two names, so the
    program compiles and the generated code decides with tensors in hand."""
    result = compile_source(GQA4D)
    kernels = result.pipelines[0].reference.kernels
    assert "must divide the query head count" in "".join(
        kernel.source for kernel in kernels
    )


def test_4d_shape_mismatches_keep_their_own_reasons():
    with pytest.raises(CompileError, match='olumn 17: 4D attention requires q:\\[B,H,L,D\\], k:\\[B,G,S,D\\], v:\\[B,G,S,V\\]') as caught:
        compile_source("""
        pipeline bad {
            input q : bf16[B, H, L, D]
            input k : bf16[B, G, S, D2]
            input v : bf16[B, G, S, V]
            o = attention(q, k, v)
            output o
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE


# ---- causal ----

def test_causal_types_like_attention():
    result = compile_source("""
    pipeline c {
        input q : bf16[B, H, L, D]
        input k : bf16[B, G, S, D]
        input v : bf16[B, G, S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """)
    assert result.pipelines[0].symbols.lookup("oc").type.shape == ["B", "H", "L", "V"]


def test_causal_needs_the_hybrid_backend_like_attention():
    result = compile_source("""
    pipeline c {
        input q : bf16[L, D]
        input k : bf16[S, D]
        input v : bf16[S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """)
    refusals = result.pipelines[0].triton.refusals
    assert {r.operator for r in refusals} == {"causal_attention"}
    assert "hybrid" in refusals[0].reason


# ---- the reference: executed, not just emitted ----

def run_reference_pipeline(source, sizes, seed=11):
    program = compile_source(source).pipelines[0]
    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<r>", "exec"), namespace)
    bf16 = namespace["bf16"]
    random = __import__("random")
    random.seed(seed)
    inputs = {}
    for name, tensor_type in program.reference.inputs:
        count = 1
        for dimension in tensor_type.shape:
            count *= sizes[str(dimension)]
        inputs[name] = [random.uniform(-1.0, 1.0) for _ in range(count)]
    arguments = [inputs[name] for name, _ in program.reference.inputs]
    arguments += [sizes[name] for name in program.reference.dimensions]
    host = namespace[f"run_{program.reference.pipeline}"]
    return program, inputs, host(*arguments)


def test_reference_4d_gqa_matches_independent_math():
    B, H, G, L, S, D, V = 2, 4, 2, 3, 6, 5, 3
    sizes = {"B": B, "H": H, "G": G, "L": L, "S": S, "D": D, "V": V}
    program, inputs, outputs = run_reference_pipeline(GQA4D, sizes)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<r>", "exec"), namespace)
    bf16 = namespace["bf16"]

    for flat, got in enumerate(outputs["o"]):
        c = flat % V
        i = flat // V % L
        h = flat // V // L % H
        b = flat // V // L // H
        g = h // (H // G)
        scores = [sum(q[((b*H+h)*L+i)*D+d] * k[((b*G+g)*S+j)*D+d] for d in range(D)) / math.sqrt(D)
                  for j in range(S)]
        largest = max(scores)
        weights = [math.exp(s - largest) for s in scores]
        total = sum(weights)
        expected = bf16(sum(weights[j] * v[((b*G+g)*S+j)*V+c] for j in range(S)) / total)
        assert got == expected


def test_reference_causal_uses_the_kv_cache_offset():
    """With S > L the window is i + (S - L) + 1: the query sees the cached
    prefix. With S == L it is the plain triangle."""
    B, H, G, L, S, D, V = 1, 2, 1, 3, 6, 4, 2
    sizes = {"B": B, "H": H, "G": G, "L": L, "S": S, "D": D, "V": V}
    program, inputs, outputs = run_reference_pipeline("""
    pipeline c {
        input q : bf16[B, H, L, D]
        input k : bf16[B, G, S, D]
        input v : bf16[B, G, S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """, sizes)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<r>", "exec"), namespace)
    bf16 = namespace["bf16"]

    for flat, got in enumerate(outputs["oc"]):
        c = flat % V
        i = flat // V % L
        h = flat // V // L % H
        b = flat // V // L // H
        # the GQA head mapping the kernel implements: h // (H // G)
        g = h // (H // G)
        limit = i + S - L + 1
        scores = [sum(q[((b*H+h)*L+i)*D+d] * k[((b*G+g)*S+j)*D+d] for d in range(D)) / math.sqrt(D)
                  for j in range(limit)]
        largest = max(scores)
        weights = [math.exp(s - largest) for s in scores]
        total = sum(weights)
        expected = bf16(sum(weights[j] * v[((b*G+g)*S+j)*V+c] for j in range(limit)) / total)
        assert got == expected


def test_reference_causal_rejects_keys_shorter_than_queries():
    program = compile_source("""
    pipeline c {
        input q : bf16[B, H, L, D]
        input k : bf16[B, G, S, D]
        input v : bf16[B, G, S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """).pipelines[0]
    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<r>", "exec"), namespace)
    with pytest.raises(ValueError, match="at least the query length"):
        namespace["run_c"]([0.0] * (2*2*7*4), [0.0] * (2*1*6*4), [0.0] * (2*1*6*2),
                           2, 2, 7, 4, 1, 6, 2)


# ---- the Triton hybrid lowering ----

def test_hybrid_emits_4d_gqa_causal_and_parses():
    for impl in ("math", "sdpa"):
        program = compile_source("""
        pipeline a4 {
            input q : bf16[B, H, L, D]
            input k : bf16[B, G, S, D]
            input v : bf16[B, G, S, V]
            o = attention(q, k, v)
            oc = causal_attention(q, k, v)
            output o
            output oc
        }
        """, gpu_backend="hybrid", attention_impl=impl).pipelines[0]
        code = triton_backend.module([program.triton])
        ast.parse(code)  # the emitted file is valid Python
        kinds = {kernel.kind for kernel in program.triton.kernels}
        assert kinds == {"library_attention", "library_causal_attention"}


def test_hybrid_gqa_emits_the_divisibility_check():
    program = compile_source(GQA4D, gpu_backend="hybrid").pipelines[0]
    kernel = program.triton.kernels[0]
    assert "must divide the query head count" in kernel.source
    assert "repeat_interleave" in kernel.source
    # The check must be the live condition, not an unreachable raise: the
    # message alone would pass with the test folded to a constant.
    assert ".shape[1] % " in kernel.source


def test_hybrid_gqa_ratio_is_computed_before_either_repeat():
    """The A800 run caught this as a wrong answer, not a crash: reading
    k's head count AFTER k was repeated made v's factor H // H == 1, so v
    stayed un-grouped while every local test passed (nothing here executes
    torch). The structural guard: the ratio is bound to a name before the
    first repeat_interleave, so neither repeat reads a mutated operand."""
    program = compile_source(GQA4D, gpu_backend="hybrid").pipelines[0]
    source = program.triton.kernels[0].source
    ratio_line = next(
        line for line in source.splitlines()
        if line.strip().endswith(":") is False and " = " in line
        and ".shape[1] // " in line
    )
    first_repeat = source.index("repeat_interleave")
    ratio_name = ratio_line.strip().split(" = ")[0]
    assert source.index(f"{ratio_name} = ") < first_repeat
    # and the repeats use the bound name, not a shape expression
    for line in source.splitlines():
        if "repeat_interleave" in line:
            assert ".shape[1]" not in line


def test_hybrid_causal_guard_is_the_live_condition():
    program = compile_source("""
    pipeline c {
        input q : bf16[B, H, L, D]
        input k : bf16[B, G, S, D]
        input v : bf16[B, G, S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """, gpu_backend="hybrid").pipelines[0]
    source = program.triton.kernels[0].source
    assert ".shape[-2] < " in source
    assert "at least the query length" in source


def test_hybrid_causal_emits_the_offset_tril_mask():
    program = compile_source("""
    pipeline c {
        input q : bf16[B, H, L, D]
        input k : bf16[B, G, S, D]
        input v : bf16[B, G, S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """, gpu_backend="hybrid").pipelines[0]
    kernel = program.triton.kernels[0]
    assert "torch.tril" in kernel.source
    # diagonal = S - L is the kv-cache offset, not the plain triangle.
    assert "attn_s - attn_l" in kernel.source


def test_hybrid_sdpa_names_the_conventions():
    program = compile_source(GQA4D, gpu_backend="hybrid", attention_impl="sdpa").pipelines[0]
    kernel = program.triton.kernels[0]
    assert "enable_gqa=True" in kernel.source
    # causal rides as an explicit mask (torch's is_causal flag has the
    # offset-free semantics when L != S, not the language's kv-cache one)
    assert "is_causal" not in kernel.source
    assert "attn_mask" not in kernel.source


def test_causal_2d_sdpa_keeps_the_batch_promotion():
    """2D rides SDPA through [None, None] promotion, as v0.1 did."""
    program = compile_source("""
    pipeline c2 {
        input q : bf16[L, D]
        input k : bf16[S, D]
        input v : bf16[S, V]
        oc = causal_attention(q, k, v)
        output oc
    }
    """, gpu_backend="hybrid", attention_impl="sdpa").pipelines[0]
    source = program.triton.kernels[0].source
    assert "[None, None]" in source
    # the mask, not the flag: the flag would forget the cached prefix
    assert "attn_mask=" in source
    assert "torch.tril" in source
    assert "is_causal" not in source
