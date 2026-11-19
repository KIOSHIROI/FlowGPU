"""The CLI is the project's face: exit codes and stream discipline matter."""

import json
from pathlib import Path

import pytest

from flowgpu.compiler.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
RESIDUAL_NORM = EXAMPLES / "residual_norm.fgpu"


def test_compiles_shipped_example(capsys):
    code = main([str(RESIDUAL_NORM)])
    captured = capsys.readouterr()

    assert code == 0
    assert "bf16[N,D]" in captured.out
    assert "OK" in captured.out
    assert captured.err == ""


def test_reports_error_on_stderr_and_exits_1(capsys):
    code = main([str(EXAMPLES / "errors" / "dtype_mismatch.fgpu")])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert "type error" in captured.err


def test_missing_file_exits_2(capsys):
    code = main([str(EXAMPLES / "does_not_exist.fgpu")])
    captured = capsys.readouterr()

    assert code == 2
    assert "cannot read" in captured.err


def test_json_output_is_machine_readable(capsys):
    code = main([str(RESIDUAL_NORM), "--emit", "json"])
    captured = capsys.readouterr()

    assert code == 0
    # No "OK" banner: stdout must stay parseable for an agent repair loop.
    payload = json.loads(captured.out)
    assert len(payload["pipelines"]) == 1

    pipeline = payload["pipelines"][0]
    assert pipeline["pipeline"] == "residual_norm"
    assert {"from": "y", "to": "z"} in pipeline["edges"]
    assert pipeline["outputs"][0]["type"] == {
        "dtype": "bf16",
        "shape": ["N", "D"],
    }


def test_several_pipelines_in_one_file(tmp_path, capsys):
    path = tmp_path / "two.fgpu"
    path.write_text(
        "pipeline a { input x : bf16[N,D] output x }\n"
        "pipeline b { input y : bf16[N,D] output y }\n",
        encoding="utf-8",
    )

    code = main([str(path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "pipeline a" in captured.out
    assert "pipeline b" in captured.out
    assert "(2 pipelines)" in captured.out


def test_error_names_the_pipeline_that_failed(tmp_path, capsys):
    path = tmp_path / "two.fgpu"
    path.write_text(
        "pipeline good { input x : bf16[N,D] output x }\n"
        "pipeline bad { input y : bf16[N,D] z = matmul(y, y) output z }\n",
        encoding="utf-8",
    )

    code = main([str(path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "in pipeline 'bad'" in captured.err


def test_tokens_emit_skips_comments(capsys):
    code = main([str(RESIDUAL_NORM), "--emit", "tokens"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.splitlines()[0].startswith("PIPELINE")
    assert "#" not in captured.out


def test_syntax_error_reports_a_position(tmp_path, capsys):
    broken = tmp_path / "broken.fgpu"
    broken.write_text(
        "pipeline p {\n  input x bf16[N,D]\n  output x\n}\n",
        encoding="utf-8",
    )

    code = main([str(broken)])
    captured = capsys.readouterr()

    assert code == 1
    assert "syntax error" in captured.err
    assert "line 2" in captured.err


def test_stdin_is_accepted(capsys, monkeypatch):
    import io
    import sys

    monkeypatch.setattr(
        sys, "stdin", io.StringIO("pipeline p { input x : bf16[N,D] output x }\n")
    )

    code = main(["-"])
    captured = capsys.readouterr()

    assert code == 0
    assert "pipeline p" in captured.out


@pytest.mark.parametrize(
    "mode",
    [
        "summary",
        "ast",
        "symbols",
        "dataflow",
        "fusion",
        "buffers",
        "triton",
        "tokens",
        "json",
    ],
)
def test_every_emit_mode_handles_infix_operators(tmp_path, capsys, mode):
    """中缀算术必须在每个 stage 都活着，而不只是类型检查那一关。

    它要从词法一路走到 IR：降级成算子调用、类型检查、渲染、降级成 IR
    节点，嵌套时还要被扁平化成一个临时值。这正是"新增一个节点种类"要
    改好几处、而"降级到已有节点"一处都不用改的原因 —— 端到端跑遍所有
    emit 模式，是唯一能一次覆盖全部关卡的守卫。

    13.6 与 13.7 也被这一圈覆盖：`y = x - r * s` 里 sub 与 mul 都是逐元素
    算子，中间值 `%0`（`r * s`）只被读一次，所以它应当被融合掉、不占 buffer。
    13.8 再往下走一步：融合的结果必须是**一个** kernel，而且 `%0` 在里面
    只能是一个寄存器表达式，不能是指针。
    """
    path = tmp_path / "arith.fgpu"
    path.write_text(
        "pipeline arith {\n"
        "    input x : bf16[N,D]\n"
        "    input r : bf16[N,D]\n"
        "    input s : bf16[N,D]\n"
        "    y = x - r * s\n"
        "    output y\n"
        "}\n",
        encoding="utf-8",
    )

    code = main([str(path), "--emit", mode])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    # 渲染是规范的：中缀进入 AST 后就是一个算子调用。
    if mode == "summary":
        assert "= sub(x, mul(r, s))" in captured.out

    if mode == "dataflow":
        assert "mul(r, s)" in captured.out
        assert "sub(x, %0)" in captured.out

    if mode == "fusion":
        assert "n3 mul -> n4 sub" in captured.out
        assert "fused  %0" in captured.out

    if mode == "buffers":
        assert "fused into k0" in captured.out

    if mode == "triton":
        # One kernel for both operators, and the temporary exists only as a
        # register: no pointer, because 13.7 gave it no buffer.
        # The input-layout contract appends constexpr layout metadata
        # (contiguous flags + shape/stride) for every external input.
        assert "def arith_k0(r_ptr, s_ptr, x_ptr, y_ptr, numel, BLOCK: tl.constexpr" in captured.out
        assert "r_contiguous: tl.constexpr" in captured.out
        assert "x_stride1: tl.constexpr" in captured.out
        assert "%0: fused, 13.7 keeps it in registers" in captured.out
        assert "_0_ptr" not in captured.out

    if mode == "json":
        pipeline = json.loads(captured.out)["pipelines"][0]
        assert pipeline["values"][0]["expr"] == "sub(x, mul(r, s))"

        # 依赖来自 IR，而不是把树再读一遍。`r * s` 先成为一个真正的中间
        # 值，所以依赖是 r -> %0 -> y。旧替身直接从 AST 上取名字，会把
        # 中间值整条跳过，报出并不存在的 r -> y。
        assert {"from": "r", "to": "%0"} in pipeline["edges"]
        assert {"from": "s", "to": "%0"} in pipeline["edges"]
        assert {"from": "%0", "to": "y"} in pipeline["edges"]
        assert {"from": "x", "to": "y"} in pipeline["edges"]
        assert {"from": "r", "to": "y"} not in pipeline["edges"]

        # 13.6 与 13.7 也出现在 JSON 里，按 §17.4 的修复循环可消费的形状。
        assert pipeline["fusion"]["fused_values"] == ["%0"]
        assert pipeline["fusion"]["kernels"][0]["members"] == [3, 4]

        buffers = {
            entry["value"]: entry for entry in pipeline["buffers"]["values"]
        }
        assert buffers["%0"]["storage"] is None
        assert buffers["%0"]["note"] == "fused into k0"
        assert buffers["x"]["storage"] == "input"
        assert buffers["y"]["size"] == "2*N*D"

        # 13.8 的结果也在 JSON 里：每个 kernel 由哪些算子组成、参数是哪些
        # 值，以及不能 lower 的算子与原因（这里没有）。
        backend = pipeline["triton"]
        assert backend["complete"] is True
        assert backend["refusals"] == []
        assert backend["kernels"][0]["ops"] == ["mul", "sub"]
        assert backend["kernels"][0]["results"] == ["y"]
        assert "%0" not in backend["kernels"][0]["operands"]


BROKEN = (
    "pipeline p {\n"
    "    input x : bf16[N,D]\n"
    "    input r : fp16[N,D]\n"
    "    y = add(x, r)\n"
    "    output y\n"
    "}\n"
)


def test_the_prose_diagnostic_names_the_kind_and_the_position(tmp_path, capsys):
    path = tmp_path / "bad.fgpu"
    path.write_text(BROKEN, encoding="utf-8")

    code = main([str(path)])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    # With poisoned symbol recovery, the failed definition of `y` does not
    # cause a spurious undefined-symbol cascade at `output y`.
    assert captured.err == (
        "error: type error in pipeline 'p' at line 4, column 9: "
        "Cannot add bf16 and fp16\n"
    )


def test_json_emit_reports_a_structured_diagnostic(tmp_path, capsys):
    """A caller that asked for JSON gets JSON when compilation fails too.

    Design.md 17.4's repair loop consumes the category and the position to
    decide what to fix and where; recovering them by parsing the English
    back apart is not a diagnostic channel.
    """
    path = tmp_path / "bad.fgpu"
    path.write_text(BROKEN, encoding="utf-8")

    code = main([str(path), "--emit", "json"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    payload = json.loads(captured.err)

    # With cascading errors suppressed, BROKEN has only one genuine error,
    # so it arrives as a single JSON object.
    assert payload == {
        "pipeline": "p",
        "kind": "type",
        "message": "Cannot add bf16 and fp16",
        "line": 4,
        "column": 9,
    }


def test_a_lexical_error_is_also_a_structured_diagnostic(tmp_path, capsys):
    """--emit tokens runs the lexical stage on its own; it reports the same way."""
    path = tmp_path / "lex.fgpu"
    path.write_text("pipeline p { input x : bf16[N,D] @ }\n", encoding="utf-8")

    code = main([str(path), "--emit", "tokens", "--emit", "json"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""

    payload = json.loads(captured.err)
    assert payload["kind"] == "syntax"
    assert payload["line"] == 1
    assert payload["column"] == 34
    assert "at line" not in payload["message"]


def test_a_failed_token_run_prints_no_tokens(tmp_path, capsys):
    """All or nothing: a lexical error partway through used to print the
    tokens it had already produced and then the error, mixing the two."""
    path = tmp_path / "lex.fgpu"
    path.write_text("pipeline p { input x : bf16[N,D] @ }\n", encoding="utf-8")

    main([str(path), "--emit", "tokens"])
    captured = capsys.readouterr()

    assert captured.out == ""
