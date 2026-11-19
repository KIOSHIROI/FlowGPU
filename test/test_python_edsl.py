"""Python eDSL frontend: builds core AST and compiles without the DSL parser."""
from __future__ import annotations

import subprocess
import sys
import pathlib

import pytest

from compiler.frontends.python_edsl import T, program
from compiler.compile import compile_pipelines

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _edsl_pipeline():
    with program("p") as p:
        x = p.input("x", "fp32[N]")
        y = p.relu(x)
        z = y + x
        p.output(z)
    return p.pipelines


def test_edsl_builds_the_same_core_ast_and_compiles():
    pipelines = _edsl_pipeline()
    assert isinstance(pipelines, tuple) and pipelines[0].name == "p"
    result = compile_pipelines(pipelines)
    pipeline = result.pipelines[0]
    assert pipeline.triton.complete
    assert [node.op for node in pipeline.dataflow.nodes] == ["input", "relu", "add", "output"]
    # differential check: the DSL lowers the same graph for the same program
    from compiler import compile_source
    dsl = compile_source(
        "pipeline p { input x:fp32[N] t1=relu(x) t2=add(t1,x) output t2 }"
    ).pipelines[0]
    assert [node.op for node in dsl.dataflow.nodes] == [node.op for node in pipeline.dataflow.nodes]
    assert dsl.triton.complete == pipeline.triton.complete


def test_edsl_supports_operator_overloads_and_literal_helpers():
    with program("p") as p:
        x = p.input("x", "fp32[N]")
        c = p.input("c", "fp32")
        y = p.relu(x)
        z = y * c
        w = z - x
        p.output(w)
    result = compile_pipelines(p.pipelines)
    assert result.pipelines[0].triton.complete


def test_edsl_supports_scalar_inputs():
    with program("scalar") as p:
        x = p.input("x", "fp32[N]")
        scale = p.input("scale", "fp32")
        y = x * scale
        p.output(y)
    result = compile_pipelines(p.pipelines)
    assert result.pipelines[0].triton.complete
    assert "scale" in result.pipelines[0].triton.kernels[0].source


def test_edsl_rejects_unknown_operator_and_bad_output():
    with program("bad") as p:
        x = p.input("x", "fp32[N]")
        with pytest.raises(ValueError):
            p.call("definitely_not_an_operator", x)
        with pytest.raises(ValueError):
            p.output("missing")
    with program("no_output") as p:
        p.input("x", "fp32[N]")
    with pytest.raises(ValueError):
        p.pipelines


def test_edsl_import_and_compile_do_not_load_the_dsl_parser():
    code = (
        "import sys; "
        "from compiler.frontends.python_edsl import program; "
        "from compiler.compile import compile_pipelines; "
        "assert 'compiler.frontend.parser' not in sys.modules, sorted(sys.modules); "
        "assert 'lark' not in sys.modules, 'lark was imported'; "
        "p = program('p'); "
        "x = p.input('x', 'fp32[N]'); "
        "p.output(p.relu(x)); "
        "r = compile_pipelines(p.pipelines); "
        "assert r.pipelines[0].triton.complete; "
        "assert 'lark' not in sys.modules, 'compile loaded lark'; "
        "print('ok')"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_type_helper():
    assert T("fp32[N]").shape == ["N"]
    assert T("fp32").is_scalar
    assert str(T("fp32[N,D]")) == "fp32[N,D]"
