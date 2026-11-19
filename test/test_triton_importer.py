"""Triton importer: pointwise/reduction subset into the core AST."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from compiler.compile import compile_pipelines
from compiler.frontends.triton_importer import import_triton

ROOT = pathlib.Path(__file__).resolve().parents[1]

ADD = '''
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)
'''

ROW_SUM = '''
@triton.jit
def row_sum(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + offsets, s)
'''

SCALE = '''
@triton.jit
def scale_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = x * 2.0
    tl.store(out_ptr + offsets, y)
'''


def test_imports_pointwise_and_matches_the_dsl_graph():
    imported = import_triton(ADD, input_types={"x_ptr": "fp32[N]", "y_ptr": "fp32[N]"},
                             output_types={"out_ptr": "fp32[N]"})
    result = compile_pipelines(imported.pipelines)
    pipeline = result.pipelines[0]
    assert pipeline.triton.complete
    assert [node.op for node in pipeline.dataflow.nodes] == ["input", "input", "add", "output"]
    # same logical graph as the DSL expression
    from compiler import compile_source
    dsl = compile_source(
        "pipeline p { input x:fp32[N] input y:fp32[N] z=add(x,y) output z }"
    ).pipelines[0]
    assert [node.op for node in dsl.dataflow.nodes] == [node.op for node in pipeline.dataflow.nodes]


def test_imports_reduction():
    imported = import_triton(ROW_SUM, input_types={"x_ptr": "fp32[N]"},
                             output_types={"out_ptr": "fp32[1]"})
    result = compile_pipelines(imported.pipelines)
    assert result.pipelines[0].triton.complete
    assert [node.op for node in result.pipelines[0].dataflow.nodes] == ["input", "sum_last", "output"]


def test_numeric_constants_become_explicit_scalar_inputs():
    imported = import_triton(SCALE, input_types={"x_ptr": "fp32[N]"},
                             output_types={"out_ptr": "fp32[N]"})
    assert imported.constants == {"c0": 2.0}
    result = compile_pipelines(imported.pipelines)
    assert result.pipelines[0].triton.complete
    assert result.pipelines[0].dataflow.nodes[-2].op == "mul"
    assert "c0" in [statement.name for statement in imported.pipelines[0].statements
                    if hasattr(statement, "name")]


def test_importer_rejects_out_of_subset_and_bad_metadata():
    with pytest.raises(ValueError):
        import_triton("def not_a_kernel():\n    pass\n",
                      input_types={}, output_types={"out": "fp32[N]"})
    with pytest.raises(ValueError):
        import_triton(ADD, input_types={"wrong": "fp32[N]"}, output_types={"out_ptr": "fp32[N]"})
    with pytest.raises(ValueError):
        # control flow is outside the subset
        import_triton('''
@triton.jit
def k(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    if x.sum() > 0:
        tl.store(out_ptr + offsets, x)
''', input_types={"x_ptr": "fp32[N]"}, output_types={"out_ptr": "fp32[N]"})
    with pytest.raises(ValueError):
        # output never stored
        import_triton('''
@triton.jit
def k(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
''', input_types={"x_ptr": "fp32[N]"}, output_types={"out_ptr": "fp32[N]"})


def test_importer_does_not_load_the_dsl_or_lark():
    code = (
        "import sys; "
        "from compiler.frontends.triton_importer import import_triton; "
        "from compiler.compile import compile_pipelines; "
        "src = '''\n"
        "@triton.jit\n"
        "def k(x_ptr, out_ptr, BLOCK: tl.constexpr):\n"
        "    offsets = tl.arange(0, BLOCK)\n"
        "    x = tl.load(x_ptr + offsets)\n"
        "    y = x + x\n"
        "    tl.store(out_ptr + offsets, y)\n"
        "'''; "
        "k = import_triton(src, input_types={'x_ptr': 'fp32[N]'}, output_types={'out_ptr': 'fp32[N]'}); "
        "r = compile_pipelines(k.pipelines); "
        "assert r.pipelines[0].triton.complete; "
        "assert 'compiler.frontend.parser' not in sys.modules, sorted(sys.modules); "
        "assert 'lark' not in sys.modules, 'lark was imported'; "
        "print('ok')"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


WHERE = '''
@triton.autotune(configs=[], key=["n_elements"])
@triton.jit
def where_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0.0, x, 0.0 - x)
    tl.store(out_ptr + offsets, y, mask=mask)
'''

MULTI = '''
@triton.jit
def two_out(x_ptr, a_ptr, b_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = x * 2.0
    z = x + 1.0
    tl.store(a_ptr + offsets, y)
    tl.store(b_ptr + offsets, z)
'''


def test_imports_tl_where_and_comparisons_under_autotune():
    imported = import_triton(WHERE, input_types={"x_ptr": "fp32[1024]"},
                             output_types={"out_ptr": "fp32[1024]"})
    result = compile_pipelines(imported.pipelines)
    ops = [node.op for node in result.pipelines[0].dataflow.nodes]
    assert result.pipelines[0].triton.complete
    assert "gt" in ops and "where" in ops
    # the two 0.0 literals are explicitly supplied scalar inputs
    assert imported.constants == {"c0": 0.0, "c1": 0.0}


def test_imports_multiple_stores_and_constants():
    imported = import_triton(MULTI, input_types={"x_ptr": "fp32[1024]"},
                             output_types={"a_ptr": "fp32[1024]", "b_ptr": "fp32[1024]"})
    result = compile_pipelines(imported.pipelines)
    assert result.pipelines[0].triton.complete
    assert [node.op for node in result.pipelines[0].dataflow.nodes][-2:] == ["output", "output"]
    assert imported.constants == {"c0": 2.0, "c1": 1.0}
