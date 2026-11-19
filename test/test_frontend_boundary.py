"""Phase 0 of de-DSL: core must not depend on the text DSL."""
from __future__ import annotations

import ast
import pathlib

import pytest

import flowgpu.compiler.core
import flowgpu.compiler.core.frontend as frontend_api
import flowgpu.compiler.frontend.ast_node as ast_node
import flowgpu.compiler.frontend.diagnostic as dsl_diagnostic
import flowgpu.compiler.frontend.position as dsl_position
from flowgpu.compiler.frontend.dsl import DslFrontend

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "flowgpu/compiler/core"


def _imports(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_core_never_imports_a_frontend():
    for path in CORE.glob("*.py"):
        imports = _imports(path)
        offending = [name for name in imports if name == "flowgpu.compiler.frontend" or name.startswith("flowgpu.compiler.frontend.")]
        assert not offending, f"{path.name} imports a frontend: {offending}"


def test_core_imports_without_the_dsl_package():
    # Importing core in a fresh interpreter must not load the DSL frontend or
    # Lark.  (Source-level import checks alone would miss flowgpu/compiler/__init__.py,
    # which used to import compile_source eagerly.)
    import subprocess
    import sys

    code = (
        "import sys, flowgpu.compiler.core; "
        "assert 'flowgpu.compiler.frontend.dsl' not in sys.modules, sorted(sys.modules); "
        "assert 'flowgpu.compiler.frontend.parser' not in sys.modules; "
        "assert 'lark' not in sys.modules, 'lark was imported'; "
        "print(flowgpu.compiler.core.TensorType('fp32', []).is_scalar)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"
    imports = set()
    for path in CORE.glob("*.py"):
        imports.update(_imports(path))
    assert "lark" not in imports


def test_programmatic_frontend_does_not_load_the_dsl_parser():
    import subprocess
    import sys

    code = (
        "import sys; "
        "from flowgpu.compiler.frontends.programmatic import ProgrammaticFrontend; "
        "assert 'flowgpu.compiler.frontend.parser' not in sys.modules, sorted(sys.modules); "
        "assert 'lark' not in sys.modules, 'lark was imported'; "
        "print(ProgrammaticFrontend().name)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "programmatic"


def test_dsl_primitives_are_the_core_objects():
    assert flowgpu.compiler.core.CompileError is dsl_diagnostic.CompileError
    assert flowgpu.compiler.core.DiagnosticKind is dsl_diagnostic.DiagnosticKind
    assert flowgpu.compiler.core.RuleError is dsl_diagnostic.RuleError
    assert flowgpu.compiler.core.Position is dsl_position.Position
    assert flowgpu.compiler.core.TensorType is ast_node.TensorType


def test_dsl_frontend_satisfies_the_protocol_and_parses():
    frontend = DslFrontend()
    assert isinstance(frontend, frontend_api.Frontend)
    parsed = frontend.parse("pipeline p { input x:fp32[N] y = relu(x) output y }")
    assert parsed[0].name == "p"


def test_a_non_dsl_frontend_can_register_without_touching_the_dsl():
    class FakeFrontend:
        name = "fake"
        description = "a frontend that does not parse FlowGPU text"
        source_extension = ".fake"

        def parse(self, source):
            return {"source": source}

    frontend_api.clear_frontends()
    try:
        info = frontend_api.register_frontend(FakeFrontend())
        assert info.name == "fake"
        assert frontend_api.get_frontend("fake").parse("hello") == {"source": "hello"}
        assert frontend_api.available_frontends() == ("fake",)
        assert isinstance(FakeFrontend(), frontend_api.Frontend)
        with pytest.raises(KeyError):
            frontend_api.get_frontend("missing")
    finally:
        frontend_api.clear_frontends()


def test_ir_does_not_import_the_lexer_or_parser():
    # Phase 0 boundary: ir may still consume DSL AST node classes (Phase 1 moves
    # those), but it must not own syntax: no lexer/parser/ast_builder imports.
    forbidden = ("flowgpu.compiler.frontend.lexer", "flowgpu.compiler.frontend.parser",
                 "flowgpu.compiler.frontend.ast_builder")
    for path in (ROOT / "flowgpu/compiler/ir").glob("*.py"):
        imports = _imports(path)
        offending = [name for name in imports if name.startswith(forbidden)]
        assert not offending, f"{path.name} imports syntax: {offending}"


def test_programmatic_frontend_builds_the_core_graph_without_the_dsl_parser():
    from flowgpu.compiler.frontends.programmatic import ProgrammaticFrontend

    spec = {
        "pipeline": "programmatic_p",
        "inputs": {"x": "fp32[N]", "y": "fp32[N]"},
        "nodes": [
            {"op": "add", "inputs": ["x", "y"], "output": "z",
             "output_type": "fp32[N]"},
            {"op": "relu", "inputs": ["z"], "output": "w",
             "output_type": "fp32[N]"},
        ],
        "outputs": ["w"],
    }
    graph = ProgrammaticFrontend().parse(spec)
    assert graph.pipeline == "programmatic_p"
    assert [node.op for node in graph.nodes] == ["input", "input", "add", "relu", "output"]
    assert graph.values["w"] == ast_node.TensorType("fp32", ["N"])
    # graph-level passes consume it without any DSL source text
    from flowgpu.compiler.optimization.fusion import fuse
    report = fuse(graph)
    assert len(report.groups) == 1 and report.groups[0].fused


def test_programmatic_frontend_accepts_json_and_rejects_bad_specs():
    import json as _json

    from flowgpu.compiler.frontends.programmatic import ProgrammaticFrontend

    frontend = ProgrammaticFrontend()
    graph = frontend.parse(_json.dumps({
        "pipeline": "j", "inputs": {"x": "fp32[2]"},
        "nodes": [{"op": "relu", "inputs": ["x"], "output": "y",
                   "output_type": "fp32[2]"}],
        "outputs": ["y"],
    }))
    assert isinstance(graph, flowgpu.compiler.core.ir.DataflowGraph)
    with pytest.raises(ValueError):
        frontend.parse({"pipeline": "bad", "inputs": {"x": "fp32[2]"},
                        "nodes": [{"op": "relu", "inputs": ["nope"],
                                   "output": "y", "output_type": "fp32[2]"}],
                        "outputs": ["y"]})
    with pytest.raises(ValueError):
        frontend.parse({"pipeline": "bad", "inputs": {"x": "fp32[2]"},
                        "nodes": [{"op": "relu", "inputs": ["x"], "output": "y"}],
                        "outputs": ["y"]})


def test_programmatic_frontend_source_does_not_import_syntax():
    from flowgpu.compiler.frontends import programmatic
    source = pathlib.Path(programmatic.__file__).read_text()
    for forbidden in ("flowgpu.compiler.frontend.lexer", "flowgpu.compiler.frontend.parser",
                      "flowgpu.compiler.frontend.ast_builder", "lark"):
        assert forbidden not in source


def test_ir_optimization_and_analysis_are_frontend_free():
    for relative in ("flowgpu/compiler/ir", "flowgpu/compiler/optimization", "flowgpu/compiler/analysis"):
        for path in (ROOT / relative).glob("*.py"):
            imports = _imports(path)
            offending = [name for name in imports
                         if name == "flowgpu.compiler.frontend" or name.startswith("flowgpu.compiler.frontend.")]
            assert not offending, f"{relative}/{path.name} imports a frontend: {offending}"


def test_core_ast_builder_does_not_load_the_dsl_or_lark():
    import subprocess
    import sys

    code = (
        "import sys; "
        "from flowgpu.compiler.core.ast import PipelineDecl, InputDecl, Assignment, Call, Name, OutputDecl, Symbol; "
        "from flowgpu.compiler.core.symbol_table import SymbolTable; "
        "from flowgpu.compiler.core.types import TensorType; "
        "from flowgpu.compiler.ir.dataflow_builder import DataflowBuilder; "
        "assert 'flowgpu.compiler.frontend.dsl' not in sys.modules, sorted(sys.modules); "
        "assert 'flowgpu.compiler.frontend.parser' not in sys.modules; "
        "assert 'lark' not in sys.modules, 'lark was imported'; "
        "st = SymbolTable(); "
        "st.define(Symbol('x', 'input', TensorType('fp32', ['N']))); "
        "st.define(Symbol('y', 'value', TensorType('fp32', ['N']))); "
        "p = PipelineDecl('p', ["
        "InputDecl('x', TensorType('fp32', ['N'])), "
        "Assignment(['y'], Call('relu', [Name('x')])), "
        "OutputDecl('y')]); "
        "g = DataflowBuilder().build(p, st); "
        "print([node.op for node in g.nodes])"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "['input', 'relu', 'output']" in completed.stdout


def test_dataflow_builder_consumes_core_ast_directly():
    from flowgpu.compiler.core.ast import Assignment, Call, InputDecl, Name, OutputDecl, PipelineDecl, Symbol
    from flowgpu.compiler.core.symbol_table import SymbolTable
    from flowgpu.compiler.core.types import TensorType
    from flowgpu.compiler.ir.dataflow_builder import DataflowBuilder

    symbols = SymbolTable()
    symbols.define(Symbol("x", "input", TensorType("fp32", ["N"])))
    symbols.define(Symbol("y", "value", TensorType("fp32", ["N"])))
    pipeline = PipelineDecl("direct", [
        InputDecl("x", TensorType("fp32", ["N"])),
        Assignment(["y"], Call("relu", [Name("x")])),
        OutputDecl("y"),
    ])
    graph = DataflowBuilder().build(pipeline, symbols)
    assert [node.op for node in graph.nodes] == ["input", "relu", "output"]
    assert graph.values["y"] == TensorType("fp32", ["N"])


def test_backend_does_not_import_the_frontend():
    for path in (ROOT / "flowgpu/compiler/backend").glob("*.py"):
        imports = _imports(path)
        offending = [name for name in imports
                     if name == "flowgpu.compiler.frontend" or name.startswith("flowgpu.compiler.frontend.")]
        assert not offending, f"backend/{path.name} imports a frontend: {offending}"


def test_compile_module_has_no_top_level_frontend_import():
    # parser/ASTBuilder are imported lazily inside the parse branch; an AST or
    # eDSL caller must not need them at module import time.
    tree = ast.parse((ROOT / "flowgpu/compiler/compile.py").read_text())
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    offending = [name for name in imports
                 if name == "flowgpu.compiler.frontend" or name.startswith("flowgpu.compiler.frontend.")]
    assert not offending, f"compile.py imports a frontend at module level: {offending}"
    import subprocess
    import sys
    code = ("import sys, flowgpu.compiler.compile; "
            "assert 'lark' not in sys.modules, 'lark was imported'; "
            "print('ok')")
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
