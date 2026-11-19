"""Course-demo frontend inspector: token stream, AST JSON and DOT output."""
from __future__ import annotations

import json
import pathlib

from tools import dsl_inspect

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/mapping_pointwise.fgpu"


def test_tokens(capsys):
    assert dsl_inspect.main([str(EXAMPLE), "--tokens"]) == 0
    out = capsys.readouterr().out
    assert "PIPELINE" in out and "mapping_pointwise" in out


def test_ast_json(capsys):
    assert dsl_inspect.main([str(EXAMPLE), "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload[0]["name"] == "mapping_pointwise"
    assert payload[0]["statements"][0]["type"]["dtype"] == "int32"


def test_ast_dot(tmp_path):
    dot_path = tmp_path / "ast.dot"
    assert dsl_inspect.main([str(EXAMPLE), "--dot", str(dot_path)]) == 0
    dot = dot_path.read_text()
    assert "digraph flowgpu_ast" in dot
    assert "PipelineDecl" in dot and "InputDecl" in dot and "Assignment" in dot


def test_ast_repr(capsys):
    assert dsl_inspect.main([str(EXAMPLE), "--ast"]) == 0
    out = capsys.readouterr().out
    assert "PipelineDecl" in out and "Assignment" in out
