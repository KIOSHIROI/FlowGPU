"""Effect-aware graph passes: CSE, DCE, identity folding and planning."""
from dataclasses import replace

import pytest

from flowgpu.compiler.analysis.planning import summarize_plan
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.ir import DataflowGraph, Node
from flowgpu.compiler.core.types import TensorType
from flowgpu.compiler.ir.effects import Purity, effect_of, verify_effects
from flowgpu.compiler.ir.verification import verify_graph
from flowgpu.compiler.ir.semantic_ir import OperationAttributes
from flowgpu.compiler.optimization.cse import common_subexpression_elimination
from flowgpu.compiler.optimization.dce import eliminate_dead_code
from flowgpu.compiler.optimization.identity import fold_identities
from flowgpu.compiler.optimization.optimizer import optimize_graph


def _graph(*nodes, values=None, aliases=None):
    value_table = {
        "x": TensorType("fp32", ["4"]),
        "y": TensorType("fp32", ["4"]),
    }
    value_table.update(values or {})
    return DataflowGraph("p", value_table, list(nodes), aliases or {})


def test_effects_classify_structural_pure_and_stateful_nodes():
    graph = _graph(
        Node("input", [], "x"),
        Node("input", [], "y"),
        Node("add", ["x", "y"], "a"),
        Node("store_cache_4d", ["x"], "cache"),
        Node("output", ["a"], None),
        values={"a": TensorType("fp32", ["4"]),
                "cache": TensorType("fp32", ["4"])},
    )
    effects = [effect_of(node) for node in graph.nodes]
    assert [effect.purity for effect in effects] == [
        Purity.STRUCTURAL, Purity.STRUCTURAL, Purity.PURE,
        Purity.STATEFUL, Purity.STRUCTURAL]
    assert effects[3].mutates_inputs is True
    verify_effects(graph)
    verify_graph(graph)


def test_cse_merges_only_pure_duplicates():
    graph = _graph(
        Node("input", [], "x"),
        Node("input", [], "y"),
        Node("add", ["x", "y"], "a"),
        Node("add", ["x", "y"], "b"),
        Node("store_cache_4d", ["x"], "c1"),
        Node("store_cache_4d", ["x"], "c2"),
        Node("output", ["b"], None),
        values={name: TensorType("fp32", ["4"])
                for name in ("a", "b", "c1", "c2")},
    )
    optimized, decisions = common_subexpression_elimination(graph)
    verify_graph(optimized)
    assert [node.op for node in optimized.nodes] == [
        "input", "input", "add", "store_cache_4d", "store_cache_4d", "output"]
    assert optimized.resolve("b") == "a"
    assert len(decisions) == 1


def test_dce_removes_dead_pure_nodes_but_keeps_stateful_writes():
    graph = _graph(
        Node("input", [], "x"),
        Node("input", [], "y"),
        Node("add", ["x", "y"], "dead_add"),
        Node("mul", ["x", "y"], "live"),
        Node("store_cache_4d", ["x"], "stateful_dead_result"),
        Node("output", ["live"], None),
        values={"dead_add": TensorType("fp32", ["4"]),
                "live": TensorType("fp32", ["4"]),
                "stateful_dead_result": TensorType("fp32", ["4"])},
    )
    optimized, decisions = eliminate_dead_code(graph)
    verify_graph(optimized)
    assert "dead_add" not in {node.output for node in optimized.nodes}
    assert "stateful_dead_result" in {node.output for node in optimized.nodes}
    assert optimized.producer_of("live").op == "mul"
    assert any("dead_add" in decision.reason for decision in decisions)


def test_identity_fold_keeps_an_exported_storage_boundary():
    graph = _graph(
        Node("input", [], "x"),
        Node("input", [], "y"),
        Node("reshape", ["x"], "reshaped",
             attributes=OperationAttributes(target_shape=("4",))),
        Node("output", ["reshaped"], None),
        values={"reshaped": TensorType("fp32", ["4"])},
    )
    optimized, decisions = fold_identities(graph)
    assert any(node.op == "reshape" for node in optimized.nodes)
    assert decisions == ()


def test_optimizer_records_cse_and_dce():
    result = compile_source(
        "pipeline p { input x:fp16[64] input y:fp16[64] "
        "a = add(x, y) b = add(x, y) out = add(a, b) output out }")
    pipeline = result.pipelines[0]
    assert [node.op for node in pipeline.dataflow.nodes] == [
        "input", "input", "add", "add", "output"]
    names = [record.name for record in pipeline.optimization]
    assert "cse" in names and "dce" in names
    assert pipeline.planning is not None
    assert pipeline.planning.launches == len(pipeline.fusion.groups)


def test_optimize_false_keeps_the_conservative_graph():
    source = ("pipeline p { input x:fp16[64] input y:fp16[64] "
              "a = add(x, y) b = add(x, y) out = add(a, b) output out }")
    optimized = compile_source(source).pipelines[0]
    conservative = compile_source(source, optimize=False).pipelines[0]
    assert len(conservative.dataflow.nodes) > len(optimized.dataflow.nodes)
    assert conservative.optimization == ()


def test_planning_summary_prices_the_chosen_plan():
    pipeline = compile_source(
        "pipeline p { input x:fp16[64] input y:fp16[64] out = add(x, y) output out }").pipelines[0]
    summary = summarize_plan(pipeline.dataflow, pipeline.fusion, pipeline.memory)
    assert summary.launches == len(pipeline.fusion.groups)
    assert summary.slots == len(pipeline.memory.slots)
    assert summary.materialized_buffers == len(pipeline.memory.materialized)
    assert summary.traffic_a >= 1.0


def test_cli_planning_and_optimizer_controls(tmp_path):
    import json
    import subprocess
    import sys

    source = tmp_path / "program.fgpu"
    source.write_text(
        "pipeline p { input x:fp16[64] input y:fp16[64] "
        "a = add(x, y) b = add(x, y) out = add(a, b) output out }\n")
    command = [sys.executable, "-m", "flowgpu", str(source)]
    result = subprocess.run(command + ["--emit", "planning"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cost"]["launches"] >= 1
    assert any(record["name"] == "cse" for record in payload["optimization"])

    conservative = subprocess.run(
        command + ["--emit", "json", "--no-optimize"],
        capture_output=True, text=True, timeout=120)
    assert conservative.returncode == 0, conservative.stderr
    dataflow = json.loads(conservative.stdout)["pipelines"][0]["dataflow"]
    assert len(dataflow["nodes"]) == 6


def test_structural_ablation_reports_optimizer_effects(tmp_path):
    import json
    import subprocess
    import sys

    source = tmp_path / "program.fgpu"
    source.write_text(
        "pipeline p { input x:fp16[64] input y:fp16[64] "
        "a = add(x, y) b = add(x, y) out = add(a, b) output out }\n")
    result = subprocess.run(
        [sys.executable, "tools/optimization_ablation.py", str(source), "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=__import__("pathlib").Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["default"]["plan"]["kernels"] >= 1
    assert report["no_graph_pipeline"]["optimization"] == []
    assert report["default"]["optimization"]
