"""A frontend that builds the Dataflow IR from a declarative Python/JSON spec.

Phase 1a's forcing function: the core graph must be constructible without the
DSL lexer, parser, AST builder or type inference.  The spec is deliberately
small and explicit (input types, node ops, output types, outputs); type
inference for this frontend is future work, not hidden DSL reuse.

Example::

    {
      "pipeline": "p",
      "inputs": {"x": "fp32[N]", "y": "fp32[N]"},
      "nodes": [
        {"op": "add", "inputs": ["x", "y"], "output": "z",
         "output_type": "fp32[N]"},
        {"op": "relu", "inputs": ["z"], "output": "w",
         "output_type": "fp32[N]"}
      ],
      "outputs": ["w"]
    }

The builder registers every non-structural op name with the core vocabulary
and validates the graph through ``DataflowGraph.validate``.  Compiling this
graph through the backend pipeline is Phase 2; Phase 1a proves the IR boundary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from flowgpu.compiler.core.ir import INPUT, OUTPUT, DataflowGraph, Node
from flowgpu.compiler.core.operators import register_operator_name
from flowgpu.compiler.core.types import TensorType


def parse_type(text: str) -> TensorType:
    """Parse ``fp32`` / ``fp32[N]`` / ``fp32[N,D]`` without the DSL parser."""
    text = str(text).strip()
    if "[" not in text:
        return TensorType(dtype=text, shape=[])
    dtype, rest = text.split("[", 1)
    rest = rest.rstrip("]").strip()
    shape = [part.strip() for part in rest.split(",") if part.strip()]
    return TensorType(dtype=dtype.strip(), shape=shape)


def _spec_dict(source) -> dict:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, str):
        return json.loads(source)
    raise TypeError("programmatic frontend expects a mapping or a JSON string")


@dataclass(frozen=True)
class ProgrammaticFrontend:
    name: str = "programmatic"
    description: str = "builds the Dataflow IR directly from a Python/JSON spec"
    source_extension: str = ".json"

    def parse(self, source) -> DataflowGraph:
        spec = _spec_dict(source)
        pipeline = spec.get("pipeline", "programmatic")
        inputs = spec.get("inputs", {})
        values: dict[str, TensorType] = {}
        nodes: list[Node] = []
        for name, type_text in inputs.items():
            values[name] = parse_type(type_text)
            nodes.append(Node(op=INPUT, inputs=[], output=name))
        for index, raw in enumerate(spec.get("nodes", [])):
            if not isinstance(raw, Mapping):
                raise TypeError(f"node {index} must be a mapping")
            op = raw.get("op")
            if not op or op in (INPUT, OUTPUT):
                raise ValueError(f"node {index}: an explicit input/output op is not allowed here")
            out = raw.get("output")
            if not out:
                raise ValueError(f"node {index}: missing output name")
            if out in values:
                raise ValueError(f"node {index}: output {out!r} is already defined")
            type_text = raw.get("output_type")
            if not type_text:
                raise ValueError(f"node {index}: output_type is required (this frontend does no inference)")
            literals = list(raw.get("inputs", []))
            for operand in literals:
                if operand not in values:
                    raise ValueError(f"node {index}: unknown operand {operand!r}")
            values[out] = parse_type(type_text)
            register_operator_name(op)
            nodes.append(Node(op=op, inputs=literals, output=out))
        outputs = list(spec.get("outputs", []))
        if not outputs:
            raise ValueError("spec needs at least one output")
        for name in outputs:
            if name not in values:
                raise ValueError(f"unknown output {name!r}")
            nodes.append(Node(op=OUTPUT, inputs=[name], output=None))
        return DataflowGraph(pipeline=pipeline, values=values, nodes=nodes, aliases={})
