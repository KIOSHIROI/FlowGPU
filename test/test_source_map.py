"""Kernel-to-DSL line mapping covers split's two outputs."""
from pathlib import Path

from flowgpu.compiler.source_map import build_source_map, render_markdown


def test_split_outputs_map_to_the_same_line():
    path = Path("examples/batch10/swiglu_oai_split.fgpu")
    source = path.read_text()
    report = build_source_map(source)
    kernel = report["kernels"][0]
    line = kernel["values"]["gate"]
    assert kernel["values"]["up"] == line
    assert line in kernel["source_lines"]
    text = render_markdown(report, source)
    assert "gate, up = split(x)" in text
