"""The audit report is derived from compiled kernels, guards and reference."""
from flowgpu.compiler.audit import build_audit_report, render_markdown


def test_audit_report_exposes_kernels_fusion_guards_and_reference():
    source = """
    pipeline p {
      input x: bf16[4,64]
      input w1: bf16[64,128]
      input w2: bf16[128,64]
      h = matmul(x, w1)
      h2 = silu(h)
      y = matmul(h2, w2)
      output y
    }
    """
    report = build_audit_report(source)
    assert report["kernel_count"] == 2
    assert report["reference_count"] == 2
    assert any(group["anchor"] == "matmul" and "silu" in group["members"]
               for group in report["fusion_groups"])
    assert report["host_guards"]
    text = render_markdown(report)
    assert "# Audit: p" in text and "## Host guards" in text
