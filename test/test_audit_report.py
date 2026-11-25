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


def test_audit_from_result_does_not_require_source_again():
    from flowgpu.compiler.compile import compile_source
    from flowgpu.compiler.audit import audit_from_result
    source = """
    pipeline p {
      input x: bf16[2,4]
      y = relu(x)
      output y
    }
    """
    result = compile_source(source)
    report = audit_from_result(result)
    assert report["kernel_count"] == 1


def test_audit_cli_accepts_shape_bindings(tmp_path, capsys):
    from flowgpu.compiler.audit import main
    path = tmp_path / "p.fgpu"
    path.write_text("""
    pipeline p {
      input x: bf16[T,D]
      y = relu(x)
      output y
    }
    """)
    assert main([str(path), "--shape", "T=2", "--shape", "D=4"]) == 0
    assert "# Audit: p" in capsys.readouterr().out
