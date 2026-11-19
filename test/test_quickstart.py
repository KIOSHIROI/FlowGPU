"""The one-command quickstart: compile, reference run, device run guard."""
from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/quickstart/add.fgpu"


def _cli_main():
    spec = importlib.util.spec_from_file_location("flowgpu_cli", ROOT / "compiler/cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_the_quickstart_example_compiles_to_a_complete_module():
    from compiler import compile_source
    from compiler.backend.triton_backend import module

    program = compile_source(EXAMPLE.read_text()).pipelines[0].triton
    assert program.complete
    text = module([program])
    assert "def run_quickstart_add(" in text
    assert "quickstart_add_k0[" in text


def test_the_reference_run_works_without_any_device(capsys):
    assert _cli_main()([str(EXAMPLE), "--run"]) == 0
    captured = capsys.readouterr()
    assert "quickstart_add" in captured.out and "out" in captured.out


def test_emit_summary_is_the_default_product_entry(capsys):
    assert _cli_main()([str(EXAMPLE)]) == 0
    assert "OK  " in capsys.readouterr().out


def test_run_gpu_fails_cleanly_when_the_device_stack_is_absent(capsys):
    import shutil

    code = _cli_main()([str(EXAMPLE), "--run-gpu"])
    captured = capsys.readouterr()
    if code == 0:
        # A real CUDA machine: the comparison must have passed.
        assert "OK  generated Triton" in captured.out
        return
    assert code == 2
    assert "PyTorch" in captured.err or "CUDA device" in captured.err


def test_run_gpu_cannot_be_combined_with_other_modes(capsys):
    code = _cli_main()([str(EXAMPLE), "--run-gpu", "--emit", "triton"])
    assert code == 2
    assert "do not combine" in capsys.readouterr().err


def test_the_cli_reports_the_package_version(capsys):
    import pytest
    pyproject = (ROOT / "pyproject.toml").read_text()
    declared = pyproject.split('version = "')[1].split('"')[0]
    import compiler
    assert compiler.__version__ == declared

    with pytest.raises(SystemExit) as exit_info:
        _cli_main()(["--version"])
    assert exit_info.value.code == 0
    output = capsys.readouterr()
    assert declared in output.out + output.err
