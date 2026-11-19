from pathlib import Path

from compiler import compile_source

ROOT = Path(__file__).resolve().parents[1]
# Most examples use the default Triton backend; the SDPA example explicitly
# documents that it needs hybrid library math for attention.
OPTIONS = {"sdpa_decoder.fgpu": {"gpu_backend": "hybrid"}}


def test_every_public_example_compiles():
    failures = []
    examples = sorted((ROOT / "examples").rglob("*.fgpu"))
    assert examples, "no examples found"
    for path in examples:
        if "errors" in path.parts:
            continue
        try:
            options = OPTIONS.get(path.name, {})
            compiled = compile_source(path.read_text(), source_path=str(path), **options)
        except Exception as error:  # noqa: BLE001 - report the example path
            failures.append(f"{path.relative_to(ROOT)}: {type(error).__name__}: {error}")
            continue
        for pipeline in compiled.pipelines:
            assert pipeline.triton.complete, (path, pipeline.triton.messages())
    assert not failures, "\n".join(failures)
