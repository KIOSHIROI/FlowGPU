from pathlib import Path


def test_layer_packages_and_canonical_grammar_exist():
    root = Path(__file__).resolve().parents[1] / "flowgpu" / "compiler"
    for layer in ("frontend", "ir", "analysis", "optimization", "backend", "runtime"):
        assert (root / layer / "__init__.py").exists()
    assert (root / "grammar.lark").exists()
    assert not (root / "frontend" / "grammar.lark").exists()


def test_legacy_imports_resolve_to_layer_implementations():
    from flowgpu.compiler.ast_node import TensorType as OldTensorType
    from flowgpu.compiler.frontend.ast_node import TensorType as NewTensorType
    from flowgpu.compiler.triton_backend import TritonProgram as OldProgram
    from flowgpu.compiler.backend.triton_backend import TritonProgram as NewProgram
    assert OldTensorType is NewTensorType
    assert OldProgram is NewProgram
