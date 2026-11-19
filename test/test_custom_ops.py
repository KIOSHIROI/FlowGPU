"""The torch.library section: what it registers, and what it must not.

This machine has no torch, so these tests cannot import the generated module
and call the op. What they CAN hold is the contract the section must satisfy
to be registrable at all -- the emitted text is valid Python, the call into
the host entry uses the host's own parameter names, the fake returns the
declared output shapes, and nothing is registered for a program the backend
refused to lower. The run-time half (does torch accept the registration and
does the fake trace) is verified on a machine with torch, and the evidence
belongs with that run.
"""

import ast
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.triton_backend import module


POINTWISE = """
pipeline chain {
    input x : bf16[N,D]
    y = relu(x)
    output y
}
"""

TWO_OUTPUTS = """
pipeline twice {
    input x : bf16[N,D]
    y = relu(x)
    z = silu(x)
    output y
    output z
}
"""

ALIASING = """
pipeline alias {
    input x : bf16[N,D]
    y = x
    output y
}
"""

NAME_COLLISION = """
pipeline collision {
    input x : bf16[x,D]
    y = relu(x)
    output y
}
"""

REFUSED = """
pipeline refused {
    input q : bf16[L,D]
    input k : bf16[S,D]
    input v : bf16[S,V]
    y = attention(q, k, v)
    output y
}
"""

MATMUL = """
pipeline gemm {
    input a : bf16[M,K]
    input b : bf16[K,N]
    c = matmul(a, b)
    d = relu(c)
    output d
}
"""


def emitted(source, **kwargs):
    result = compile_source(source, **kwargs)
    return module([pipeline.triton for pipeline in result.pipelines])


def section(source, **kwargs):
    return compile_source(source, custom_ops=True, **kwargs).pipelines[0].triton.custom_ops


# ---- It is absent unless asked for ----

def test_default_emission_has_no_registration():
    """The section is opt-in, so every existing golden stays byte-identical."""
    assert compile_source(POINTWISE).pipelines[0].triton.custom_ops == ""
    assert "custom_op" not in emitted(POINTWISE)


def test_flag_puts_the_registration_in_the_module():
    text = emitted(POINTWISE, custom_ops=True)
    assert 'torch.library.custom_op(' in text
    assert '"flowgpu::chain"' in text


# ---- What it registers ----

def test_registration_names_the_pipeline_and_calls_the_host_entry():
    body = section(POINTWISE)
    assert '"flowgpu::chain"' in body
    assert 'return run_chain(x)["y"]' in body


def test_the_fake_returns_the_declared_output_shape_and_dtype():
    body = section(POINTWISE)
    assert "N = x.shape[0]" in body
    assert "D = x.shape[1]" in body
    assert "x.new_empty((N, D), dtype=torch.bfloat16)" in body


def test_literal_dimensions_need_no_shape_reads():
    """A fully literal program binds no symbols, so the fake reads no shapes."""
    source = """
    pipeline fixed {
        input x : bf16[4,8]
        y = relu(x)
        output y
    }
    """
    body = section(source)
    assert "x.shape[" not in body
    assert "x.new_empty((4, 8), dtype=torch.bfloat16)" in body


def test_two_outputs_are_returned_as_a_tuple():
    body = section(TWO_OUTPUTS)
    assert "-> tuple[torch.Tensor, torch.Tensor]:" in body
    assert '_flowgpu_result = run_twice(x)' in body
    assert 'return (_flowgpu_result["y"], _flowgpu_result["z"])' in body
    assert "x.new_empty((N, D), dtype=torch.bfloat16), x.new_empty((N, D), dtype=torch.bfloat16)" in body


@pytest.mark.parametrize("count", [2, 3])
def test_multi_output_schema_declares_each_tensor_without_ellipsis(count):
    from typing import get_args
    from flowgpu.compiler.custom_ops import returns_annotation
    tensor = object()
    annotation = eval(returns_annotation(count), {"torch": SimpleNamespace(Tensor=tensor)})
    assert get_args(annotation) == (tensor,) * count


@pytest.mark.parametrize("count", [2, 3])
def test_multiple_outputs_execute_one_host_call_and_share_its_results(count):
    source = TWO_OUTPUTS.replace("output z", "output z\noutput x") if count == 3 else TWO_OUTPUTS
    implementation = next(node for node in ast.parse(section(source)).body
                          if isinstance(node, ast.FunctionDef) and node.name.endswith("_impl"))
    calls = []

    def host(value):
        calls.append(value)
        # Distinct invocations return distinct objects, exposing repeated execution.
        return {name: object() for name in ("x", "y", "z")}

    namespace = {"torch": SimpleNamespace(Tensor=object), "run_twice": host}
    exec(compile(ast.Module(body=[implementation], type_ignores=[]), "<custom-op>", "exec"), namespace)
    argument = object()
    result = namespace[implementation.name](argument)
    assert calls == [argument]
    assert isinstance(result, tuple) and len(result) == count


def test_an_aliasing_output_reports_the_input_shape():
    """`y = x` emits no node, so the output's shape is the input's."""
    body = section(ALIASING)
    assert "x.new_empty((N, D), dtype=torch.bfloat16)" in body


def test_the_parameter_keeps_the_name_the_host_uses():
    """The host renames an input that collides with a dimension symbol, and
    the registration has to call it by that name -- otherwise the generated
    module raises NameError the first time the op is called."""
    text = emitted(NAME_COLLISION, custom_ops=True)
    # The host parameter is renamed; both the host and the op must agree.
    assert "def run_collision(" in text
    host_params = text.split("def run_collision(")[1].split(")")[0]
    op_call = "return run_collision(" + host_params.split(",")[0].strip() + ")" in text
    assert op_call, host_params


def test_mutates_args_is_empty_by_construction():
    """Every slot is a fresh torch.empty, so a caller's tensor is never
    written in place -- which is the condition for registering without a
    mutation declaration."""
    body = section(POINTWISE)
    assert "mutates_args=()" in body


# ---- What it refuses to register ----

def test_a_refused_lowering_registers_nothing():
    """Registering an op the backend could not emit would promise an
    implementation that does not exist."""
    program = compile_source(REFUSED, custom_ops=True).pipelines[0].triton
    assert program.refusals, "expected this program to be refused"
    assert program.custom_ops == ""
    assert "custom_op" not in emitted(REFUSED, custom_ops=True)


def test_an_empty_pipeline_registers_nothing():
    """`statement*` allows an empty body, and there is nothing to call."""
    body = section("pipeline nothing { }")
    assert "custom_op" not in body
    assert "nothing to register as a torch op" in body


def test_a_pipeline_with_no_outputs_registers_nothing():
    """No `output` decl means the op would have no return contract."""
    body = section("pipeline silent { input x : bf16[2,2] }")
    assert "custom_op" not in body
    assert "no outputs" in body


# ---- It is valid Python ----

@pytest.mark.parametrize(
    "source", [POINTWISE, TWO_OUTPUTS, ALIASING, MATMUL],
    ids=["pointwise", "two-outputs", "aliasing", "matmul"],
)
def test_the_emitted_module_parses(source):
    ast.parse(emitted(source, custom_ops=True))


def test_the_registration_is_registrable_shape():
    """`torch.library.custom_op(name, fn, mutates_args=..., device_types=...)`
    then `.register_fake` -- the two-call shape the runtime API accepts."""
    body = section(POINTWISE)
    tree = ast.parse(body)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "custom_op"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 2
    keywords = {kw.arg for kw in calls[0].keywords}
    assert keywords == {"mutates_args", "device_types"}
