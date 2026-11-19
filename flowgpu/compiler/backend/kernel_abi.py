"""Conservative argument liveness for compiler-owned, fixed Triton kernels.

Keep the kernel body and its graph-level operands/results unchanged. Rewrite
only the formal signature and its paired launch arguments, before host emission.
Autotune supplies parameters separately and is deliberately outside this pass.
"""
import ast
from dataclasses import replace


def prune_unused_parameters(kernel):
    """Return (kernel, audit); unsupported signatures fail before any rewriting.

    Every reference, including assignments and nested control-flow, keeps a
    parameter live. No dead-code or arithmetic rewrite is performed. This is
    conservative formal liveness, not a memory-access or whole-IR liveness pass.
    """
    if kernel.autotune_family:
        raise ValueError('ABI pruning requires a fixed kernel, not an autotuner')
    tree = ast.parse(kernel.source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != kernel.name):
        raise ValueError('ABI pruning requires exactly one named kernel')
    fn = tree.body[0]
    args = fn.args
    if (args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs
            or args.defaults or args.kw_defaults or fn.returns):
        raise ValueError('ABI pruning requires positional formals without defaults')
    if ([ast.unparse(d) for d in fn.decorator_list] != ['triton.jit']
            or len(args.args) != len(kernel.parameters)
            or len(kernel.parameters) != len(kernel.launch)):
        raise ValueError('ABI signature/launch contract mismatch or unsupported decorator')
    for actual, expected in zip(args.args, kernel.parameters):
        if actual.annotation and ast.unparse(actual.annotation) != 'tl.constexpr':
            raise ValueError('ABI pruning only supports tl.constexpr annotations')
        canonical = actual.arg + (': tl.constexpr' if actual.annotation else '')
        if expected.strip() != canonical:
            raise ValueError('ABI source formals differ from parameter metadata')
    body = ast.Module(body=fn.body, type_ignores=[])
    referenced = set()
    for node in ast.walk(body):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Lambda, ast.Global, ast.Nonlocal)):
            raise ValueError('ABI pruning does not support nested scopes')
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {'locals', 'globals', 'eval', 'exec', 'vars'}):
            raise ValueError('ABI pruning does not support dynamic name access')
    kept = tuple(i for i, arg in enumerate(args.args) if arg.arg in referenced)
    removed = tuple(arg.arg for i, arg in enumerate(args.args) if i not in kept)
    if not removed:
        return kernel, dict(before=len(args.args), after=len(args.args), removed=[],
                            retained_indices=list(kept), body_unchanged=True)
    # Our emitter writes a one-line signature. Keeping all other bytes also
    # preserves comments, numerical body spelling and decorator behavior.
    lines = kernel.source.splitlines(keepends=True)
    header = lines[fn.lineno-1]
    if not header.startswith('def '+kernel.name+'(') or not header.rstrip().endswith('):'):
        raise ValueError('ABI pruning requires the compiler one-line signature')
    parameters = tuple(kernel.parameters[i] for i in kept)
    lines[fn.lineno-1] = 'def '+kernel.name+'('+', '.join(parameters)+'):\n'
    source = ''.join(lines)
    # Only one header line was replaced; body bytes are preserved directly.
    # Integration tests independently compare parsed bodies and execute calls.
    result = replace(kernel, source=source, parameters=parameters,
                     launch=tuple(kernel.launch[i] for i in kept))
    return result, dict(before=len(args.args), after=len(parameters), removed=list(removed),
                        retained_indices=list(kept), body_unchanged=True)
