"""Freeze measured autotune configurations into ordinary guarded JIT launches.

This transforms generated source, never numerical expressions. Configurations
must belong to the emitted search space. META grids are specialized together
with launch kwargs, so a new tile cannot retain the old program count.
"""
import ast
import copy

_LAUNCH_CACHE = "_LAUNCH_CACHE"


def _is_launch_cache_declaration(statement):
    return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == _LAUNCH_CACHE)


def _is_name(expression, prefix, *, is_not=None):
    """True for `_fg_<prefix>... [is not] constant` comparisons."""
    if (not isinstance(expression, ast.Compare) or len(expression.ops) != 1
            or len(expression.comparators) != 1
            or not isinstance(expression.left, ast.Name)
            or not expression.left.id.startswith(f"_fg_{prefix}")
            or not isinstance(expression.comparators[0], ast.Constant)):
        return False
    if is_not is True and not isinstance(expression.ops[0], ast.IsNot):
        return False
    if is_not is False and not isinstance(expression.ops[0], ast.Is):
        return False
    return True


def _assigns_generated_fast_path(statement):
    """True when a statement binds one of the launch-cache helper names."""
    if not isinstance(statement, ast.Assign):
        return False
    return all(isinstance(target, ast.Name) and target.id.startswith("_fg_")
               for target in statement.targets)


def _plain_launch_host(host):
    """Replace the emitted compiled-handle fast path with its plain fallback.

    When the launch cache is opted into, `triton_backend.module` emits every
    plain JIT launch as a cached compiled handle whose last two statements are
    `if _fg_launch is None:` followed by the launch that was emitted before the
    cache existed (a default lowering emits no fast path, so this is a no-op
    there). Prepared deployment
    source has a stricter contract than an imported module: it must contain one
    visible fixed native launch per kernel (flowgpu/compiler/runtime/execution_plan.py
    parses it and rejects any other host operation), and it must be capturable
    in a CUDA graph, where the first call cannot lazily compile a kernel. So
    the prepared form keeps the fallback branch and drops the cache wrapper.
    Nothing numerical changes -- the retained launch text is exactly the plain
    launch and its argument list.
    """
    result = []
    for statement in host.body:
        if isinstance(statement, ast.If) and _is_name(statement.test, "launch", is_not=False):
            result.extend(_plain_launch_host(ast.Module(body=statement.body, type_ignores=[])))
            continue
        if isinstance(statement, ast.If) and _is_name(statement.test, "entry", is_not=True):
            continue
        if isinstance(statement, ast.With):
            # Overlap hosts emit the launch inside a stream context; the fast
            # path there is inside the `with` body, not at the host's top level.
            statement.body = _plain_launch_host(
                ast.Module(body=statement.body, type_ignores=[]))
            result.append(statement)
            continue
        if _assigns_generated_fast_path(statement):
            continue
        if any(isinstance(node, ast.Name) and node.id == _LAUNCH_CACHE
               for node in ast.walk(statement)):
            continue
        result.append(statement)
    return result


def _strip_launch_cache(tree):
    tree.body = [statement for statement in tree.body
                 if not _is_launch_cache_declaration(statement)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_"):
            node.body = _plain_launch_host(node)
    return tree


def _generated_cache_options(tree, value):
    """Recognize only our exact optional cache keyword, never arbitrary hooks."""
    from flowgpu.compiler.backend.autotune import cache_options_source
    name = '_FG_AUTOTUNE_CACHE_KWARGS'
    if not isinstance(value, ast.Name) or value.id != name:
        return False
    expected = next(n for n in ast.parse(cache_options_source()).body if isinstance(n, ast.Assign))
    bindings = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets)==1
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    if len(bindings)!=1 or ast.dump(bindings[0])!=ast.dump(expected):
        return False
    allowed = {id(bindings[0].targets[0])}
    for fn in (n for n in tree.body if isinstance(n,ast.FunctionDef)):
        for decorator in fn.decorator_list:
            if isinstance(decorator,ast.Call) and ast.unparse(decorator.func)=='triton.autotune':
                allowed.update(id(k.value) for k in decorator.keywords if k.arg is None
                               and isinstance(k.value,ast.Name) and k.value.id==name)
    # An extra reference could mutate or alias the dictionary before tuning.
    return all(id(n) in allowed for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id==name)


def _configs(node, assignments):
    if isinstance(node, ast.Name):
        node = assignments[node.id]
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ValueError('unsupported autotune config table')
    configs = []
    for call in node.elts:
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != 'triton.Config':
            raise ValueError('unsupported autotune configuration')
        values = ast.literal_eval(call.args[0])
        values.update(num_warps=4, num_stages=3, num_ctas=1)
        for kw in call.keywords:
            if kw.arg not in {'num_warps', 'num_stages', 'num_ctas'}:
                raise ValueError('configuration hooks/resources are not supported')
            values[kw.arg] = ast.literal_eval(kw.value)
        configs.append(values)
    return configs


def freeze_autotune(code, selections, *, pipeline, context, device_key, specialize_metadata=True):
    """Return standalone source with exact input metadata/device guards.

    context contains shape, stride, dtype, device and pointer alignment, not
    tensor contents or addresses. Original generated semantic guards remain.
    """
    tree = ast.parse(code)
    # Prepared deployment source keeps one visible plain launch per kernel: the
    # execution-plan parser requires it and a graph capture cannot compile on
    # its first call. See `_plain_launch_host`.
    tree = _strip_launch_cache(tree)
    assignments = {n.targets[0].id: n.value for n in tree.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    tuned = {}
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        decorators = [d for d in fn.decorator_list if isinstance(d, ast.Call)
                      and ast.unparse(d.func) == 'triton.autotune']
        if not decorators:
            continue
        if len(decorators) != 1 or fn.name not in selections:
            raise ValueError('one measured configuration required per autotuned kernel')
        decorator = decorators[0]
        kws = {k.arg: k.value for k in decorator.keywords}
        expansions = [k.value for k in decorator.keywords if k.arg is None]
        if expansions:
            if len(expansions)!=1 or not _generated_cache_options(tree, expansions[0]):
                raise ValueError('autotune hooks/reset/restore are not supported')
            kws.pop(None)
        if set(kws) - {'configs', 'key', 'cache_results'}:
            raise ValueError('autotune hooks/reset/restore are not supported')
        selected = selections[fn.name]
        if selected not in _configs(kws['configs'], assignments):
            raise ValueError('selected configuration is outside emitted search space')
        if any(type(v) is not int or v <= 0 for v in selected.values()):
            raise ValueError('configuration values must be positive integers')
        tuned[fn.name] = selected
        fn.decorator_list.remove(decorator)
    if set(tuned) != set(selections):
        raise ValueError('selection names do not match generated autotuned kernels')
    if not any(isinstance(n,ast.FunctionDef) and n.name=='_flowgpu_device_key' for n in tree.body):
        raise ValueError('prepared source requires a device/software identity helper')

    counts = dict.fromkeys(tuned, 0)

    class Launches(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Subscript) or not isinstance(node.func.value, ast.Name):
                return node
            name = node.func.value.id
            if name not in tuned:
                return node
            selected = tuned[name]
            grid = node.func.slice
            if not isinstance(grid, ast.Lambda) or len(grid.args.args) != 1:
                raise ValueError('autotune launch requires a META grid')
            meta = grid.args.args[0].arg

            class Meta(ast.NodeTransformer):
                def visit_Subscript(self, expr):
                    if isinstance(expr.value, ast.Name) and expr.value.id == meta:
                        key = ast.literal_eval(expr.slice)
                        if key not in selected:
                            raise ValueError('grid knob missing from selected configuration')
                        return ast.copy_location(ast.Constant(selected[key]), expr)
                    return self.generic_visit(expr)
            node.func.slice = Meta().visit(copy.deepcopy(grid.body))
            if any(isinstance(n, ast.Name) and n.id == meta for n in ast.walk(node.func.slice)):
                raise ValueError('unspecialized META expression')
            if set(selected) & {kw.arg for kw in node.keywords}:
                raise ValueError('host and selected config both own a launch parameter')
            node.keywords += [ast.keyword(arg=k, value=ast.Constant(v)) for k, v in sorted(selected.items())]
            counts[name] += 1
            return node

    tree = Launches().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError('each tuned kernel must have exactly one host launch')
    host = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_' + pipeline)
    if len(host.args.args) != len(context):
        raise ValueError('preparation currently requires tensor-only pipeline inputs')
    if specialize_metadata:
        from flowgpu.compiler.backend.metadata_specialization import specialize_host
        specialize_host(host, context)
    guards = []
    for arg, item in zip(host.args.args, context):
        name = arg.arg
        if item.get('kind') == 'scalar':
            guards.append(f"if type({name}) not in (int, float):\n"
                          "    raise ValueError('prepared scalar input changed; prepare again')")
            continue
        guards.append(f"if ({name}.layout != torch.strided or tuple({name}.shape) != {tuple(item['shape'])!r} or "
                      f"tuple({name}.stride()) != {tuple(item['stride'])!r} or "
                      f"str({name}.dtype) != {item['dtype']!r} or "
                      f"str({name}.device) != {item['device']!r} or "
                      f"{name}.data_ptr() % 128 != {item['alignment_mod128']!r}):\n"
                      "    raise ValueError('prepared input context changed; prepare again')")
    first = next(arg.arg for arg, item in zip(host.args.args, context) if item.get('kind') != 'scalar')
    guards.append(f"if _flowgpu_device_key({first}.device.index) != {device_key!r}:\n"
                  "    raise ValueError('prepared device/software context changed; prepare again')")
    # Preserve the docstring, and run guards before any allocation or launch.
    pos = int(bool(host.body and isinstance(host.body[0], ast.Expr)
                   and isinstance(host.body[0].value, ast.Constant)
                   and isinstance(host.body[0].value.value, str)))
    host.body[pos:pos] = [n for guard in guards for n in ast.parse(guard).body]
    return '# FlowGPU measured configuration specialization; no output caching.\n' + ast.unparse(ast.fix_missing_locations(tree)) + '\n'
