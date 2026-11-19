"""Dedicated one-launch lowering for the opaque store_cache_4d scatter."""

from compiler.backend.backends import Argument, Names, identifier, group_operands, group_results
from compiler.backend.platform_configs import emitted_index_dtype

def triton_kernel(graph, group, memory, name, config):
    from compiler.backend.triton_backend import TritonKernel
    assert len(group.members) == 2
    assert all(node.op == 'store_cache_4d' for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 5 and len(results) == 2
    k_view, v_view, cache_k, cache_v, loc = operands
    k_out, v_out = results
    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + '_ptr') for value in operands + results}
    pairs = [(ptr[value], Argument('tensor', value)) for value in operands + results]

    def constexpr(name_hint, value, argument):
        name = names.fresh(name_hint)
        pairs.append((f'{name}: tl.constexpr', argument))
        return name

    p = constexpr('P', graph.values[k_view].shape[0], Argument('shape', f'{k_view}:0'))
    page = names.fresh('PAGE')
    pairs.append((page, Argument('value', 'page_size')))
    heads = constexpr('H', graph.values[cache_k].shape[1], Argument('shape', f'{cache_k}:1'))
    dim = constexpr('D', graph.values[cache_k].shape[2], Argument('shape', f'{cache_k}:2'))
    n = constexpr('N', graph.values[loc].shape[0], Argument('shape', f'{loc}:0'))
    block = names.fresh('BLOCK')
    cache_block = config.default_block_size
    pairs.append((f'{block}: tl.constexpr', Argument('value', str(cache_block))))

    stride_names = {}
    for value in operands + results:
        rank = len(graph.values[value].shape)
        stride_names[value] = []
        for axis in range(rank):
            stride_names[value].append(constexpr(
                identifier(value) + f'_s{axis}', graph.values[value].shape[axis],
                Argument('stride', f'{value}:{axis}')))

    idx = names.fresh('idx')
    valid = names.fresh('valid')
    token = names.fresh('token')
    rem = names.fresh('rem')
    head = names.fresh('head')
    lane = names.fresh('lane')
    location = names.fresh('location')
    page_index = names.fresh('page_index')
    offset = names.fresh('offset')
    ks, vs, cks, cvs, ls = (stride_names[x] for x in (k_out, v_out, cache_k, cache_v, loc))
    # The dtype of the data-derived location value, folded into the platform
    # policy instead of staying a branch on device_type. 华为 measured the
    # int32 form (the store_cache_4d submission); the CUDA-class targets keep
    # the unchanged int64 one. This is a *value* dtype, not a proven program
    # index, so it follows the platform preference directly -- and this ABI is
    # only emitted on targets whose allowed_kernel_kinds admit 'cache_store'
    # (cuda and npu today), so no unproven int32 path is newly reachable.
    # `proven=False` is deliberate: this value is data-derived, not the
    # program index a host bound covers, so only an explicit 'int32' platform
    # request selects it and 'auto' keeps the int64 form.
    index_type = ('tl.int32' if emitted_index_dtype(config, proven=False) == 'int32'
                  else 'tl.int64')
    if config.device_type == 'npu':
        lines = [
            f'    {token} = tl.program_id(0)',
            f'    {idx} = tl.program_id(1) * {block} + tl.arange(0, {block})',
            f'    kv_axis = tl.program_id(2)',
            f'    {valid} = {idx} < {heads} * {dim}',
            f'    {head} = {idx} // {dim}',
            f'    {lane} = {idx} % {dim}',
            f'    {location} = tl.load({ptr[loc]} + {token} * {ls[0]}, mask={valid}, other=0).to(tl.int32)',
            f'    {page_index} = {location} // {page}',
            f'    {offset} = {location} - {page_index} * {page}',
            f'    if kv_axis == 0:',
            f'        value = tl.load({ptr[cache_k]} + {token} * {cks[0]} + {head} * {cks[1]} + {lane} * {cks[2]}, mask={valid}, other=0)',
            f'        tl.store({ptr[k_out]} + {page_index} * {ks[0]} + {offset} * {ks[1]} + {head} * {ks[2]} + {lane} * {ks[3]}, value, mask={valid})',
            f'    else:',
            f'        value = tl.load({ptr[cache_v]} + {token} * {cvs[0]} + {head} * {cvs[1]} + {lane} * {cvs[2]}, mask={valid}, other=0)',
            f'        tl.store({ptr[v_out]} + {page_index} * {vs[0]} + {offset} * {vs[1]} + {head} * {vs[2]} + {lane} * {vs[3]}, value, mask={valid})',
        ]
        grid = f'({loc}.shape[0], triton.cdiv({cache_k}.shape[1] * {cache_k}.shape[2], {cache_block}), 2)'
    else:
        lines = [
            f'    {idx} = tl.program_id(0) * {block} + tl.arange(0, {block})',
            f'    {valid} = {idx} < {n} * {heads} * {dim}',
            f'    {token} = {idx} // ({heads} * {dim})',
            f'    {rem} = {idx} % ({heads} * {dim})',
            f'    {head} = {rem} // {dim}',
            f'    {lane} = {rem} % {dim}',
            f'    {location} = tl.load({ptr[loc]} + {token} * {ls[0]}, mask={valid}, other=0).to({index_type})',
            f'    {page_index} = {location} // {page}',
            f'    {offset} = {location} - {page_index} * {page}',
            f'    k_value = tl.load({ptr[cache_k]} + {token} * {cks[0]} + {head} * {cks[1]} + {lane} * {cks[2]}, mask={valid}, other=0)',
            f'    v_value = tl.load({ptr[cache_v]} + {token} * {cvs[0]} + {head} * {cvs[1]} + {lane} * {cvs[2]}, mask={valid}, other=0)',
            f'    tl.store({ptr[k_out]} + {page_index} * {ks[0]} + {offset} * {ks[1]} + {head} * {ks[2]} + {lane} * {ks[3]}, k_value, mask={valid})',
            f'    tl.store({ptr[v_out]} + {page_index} * {vs[0]} + {offset} * {vs[1]} + {head} * {vs[2]} + {lane} * {vs[3]}, v_value, mask={valid})',
        ]
        grid = f'(triton.cdiv({loc}.shape[0] * {cache_k}.shape[1] * {cache_k}.shape[2], {cache_block}),)'
    params = ', '.join(parameter for parameter, _ in pairs)
    source = '# store_cache_4d\n@triton.jit\n'
    source += f'def {name}({params}):\n' + '\n'.join(lines) + '\n'
    return TritonKernel(name, 'cache_store', tuple(n.op for n in group.members), operands, results,
                        tuple(parameter for parameter, _ in pairs), tuple(argument for _, argument in pairs),
                        grid, source,
                        launch_options=(('num_warps', config.num_warps),))


def reference_kernel(graph, group, memory, name):
    from compiler.backend.reference import _kernel, _dimension_parameters, _rounded
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    k_view, v_view, cache_k, cache_v, loc = operands
    k_out, v_out = results
    names = Names(graph)
    variables = {v: names.reserve(identifier(v)) for v in operands + results}
    pairs = [(variables[v], Argument('tensor', v)) for v in operands + results]
    _dimension_parameters(names, pairs, [d for v in operands + results for d in graph.values[v].shape])
    page_size = graph.values[k_view].shape[1]
    n = graph.values[loc].shape[0]
    heads = graph.values[cache_k].shape[1]
    dim = graph.values[cache_k].shape[2]
    # `loc` is the destination slot for that token; a negative slot is padding
    # and writes nowhere.  Python's floor division made `page` negative for such
    # a value and the negative index then crashed the oracle (review P1-11).
    lines = [f'    {variables[k_out]}[:] = {variables[k_view]}[:]',
             f'    {variables[v_out]}[:] = {variables[v_view]}[:]']
    lines += [f'    for token in range({n}):',
              f'        if ({variables[loc]}[token] < 0',
              f'                or {variables[loc]}[token] >= len({variables[k_out]}) // ({heads} * {dim})):',
              f'            continue',
              f'        for head in range({heads}):',
              f'            for lane in range({dim}):',
              f'                k_index = {variables[loc]}[token] * ({heads}) * ({dim}) + head * ({dim}) + lane',
              f'                c_index = token * ({heads}) * ({dim}) + head * ({dim}) + lane',
              f'                {variables[k_out]}[k_index] = {_rounded(graph, k_out, variables[cache_k] + "[c_index]")}',
              f'                {variables[v_out]}[k_index] = {_rounded(graph, v_out, variables[cache_v] + "[c_index]")}']
    return _kernel(name, 'cache_store', group, operands, results, pairs, lines)
