"""Guarded read coalescing for adjacent, same-shaped external pointwise views.

Static facts only nominate pairs. A host metadata proof is passed as a
constexpr; neither value contents nor a previous call's pointers are cached.
The ordinary two loads remain the fallback. Reduction and remapped index
domains deliberately do not participate in this first implementation.
"""
from compiler.backend.backends import Argument, identifier


def adjacent_view_condition(left, right):
    """Python host expression, after ordinary dtype/shape/device contracts.

    Retain a same-storage check even if allocators happen to place unrelated
    tensors next to one another. Storage offsets and strides refer to elements
    of the already checked same dtype; no tensor arithmetic runs here.
    Binary conjunctions are intentional for restricted Triton frontends.
    """
    checks = [
        f'{left}.stride(-1) == 2',
        f'{left}.stride() == {right}.stride()',
        f'{left}.untyped_storage().data_ptr() == {right}.untyped_storage().data_ptr()',
        f'{right}.storage_offset() == {left}.storage_offset() + 1',
    ]
    result = f'({checks[0]})'
    for check in checks[1:]:
        result = f'({result} and ({check}))'
    return result


class PairedViewLoads:
    def __init__(self, graph, operands, result, layouts, ptrs, names, pairs, *, enabled):
        self.graph, self.layouts, self.ptrs, self.names = graph, layouts, ptrs, names
        self.partners = {}
        self.loaded = {}
        if not enabled:
            return
        shape = graph.values[result].shape
        eligible = [v for v in operands if v in layouts
                    and graph.values[v].shape == shape and shape
                    and graph.values[v].dtype in {'fp16', 'bf16', 'fp32'}]
        for i, left in enumerate(eligible):
            if left in self.partners:
                continue
            right = next((v for v in eligible[i+1:] if v not in self.partners
                          and graph.values[v].dtype == graph.values[left].dtype), None)
            if right is None:
                continue
            flag = names.fresh('adjacent_views')
            pairs.append((f'{flag}: tl.constexpr', Argument('adjacent_views', f'{left}:{right}')))
            self.partners[left] = self.partners[right] = (left, right, flag)

    def load(self, value, at, mask, lines, *, indent='    ', coordinates=None):
        """Called only in the unchanged, one-dimensional output index domain."""
        from compiler.backend.triton_backend import _view_offset
        if value not in self.partners:
            return None
        coordinate_key = tuple(coordinates) if coordinates is not None else None
        key = value, at, mask, indent, coordinate_key
        if key in self.loaded:
            return self.loaded[key]
        left, right, flag = self.partners[value]
        lhs = self.names.fresh(identifier(left) + '_load')
        rhs = self.names.fresh(identifier(right) + '_load')
        packed = self.names.fresh('paired_values')
        addresses = []
        address = _view_offset(left, at, self.layouts, self.names, addresses, coordinates=coordinates)
        # Address reconstruction is shared with the old path, including row
        # gaps. The +1 read is valid exactly when the host proves adjacency.
        lines.extend(indent + line[4:] for line in addresses)
        lines += [f'{indent}if {flag}:',
                  f'{indent}    {packed} = tl.load({self.ptrs[left]} + ({address})[:, None] + tl.arange(0, 2)[None, :], mask=({mask})[:, None], other=0)',
                  f'{indent}    {lhs}, {rhs} = tl.split({packed})',
                  f'{indent}else:',
                  f'{indent}    {lhs} = tl.load({self.ptrs[left]} + {address}, mask={mask}, other=0)']
        addresses = []
        right_address = _view_offset(right, at, self.layouts, self.names, addresses, coordinates=coordinates)
        lines.extend(indent + '    ' + line[4:] for line in addresses)
        lines.append(f'{indent}    {rhs} = tl.load({self.ptrs[right]} + {right_address}, mask={mask}, other=0)')
        self.loaded[(left, at, mask, indent, coordinate_key)] = lhs
        self.loaded[(right, at, mask, indent, coordinate_key)] = rhs
        return self.loaded[key]
