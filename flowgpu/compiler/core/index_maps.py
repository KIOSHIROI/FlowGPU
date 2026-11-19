"""Axis identity contracts shared by semantic normalization and addressing.

Equal dimension sizes never prove a permutation. Legacy hand-built dataflow
without attributes uses the operator contract; typed metadata must agree.
"""


def canonical_permutation(op,rank):
    if type(rank) is not int or rank<=0:raise ValueError('positive permutation rank required')
    if op=='move_first_last':return tuple(range(1,rank))+(0,)
    if op=='transpose' and rank==2:return (1,0)
    raise ValueError('unsupported permutation operator/rank')


def permutation_of(node,rank):
    expected=canonical_permutation(node.op,rank)
    attrs=node.attributes
    if attrs is None:return expected
    actual=getattr(attrs,'permutation',None)
    if (not isinstance(actual,tuple) or any(type(a) is not int for a in actual)
            or actual!=expected):
        raise ValueError('view permutation metadata disagrees with operator/axis identities')
    return actual


def inverse_coordinates(node,coordinates):
    axes=permutation_of(node,len(coordinates))
    return tuple(coordinates[axes.index(axis)] for axis in range(len(axes)))
