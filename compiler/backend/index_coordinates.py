"""Carry known coordinates through layout operations without flatten/unflatten.

Only proven rotation, broadcast and singleton reshape maps are handled. A
reshape that merges/splits non-singleton axes returns None for the general path.
Expressions are source fragments; no tensor values or concrete shape guesses.
"""
from compiler.backend.backends import product


def _term(coordinate, tail):
    """One axis of a flattened index, with unit strides written bare.

    Spelling a unit stride as "() * (1)" hides the contiguity from Triton's axis
    analysis: the LSE kernel emitted sixteen 16-bit scalar loads per thread and no
    vector loads at all (confirmed in PTX and SASS), while a plain pointwise
    kernel in the same environment emitted 128-bit loads. The last axis is the
    contiguous one, so it must reach the compiler as a bare coordinate.
    """
    return f'({coordinate})' if str(tail) == '1' else f'({coordinate}) * ({tail})'


def flatten(shape, coordinates):
    if len(shape) != len(coordinates):
        raise ValueError('coordinate rank mismatch')
    return ' + '.join(_term(c, product(shape[i+1:]))
                      for i,c in enumerate(coordinates) if str(shape[i]) != '1') or '0'


def singleton_reshape(source, target, coordinates):
    if len(target) != len(coordinates):
        raise ValueError('coordinate rank mismatch')
    source_axes=[str(d) for d in source if str(d) != '1']
    target_axes=[str(d) for d in target if str(d) != '1']
    if source_axes != target_axes:
        return None
    axes=iter(c for d,c in zip(target,coordinates) if str(d) != '1')
    return tuple('0' if str(d) == '1' else next(axes) for d in source)


def broadcast(source, target, coordinates):
    if len(source) > len(target) or len(coordinates) != len(target):
        return None
    tail=coordinates[len(target)-len(source):]
    dimensions=target[len(target)-len(source):]
    if any(str(s) != '1' and str(s) != str(t) for s,t in zip(source,dimensions)):
        return None
    return tuple('0' if str(d) == '1' else c for d,c in zip(source,tail))
