import itertools

import pytest

from flowgpu.compiler.backend.index_coordinates import broadcast, flatten, singleton_reshape


@pytest.mark.parametrize('source,target',[
    ([2,1,3],[1,2,3,1]),([1,1],[1]),([2,3],[2,1,3]),([2,3],[2,3]),
])
def test_singleton_reshape_preserves_linear_address_for_every_coordinate(source,target):
    names=tuple('c'+str(i) for i in range(len(target)))
    mapped=singleton_reshape(source,target,names)
    assert mapped is not None
    for coords in itertools.product(*(range(d) for d in target)):
        env=dict(zip(names,coords))
        assert eval(flatten(source,mapped),{},env)==eval(flatten(target,names),{},env)


def test_general_reshape_is_not_guessed_from_equal_element_count():
    assert singleton_reshape([2,6],[3,4],('a','b')) is None
    assert singleton_reshape(['A','B'],['B','A'],('a','b')) is None
    with pytest.raises(ValueError, match='coordinate rank mismatch'):flatten([2,3],('a',))


def test_broadcast_projection_and_rotation_inverse_are_exact():
    mapped=broadcast([2,1,5],[7,2,3,5],('w','x','y','z'))
    assert mapped==('x','0','z')
    for r,b,h,d in itertools.product(range(3),range(2),range(4),range(5)):
        # Output B,H,D,R of move_first_last(input R,B,H,D).
        inverse=('r','b','h','d')
        env=dict(r=r,b=b,h=h,d=d)
        assert eval(flatten([3,2,4,5],inverse),{},env)==((r*2+b)*4+h)*5+d
    assert broadcast([3,5],[2,4,5],('a','b','c')) is None
