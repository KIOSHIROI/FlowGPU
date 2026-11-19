import ast
from types import SimpleNamespace

import pytest

from flowgpu.compiler.backend.metadata_specialization import specialize_host


CONTEXT=[dict(shape=[7,3],stride=[3,1],dtype='torch.float32',device='cuda:0',alignment_mod128=0)]


def specialized(source):
    tree=ast.parse(source);tree.body[0]=specialize_host(tree.body[0],CONTEXT)
    result=ast.unparse(ast.fix_missing_locations(tree));ns={};exec(result,ns)
    return ns['run'],result


def test_mutating_shape_dictionary_keeps_live_reads_and_stores():
    source='''def run(x):
        shape = {}
        if 'R' in shape and shape['R'] != x.shape[0]:
            raise ValueError('shape mismatch')
        shape['R'] = x.shape[0]
        return shape['R']
    '''
    run,text=specialized(source)
    assert run(SimpleNamespace(shape=(7,3)))==7
    assert "shape['R']" in text


@pytest.mark.parametrize('branch',[False,True])
def test_conditional_reassignment_does_not_reuse_previous_constant(branch):
    run,_=specialized('''def run(x):
        n = 1
        if x.flag:
            n = 2
        return n
    ''')
    assert run(SimpleNamespace(flag=branch))==(2 if branch else 1)


def test_mutable_container_and_alias_do_not_turn_into_copies():
    run,_=specialized('''def run(x):
        values = [1]
        alias = values
        alias[0] = 3
        return values[0], alias is values
    ''')
    assert run(None)==(3,True)


def test_reassigned_input_and_untaken_invalid_subscript_stay_live():
    run,_=specialized('''def run(x):
        x = x.other
        dimensions = (3,)
        if x.flag:
            return dimensions[4]
        return x.shape[0]
    ''')
    x=SimpleNamespace(other=SimpleNamespace(shape=(13,),flag=False))
    assert run(x)==13
    x.other.flag=True
    with pytest.raises(IndexError, match='tuple index out of range'):run(x)


def test_simple_immutable_metadata_still_specializes():
    run,text=specialized('''def run(x):
        dimensions = x.shape
        first = dimensions[0]
        return first
    ''')
    assert run(None)==7 and 'return 7' in text
