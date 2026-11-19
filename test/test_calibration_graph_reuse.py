"""CPU graph doubles check lifecycle/counts, not GPU latency or equivalence."""
import copy
import sys
from types import SimpleNamespace
import weakref

import pytest

from flowgpu.compiler.optimization import measurement
from flowgpu.compiler.optimization.measurement import measure_calls,digest
from flowgpu.compiler.optimization.strategy_tuning import policy_identity,bound_comparison
from test.test_strategy_tuning import observed,SOURCE


def fake_cuda(monkeypatch, *, bytes_per_graph=64, replay_error=False):
    refs=[];state=SimpleNamespace(active=None,replays=0,captures=0)
    class Graph:
        def __init__(self):refs.append(weakref.ref(self));self.size=bytes_per_graph
        def replay(self):
            state.replays+=1
            if replay_error:raise RuntimeError('replay sentinel')
    class Context:
        def __init__(self,g):self.g=g
        def __enter__(self):state.active=self.g;state.captures+=1
        def __exit__(self,*args):state.active=None
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
        def synchronize(self):pass
        def elapsed_time(self,other):return 1.
    cuda=SimpleNamespace(synchronize=lambda:None,CUDAGraph=Graph,graph=Context,Event=Event,
        memory_allocated=lambda:sum(r().size for r in refs if r() is not None))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    return refs,state


@pytest.mark.parametrize('reuse,expected',[(False,12),(True,3)])
def test_capture_count_and_full_paired_rounds(monkeypatch,reuse,expected):
    refs,state=fake_cuda(monkeypatch)
    counts={n:0 for n in 'abc'}
    def call(n):counts[n]+=1;return {'result':n}
    stats={};samples=measure_calls({n:lambda n=n:call(n) for n in counts},
        rounds=4,graph_replays=2,wall_calls=3,reuse_graphs=reuse,capture_stats=stats)
    assert state.captures==stats['captures']==expected
    assert all(len(v[m])==4 for v in samples.values() for m in ('graph_us','wall_us'))
    assert all(n==(43 if reuse else 46) for n in counts.values())
    assert all(r() is None for r in refs)


def test_retention_budget_falls_back_to_full_recapture_without_dropping_candidates(monkeypatch):
    refs,state=fake_cuda(monkeypatch,bytes_per_graph=40*1024*1024)
    stats={};samples=measure_calls({'a':lambda:None,'b':lambda:None,'c':lambda:None},
        rounds=4,graph_replays=2,wall_calls=1,reuse_graphs=True,capture_stats=stats)
    assert set(samples)=={'a','b','c'} and all(len(v['graph_us'])==4 for v in samples.values())
    assert stats['mode']=='per_round' and stats['captures']==14
    assert stats['fallback_reason']=='retained allocation budget exceeded'
    assert all(r() is None for r in refs)


def test_runtime_failure_propagates_and_releases_retained_graphs(monkeypatch):
    refs,_=fake_cuda(monkeypatch,replay_error=True)
    with pytest.raises(RuntimeError,match='replay sentinel'):
        measure_calls({'a':lambda:None,'b':lambda:None},reuse_graphs=True)
    assert all(r() is None for r in refs)


@pytest.mark.parametrize('bad',[1,None,'yes'])
def test_invalid_opt_in_rejected_without_cuda(bad):
    with pytest.raises(ValueError, match='reuse_graphs must be boolean'):measure_calls({},reuse_graphs=bad)
    with pytest.raises(ValueError, match='reuse_calibration_graphs must be boolean'):policy_identity(SOURCE,'lse',{}, {}, {},objective='graph_us',
        workload_tag='unit',reuse_calibration_graphs=bad)


def test_capture_mode_is_a_distinct_cache_identity():
    args=(SOURCE,'lse',{}, {}, {})
    kw=dict(objective='graph_us',workload_tag='unit')
    assert digest(policy_identity(*args,**kw))!=digest(policy_identity(*args,**kw,reuse_calibration_graphs=True))


def test_reused_policy_requires_consistent_capture_evidence():
    identity,evidence=observed();identity=copy.deepcopy(identity)
    identity['measurement']['reuse_calibration_graphs']=True
    with pytest.raises(ValueError,match='capture evidence'):bound_comparison(SOURCE,identity,evidence)
    evidence['calibration_capture']=dict(requested_reuse=True,captures=5,retained_bytes_peak=1024,
        retention_budget_bytes=64*1024*1024,mode='per_calibration',fallback_reason=None)
    assert len(bound_comparison(SOURCE,identity,evidence)['candidates'])==5
    evidence['calibration_capture']['captures']=4
    with pytest.raises(ValueError,match='capture evidence'):bound_comparison(SOURCE,identity,evidence)
