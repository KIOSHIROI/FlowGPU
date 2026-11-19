import pytest

from flowgpu.compiler.optimization.preparation_cost import PhaseLedger,compare_lifetime_cost


def test_phase_clock_records_failures_and_keeps_unattributed_overhead():
    ticks=iter([0.,.002,.003,.008])
    ledger=PhaseLedger(clock=lambda:next(ticks))
    assert ledger.call('compile',lambda:7)==7
    def fail():raise ValueError('validation failed')
    with pytest.raises(ValueError, match='validation failed'):ledger.call('validation',fail)
    report=ledger.finish(10.)
    assert report['clock']=='host_elapsed_ms'
    assert [e['success'] for e in report['entries']]==[True,False]
    assert report['unattributed_ms']==pytest.approx(3.)


def test_reuse_changes_projected_winner_without_changing_dispatch():
    # Synthetic observations test arithmetic, not device speed.
    records=[dict(name='fixed',setup_ms=1.,wall_us=10.,graph_us=2.),
             dict(name='search',setup_ms=101.,wall_us=5.,graph_us=3.)]
    assert compare_lifetime_cost(records,calls=1,objective='wall_us')['projected_best']=='fixed'
    long=compare_lifetime_cost(records,calls=100000,objective='wall_us')
    assert long['projected_best']=='search' and long['automatic_dispatch'] is False
    assert compare_lifetime_cost(records,calls=100000,objective='graph_us')['projected_best']=='fixed'


@pytest.mark.parametrize('calls',[0,-1,True,1.5])
def test_bad_reuse_counts_cannot_enter_cost_comparison(calls):
    with pytest.raises(ValueError, match='positive expected call count required'):compare_lifetime_cost([dict(name='x',setup_ms=0,wall_us=1)],calls=calls,objective='wall_us')


@pytest.mark.parametrize('setup,hot',[(float('nan'),1),(0,float('inf')),(-1,1),(0,0)])
def test_missing_or_invalid_timings_are_not_zero_cost(setup,hot):
    with pytest.raises(ValueError, match='finite nonnegative setup and positive runtime required'):compare_lifetime_cost([dict(name='x',setup_ms=setup,wall_us=hot)],calls=1,objective='wall_us')
