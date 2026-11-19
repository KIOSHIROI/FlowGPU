"""Host phase accounting and explicit reuse-cost comparisons.

Host elapsed may include synchronization/JIT; it is never labeled GPU time.
Projected lifetime cost is a model of supplied observations, not a new timing.
"""
import math
import time


class PhaseLedger:
    def __init__(self,clock=None):
        self.clock=clock or time.perf_counter
        self.entries=[]

    def call(self,name,function,*args,**kwargs):
        start=self.clock();success=False
        try:
            result=function(*args,**kwargs);success=True
            return result
        finally:
            self.entries.append(dict(phase=name,elapsed_ms=(self.clock()-start)*1000,success=success))

    def finish(self,total_ms):
        attributed=sum(e['elapsed_ms'] for e in self.entries)
        return dict(clock='host_elapsed_ms',total_ms=total_ms,entries=list(self.entries),
            unattributed_ms=max(0.,total_ms-attributed),
            note='Calls are sequential, phases are not device-event/JIT-only measurements')


def compare_lifetime_cost(observations,*,calls,objective):
    """Compare measured alternatives for a stated reuse count, without dispatch."""
    if type(calls) is not int or calls<=0:raise ValueError('positive expected call count required')
    if objective not in {'graph_us','wall_us'}:raise ValueError('separate graph/wall objective required')
    if not observations:raise ValueError('cost observations required')
    names=set();rows=[]
    for item in observations:
        name=item['name'];setup=item['setup_ms'];hot=item[objective]
        if name in names:raise ValueError('unique cost alternative names required')
        names.add(name)
        if any(type(v) not in {int,float} or not math.isfinite(v) or v<0 for v in (setup,hot)) or hot==0:
            raise ValueError('finite nonnegative setup and positive runtime required')
        rows.append(dict(name=name,setup_ms=setup,per_call_us=hot,
                         projected_total_ms=setup+calls*hot/1000))
    return dict(calls=calls,objective=objective,alternatives=rows,
        projected_best=min(rows,key=lambda r:r['projected_total_ms'])['name'],automatic_dispatch=False,
        evidence='projection from caller-supplied observations; validate against actual reuse distribution')
