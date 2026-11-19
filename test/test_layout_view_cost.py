"""P0 cost model for view vs materialise."""
from __future__ import annotations

from compiler.optimization.layout_view_cost import (LayoutViewCostModel,
                                                    materialize_cost,
                                                    should_fuse_layout_view,
                                                    view_cost)


def test_view_wins_for_the_measured_corpus_sizes():
    for numel, layout_ops in ((4096, 1), (4096, 3), (1_048_576, 2)):
        decision, costs = should_fuse_layout_view(numel, layout_ops, consumer_ops=1)
        assert decision is True
        assert costs["view_us"] < costs["materialize_us"]


def test_a_pathological_index_model_can_prefer_materialisation():
    model = LayoutViewCostModel(index_us_per_element_per_layout=5e-2,
                                consumer_us_per_element=0.0)
    decision, costs = should_fuse_layout_view(1_048_576, 10, 1, model=model)
    assert decision is False
    assert costs["view_us"] > costs["materialize_us"]


def test_cost_components_are_monotone_in_extent():
    assert view_cost(4096, 2, 1) < view_cost(8192, 2, 1)
    assert materialize_cost(4096, 2, 1) < materialize_cost(8192, 2, 1)
