"""Cost model for P0's view-vs-materialise decision.

The measured choice is usually obvious: a layout anchor plus its pointwise
consumer is one kernel as a view, versus two kernels (layout copy + consumer)
when materialised.  The model still exists because a view is not free: it
replaces a contiguous load/store with output-index remapping, whose cost grows
with the number of layout operators and the output extent.  The model is used
by ``layout_view_policy="cost"``; ``"always"`` (the default when layout_views is
enabled) keeps the measured-good behaviour.

Coefficients are deliberately documented as estimates, not measured constants:
H100 views of 4096 elements were ~20-27 us and the extra launch alone is
~15 us, so the launch term dominates at this corpus size.  The per-element
index term only matters for much larger outputs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutViewCostModel:
    #: One kernel launch's fixed host+device cost (H100 calibration: ~15.5 us).
    launch_us: float = 15.5
    #: Effective bandwidth for a materialised copy (bytes/us; 1 TB/s).
    bandwidth_bytes_per_us: float = 1_000.0
    #: Index-remap arithmetic per layout operator per output element.
    index_us_per_element_per_layout: float = 5e-5
    #: Pointwise consumer arithmetic per element (estimated, not measured).
    consumer_us_per_element: float = 1e-5


def view_cost(numel: int, layout_ops: int, consumer_ops: int,
              model: LayoutViewCostModel | None = None) -> float:
    """Estimated cost of one fused indexed region (one launch)."""
    model = model or LayoutViewCostModel()
    index = numel * layout_ops * model.index_us_per_element_per_layout
    consumer = numel * consumer_ops * model.consumer_us_per_element
    return model.launch_us + index + consumer


def materialize_cost(numel: int, layout_ops: int, consumer_ops: int,
                     dtype_bytes: int = 4,
                     model: LayoutViewCostModel | None = None) -> float:
    """Two launches: the layout copy and the consumer, plus copy traffic."""
    model = model or LayoutViewCostModel()
    # materialise writes then reads the intermediate: 2 * numel * dtype bytes
    traffic = 2 * numel * dtype_bytes / model.bandwidth_bytes_per_us
    consumer = numel * consumer_ops * model.consumer_us_per_element
    return 2 * model.launch_us + traffic + consumer


def should_fuse_layout_view(numel: int, layout_ops: int, consumer_ops: int,
                            *, dtype_bytes: int = 4,
                            model: LayoutViewCostModel | None = None
                            ) -> tuple[bool, dict]:
    model = model or LayoutViewCostModel()
    view = view_cost(numel, layout_ops, consumer_ops, model)
    material = materialize_cost(numel, layout_ops, consumer_ops, dtype_bytes, model)
    return view < material, {"view_us": round(view, 3),
                             "materialize_us": round(material, 3),
                             "saving_us": round(material - view, 3)}
