# Competitive probe

This is the first two CPU-only layers of the competitive benchmark matrix. It
compiles the public cases at their declared shapes and records compile time,
kernels, slots and modeled traffic. It also runs the FlowGPU reference backend
against independent flat-list oracles for all six cases. It does **not** claim a
device latency or a win against Triton/TVM/FlagOS; those need a pinned competitor
runner.

```sh
python benchmarks/competitive/probe.py
python benchmarks/competitive/probe.py --json
python benchmarks/competitive/probe.py --check
```

The case list is `cases.json`; it uses only public examples, so the probe is
reproducible without private fixtures.

Next layers:

1. correctness: FlowGPU reference vs an independent PyTorch oracle on the same
   numbers;
2. device latency: FlowGPU Triton vs Torch eager / `torch.compile` / a pinned
   Triton baseline on the same shapes;
3. dynamic-shape dispatch: per-shape cache hit rate and rebuild cost;
4. competitor columns (TVM/Relax, FlagOS/Triton-Ascend) only after their
   installation, versions and tolerances are pinned.
