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

The M2 device runner is `device_runner.py`:

```sh
python benchmarks/competitive/device_runner.py --check   # CPU/CI matrix validation
python benchmarks/competitive/device_runner.py --json    # CUDA measurements
python benchmarks/competitive/repeat.py --seeds 0,1,2 --json  # medians
```

It uses the same generated tensors for every backend and the same float64
oracle/tolerance. It separates **CUDA Graph replay** (no host guards/allocation/
dispatch) from **full-call** (guarded host call) and reports both. Backends are
FlowGPU-generated Triton, prepared FlowGPU, static zero-copy CUDA Graph replay,
PyTorch eager, `torch.compile` (TorchInductor) and an explicit hand-written
Triton baseline (`triton_baselines.py`).

The device runner is not executed in CI because CI has no GPU. A handwritten
Triton baseline is included; TVM/FlagOS columns can be added after their
versions and tolerances are pinned. The first H100 GPU1 result is stored in
`measurements/m2-h100-gpu1-20261122.json` and discussed in
`docs/user-guide/competitive-m2-h100.md`.
