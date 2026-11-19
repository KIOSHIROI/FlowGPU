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
```

It uses the same generated tensors for every backend and the same float64
oracle/tolerance. It separates **CUDA Graph replay** (no host guards/allocation/
dispatch) from **full-call** (guarded host call) and reports both. Backends are
FlowGPU-generated Triton, PyTorch eager, and `torch.compile` (TorchInductor).

The device runner is not executed in CI because CI has no GPU. A handwritten
Triton baseline and TVM/FlagOS columns can be added after their versions and
tolerances are pinned.
