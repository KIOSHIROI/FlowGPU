# Container image

A CPU image is provided for trying the compiler and CLI without a local Python
environment:

```bash
docker build -t flowgpu .
docker run --rm flowgpu --version
docker run --rm -v "$PWD:/work" flowgpu --emit summary /work/program.fgpu
```

The image installs the package from this repository and exposes the `flowgpu`
entry point. It is **CPU/CLI only**: GPU execution still needs the host CUDA
driver and a Triton-capable device. The CI `docker` workflow builds the image
and runs `--version` on every push.
