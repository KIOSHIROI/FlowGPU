# CPU image: compile and inspect FlowGPU programs. GPU execution still needs
# the host's CUDA driver and a Triton-capable device; this image is for
# adoption, CLI use and CI, not a substitute for the GPU runtime.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY flowgpu ./flowgpu
RUN python -m pip install --upgrade pip && python -m pip install .

ENTRYPOINT ["flowgpu"]
CMD ["--help"]
