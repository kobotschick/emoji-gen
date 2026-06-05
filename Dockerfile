FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Install PyTorch with CUDA 12.1 first so that open-clip-torch links against
# the GPU build rather than the default CPU wheel from PyPI.
RUN uv pip install --system --no-cache \
    "torch>=2.2" "torchvision>=0.18" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy project files (pyproject.toml first for layer-cache efficiency)
COPY pyproject.toml .
COPY ssfm/       ssfm/
COPY data/       data/
COPY experiments/ experiments/

# Install remaining deps: jax[cuda12], equinox, optax, open-clip-torch, wandb, …
# open-clip-torch will reuse the torch already installed above.
RUN uv pip install --system --no-cache .

CMD ["python", "experiments/cifar10_text/main.py"]
