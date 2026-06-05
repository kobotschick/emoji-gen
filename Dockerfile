FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Copy project files (pyproject.toml first for layer-cache efficiency)
COPY pyproject.toml .
COPY ssfm/       ssfm/
COPY data/       data/
COPY experiments/ experiments/

# PyTorch 2.8.0 + CUDA 12.8 already in base image.
# Install remaining deps: jax[cuda12], equinox, optax, open-clip-torch, wandb, …
RUN uv pip install --system --no-cache \
    "jax[cuda12]>=0.4.25" \
    "equinox>=0.11" \
    "optax>=0.2" \
    "open-clip-torch>=2.24" \
    "wandb>=0.17" \
    "pillow>=10.0" \
    "numpy>=1.26"

CMD ["python", "experiments/cifar10_text/main.py"]
