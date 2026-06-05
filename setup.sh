#!/usr/bin/env bash
# Run this on your RunPod pod after SSH-ing in.
# Usage: bash setup.sh [--wandb-key YOUR_KEY] [--anthropic-key YOUR_KEY]
set -euo pipefail

REPO_URL="https://github.com/kobotschick/emoji-gen.git"
BRANCH="claude/text-conditioned-ssfm-VnVyS"
WORKDIR="$HOME/emoji-gen"
WANDB_KEY=""
ANTHROPIC_KEY=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb-key)     WANDB_KEY="$2";     shift 2 ;;
    --anthropic-key) ANTHROPIC_KEY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "==> Cloning repo"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "==> Installing uv"
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "==> Installing project deps (PyTorch already present on pod)"
# Exclude torch/torchvision — the pod ships PyTorch 2.8.0 + CUDA 12.8.1.
# open-clip-torch will detect and reuse the existing installation.
uv pip install --system --no-cache \
  "jax[cuda12]>=0.4.25" \
  "equinox>=0.11" \
  "optax>=0.2" \
  "open-clip-torch>=2.24" \
  "wandb>=0.17" \
  "pillow>=10.0" \
  "numpy>=1.26"

echo "==> Verifying GPU is visible to JAX"
python - <<'EOF'
import jax
print("JAX devices:", jax.devices())
EOF

if [ -n "$WANDB_KEY" ]; then
  echo "==> Logging in to W&B"
  python -m wandb login "$WANDB_KEY"
fi

echo "==> Installing tmux + code-server"
apt-get install -y --no-install-recommends tmux
if ! command -v code-server &>/dev/null; then
  curl -fsSL https://code-server.dev/install.sh | sh
fi

echo "==> Installing Node.js (required for Claude Code)"
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y --no-install-recommends nodejs
fi

echo "==> Installing Claude Code"
npm install -g @anthropic-ai/claude-code

if [ -n "$ANTHROPIC_KEY" ]; then
  echo "==> Setting ANTHROPIC_API_KEY"
  echo "export ANTHROPIC_API_KEY=$ANTHROPIC_KEY" >> ~/.bashrc
  export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
fi

echo ""
echo "Setup complete."
echo ""
echo "  Train:        cd $WORKDIR && python experiments/cifar10_text/main.py"
echo "  Claude Code:  cd $WORKDIR && claude"
echo "  IDE:          code-server --bind-addr 0.0.0.0:8080 --auth none $WORKDIR"
echo "                then open RunPod's port 8080 proxy URL in your browser"
