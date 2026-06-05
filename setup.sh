#!/usr/bin/env bash
# Run this on your RunPod pod after SSH-ing in.
# Usage: bash setup.sh [--wandb-key YOUR_KEY]
set -euo pipefail

REPO_URL="https://github.com/kobotschick/emoji-gen.git"
BRANCH="claude/text-conditioned-ssfm-VnVyS"
WORKDIR="$HOME/emoji-gen"
WANDB_KEY=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb-key) WANDB_KEY="$2"; shift 2 ;;
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

echo "==> Installing PyTorch (cu121)"
uv pip install --system --no-cache \
  "torch>=2.2" "torchvision>=0.18" \
  --extra-index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project deps"
uv pip install --system --no-cache .

echo "==> Verifying GPU is visible to JAX"
python - <<'EOF'
import jax
print("JAX devices:", jax.devices())
EOF

if [ -n "$WANDB_KEY" ]; then
  echo "==> Logging in to W&B"
  python -m wandb login "$WANDB_KEY"
fi

echo ""
echo "Setup complete. To start training:"
echo "  cd $WORKDIR && python experiments/cifar10_text/main.py"
