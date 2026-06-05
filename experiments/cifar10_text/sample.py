"""
Sampling script: generate images from arbitrary text prompts.

Usage:
    uv run python experiments/cifar10_text/sample.py \
        --checkpoint experiments/cifar10_text/checkpoints/model_ema_final.eqx \
        --prompt "a photo of a red sports car" \
        --n-samples 8 \
        --guidance-scale 5.0 \
        --steps 2
"""

import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from pathlib import Path

try:
    import open_clip
    import torch
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ssfm.text_ssfm import TextSSFM, sample
from experiments.cifar10_text.main import cfg


def encode_prompt(
    prompt: str,
    clip_model,
    tokenizer,
    device: str,
    context_length: int = 77,
    text_dim: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """
    Encode a free-text prompt via CLIP.
    Returns (token_feats [L, D], pooled [D]) as JAX arrays.
    """
    tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        pooled = clip_model.encode_text(tokens, normalize=False)
    pooled_np = pooled.cpu().numpy().squeeze(0)  # [D]
    # Broadcast pooled to fake per-token features [L, D]
    token_feats_np = np.broadcast_to(pooled_np[None], (context_length, text_dim))
    return jnp.array(token_feats_np), jnp.array(pooled_np)


def denormalize(x: jax.Array) -> np.ndarray:
    """[-1,1] float → [0,255] uint8"""
    x = np.array(x)
    x = (x + 1.0) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


def make_grid(images: list[np.ndarray], ncols: int = 4) -> np.ndarray:
    """Stack [H,W,C] images into a grid."""
    H, W, C = images[0].shape
    nrows = (len(images) + ncols - 1) // ncols
    grid = np.zeros((nrows * H, ncols * W, C), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="a photo of a cat")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="samples.png")
    args = parser.parse_args()

    assert HAS_DEPS, "Missing deps. Run: uv sync"

    # ── Load model ──────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    model = TextSSFM(
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        patch_size=cfg.patch_size,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        text_dim=cfg.text_dim,
        n_brownian_coeffs=cfg.n_brownian_coeffs,
        context_length=cfg.context_length,
        cfg_drop_prob=cfg.cfg_drop_prob,
        key=model_key,
    )
    model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── CLIP encoder ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    token_feats, text_pooled = encode_prompt(
        args.prompt, clip_model, tokenizer, device,
        cfg.context_length, cfg.text_dim)

    print(f"Prompt: '{args.prompt}'")
    print(f"Generating {args.n_samples} images with {args.steps} NFE, "
          f"guidance={args.guidance_scale}...")

    # ── Sample ──────────────────────────────────────────────────────────────
    images = []
    for i in range(args.n_samples):
        key, k = jax.random.split(key)
        img = sample(
            model,
            prompt_tokens=token_feats,
            prompt_pooled=text_pooled,
            img_size=cfg.img_size,
            in_channels=cfg.in_channels,
            n_steps=args.steps,
            guidance_scale=args.guidance_scale,
            key=k,
        )
        images.append(denormalize(img))

    # ── Save grid ───────────────────────────────────────────────────────────
    grid = make_grid(images, ncols=min(4, args.n_samples))
    Image.fromarray(grid).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
