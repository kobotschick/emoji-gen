"""
Training script: Text-conditioned SSFM on CIFAR-10.

Each CIFAR-10 class gets a natural-language prompt (e.g. "a photo of a dog").
CLIP text embeddings are pre-computed once and cached.

Usage:
    uv run python experiments/cifar10_text/main.py

Logs to Weights & Biases project "Text-SSFM".
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from functools import partial
import wandb

# Lazy import — only needed at runtime
try:
    import open_clip
    import torchvision
    import torch
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# Add repo root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ssfm.text_ssfm import TextSSFM, total_loss, LossComponents
from ssfm.text_conditioning import CIFAR10_PROMPTS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Model
    img_size: int = 32
    in_channels: int = 3
    patch_size: int = 4
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 12
    text_dim: int = 512          # CLIP ViT-B/32
    n_brownian_coeffs: int = 4
    context_length: int = 77
    cfg_drop_prob: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 400_000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    eta: float = 0.5             # fraction of batch → matching loss
    delta_t: float = 0.05        # matching/distillation threshold

    # Noise schedule (VP-SDE linear schedule)
    beta_min: float = 0.1
    beta_max: float = 20.0

    # Logging
    log_every: int = 100
    sample_every: int = 5000
    save_every: int = 10000
    output_dir: str = "experiments/cifar10_text/checkpoints"
    wandb_project: str = "Text-SSFM"

    # Inference
    guidance_scale: float = 5.0
    sample_steps: int = 2
    n_sample_images: int = 8


cfg = Config()


# ---------------------------------------------------------------------------
# Noise schedule (VP-SDE)
# ---------------------------------------------------------------------------

def _beta_integral(t, beta_min, beta_max):
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2

def alpha_fn(t):
    """αₜ = exp(-½ ∫₀ᵗ β(s)ds)"""
    return jnp.exp(-0.5 * _beta_integral(t, cfg.beta_min, cfg.beta_max))

def sigma_fn(t):
    """σₜ = sqrt(1 - αₜ²)"""
    return jnp.sqrt(jnp.clip(1 - alpha_fn(t) ** 2, 1e-6, 1.0))

def f_sde(t, x_t, x_1):
    """
    Drift of the reverse diffusion SDE conditioned on endpoint X₁.
    f(t, Xₜ | X₁) = (ȧₜ/αₜ)·Xₜ - (ȧₜ·σₜ/αₜ + σ̇ₜ)·(Xₜ - αₜ·X₁)/σₜ

    For the affine Gaussian path  Xₜ = αₜ·X₁ + σₜ·X₀,
    the conditional drift simplifies to:
        ṁₜ = ȧₜ·X₁ + σ̇ₜ·(Xₜ - αₜ·X₁)/σₜ
    """
    alpha_t = alpha_fn(t)
    sigma_t = sigma_fn(t)
    # Numerical differentiation for simplicity; swap for analytic in production
    eps = 1e-5
    alpha_dot = (alpha_fn(t + eps) - alpha_fn(t - eps)) / (2 * eps)
    sigma_dot = (sigma_fn(t + eps) - sigma_fn(t - eps)) / (2 * eps)
    x_0_hat = (x_t - alpha_t * x_1) / jnp.clip(sigma_t, 1e-6)
    return alpha_dot * x_1 + sigma_dot * x_0_hat

def g_sde(t):
    """Diffusion coefficient νₜ of the SDE (scalar)."""
    eps = 1e-5
    alpha_dot = (alpha_fn(t + eps) - alpha_fn(t - eps)) / (2 * eps)
    sigma_dot = (sigma_fn(t + eps) - sigma_fn(t - eps)) / (2 * eps)
    alpha_t = alpha_fn(t)
    sigma_t = sigma_fn(t)
    nu2 = 2 * alpha_dot / alpha_t * sigma_t ** 2 - 2 * sigma_t * sigma_dot
    return jnp.sqrt(jnp.clip(nu2, 1e-6))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_cifar10_dataloader(batch_size: int, train: bool = True):
    """Returns a torchvision CIFAR-10 DataLoader with class labels."""
    assert HAS_DEPS, "torchvision not installed — run: uv sync --extra metrics"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="experiments/cifar10/data", train=train,
        download=True, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)


def precompute_clip_embeddings(clip_model, tokenizer, device) -> dict:
    """
    Pre-compute CLIP text embeddings for all 10 CIFAR-10 class prompts.
    Returns dict: class_idx → (token_feats [L, D], pooled [D]) as numpy arrays.
    """
    embeddings = {}
    for cls_idx, prompt in CIFAR10_PROMPTS.items():
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            # open_clip returns (text_features, ) — we need the full hidden states
            # For full token features we need a forward hook
            feats = clip_model.encode_text(tokens, normalize=False)
        embeddings[cls_idx] = feats.cpu().numpy()   # [1, D]
    return embeddings


def batch_to_jax(images, labels, clip_embeddings, context_length: int, text_dim: int):
    """
    Convert a torch batch to JAX arrays, look up CLIP embeddings by class.

    Returns dict with x_1, text_tokens, text_pooled.
    """
    B = images.shape[0]
    imgs = jnp.array(images.numpy()).transpose(0, 2, 3, 1)  # [B, H, W, C]

    # For simplicity: use pooled embedding only, broadcast to [B, 1, D] as tokens
    # A production version would use full per-token features from CLIP's transformer
    pooled = np.stack([clip_embeddings[int(l)] for l in labels], axis=0)  # [B, D]
    pooled = pooled.squeeze(1)
    # Fake per-token features: repeat pooled across context_length
    tokens = np.broadcast_to(pooled[:, None, :], (B, context_length, text_dim))

    return {
        "x_1": imgs,
        "text_tokens": jnp.array(tokens),
        "text_pooled": jnp.array(pooled),
    }


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@eqx.filter_jit
def ema_update(model: TextSSFM, model_ema: TextSSFM, decay: float) -> TextSSFM:
    params = eqx.filter(model, eqx.is_array)
    params_ema = eqx.filter(model_ema, eqx.is_array)
    new_params_ema = jax.tree.map(
        lambda p_ema, p: decay * p_ema + (1 - decay) * p,
        params_ema, params)
    return eqx.apply_updates(model_ema, new_params_ema)


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(model, model_ema, opt_state, batch, key, optimizer):
    def loss_fn(m):
        lc = total_loss(
            m, model_ema, batch, key,
            alpha_fn, sigma_fn, f_sde, g_sde,
            eta=cfg.eta, delta_t=cfg.delta_t,
        )
        return lc.loss_total, lc

    (loss_val, lc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(
        eqx.filter(grads, eqx.is_array),
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, lc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert HAS_DEPS, "Missing deps. Run: uv sync && uv sync --extra metrics"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # ── CLIP setup ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_embeddings = precompute_clip_embeddings(clip_model, tokenizer, device)

    # ── Model setup ─────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(42)
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
    model_ema = model   # EMA starts as a copy

    # ── Optimizer ───────────────────────────────────────────────────────────
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.total_steps,
        end_value=cfg.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ── Dataloader ──────────────────────────────────────────────────────────
    loader = get_cifar10_dataloader(cfg.batch_size, train=True)

    # ── Training loop ───────────────────────────────────────────────────────
    step = 0
    while step < cfg.total_steps:
        for images, labels in loader:
            if step >= cfg.total_steps:
                break

            batch = batch_to_jax(
                images, labels, clip_embeddings,
                cfg.context_length, cfg.text_dim)

            key, step_key = jax.random.split(key)
            model, opt_state, lc = train_step(
                model, model_ema, opt_state, batch, step_key, optimizer)

            model_ema = ema_update(model, model_ema, cfg.ema_decay)

            if step % cfg.log_every == 0:
                wandb.log({
                    "loss/total": float(lc.loss_total),
                    "loss/matching": float(lc.loss_fg),
                    "loss/distillation": float(lc.loss_D),
                    "step": step,
                })
                print(f"Step {step:6d} | "
                      f"L={float(lc.loss_total):.4f} "
                      f"(fg={float(lc.loss_fg):.4f}, D={float(lc.loss_D):.4f})")

            if step % cfg.save_every == 0 and step > 0:
                eqx.tree_serialise_leaves(
                    output_dir / f"model_{step:07d}.eqx", model)
                eqx.tree_serialise_leaves(
                    output_dir / f"model_ema_{step:07d}.eqx", model_ema)

            step += 1

    # Final save
    eqx.tree_serialise_leaves(output_dir / "model_final.eqx", model)
    eqx.tree_serialise_leaves(output_dir / "model_ema_final.eqx", model_ema)
    wandb.finish()


if __name__ == "__main__":
    main()
