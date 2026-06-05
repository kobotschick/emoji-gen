"""
Text conditioning for SSFM.

Wraps a frozen CLIP text encoder and provides:
  - encode_text(prompts)  →  (tokens: [B, L, D], pooled: [B, D])
  - cfg_dropout(...)       →  randomly null-out conditioning during training
  - null_embedding(...)    →  the learned "unconditional" embedding for CFG at inference
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple

# ---------------------------------------------------------------------------
# CLIP text encoder (frozen)
# ---------------------------------------------------------------------------

class CLIPTextConfig(NamedTuple):
    vocab_size: int = 49408
    context_length: int = 77
    embed_dim: int = 512        # ViT-B/32 text width
    num_heads: int = 8
    num_layers: int = 12
    hidden_dim: int = 512


class CLIPTextEncoder(eqx.Module):
    """
    Minimal frozen CLIP text encoder.

    In practice, load weights from `open_clip` or `transformers`:

        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
        encoder = CLIPTextEncoder.from_open_clip(model.token_embedding,
                                                  model.transformer,
                                                  model.ln_final,
                                                  model.text_projection)

    The module is always run with eqx.nn.inference_mode — gradients are
    never computed through it.
    """

    token_embedding: eqx.nn.Embedding
    positional_embedding: jax.Array          # [L, D]
    transformer_blocks: list                 # list of eqx.nn.TransformerEncoderLayer
    ln_final: eqx.nn.LayerNorm
    text_projection: jax.Array               # [D, D_proj]
    context_length: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, cfg: CLIPTextConfig, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, cfg.num_layers + 3)
        self.token_embedding = eqx.nn.Embedding(cfg.vocab_size, cfg.embed_dim, key=keys[0])
        self.positional_embedding = jax.random.normal(keys[1], (cfg.context_length, cfg.embed_dim)) * 0.01
        self.transformer_blocks = [
            eqx.nn.TransformerEncoderLayer(
                input_size=cfg.embed_dim,
                intermediate_size=cfg.hidden_dim * 4,
                num_heads=cfg.num_heads,
                dropout_p=0.0,
                key=keys[i + 2],
            )
            for i in range(cfg.num_layers)
        ]
        self.ln_final = eqx.nn.LayerNorm(cfg.embed_dim)
        self.text_projection = jax.random.normal(keys[-1], (cfg.embed_dim, cfg.embed_dim)) * 0.02
        self.context_length = cfg.context_length
        self.embed_dim = cfg.embed_dim

    def __call__(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            tokens: [B, L] int32 token ids (padded to context_length with 0)

        Returns:
            token_feats:  [B, L, D]   per-token features
            pooled:       [B, D]      EOS-token feature projected
        """
        B, L = tokens.shape
        x = jax.vmap(self.token_embedding)(tokens)          # [B, L, D]
        x = x + self.positional_embedding[:L]               # [B, L, D]

        # causal mask — CLIP uses causal attention in its text transformer
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))

        for block in self.transformer_blocks:
            x = jax.vmap(lambda xi: block(xi, mask=mask))(x)

        x = jax.vmap(jax.vmap(self.ln_final))(x)           # [B, L, D]

        # Pool at the EOS token (highest token id position)
        eos_idx = jnp.argmax(tokens, axis=-1)               # [B]
        pooled = x[jnp.arange(B), eos_idx]                  # [B, D]
        pooled = pooled @ self.text_projection               # [B, D]

        return x, pooled                                     # (token_feats, pooled)


# ---------------------------------------------------------------------------
# Null (unconditional) embedding
# ---------------------------------------------------------------------------

class NullEmbedding(eqx.Module):
    """
    A single learned vector representing the unconditional signal.
    Used during CFG: replace conditioning with this when text is dropped.
    """
    token_feats: jax.Array    # [L, D]
    pooled: jax.Array         # [D]

    def __init__(self, context_length: int, embed_dim: int, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.token_feats = jax.random.normal(k1, (context_length, embed_dim)) * 0.02
        self.pooled = jax.random.normal(k2, (embed_dim,)) * 0.02

    def expand(self, batch_size: int) -> tuple[jax.Array, jax.Array]:
        """Return [B, L, D] and [B, D] null embeddings."""
        return (
            jnp.broadcast_to(self.token_feats[None], (batch_size, *self.token_feats.shape)),
            jnp.broadcast_to(self.pooled[None], (batch_size, self.pooled.shape[0])),
        )


# ---------------------------------------------------------------------------
# Classifier-free guidance utilities
# ---------------------------------------------------------------------------

def cfg_dropout(
    token_feats: jax.Array,     # [B, L, D]
    pooled: jax.Array,          # [B, D]
    null_token_feats: jax.Array,  # [B, L, D]
    null_pooled: jax.Array,       # [B, D]
    drop_prob: float,
    key: jax.random.PRNGKey,
) -> tuple[jax.Array, jax.Array]:
    """
    Randomly replace conditioning with null embedding for each sample in batch.
    Called once per training step.

    With probability `drop_prob`, each sample's text embedding is replaced
    by the null embedding — teaching the model to also work unconditionally.
    """
    B = token_feats.shape[0]
    mask = jax.random.bernoulli(key, p=drop_prob, shape=(B,))   # True = drop
    mask_tokens = mask[:, None, None]
    mask_pooled = mask[:, None]

    token_feats = jnp.where(mask_tokens, null_token_feats, token_feats)
    pooled = jnp.where(mask_pooled, null_pooled, pooled)

    return token_feats, pooled


def cfg_guidance(
    model_fn,                   # callable: (x, s, t, token_feats, pooled) -> x_pred
    x: jax.Array,               # [B, H, W, C]
    s: float,
    t: float,
    token_feats: jax.Array,     # [B, L, D]  conditional
    pooled: jax.Array,          # [B, D]
    null_token_feats: jax.Array,
    null_pooled: jax.Array,
    guidance_scale: float = 5.0,
) -> jax.Array:
    """
    Classifier-free guidance at inference.

    output = uncond + scale * (cond - uncond)

    Pass guidance_scale=1.0 to disable guidance (conditional only).
    Pass guidance_scale=0.0 for unconditional only.
    """
    cond_out = model_fn(x, s, t, token_feats, pooled)
    uncond_out = model_fn(x, s, t, null_token_feats, null_pooled)
    return uncond_out + guidance_scale * (cond_out - uncond_out)


# ---------------------------------------------------------------------------
# CIFAR-10 text labels (for training without a caption dataset)
# ---------------------------------------------------------------------------

CIFAR10_PROMPTS = {
    0: "a photo of an airplane",
    1: "a photo of a car",
    2: "a photo of a bird",
    3: "a photo of a cat",
    4: "a photo of a deer",
    5: "a photo of a dog",
    6: "a photo of a frog",
    7: "a photo of a horse",
    8: "a photo of a ship",
    9: "a photo of a truck",
}


def class_to_prompt(class_idx: int) -> str:
    return CIFAR10_PROMPTS[int(class_idx)]
