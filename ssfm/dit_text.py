"""
Text-conditioned DiT backbone for SSFM.

Extends the original DiT with:
  1. Cross-attention to CLIP text token features (per-token, L tokens)
  2. AdaLN-Zero conditioned on (timestep_emb + pooled_text_emb) jointly
  3. Brownian coefficient injection via a small MLP, added to the
     time+text conditioning vector before AdaLN

Architecture per block:
    x → LayerNorm (adaLN-modulated) → Self-Attention → residual
      → LayerNorm (adaLN-modulated) → Cross-Attention(text) → residual
      → LayerNorm (adaLN-modulated) → FFN → residual

The AdaLN parameters (scale, shift, gate) are predicted from a conditioning
vector c = MLP(t_emb + text_pooled_emb + brownian_emb).
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


# ---------------------------------------------------------------------------
# Positional / time embeddings
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    """Scalar or [B] float → [B, dim] sinusoidal embedding."""
    t = jnp.atleast_1d(t)
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)
    args = t[:, None] * freqs[None]           # [B, half]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # [B, dim]


class TimestepEmbedding(eqx.Module):
    """Maps scalar timestep → D-dimensional embedding via sinusoidal + MLP."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.linear1 = eqx.nn.Linear(dim, dim * 4, key=k1)
        self.linear2 = eqx.nn.Linear(dim * 4, dim, key=k2)

    def __call__(self, t: jax.Array) -> jax.Array:
        """t: scalar float → [D]"""
        x = sinusoidal_embedding(jnp.atleast_1d(t), self.dim)[0]  # [D]
        x = jax.nn.silu(self.linear1(x))
        return self.linear2(x)


class BrownianEmbedding(eqx.Module):
    """
    Maps N Brownian coefficient vectors (each [H*W*C]) → single [D] embedding.

    The key insight: the Brownian coefficients I^(n)_{s,t} have decreasing
    variance with n, so we weight them accordingly and project down to D.
    """
    proj: eqx.nn.Linear
    dim: int = eqx.field(static=True)
    n_coeffs: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)

    def __init__(self, n_coeffs: int, spatial_dim: int, dim: int, *, key: jax.random.PRNGKey):
        self.n_coeffs = n_coeffs
        self.spatial_dim = spatial_dim
        self.dim = dim
        self.proj = eqx.nn.Linear(n_coeffs * spatial_dim, dim, key=key)

    def __call__(self, brownian_coeffs: jax.Array) -> jax.Array:
        """
        brownian_coeffs: [N, H*W*C]  — the N polynomial Legendre coefficients
                                       flattened per coefficient
        Returns: [D]
        """
        # Weight by 1/sqrt(2n+1) to account for decreasing variance
        n = jnp.arange(self.n_coeffs, dtype=jnp.float32)
        weights = 1.0 / jnp.sqrt(2 * n + 1)          # [N]
        weighted = brownian_coeffs * weights[:, None]  # [N, spatial_dim]
        flat = weighted.reshape(-1)                    # [N * spatial_dim]
        return jax.nn.silu(self.proj(flat))            # [D]


# ---------------------------------------------------------------------------
# AdaLN-Zero modulation
# ---------------------------------------------------------------------------

class AdaLNZero(eqx.Module):
    """
    Produces (scale, shift, gate) for AdaLN-Zero modulation from a
    conditioning vector c.  Applied as:
        x_mod = gate * LayerNorm(x) * (1 + scale) + shift
    """
    linear: eqx.nn.Linear

    def __init__(self, cond_dim: int, x_dim: int, *, key: jax.random.PRNGKey):
        # 3 * x_dim outputs: scale, shift, gate
        self.linear = eqx.nn.Linear(cond_dim, 3 * x_dim, key=key,
                                     use_bias=True)

    def __call__(self, c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """c: [D_cond] → scale, shift, gate each [D_x]"""
        out = self.linear(c)
        scale, shift, gate = jnp.split(out, 3, axis=-1)
        gate = jax.nn.tanh(gate)   # gate ∈ (-1, 1); initialised near 0
        return scale, shift, gate


def adaLN_modulate(x: jax.Array, ln: eqx.nn.LayerNorm,
                   scale: jax.Array, shift: jax.Array) -> jax.Array:
    return ln(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Cross-attention to text tokens
# ---------------------------------------------------------------------------

class CrossAttention(eqx.Module):
    """Multi-head cross-attention: queries from x, keys/values from text."""
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, x_dim: int, text_dim: int, num_heads: int, *,
                 key: jax.random.PRNGKey):
        assert x_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(x_dim, x_dim, key=k1)
        self.k_proj = eqx.nn.Linear(text_dim, x_dim, key=k2)
        self.v_proj = eqx.nn.Linear(text_dim, x_dim, key=k3)
        self.out_proj = eqx.nn.Linear(x_dim, x_dim, key=k4)

    def __call__(self, x: jax.Array, text: jax.Array,
                 text_mask: Optional[jax.Array] = None) -> jax.Array:
        """
        x:    [N, D]     image patch tokens
        text: [L, D_t]   CLIP text token features
        Returns [N, D]
        """
        N, D = x.shape
        L = text.shape[0]
        H, Hd = self.num_heads, self.head_dim

        Q = jax.vmap(self.q_proj)(x).reshape(N, H, Hd)           # [N, H, Hd]
        K = jax.vmap(self.k_proj)(text).reshape(L, H, Hd)        # [L, H, Hd]
        V = jax.vmap(self.v_proj)(text).reshape(L, H, Hd)        # [L, H, Hd]

        # Scaled dot-product attention
        scale = Hd ** -0.5
        attn = jnp.einsum("nhd,lhd->nhl", Q, K) * scale          # [N, H, L]
        if text_mask is not None:
            attn = jnp.where(text_mask[None, None, :], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("nhl,lhd->nhd", attn, V)                 # [N, H, Hd]
        out = out.reshape(N, D)
        return jax.vmap(self.out_proj)(out)


# ---------------------------------------------------------------------------
# Single DiT block with text cross-attention
# ---------------------------------------------------------------------------

class DiTBlockText(eqx.Module):
    """
    DiT block extended with a cross-attention sublayer for text conditioning.

    Conditioning vector c = t_emb + text_pooled_emb + brownian_emb
    drives three independent AdaLN-Zero modules (self-attn, cross-attn, FFN).
    """
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    self_attn: eqx.nn.MultiheadAttention
    cross_attn: CrossAttention
    ffn1: eqx.nn.Linear
    ffn2: eqx.nn.Linear
    adaLN_self: AdaLNZero
    adaLN_cross: AdaLNZero
    adaLN_ffn: AdaLNZero

    def __init__(self, hidden_dim: int, num_heads: int, text_dim: int,
                 cond_dim: int, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 8)
        self.norm1 = eqx.nn.LayerNorm(hidden_dim, use_bias=False, use_weight=False)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim, use_bias=False, use_weight=False)
        self.norm3 = eqx.nn.LayerNorm(hidden_dim, use_bias=False, use_weight=False)
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=hidden_dim, key=keys[0])
        self.cross_attn = CrossAttention(
            x_dim=hidden_dim, text_dim=text_dim, num_heads=num_heads, key=keys[1])
        self.ffn1 = eqx.nn.Linear(hidden_dim, hidden_dim * 4, key=keys[2])
        self.ffn2 = eqx.nn.Linear(hidden_dim * 4, hidden_dim, key=keys[3])
        self.adaLN_self = AdaLNZero(cond_dim, hidden_dim, key=keys[4])
        self.adaLN_cross = AdaLNZero(cond_dim, hidden_dim, key=keys[5])
        self.adaLN_ffn = AdaLNZero(cond_dim, hidden_dim, key=keys[6])

    def __call__(self, x: jax.Array, c: jax.Array,
                 text_tokens: jax.Array,
                 text_mask: Optional[jax.Array] = None) -> jax.Array:
        """
        x:           [N, D]   image patch tokens
        c:           [D_c]    conditioning vector (time + text_pooled + brownian)
        text_tokens: [L, D_t] CLIP text token features
        """
        # --- 1. Self-attention with AdaLN ---
        s1, sh1, g1 = self.adaLN_self(c)
        x_mod = adaLN_modulate(x, self.norm1, s1, sh1)
        x = x + g1 * self.self_attn(x_mod, x_mod, x_mod)

        # --- 2. Cross-attention to text ---
        s2, sh2, g2 = self.adaLN_cross(c)
        x_mod = adaLN_modulate(x, self.norm2, s2, sh2)
        x = x + g2 * self.cross_attn(x_mod, text_tokens, text_mask)

        # --- 3. FFN ---
        s3, sh3, g3 = self.adaLN_ffn(c)
        x_mod = adaLN_modulate(x, self.norm3, s3, sh3)
        ffn_out = jax.vmap(lambda xi: self.ffn2(jax.nn.gelu(self.ffn1(xi))))(x_mod)
        x = x + g3 * ffn_out

        return x


# ---------------------------------------------------------------------------
# Full text-conditioned DiT
# ---------------------------------------------------------------------------

class TextConditionedDiT(eqx.Module):
    """
    Full DiT for text-conditioned SSFM on images.

    Inputs:
        x_noisy:          [H, W, C]          noisy image at time s
        brownian_coeffs:  [N, H*W*C]         Legendre coefficients
        s:                float              start time
        t:                float              end time
        text_tokens:      [L, D_text]        CLIP per-token features (single sample)
        text_pooled:      [D_text]           CLIP pooled embedding (single sample)

    Output:
        [H, W, C]  predicted image at time t
    """
    # Patch embedding
    patch_embed: eqx.nn.Conv2d
    # Time embeddings for s and t independently
    s_embed: TimestepEmbedding
    t_embed: TimestepEmbedding
    # Text projection (CLIP dim → hidden_dim)
    text_token_proj: eqx.nn.Linear
    text_pooled_proj: eqx.nn.Linear
    # Brownian embedding
    brownian_embed: BrownianEmbedding
    # Conditioning MLP: combines t_emb, s_emb, text_pooled_emb, brownian_emb → cond_dim
    cond_mlp1: eqx.nn.Linear
    cond_mlp2: eqx.nn.Linear
    # Transformer blocks
    blocks: list
    # Final projection
    final_norm: eqx.nn.LayerNorm
    final_adaLN: AdaLNZero
    final_linear: eqx.nn.Linear
    # Unpatch
    unpatch: eqx.nn.ConvTranspose2d

    # Static fields
    patch_size: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    img_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        text_dim: int = 512,      # CLIP ViT-B/32 dimension
        n_brownian_coeffs: int = 4,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 20)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.in_channels = in_channels

        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        cond_dim = hidden_dim

        # Patchify
        self.patch_embed = eqx.nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size, key=keys[0])

        # Timestep embeddings (separate for s and t)
        self.s_embed = TimestepEmbedding(hidden_dim, key=keys[1])
        self.t_embed = TimestepEmbedding(hidden_dim, key=keys[2])

        # Text projections
        self.text_token_proj = eqx.nn.Linear(text_dim, hidden_dim, key=keys[3])
        self.text_pooled_proj = eqx.nn.Linear(text_dim, hidden_dim, key=keys[4])

        # Brownian embedding
        spatial_dim = img_size * img_size * in_channels
        self.brownian_embed = BrownianEmbedding(
            n_brownian_coeffs, spatial_dim, hidden_dim, key=keys[5])

        # Conditioning MLP: [4 * hidden_dim] → cond_dim
        self.cond_mlp1 = eqx.nn.Linear(4 * hidden_dim, 2 * hidden_dim, key=keys[6])
        self.cond_mlp2 = eqx.nn.Linear(2 * hidden_dim, cond_dim, key=keys[7])

        # Transformer blocks
        self.blocks = [
            DiTBlockText(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                text_dim=hidden_dim,   # after projection
                cond_dim=cond_dim,
                key=keys[8 + i],
            )
            for i in range(num_layers)
        ]

        # Final layer
        self.final_norm = eqx.nn.LayerNorm(hidden_dim, use_bias=False, use_weight=False)
        self.final_adaLN = AdaLNZero(cond_dim, hidden_dim, key=keys[8 + num_layers])
        self.final_linear = eqx.nn.Linear(
            hidden_dim, patch_size * patch_size * in_channels,
            key=keys[9 + num_layers])
        self.unpatch = eqx.nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=patch_size, stride=patch_size,
            key=keys[10 + num_layers])

    def __call__(
        self,
        x_noisy: jax.Array,         # [H, W, C]
        brownian_coeffs: jax.Array,  # [N, H*W*C]
        s: jax.Array,                # scalar
        t: jax.Array,                # scalar
        text_tokens: jax.Array,      # [L, D_text]
        text_pooled: jax.Array,      # [D_text]
    ) -> jax.Array:                  # [H, W, C]

        H, W, C = x_noisy.shape
        P = self.patch_size
        n_p = H // P  # patches per side

        # 1. Patchify: [H, W, C] → [N_patches, hidden_dim]
        x = jnp.moveaxis(x_noisy, -1, 0)              # [C, H, W]
        x = self.patch_embed(x)                         # [D, n_p, n_p]
        x = jnp.moveaxis(x, 0, -1)                     # [n_p, n_p, D]
        x = x.reshape(n_p * n_p, self.hidden_dim)       # [N, D]

        # 2. Build conditioning vector
        s_emb = self.s_embed(s)                          # [D]
        t_emb = self.t_embed(t)                          # [D]
        text_p_emb = self.text_pooled_proj(text_pooled)  # [D]
        bm_emb = self.brownian_embed(brownian_coeffs)    # [D]

        cond_in = jnp.concatenate([s_emb, t_emb, text_p_emb, bm_emb])  # [4D]
        c = jax.nn.silu(self.cond_mlp1(cond_in))
        c = self.cond_mlp2(c)                            # [D]

        # 3. Project text tokens to hidden_dim
        text_proj = jax.vmap(self.text_token_proj)(text_tokens)  # [L, D]

        # 4. Transformer blocks
        for block in self.blocks:
            x = block(x, c, text_proj)                  # [N, D]

        # 5. Final projection back to pixels
        s_f, sh_f, _ = self.final_adaLN(c)
        x = adaLN_modulate(x, self.final_norm, s_f, sh_f)
        x = jax.vmap(self.final_linear)(x)              # [N, P*P*C]

        # 6. Unpatchify
        x = x.reshape(n_p, n_p, P, P, C)
        x = jnp.einsum("ijpqc->ipcjqc", x).reshape(H, W, C)  # [H, W, C]

        return x
