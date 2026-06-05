"""
Text-conditioned SSFM.

Wraps the TextConditionedDiT backbone in the SSFM parameterisation:

    Ψθ_{s,t}(Xₛ, I^(N)_{s,t}, c) = Xₛ
                                   + f^θ_{s,t}(Xₛ, I^(N)_{s,t}, c) · (t - s)
                                   + g^θ_{s,t}(I^(N)_{s,t}, c) · (Wt - Ws)

where c = (text_tokens, text_pooled) is the CLIP text conditioning.

Two networks share the same DiT backbone but with different output heads:
  - f_net: predicts the drift integral (depends on Xₛ)
  - g_net: predicts the diffusion integral (depends only on path, not Xₛ)

In practice we use a single network and split the output channels.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import NamedTuple

from .dit_text import TextConditionedDiT
from .text_conditioning import NullEmbedding, cfg_dropout, cfg_guidance


# ---------------------------------------------------------------------------
# Brownian coefficient utilities  (mirror ssfm/diffusions.py conventions)
# ---------------------------------------------------------------------------

def sample_brownian_coeffs(
    s: float,
    t: float,
    n_coeffs: int,
    spatial_shape: tuple,
    key: jax.random.PRNGKey,
) -> jax.Array:
    """
    Sample I^(N)_{s,t} ~ N(0, (t-s)/(2n+1) · I) independently per coefficient.

    Returns [N, prod(spatial_shape)] flattened.
    """
    spatial_dim = 1
    for d in spatial_shape:
        spatial_dim *= d

    keys = jax.random.split(key, n_coeffs)
    coeffs = jnp.stack([
        jax.random.normal(keys[n], (spatial_dim,)) * jnp.sqrt((t - s) / (2 * n + 1))
        for n in range(n_coeffs)
    ])  # [N, spatial_dim]
    return coeffs


def chen_combine(
    coeffs_su: jax.Array,   # [N, D]  coefficients over [s, u]
    coeffs_ut: jax.Array,   # [N, D]  coefficients over [u, t]
    s: float, u: float, t: float,
) -> jax.Array:
    """
    Chen relations for Legendre polynomial approximation.

    The zeroth coefficient (Brownian increment) combines additively:
        I^(0)_{s,t} = I^(0)_{s,u} + I^(0)_{u,t}

    Higher coefficients follow the Chen shuffle product. For the piecewise
    polynomial case with Legendre basis, the combination is a linear map
    determined by (s, u, t) — here we implement the first two orders exactly
    and use a linear approximation for higher orders.

    Reference: Foster et al. (2020), Theorem 3 / Eq. (22) in the SSFM paper.
    """
    N = coeffs_su.shape[0]
    h1 = u - s
    h2 = t - u
    h = t - s

    combined = []
    for n in range(N):
        if n == 0:
            # I^(0) = Brownian increment — simply additive
            combined.append(coeffs_su[0] + coeffs_ut[0])
        elif n == 1:
            # I^(1) = ∫ (2u-1) dW  — Chen gives cross term
            # c_{s,t}^(1) = (h1/h)·c_{s,u}^(1) + (h2/h)·c_{u,t}^(1)
            #             + (h1*h2/h²) · correction from I^(0) terms
            alpha = h1 / h
            beta = h2 / h
            cross = (h1 * h2 / h**2) * (coeffs_su[0] - coeffs_ut[0]) * 0.5
            combined.append(alpha * coeffs_su[1] + beta * coeffs_ut[1] + cross)
        else:
            # Higher orders: linear combination (approximate)
            alpha = (h1 / h) ** (n + 0.5)
            beta = (h2 / h) ** (n + 0.5)
            combined.append(alpha * coeffs_su[n] + beta * coeffs_ut[n])

    return jnp.stack(combined)  # [N, D]


# ---------------------------------------------------------------------------
# The SSFM flow map
# ---------------------------------------------------------------------------

class TextSSFM(eqx.Module):
    """
    Text-conditioned Strong Stochastic Flow Map.

    Uses a single DiT backbone that outputs 2*C channels:
        - first C channels  → drift integral f^θ
        - last  C channels  → diffusion integral g^θ

    The SSFM parameterisation then gives:
        Ψθ_{s,t}(Xₛ, I, c) = Xₛ + f^θ·(t-s) + g^θ·I^(0)_{s,t}

    where I^(0)_{s,t} = Wt - Ws is the zeroth Legendre coefficient.
    """
    backbone: TextConditionedDiT
    null_embedding: NullEmbedding

    img_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    n_brownian_coeffs: int = eqx.field(static=True)
    cfg_drop_prob: float = eqx.field(static=True)

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        text_dim: int = 512,
        n_brownian_coeffs: int = 4,
        context_length: int = 77,
        cfg_drop_prob: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2 = jax.random.split(key)
        self.backbone = TextConditionedDiT(
            img_size=img_size,
            in_channels=in_channels * 2,      # 2× channels: f and g heads
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            text_dim=text_dim,
            n_brownian_coeffs=n_brownian_coeffs,
            key=k1,
        )
        self.null_embedding = NullEmbedding(context_length, text_dim, key=k2)
        self.img_size = img_size
        self.in_channels = in_channels
        self.n_brownian_coeffs = n_brownian_coeffs
        self.cfg_drop_prob = cfg_drop_prob

    def _split_output(self, out: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Split [H, W, 2C] backbone output into (f_integral, g_integral)."""
        f, g = jnp.split(out, 2, axis=-1)
        return f, g

    def flow_map(
        self,
        x_s: jax.Array,             # [H, W, C]
        brownian_coeffs: jax.Array,  # [N, H*W*C]
        s: jax.Array,
        t: jax.Array,
        text_tokens: jax.Array,      # [L, D_text]
        text_pooled: jax.Array,      # [D_text]
    ) -> jax.Array:                  # [H, W, C]
        """
        Single-sample flow map evaluation: Ψθ_{s,t}(Xₛ, I, c)
        """
        H, W, C = x_s.shape
        # Expand noisy image to 2C for backbone input (f and g share the same x_s)
        x_in = jnp.concatenate([x_s, x_s], axis=-1)  # [H, W, 2C]

        out = self.backbone(x_in, brownian_coeffs, s, t, text_tokens, text_pooled)
        f_int, g_int = self._split_output(out)  # each [H, W, C]

        # Wt - Ws = I^(0)_{s,t}  (zeroth Legendre coefficient reshaped)
        wt_ws = brownian_coeffs[0].reshape(H, W, C)

        return x_s + f_int * (t - s) + g_int * wt_ws

    def flow_map_batched(
        self,
        x_s: jax.Array,              # [B, H, W, C]
        brownian_coeffs: jax.Array,  # [B, N, H*W*C]
        s: jax.Array,                # [B] or scalar
        t: jax.Array,                # [B] or scalar
        text_tokens: jax.Array,      # [B, L, D_text]
        text_pooled: jax.Array,      # [B, D_text]
    ) -> jax.Array:                  # [B, H, W, C]
        return jax.vmap(self.flow_map)(x_s, brownian_coeffs, s, t, text_tokens, text_pooled)


# ---------------------------------------------------------------------------
# Training losses  (Algorithm 1 from the paper, extended for text)
# ---------------------------------------------------------------------------

class LossComponents(NamedTuple):
    loss_fg: jax.Array      # matching loss (diagonal)
    loss_D: jax.Array       # distillation loss (semigroup)
    loss_total: jax.Array


def matching_loss(
    model: TextSSFM,
    x_1: jax.Array,          # [B, H, W, C]  real images
    x_0: jax.Array,          # [B, H, W, C]  Gaussian noise
    s: jax.Array,            # [B]
    t: jax.Array,            # [B]  t is near s  (t < s + Δt)
    brownian_coeffs: jax.Array,  # [B, N, H*W*C]
    text_tokens: jax.Array,  # [B, L, D]
    text_pooled: jax.Array,  # [B, D]
    alpha: callable,         # αₜ schedule
    sigma: callable,         # σₜ schedule
    f_sde: callable,         # f(t, Xₜ, X₁): drift of diffusion SDE
    g_sde: callable,         # g(t): diffusion coefficient
    null_token_feats: jax.Array,
    null_pooled: jax.Array,
    cfg_key: jax.random.PRNGKey,
) -> jax.Array:
    """
    L_f,g: network output at s≈t must match one Euler-Maruyama step.

    Target: X̂ₜ = Xₛ + f(s, Xₛ, X₁)·(t-s) + g(s)·(Wt-Ws)
    """
    B, H, W, C = x_1.shape

    # Interpolate to get Xₛ
    alpha_s = alpha(s)[:, None, None, None]
    sigma_s = sigma(s)[:, None, None, None]
    x_s = alpha_s * x_1 + sigma_s * x_0   # [B, H, W, C]

    # Euler-Maruyama target
    wt_ws = brownian_coeffs[:, 0].reshape(B, H, W, C)
    drift = jax.vmap(lambda xi, x1i, si: f_sde(si, xi, x1i))(x_s, x_1, s)
    diff = jax.vmap(g_sde)(s)[:, None, None, None]
    x_t_target = x_s + drift * (t - s)[:, None, None, None] + diff * wt_ws

    # CFG dropout
    text_tokens_c, text_pooled_c = cfg_dropout(
        text_tokens, text_pooled, null_token_feats, null_pooled,
        model.cfg_drop_prob, cfg_key)

    # Network prediction
    x_t_pred = model.flow_map_batched(
        x_s, brownian_coeffs, s, t, text_tokens_c, text_pooled_c)

    # Loss: MSE scaled by 1/(t-s)
    dt = (t - s)[:, None, None, None]
    loss = jnp.mean((x_t_target - x_t_pred) ** 2 / dt)
    return loss


def distillation_loss(
    model: TextSSFM,
    model_ema: TextSSFM,       # EMA (frozen) network
    x_1: jax.Array,            # [B, H, W, C]
    x_0: jax.Array,            # [B, H, W, C]
    s: jax.Array,              # [B]
    t: jax.Array,              # [B]  t >> s  (t > s + Δt)
    brownian_su: jax.Array,    # [B, N, H*W*C]  path over [s, u]
    brownian_ut: jax.Array,    # [B, N, H*W*C]  path over [u, t]
    brownian_st: jax.Array,    # [B, N, H*W*C]  combined via Chen, over [s, t]
    u: jax.Array,              # [B]  midpoint  u = (s+t)/2
    text_tokens: jax.Array,    # [B, L, D]
    text_pooled: jax.Array,    # [B, D]
    alpha: callable,
    sigma: callable,
    null_token_feats: jax.Array,
    null_pooled: jax.Array,
    cfg_key: jax.random.PRNGKey,
) -> jax.Array:
    """
    L_D: semigroup consistency.

    Target (EMA one-hop):  X_tgt = stop_grad( Ψ_EMA(Xₛ, I_{s,t}, c) )
    Pred  (live two-hop):  X_pred = Ψθ( Ψθ(Xₛ, I_{s,u}, c), I_{u,t}, c )
    """
    B, H, W, C = x_1.shape

    # Interpolate Xₛ
    alpha_s = alpha(s)[:, None, None, None]
    sigma_s = sigma(s)[:, None, None, None]
    x_s = alpha_s * x_1 + sigma_s * x_0

    # CFG dropout — same mask for both hops (same path, same conditioning)
    text_tokens_c, text_pooled_c = cfg_dropout(
        text_tokens, text_pooled, null_token_feats, null_pooled,
        model.cfg_drop_prob, cfg_key)

    # One-hop target via EMA network (no gradient)
    x_tgt = jax.lax.stop_gradient(
        model_ema.flow_map_batched(
            x_s, brownian_st, s, t, text_tokens_c, text_pooled_c)
    )

    # Two-hop prediction via live network
    x_u = model.flow_map_batched(x_s, brownian_su, s, u, text_tokens_c, text_pooled_c)
    x_pred = model.flow_map_batched(x_u, brownian_ut, u, t, text_tokens_c, text_pooled_c)

    # Loss: MSE scaled by 1/(t-s)
    dt = (t - s)[:, None, None, None]
    loss = jnp.mean((x_tgt - x_pred) ** 2 / dt)
    return loss


def total_loss(
    model: TextSSFM,
    model_ema: TextSSFM,
    batch: dict,
    key: jax.random.PRNGKey,
    alpha: callable,
    sigma: callable,
    f_sde: callable,
    g_sde: callable,
    eta: float = 0.5,          # fraction of batch used for matching loss
    delta_t: float = 0.05,     # threshold between short/long intervals
) -> LossComponents:
    """
    Combined training loss: L = L_f,g + L_D

    batch keys: x_1 [B,H,W,C], text_tokens [B,L,D], text_pooled [B,D],
                class_labels (optional)
    """
    x_1 = batch["x_1"]
    text_tokens = batch["text_tokens"]
    text_pooled = batch["text_pooled"]
    B, H, W, C = x_1.shape

    keys = jax.random.split(key, 8)

    # Sample noise
    x_0 = jax.random.normal(keys[0], x_1.shape)

    # Null embeddings for CFG dropout
    null_tokens, null_pooled = model.null_embedding.expand(B)

    # ── Matching loss (η fraction of batch) ──────────────────────────────
    B_fg = max(1, int(eta * B))
    s_fg = jax.random.uniform(keys[1], (B_fg,))
    t_fg = s_fg + jax.random.uniform(keys[2], (B_fg,)) * delta_t
    t_fg = jnp.clip(t_fg, 0, 1)

    coeffs_fg = jax.vmap(
        lambda k, si, ti: sample_brownian_coeffs(si, ti, model.n_brownian_coeffs, (H * W * C,), k)
    )(jax.random.split(keys[3], B_fg), s_fg, t_fg)

    l_fg = matching_loss(
        model,
        x_1[:B_fg], x_0[:B_fg],
        s_fg, t_fg, coeffs_fg,
        text_tokens[:B_fg], text_pooled[:B_fg],
        alpha, sigma, f_sde, g_sde,
        null_tokens[:B_fg], null_pooled[:B_fg],
        keys[4],
    )

    # ── Distillation loss (remaining batch) ──────────────────────────────
    B_D = B - B_fg
    s_D = jax.random.uniform(keys[5], (B_D,))
    t_D = s_D + delta_t + jax.random.uniform(keys[5], (B_D,)) * (1 - s_D - delta_t)
    t_D = jnp.clip(t_D, 0, 1)
    u_D = 0.5 * (s_D + t_D)

    def _sample_pair(k, si, ui, ti):
        k1, k2 = jax.random.split(k)
        c_su = sample_brownian_coeffs(si, ui, model.n_brownian_coeffs, (H * W * C,), k1)
        c_ut = sample_brownian_coeffs(ui, ti, model.n_brownian_coeffs, (H * W * C,), k2)
        c_st = chen_combine(c_su, c_ut, si, ui, ti)
        return c_su, c_ut, c_st

    c_su, c_ut, c_st = jax.vmap(_sample_pair)(
        jax.random.split(keys[6], B_D), s_D, u_D, t_D)

    l_D = distillation_loss(
        model, model_ema,
        x_1[B_fg:], x_0[B_fg:],
        s_D, t_D, c_su, c_ut, c_st, u_D,
        text_tokens[B_fg:], text_pooled[B_fg:],
        alpha, sigma,
        null_tokens[B_fg:], null_pooled[B_fg:],
        keys[7],
    )

    return LossComponents(l_fg, l_D, l_fg + l_D)


# ---------------------------------------------------------------------------
# Inference / sampling
# ---------------------------------------------------------------------------

def sample(
    model: TextSSFM,
    prompt_tokens: jax.Array,    # [L, D_text]
    prompt_pooled: jax.Array,    # [D_text]
    img_size: int,
    in_channels: int,
    n_steps: int = 2,
    guidance_scale: float = 5.0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> jax.Array:
    """
    Generate a single image from a text prompt.

    For n_steps=1: single call s=0 → t=1
    For n_steps=2: s=0 → 0.5 → 1  (using Chen-combined paths at each hop)

    Returns [H, W, C] float in [-1, 1] approx.
    """
    H = W = img_size
    C = in_channels
    spatial_dim = H * W * C
    N = model.n_brownian_coeffs

    null_tokens, null_pooled = model.null_embedding.expand(1)
    null_tokens = null_tokens[0]   # [L, D]
    null_pooled = null_pooled[0]   # [D]

    # Sample initial noise
    key, k0 = jax.random.split(key)
    x = jax.random.normal(k0, (H, W, C))

    # Build time schedule
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    for i in range(n_steps):
        s, t = float(ts[i]), float(ts[i + 1])
        key, k_bm = jax.random.split(key)
        coeffs = sample_brownian_coeffs(s, t, N, (spatial_dim,), k_bm)

        # CFG: run conditional and unconditional, combine
        x = cfg_guidance(
            model_fn=lambda xi, si, ti, tok, pool: model.flow_map(
                xi, coeffs, jnp.array(si), jnp.array(ti), tok, pool),
            x=x, s=s, t=t,
            token_feats=prompt_tokens, pooled=prompt_pooled,
            null_token_feats=null_tokens, null_pooled=null_pooled,
            guidance_scale=guidance_scale,
        )

    return x
