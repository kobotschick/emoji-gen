# Text-Conditioned SSFM

Extension of [Strong Stochastic Flow Maps](https://arxiv.org/abs/2606.01086) with CLIP text conditioning and classifier-free guidance.

## What changed vs. the original

### 1. Text conditioning in the DiT backbone (`ssfm/dit_text.py`)

Each DiT block gains a **cross-attention sublayer** between the self-attention and FFN:

```
Original block:           Text-conditioned block:
  Self-Attn (AdaLN)         Self-Attn   (AdaLN)
  FFN       (AdaLN)         Cross-Attn  (AdaLN)  ← new: attends to CLIP tokens
                            FFN         (AdaLN)
```

The AdaLN conditioning vector `c` is now a fusion of four signals:

```
c = MLP( t_embed(t) ‖ t_embed(s) ‖ text_proj(pooled) ‖ brownian_embed(I^N) )
```

The Brownian coefficients remain first-class inputs — not folded into the text conditioning — so pathwise consistency is preserved.

### 2. Text encoder (`ssfm/text_conditioning.py`)

- Frozen CLIP ViT-B/32 text encoder (loaded via `open_clip`)
- Produces `(token_feats [L, D], pooled [D])` per prompt
- Per-token features go to cross-attention; pooled goes to AdaLN

### 3. Classifier-free guidance (`ssfm/text_conditioning.py`)

**Training**: each sample's text embedding is randomly replaced with a learned null embedding with probability `cfg_drop_prob=0.1`.

**Inference**: run the flow map twice per step — conditional and unconditional — then combine:

```
output = uncond + guidance_scale × (cond − uncond)
```

### 4. Training objective (`ssfm/text_ssfm.py`)

Identical to the original Algorithm 1 — the SSFM loss is unchanged:

```
L = L_f,g  +  L_D
```

Text conditioning is just an additional input threaded through both the matching and distillation terms. The semigroup property and Chen relations are unaffected.

---

## Installation

```bash
git clone https://github.com/sammccallum/ssfm
cd ssfm
# Copy the new files in
uv sync
uv sync --extra metrics   # for FID evaluation
```

Additional dependency for CLIP:
```bash
uv add open-clip-torch
```

---

## Training

```bash
uv run python experiments/cifar10_text/main.py
```

CIFAR-10 trains with class names as prompts:

| Class | Prompt |
|-------|--------|
| airplane | "a photo of an airplane" |
| dog | "a photo of a dog" |
| … | … |

For a real text-image dataset (e.g. CC3M, LAION), replace `batch_to_jax` with your own caption loader.

---

## Sampling

```bash
uv run python experiments/cifar10_text/sample.py \
    --checkpoint experiments/cifar10_text/checkpoints/model_ema_final.eqx \
    --prompt "a photo of a red car" \
    --n-samples 8 \
    --guidance-scale 5.0 \
    --steps 2
```

`--steps 1` uses a single network call (s=0 → t=1). `--steps 2` splits at the midpoint and applies Chen combination for the Brownian path.

---

## Design decisions and tradeoffs

| Decision | Choice | Alternative |
|----------|--------|-------------|
| Text encoder | Frozen CLIP ViT-B/32 | Trainable T5 / learned class table |
| Text injection | Cross-attention + AdaLN pooled | AdaLN only (simpler, weaker) |
| CFG | Standard binary dropout | Continuous conditioning scale |
| Brownian coefficients | N=4 Legendre (same as original) | More coefficients for larger steps |
| Dataset | CIFAR-10 class names | Any captioned image dataset |

### Why cross-attention and not just AdaLN on the pooled embedding?

AdaLN on pooled text works fine for class-conditional generation (10 classes). For free-text prompts, you want the model to attend to individual words — "red car" vs. "blue car" differ at the token level, not just in the pooled vector. Cross-attention lets each image patch selectively attend to the relevant text tokens.

### Why keep CLIP frozen?

The Brownian coefficient inputs already make the training signal unusual — adding a trainable text encoder would destabilize training, especially early on. CLIP embeddings are strong priors and fine-tuning them on CIFAR-10 scale data would likely hurt generalization.

---

## File structure

```
ssfm/
  text_conditioning.py    CLIP encoder, null embedding, CFG utilities
  dit_text.py             Text-conditioned DiT backbone
  text_ssfm.py            TextSSFM model + training losses + sampler

experiments/cifar10_text/
  main.py                 Training loop
  sample.py               Inference with free-text prompts
```
