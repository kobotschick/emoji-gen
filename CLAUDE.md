# emoji-gen — Claude Code context

## Project
Text-conditioned extension of Strong Stochastic Flow Maps (arxiv 2606.01086).
Generates emoji images from text prompts using CLIP + DiT + CFG.

## Architecture
- `ssfm/text_conditioning.py` — frozen CLIP ViT-B/32 encoder, CFG dropout, null embedding
- `ssfm/dit_text.py`          — DiT with Self-Attn → Cross-Attn(CLIP) → FFN per block
- `ssfm/text_ssfm.py`         — TextSSFM model, matching + distillation losses
- `data/emoji_metadata.py`    — 446 emoji with CLDR names/keywords
- `data/build_dataset.py`     — renders via Noto Color Emoji, 8 augmentations
- `experiments/cifar10_text/` — training loop + sampling script

## Branch
Active development: `claude/text-conditioned-ssfm-VnVyS`

## Environment
- RunPod pod: `pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8.1)
- Setup: `bash setup.sh --anthropic-key ... --wandb-key ...`
- Train: `python experiments/cifar10_text/main.py`
- Sample: `python experiments/cifar10_text/sample.py --prompt "..." --guidance-scale 5.0`

## AdaLN conditioning vector
`c = MLP(t_emb ‖ s_emb ‖ text_pooled ‖ brownian_emb)`

## CFG
- Training: 10% dropout to null embedding
- Inference: `output = uncond + 5.0 * (cond - uncond)`

## Current status
<!-- Update this section as work progresses -->
- [x] Core architecture implemented
- [x] Docker + RunPod setup
- [ ] Training run on emoji dataset
