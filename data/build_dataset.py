"""
Emoji dataset builder.

Renders each emoji from the metadata at multiple sizes and augmentations,
pairs each image with a set of natural-language descriptions, and saves
the dataset to disk.

Output structure:
    data/emoji_dataset/
        images/
            0000_grinning_face_aug0.png
            0000_grinning_face_aug1.png
            ...
        metadata.json      { image_path, emoji, name, description, keywords }
        captions.csv       image_path, caption   (one row per augmentation)
        index.json         full index for DataLoader

Usage:
    python data/build_dataset.py --size 64 --augmentations 8 --out data/emoji_dataset
"""

import json
import csv
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# Load our metadata
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.emoji_metadata import EMOJI_DATA


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

NOTO_EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
# Noto renders at a fixed internal size of ~109px; we render large then resize
RENDER_SIZE = 256
FONT_SIZE = 109   # Noto Color Emoji's native resolution


def load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(NOTO_EMOJI_FONT, size)


def render_emoji(char: str, out_size: int, font: ImageFont.FreeTypeFont,
                 bg: tuple = (255, 255, 255, 0)) -> Image.Image:
    """
    Render a single emoji character to an RGBA image of `out_size` x `out_size`.
    """
    # Render at RENDER_SIZE for quality, then downscale
    canvas = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), bg)
    draw = ImageDraw.Draw(canvas)
    # Centre the glyph
    bbox = draw.textbbox((0, 0), char, font=font, embedded_color=True)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (RENDER_SIZE - w) // 2 - bbox[0]
    y = (RENDER_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, embedded_color=True)
    return canvas.resize((out_size, out_size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def augment(img: Image.Image, seed: int) -> Image.Image:
    """
    Apply a deterministic set of light augmentations to an emoji image.
    Keeps the emoji recognizable but adds variety for training.

    Augmentations applied (light — emojis are already canonical):
      - Small rotation (±15°)
      - Slight scale jitter (0.85 – 1.0 of canvas)
      - Brightness / contrast jitter
      - Horizontal flip (50%)
      - White / coloured background (vs transparent)
    """
    rng = random.Random(seed)
    size = img.size[0]

    # 1. Background fill
    bg_choice = rng.choice(["white", "light_grey", "pastel", "transparent"])
    if bg_choice == "white":
        bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    elif bg_choice == "light_grey":
        bg = Image.new("RGBA", (size, size), (245, 245, 245, 255))
    elif bg_choice == "pastel":
        r = rng.randint(200, 255)
        g = rng.randint(200, 255)
        b = rng.randint(200, 255)
        bg = Image.new("RGBA", (size, size), (r, g, b, 255))
    else:
        bg = Image.new("RGBA", (size, size), (255, 255, 255, 0))

    # 2. Scale jitter
    scale = rng.uniform(0.80, 1.0)
    new_size = int(size * scale)
    scaled = img.resize((new_size, new_size), Image.LANCZOS)
    offset_x = rng.randint(0, size - new_size)
    offset_y = rng.randint(0, size - new_size)
    bg.paste(scaled, (offset_x, offset_y), scaled)
    result = bg

    # 3. Rotation
    angle = rng.uniform(-15, 15)
    result = result.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 4. Horizontal flip
    if rng.random() < 0.5:
        result = result.transpose(Image.FLIP_LEFT_RIGHT)

    # 5. Brightness / contrast
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.90, 1.10)
    result_rgb = result.convert("RGB")
    result_rgb = ImageEnhance.Brightness(result_rgb).enhance(brightness)
    result_rgb = ImageEnhance.Contrast(result_rgb).enhance(contrast)

    return result_rgb.convert("RGB")


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

CAPTION_TEMPLATES = [
    "an emoji of {name}",
    "a {name} emoji",
    "{name}",
    "emoji showing {name}",
    "a colorful emoji depicting {name}",
    "{description}",
    "an emoji: {description}",
    "pixel art of {name}",
    "a small illustration of {name}",
    "icon of {name}",
    "{name} icon",
    "cartoon {name}",
    "emoji icon: {name}",
]

def generate_captions(name: str, description: str, keywords: list[str], n: int) -> list[str]:
    """
    Generate `n` distinct natural-language captions for an emoji.
    Uses templates + keyword injection.
    """
    rng = random.Random(abs(hash(name)))
    captions = set()

    # Template-based
    for tmpl in CAPTION_TEMPLATES:
        cap = tmpl.format(name=name, description=description)
        captions.add(cap)

    # Keyword combinations
    for kw in keywords:
        captions.add(f"a {kw} emoji")
        captions.add(f"emoji: {kw}")

    # Shuffle and pick n
    caps = list(captions)
    rng.shuffle(caps)
    # Pad with repeats if needed
    while len(caps) < n:
        caps += caps
    return caps[:n]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataset(
    out_dir: str,
    img_size: int = 64,
    n_augmentations: int = 8,
    seed: int = 42,
):
    out_path = Path(out_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(FONT_SIZE)
    random.seed(seed)

    metadata = []
    captions_rows = []   # for captions.csv
    index = []           # for index.json

    print(f"Building emoji dataset: {len(EMOJI_DATA)} emoji × {n_augmentations} augmentations")
    print(f"Output size: {img_size}×{img_size}, Output dir: {out_path}")

    skipped = 0

    for idx, (char, name, description, keywords) in enumerate(EMOJI_DATA):
        # Render base image
        try:
            base_img = render_emoji(char, RENDER_SIZE, font)
        except Exception as e:
            print(f"  SKIP {char} ({name}): {e}")
            skipped += 1
            continue

        # Check it's not blank (some emoji may not render)
        arr = np.array(base_img)
        if arr.shape[-1] == 4 and arr[:, :, 3].mean() < 5:
            print(f"  SKIP {char} ({name}): blank render")
            skipped += 1
            continue

        # Generate captions
        caps = generate_captions(name, description, keywords, n_augmentations)

        safe_name = name.replace(" ", "_").replace("/", "_")[:40]
        emoji_meta = {
            "emoji_idx": idx,
            "char": char,
            "name": name,
            "description": description,
            "keywords": keywords,
            "augmentations": [],
        }

        for aug_i in range(n_augmentations):
            aug_seed = seed * 10000 + idx * 100 + aug_i
            aug_img = augment(base_img, aug_seed)

            # Final resize to target size
            final_img = aug_img.resize((img_size, img_size), Image.LANCZOS)

            fname = f"{idx:04d}_{safe_name}_aug{aug_i:02d}.png"
            fpath = img_dir / fname
            final_img.save(fpath)

            caption = caps[aug_i % len(caps)]
            rel_path = f"images/{fname}"

            emoji_meta["augmentations"].append({
                "image_path": rel_path,
                "caption": caption,
                "aug_seed": aug_seed,
            })

            captions_rows.append({
                "image_path": rel_path,
                "caption": caption,
                "emoji": char,
                "name": name,
            })

            index.append({
                "image_path": rel_path,
                "caption": caption,
                "name": name,
                "emoji_idx": idx,
            })

        metadata.append(emoji_meta)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(EMOJI_DATA)} emoji rendered...")

    # Save metadata
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(out_path / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    with open(out_path / "captions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption", "emoji", "name"])
        writer.writeheader()
        writer.writerows(captions_rows)

    total = len(EMOJI_DATA) - skipped
    print(f"\nDone. {total} emoji × {n_augmentations} aug = {total * n_augmentations} images.")
    print(f"Skipped: {skipped}  |  Output: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Dataset loader  (PyTorch / numpy compatible)
# ---------------------------------------------------------------------------

class EmojiDataset:
    """
    Simple emoji dataset for the text-conditioned SSFM.

    Each item: { "image": [H, W, 3] float32 in [-1, 1],
                 "caption": str,
                 "name": str }

    Compatible with both PyTorch DataLoader and JAX data pipelines.
    """

    def __init__(self, dataset_dir: str, img_size: int = 64):
        self.root = Path(dataset_dir)
        with open(self.root / "index.json", encoding="utf-8") as f:
            self.index = json.load(f)
        self.img_size = img_size
        print(f"EmojiDataset loaded: {len(self.index)} samples from {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        entry = self.index[i]
        img_path = self.root / entry["image_path"]
        img = Image.open(img_path).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        # Normalize to [-1, 1]
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return {
            "image": arr,
            "caption": entry["caption"],
            "name": entry["name"],
            "emoji_idx": entry["emoji_idx"],
        }

    def collate(self, batch: list[dict]) -> dict:
        """Collate a list of items into a batch dict with numpy arrays."""
        return {
            "image": np.stack([b["image"] for b in batch]),
            "caption": [b["caption"] for b in batch],
            "name": [b["name"] for b in batch],
            "emoji_idx": np.array([b["emoji_idx"] for b in batch]),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build emoji image dataset")
    parser.add_argument("--size", type=int, default=64,
                        help="Output image size (default: 64)")
    parser.add_argument("--augmentations", type=int, default=8,
                        help="Augmentations per emoji (default: 8)")
    parser.add_argument("--out", type=str, default="data/emoji_dataset",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        out_dir=args.out,
        img_size=args.size,
        n_augmentations=args.augmentations,
        seed=args.seed,
    )
