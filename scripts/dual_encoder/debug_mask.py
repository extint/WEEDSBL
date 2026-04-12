"""
debug_mask.py — Verify mask loading logic before retraining.
Shows raw mask pixel values, old broken logic vs new fixed logic, side by side.

Run:
    python -m dual_encoder.debug_mask \
        --data_root /home/vjti-comp/Downloads/SUGARBEETS_AUGMENTED_DATASET \
        --num_samples 8 \
        --output_dir ./mask_debug
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask_raw(path):
    """Raw uint8 mask — no processing."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def old_logic(m):
    """Exactly what the broken dataloader did."""
    m = m.copy().astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    return (m > 0.5).astype(np.uint8)


def new_logic_vegetation(m):
    """Stage 1 fix: crop + weed = 1, background = 0."""
    return (m > 0).astype(np.uint8)


def new_logic_weed_only(m):
    """Stage 2: weed = 1, everything else = 0."""
    return (m == 2).astype(np.uint8)


def colorise(mask_raw):
    """Return an RGB image colouring bg/crop/weed distinctly."""
    h, w = mask_raw.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[mask_raw == 0]   = [30,  30,  30]    # background — dark grey
    colour[mask_raw == 1] = [60, 180,  60]    # crop       — green
    colour[mask_raw == 2] = [220, 50,  50]    # weed       — red
    return colour


def binary_colour(mask_bin, fg_color):
    """Colour a binary mask: 1=fg_color, 0=dark grey."""
    h, w = mask_bin.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[mask_bin == 0] = [30, 30, 30]
    colour[mask_bin == 1] = fg_color
    return colour


def pixel_stats(m_raw, old, new_veg, new_weed):
    total = m_raw.size
    n_bg   = (m_raw == 0).sum()
    n_crop = (m_raw == 1).sum()
    n_weed = (m_raw == 2).sum()
    n_other = total - n_bg - n_crop - n_weed

    lines = [
        f"── RAW PIXEL VALUES ──────────────────",
        f"  background (0)  : {n_bg:>7,}  ({n_bg/total*100:5.1f}%)",
        f"  crop      (128) : {n_crop:>7,}  ({n_crop/total*100:5.1f}%)",
        f"  weed      (255) : {n_weed:>7,}  ({n_weed/total*100:5.1f}%)",
    ]
    if n_other:
        lines.append(f"  OTHER (unexpected): {n_other:>7,}  ← check dataset!")
    lines += [
        f"",
        f"── OLD LOGIC  (m/255 > 0.5) ──────────",
        f"  foreground=1 : {old.sum():>7,}  ({old.mean()*100:5.1f}%)",
        f"  background=0 : {(old==0).sum():>7,}",
        f"  crop captured: {'YES ✓' if n_crop > 0 and (old[m_raw==1]==1).mean() > 0.9 else 'NO  ✗  ← BUG'}",
        f"  weed captured: {'YES ✓' if n_weed > 0 and (old[m_raw==2]==1).mean() > 0.9 else 'NO  ✗'}",
        f"",
        f"── NEW LOGIC  vegetation (m > 0) ─────",
        f"  foreground=1 : {new_veg.sum():>7,}  ({new_veg.mean()*100:5.1f}%)",
        f"  crop captured: {'YES ✓' if n_crop==0 or (new_veg[m_raw==1]==1).all() else 'NO  ✗'}",
        f"  weed captured: {'YES ✓' if n_weed==0 or (new_veg[m_raw==2]==1).all() else 'NO  ✗'}",
        f"",
        f"── NEW LOGIC  weed_only (m==255) ─────",
        f"  foreground=1 : {new_weed.sum():>7,}  ({new_weed.mean()*100:5.1f}%)",
        f"  weed captured: {'YES ✓' if n_weed==0 or (new_weed[m_raw==2]==1).all() else 'NO  ✗'}",
        f"  crop excluded: {'YES ✓' if n_crop==0 or (new_weed[m_raw==1]==0).all() else 'NO  ✗'}",
    ]
    return "\n".join(lines)


def debug_sample(rgb_path, mask_path, out_path, size=512):
    rgb   = cv2.resize(read_rgb(str(rgb_path)), (size, size))
    m_raw = cv2.resize(read_mask_raw(str(mask_path)), (size, size),
                       interpolation=cv2.INTER_NEAREST)

    old      = old_logic(m_raw)
    new_veg  = new_logic_vegetation(m_raw)
    new_weed = new_logic_weed_only(m_raw)

    stats = pixel_stats(m_raw, old, new_veg, new_weed)
    print(f"\n{'='*55}")
    print(f"  {Path(mask_path).name}")
    print(f"{'='*55}")
    print(stats)

    # ── figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle(Path(mask_path).name, fontsize=11, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontweight="bold"); axes[0].axis("off")

    axes[1].imshow(colorise(m_raw))
    axes[1].set_title("Raw mask\n(grey=bg  green=crop  red=weed)", fontweight="bold")
    axes[1].axis("off")
    # legend
    axes[1].legend(handles=[
        mpatches.Patch(color=[.12,.12,.12], label="background"),
        mpatches.Patch(color=[.24,.71,.24], label="crop"),
        mpatches.Patch(color=[.86,.20,.20], label="weed"),
    ], loc="lower right", fontsize=7)

    axes[2].imshow(binary_colour(old, [255, 80, 80]))
    n_bg_old = (m_raw == 128).sum()
    axes[2].set_title(
        f"OLD logic  (m/255 > 0.5)\n"
        f"{'⚠ CROP MISSING — ' + str((m_raw==1).sum()) + ' px lost!' if (m_raw==1).sum()>0 else 'no crop pixels in image'}",
        fontweight="bold", color="red" if (m_raw==1).sum() > 0 else "black"
    )
    axes[2].axis("off")

    axes[3].imshow(binary_colour(new_veg, [80, 220, 80]))
    axes[3].set_title(
        f"NEW — vegetation\n(crop+weed=1, bg=0)\n"
        f"Stage 1 ✓",
        fontweight="bold", color="green"
    )
    axes[3].axis("off")

    axes[4].imshow(binary_colour(new_weed, [255, 80, 80]))
    axes[4].set_title(
        f"NEW — weed_only\n(weed=1, rest=0)\n"
        f"Stage 2 ✓",
        fontweight="bold", color="darkred"
    )
    axes[4].axis("off")

    # stats text box on figure
    fig.text(0.01, 0.01, stats, fontsize=7, fontfamily="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--split",       default="train")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir",  default="./mask_debug")
    parser.add_argument("--size",        type=int, default=512)
    args = parser.parse_args()

    root     = Path(args.data_root)
    mask_dir = root / "masks"
    rgb_dir  = root / "rgb"
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read split file
    split_file = root / "splits" / f"{args.split}.txt"
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]

    ids = ids[:args.num_samples]
    print(f"[INFO] Checking {len(ids)} samples from '{args.split}' split")
    print(f"[INFO] Output → {out_dir}")

    all_unique = set()
    for img_id in ids:
        mask_path = mask_dir / f"mask_{img_id}.png"
        rgb_path  = rgb_dir  / f"rgb_{img_id}.png"

        if not mask_path.exists():
            print(f"[WARN] Missing mask: {mask_path}"); continue

        m_raw = read_mask_raw(str(mask_path))
        all_unique.update(np.unique(m_raw).tolist())

        out_path = out_dir / f"{img_id}_mask_debug.png"
        debug_sample(rgb_path, mask_path, out_path, size=args.size)
        print(f"  saved → {out_path}")

    print(f"\n{'='*55}")
    print(f"SUMMARY — unique pixel values across all {len(ids)} masks:")
    print(f"  {sorted(all_unique)}")
    if {0, 1, 2} == all_unique or all_unique.issubset({0, 1, 2}):
        print("  ✓ Values match expected {0=bg, 1=crop, 2=weed}")
    else:
        print("  ⚠ Unexpected values found — update BG_VAL/CROP_VAL/WEED_VAL in dataloader!")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()