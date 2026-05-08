"""Compose a one-page summary figure of the whole pipeline.

Layout:
    Block A (top): inversion-method comparison grid (target | optim | encoder)
    Block B (mid): CAM heatmaps for the 8 attributes (one row, raw abs_diff)
    Block C (btm): real-image attribute editing for 3 representative attrs

Saved as out/final/summary.png plus a metrics.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont



def _hstack_with_labels(blocks: list[np.ndarray], labels: list[str]) -> np.ndarray:
    H = blocks[0].shape[0]
    rows = []
    label_strip = np.concatenate([_label_strip(l, b.shape[1]) for b, l in zip(blocks, labels)],
                                  axis=1)
    rows.append(label_strip)
    rows.append(np.concatenate(blocks, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="out/compare/grid_target_optim_encoder.png")
    ap.add_argument("--cam-dir", default="out/cam_full")
    ap.add_argument("--edit-dir", default="out/edit_real")
    ap.add_argument("--out", default="out/final/summary.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []

    # --- Block A: inversion comparison ---
    cmp_path = Path(args.compare)
    if cmp_path.exists():
        cmp_img = np.asarray(Image.open(cmp_path).convert("RGB"))
        title = _label_strip(
            "A. inversion: target | optim (1000 step Adam) | encoder (1 forward)",
            cmp_img.shape[1], h=22, font_size=16,
        )
        panels.append(np.concatenate([title, cmp_img], axis=0))

    # --- Block B: CAM montage ---
    cam_dir = Path(args.cam_dir)
    cam_attrs = sorted([p.name for p in cam_dir.iterdir() if p.is_dir()])
    cam_blocks, cam_labels = [], []
    for attr in cam_attrs:
        p = cam_dir / attr / "abs_diff.png"
        if p.exists():
            cam_blocks.append(np.asarray(Image.open(p).convert("RGB")))
            cam_labels.append(attr)
    if cam_blocks:
        # downscale each to 128px for compact montage
        target_h = 128
        scaled = []
        for b in cam_blocks:
            ratio = target_h / b.shape[0]
            new_w = int(b.shape[1] * ratio)
            scaled.append(np.asarray(
                Image.fromarray(b).resize((new_w, target_h), Image.BILINEAR)
            ))
        block_b = _hstack_with_labels(scaled, cam_labels)
        title = _label_strip("B. CAM saliency (boundary perturbation, 64 samples)",
                              block_b.shape[1], h=22, font_size=16)
        panels.append(np.concatenate([title, block_b], axis=0))

    # --- Block C: real-image editing for 3 representative attrs ---
    edit_dir = Path(args.edit_dir)
    showcase = ["view", "indoor_lighting", "wood"]
    edit_blocks, edit_labels = [], []
    for attr in showcase:
        p = edit_dir / f"{attr}.png"
        if p.exists():
            edit_blocks.append(np.asarray(Image.open(p).convert("RGB")))
            edit_labels.append(attr)
    if edit_blocks:
        for blk, lbl in zip(edit_blocks, edit_labels):
            title = _label_strip(
                f"C. {lbl}: real | -d ... +d   (cols = distances)",
                blk.shape[1], h=22, font_size=16,
            )
            panels.append(np.concatenate([title, blk], axis=0))

    # final compose: stack panels vertically with light separators
    if not panels:
        raise SystemExit("no panels found; have all upstream scripts run?")

    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 255, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        padded.append(p)
        padded.append(np.full((6, max_w, 3), 220, dtype=np.uint8))  # spacer
    final = np.concatenate(padded[:-1], axis=0)
    Image.fromarray(final).save(out)

    # consolidated metrics
    metrics = {}
    cmp_metrics = Path("out/compare/metrics.json")
    if cmp_metrics.exists():
        metrics["inversion"] = json.loads(cmp_metrics.read_text())
    Path(out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out}  ({final.shape[1]}x{final.shape[0]})")


if __name__ == "__main__":
    main()
