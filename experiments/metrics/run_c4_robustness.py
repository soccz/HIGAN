"""C4 robustness — test whether the predictor↔non-linearity correlation
holds after dropping the high-curvature outlier attribute (view in
bedroom, pose in FFHQ).

Reads existing pair-level data from out/bedroom_c4 and out/ffhq_c4.
No GPU needed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

PAPER = Path(__file__).resolve().parents[2]


def report(name, pairs, *, log_pearson=False, also_raw=False):
    pred = np.array([p["predictor"] for p in pairs])
    nl = np.array([p["nonlinearity"] for p in pairs])
    sp_r, sp_p = spearmanr(pred, nl)
    pe_r, pe_p = pearsonr(pred, nl)
    print(f"{name:30s} n={len(pairs):2d}  "
          f"Spearman r={sp_r:+.3f} p={sp_p:.3g}  "
          f"Pearson r={pe_r:+.3f} p={pe_p:.3g}")
    out = {"n": len(pairs),
           "spearman": {"r": float(sp_r), "p": float(sp_p)},
           "pearson":  {"r": float(pe_r), "p": float(pe_p)}}
    if log_pearson:
        lp_r, lp_p = pearsonr(np.log(pred), np.log(nl))
        out["log_pearson"] = {"r": float(lp_r), "p": float(lp_p)}
        print(f"{'':30s}    log-Pearson r={lp_r:+.3f} p={lp_p:.3g}")
    if also_raw:
        raw = np.array([p["mixed_norm"] for p in pairs])
        sr_r, sr_p = spearmanr(raw, nl)
        out["spearman_raw_mixed"] = {"r": float(sr_r), "p": float(sr_p)}
        print(f"{'':30s}    raw-mixed Spearman r={sr_r:+.3f} p={sr_p:.3g}")
    return out


def analyse(domain: str, outlier_attr: str):
    print(f"\n=== {domain.upper()} (outlier = {outlier_attr}) ===")
    j = json.loads((PAPER / "experiments" / "out" / f"{domain}_c4"
                    / "metrics.json").read_text())
    pairs = j["pairs"]

    full = report(f"{domain} all pairs", pairs, log_pearson=True, also_raw=True)

    no_out = [p for p in pairs if outlier_attr not in (p["a"], p["b"])]
    drop = report(f"{domain} no-{outlier_attr}", no_out, log_pearson=True, also_raw=True)

    only_out = [p for p in pairs if outlier_attr in (p["a"], p["b"])]
    only = report(f"{domain} only-{outlier_attr}", only_out)

    return {"full": full, f"no_{outlier_attr}": drop,
            f"only_{outlier_attr}": only}


def main():
    out = PAPER / "experiments" / "out" / "c4_robustness"
    out.mkdir(parents=True, exist_ok=True)

    bedroom = analyse("bedroom", "view")
    ffhq = analyse("ffhq", "pose")

    print("\n=== summary ===")
    print("C4 holds without outlier-attribute? "
          "(answer = Spearman stays positive at p<0.05 after drop)")
    print(f"  bedroom no-view: r={bedroom['no_view']['spearman']['r']:+.3f}, "
          f"p={bedroom['no_view']['spearman']['p']:.3g}")
    print(f"  ffhq    no-pose: r={ffhq['no_pose']['spearman']['r']:+.3f}, "
          f"p={ffhq['no_pose']['spearman']['p']:.3g}")

    # Plot: pred vs nl, colour-coded by domain and outlier-membership
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    for ax, domain, outlier_attr in [(axes[0], "bedroom", "view"),
                                      (axes[1], "ffhq", "pose")]:
        j = json.loads((PAPER / "experiments" / "out" / f"{domain}_c4"
                        / "metrics.json").read_text())
        pairs = j["pairs"]
        pred = np.array([p["predictor"] for p in pairs])
        nl = np.array([p["nonlinearity"] for p in pairs])
        is_out = np.array([outlier_attr in (p["a"], p["b"]) for p in pairs])
        ax.scatter(pred[~is_out], nl[~is_out], s=60,
                   c="#0e7490", alpha=0.85, edgecolors="white",
                   linewidths=1.3, label=f"texture-only (no {outlier_attr})")
        ax.scatter(pred[is_out], nl[is_out], s=60,
                   c="#dc2626", alpha=0.85, edgecolors="white",
                   linewidths=1.3, label=f"{outlier_attr} pairs")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("mixed-Hessian predictor", fontsize=10)
        ax.set_ylabel("compositional non-linearity (1 − corr)", fontsize=10)

        store = bedroom if domain == "bedroom" else ffhq
        no_key = f"no_{outlier_attr}"
        full_r = store["full"]["spearman"]["r"]
        drop_r = store[no_key]["spearman"]["r"]
        drop_p = store[no_key]["spearman"]["p"]
        ax.set_title(f"{domain.capitalize()} C4 — full r={full_r:+.2f}, "
                     f"no-{outlier_attr} r={drop_r:+.2f} (p={drop_p:.2g})",
                     fontsize=10.5, weight="bold", pad=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("C4 robustness — does the correlation survive without the "
                 "high-curvature attribute?", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    from PIL import Image
    Image.fromarray(arr).save(out / "c4_robustness.png")
    print(f"\nsaved {out / 'c4_robustness.png'}")

    (out / "metrics.json").write_text(json.dumps(
        {"bedroom": bedroom, "ffhq": ffhq}, indent=2))


if __name__ == "__main__":
    main()
