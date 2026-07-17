"""FFHQ finite-difference vs exact second-order generator curvature.

This is the FFHQ analogue of higan_dev/scripts/29_shao_fd_vs_exact.py, but it
uses the existing InterFaceGAN FFHQ wrapper in paper/experiments/domains/ffhq.
It writes a standalone metrics.json and trilemma_table.json under the requested
output directory; it does not touch the frozen TMLR submission package.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.func import jvp

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from domains.ffhq.generator import FFHQGenerator  # noqa: E402


LAYERS_FOR = {
    "pose": list(range(0, 4)),
    "gender": list(range(0, 8)),
    "age": list(range(0, 8)),
    "eyeglasses": list(range(0, 8)),
    "smile": list(range(4, 8)),
}


def curve_fn(G, wp_base, v_layered, B):
    def f(alpha):
        return G.synthesize(wp_base + alpha.view(B, 1, 1) * v_layered.unsqueeze(0))

    return f


def exact_second(G, f, B):
    ones = torch.ones(B, device=G.device)
    a0 = torch.zeros(B, device=G.device)

    def df(alpha):
        return jvp(f, (alpha,), (torch.ones_like(alpha),))[1]

    _, second = jvp(df, (a0,), (ones,))
    _, first = jvp(f, (a0,), (ones,))
    return second.abs().mean().item(), first.abs().mean().item()


def fd_both(G, f, B, d):
    ap = torch.full((B,), +d, device=G.device)
    am = torch.full((B,), -d, device=G.device)
    a0 = torch.zeros(B, device=G.device)
    with torch.no_grad():
        gp = f(ap)
        g0 = f(a0)
        gm = f(am)
        second = (gp - 2.0 * g0 + gm) / (d * d)
        first = (gp - gm) / (2.0 * d)
    return second.abs().mean().item(), first.abs().mean().item()


def _rank(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = sum((x - mx) ** 2 for x in rx) ** 0.5
    vy = sum((y - my) ** 2 for y in ry) ** 0.5
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / (vx * vy)


def trilemma_table(records, steps):
    exact2 = [r["c_exact"] for r in records]
    exact1 = [r["c_first_exact"] for r in records]
    rows = []
    for d in steps:
        key = str(d)
        fd2 = [r["c_fd2"][key] for r in records]
        fd1 = [r["c_fd1"][key] for r in records]
        rows.append({
            "step": d,
            "second_order": {
                "mean_rel_err": mean(abs(a - b) / max(abs(b), 1e-12)
                                      for a, b in zip(fd2, exact2)),
                "mean_signed_bias": mean((a - b) / max(abs(b), 1e-12)
                                          for a, b in zip(fd2, exact2)),
                "spearman_vs_exact": spearman(fd2, exact2),
            },
            "first_order": {
                "mean_rel_err": mean(abs(a - b) / max(abs(b), 1e-12)
                                      for a, b in zip(fd1, exact1)),
                "mean_signed_bias": mean((a - b) / max(abs(b), 1e-12)
                                          for a, b in zip(fd1, exact1)),
                "spearman_vs_exact": spearman(fd1, exact1),
            },
        })

    so = [r["second_order"] for r in rows]
    return {
        "summary": {
            "n_records": len(records),
            "n_steps": len(steps),
            "second_order_magnitude_best_step":
                rows[min(range(len(rows)), key=lambda i: so[i]["mean_rel_err"])]["step"],
            "second_order_bias_best_step":
                rows[min(range(len(rows)), key=lambda i: abs(so[i]["mean_signed_bias"]))]["step"],
            "second_order_rank_best_step":
                rows[max(range(len(rows)), key=lambda i: so[i]["spearman_vs_exact"])]["step"],
        },
        "rows": rows,
    }


def load_ffhq_boundary(attr, L, D, device):
    bdir = EXPERIMENTS_DIR / "data" / "interfacegan" / "boundaries"
    for suffix in ["_w_boundary.npy", "_boundary.npy"]:
        path = bdir / f"stylegan_ffhq_{attr}{suffix}"
        if path.exists():
            b = np.load(path, allow_pickle=True).squeeze().astype(np.float32)
            b = torch.from_numpy(b).to(device)
            b = b / b.norm().clamp_min(1e-8)
            layered = torch.zeros(L, D, device=device)
            for li in LAYERS_FOR.get(attr, range(L)):
                layered[li] = b
            return path.name, layered
    raise FileNotFoundError(f"missing FFHQ boundary for {attr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=["smile", "age", "pose", "gender", "eyeglasses"])
    ap.add_argument("--n-random-dirs", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--d-grid", type=float, nargs="+",
                    default=[8.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5,
                             0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--lod-override", type=float, default=2.0,
                    help="Use lower-detail FFHQ synthesis to keep composed-JVP memory bounded.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-records", type=int, default=0,
                    help="Optional smoke-test cap; 0 means all attrs/random dirs × samples.")
    ap.add_argument("--out", default="out/ffhq_fd_vs_exact_trilemma")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    G = FFHQGenerator(device=args.device, lod_override=args.lod_override)
    L, D = G.num_layers, G.w_dim
    print(f"[{time.strftime('%H:%M:%S')}] FFHQ loaded; L={L} D={D} "
          f"res={G.resolution} lod_override={args.lod_override} device={G.device}")

    dirs = []
    for attr in args.attrs:
        name, layered = load_ffhq_boundary(attr, L, D, G.device)
        dirs.append((attr, layered, name))
    for r in range(args.n_random_dirs):
        gen = torch.Generator(device=G.device).manual_seed(1000 + r)
        rv = torch.randn(L, D, device=G.device, generator=gen)
        rv = rv / rv.norm()
        dirs.append((f"rand{r}", rv, "random-layered-w"))

    rng = torch.Generator(device=G.device).manual_seed(args.seed)
    base_wp = G.sample_wp(args.num_samples, generator=rng)

    records = []
    for dname, v_layered, source in dirs:
        for sample in range(args.num_samples):
            if args.max_records and len(records) >= args.max_records:
                break
            wp = base_wp[sample:sample + 1].detach()
            f = curve_fn(G, wp, v_layered, 1)
            c_exact, c_first = exact_second(G, f, 1)
            fds = {d: fd_both(G, f, 1, d) for d in args.d_grid}
            records.append({
                "dir": dname,
                "direction_source": source,
                "sample": sample,
                "c_exact": c_exact,
                "c_first_exact": c_first,
                "c_fd2": {str(d): fds[d][0] for d in args.d_grid},
                "c_fd1": {str(d): fds[d][1] for d in args.d_grid},
            })
            if G.device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  {dname:12s} done ({len(records)} records, {time.strftime('%H:%M:%S')})")
        if args.max_records and len(records) >= args.max_records:
            break

    def relerr(cfd, cex):
        return abs(cfd - cex) / max(abs(cex), 1e-12)

    agg = {}
    for d in args.d_grid:
        key = str(d)
        errs2 = [relerr(r["c_fd2"][key], r["c_exact"]) for r in records]
        errs1 = [relerr(r["c_fd1"][key], r["c_first_exact"]) for r in records]
        agg[key] = {
            "mean_relerr": float(np.mean(errs2)),
            "median_relerr": float(np.median(errs2)),
            "mean_c_fd": float(np.mean([r["c_fd2"][key] for r in records])),
            "mean_relerr_FIRST": float(np.mean(errs1)),
            "mean_c_fd1": float(np.mean([r["c_fd1"][key] for r in records])),
        }

    table = trilemma_table(records, args.d_grid)
    metrics = {
        "records": records,
        "agg_by_step": agg,
        "trilemma_summary": table["summary"],
        "mean_c_exact": float(np.mean([r["c_exact"] for r in records])),
        "config": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "trilemma_table.json").write_text(json.dumps(table, indent=2))

    print("\nFFHQ FD-vs-exact trilemma summary")
    print(json.dumps(table["summary"], indent=2))
    print(f"saved {out / 'metrics.json'}")
    print(f"saved {out / 'trilemma_table.json'}")


if __name__ == "__main__":
    main()
