#!/usr/bin/env python3
"""Recompute the step-selection trilemma (body Table 2 / appendix Table 3) directly
from the per-record FD-vs-exact evidence, so the headline table is reproducible from a
single released artifact.

Input : stylegan_fd_vs_exact_table2.json  (96 direction-anchor records; each has
        c_exact, c_first_exact, and c_fd2/c_fd1 dicts over 16 steps)
Output: trilemma_table.json  (per step: 2nd- and 1st-order mean relative error,
        mean signed bias, and Spearman rank correlation vs the exact composed-JVP,
        averaged over all 96 records)

No new experiments: this is a pure re-aggregation. Spearman is implemented locally
(average-rank ties) so the script has no third-party dependency.
"""
import json
import os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "stylegan_fd_vs_exact_table2.json")
OUT = os.path.join(HERE, "trilemma_table.json")


def _rank(vals):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = _rank(xs), _rank(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy)


def main():
    d = json.load(open(SRC))
    recs = d["records"]
    steps = list(d["agg_by_step"].keys())
    ce = [r["c_exact"] for r in recs]
    cf = [r["c_first_exact"] for r in recs]

    rows = []
    for s in steps:
        f2 = [r["c_fd2"][s] for r in recs]
        f1 = [r["c_fd1"][s] for r in recs]
        rows.append({
            "step": float(s),
            "second_order": {
                "mean_rel_err": mean(abs(a - b) / abs(b) for a, b in zip(f2, ce)),
                "mean_signed_bias": mean((a - b) / b for a, b in zip(f2, ce)),
                "spearman_vs_exact": spearman(f2, ce),
            },
            "first_order": {
                "mean_rel_err": mean(abs(a - b) / abs(b) for a, b in zip(f1, cf)),
                "mean_signed_bias": mean((a - b) / b for a, b in zip(f1, cf)),
                "spearman_vs_exact": spearman(f1, cf),
            },
        })

    so = [r["second_order"] for r in rows]
    summary = {
        "n_records": len(recs),
        "n_steps": len(steps),
        "second_order_magnitude_best_step":
            rows[min(range(len(rows)), key=lambda i: so[i]["mean_rel_err"])]["step"],
        "second_order_bias_best_step":
            rows[min(range(len(rows)), key=lambda i: abs(so[i]["mean_signed_bias"]))]["step"],
        "second_order_rank_best_step":
            rows[max(range(len(rows)), key=lambda i: so[i]["spearman_vs_exact"])]["step"],
    }

    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("magnitude-best step:", summary["second_order_magnitude_best_step"])
    print("bias-best step     :", summary["second_order_bias_best_step"])
    print("rank-best step     :", summary["second_order_rank_best_step"])


if __name__ == "__main__":
    main()
