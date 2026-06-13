# Round 4 Status — verified safe to leave running

**Launched**: 2026-05-21 20:54 KST
**PID**: 659921 (bash run_round4.sh)
**Detach state**: PPID=1 (init), SID=659921, no TTY — fully detached

---

## 5-point strict verification (all ✅)

### 1. Background fully detached
- `setsid + nohup + disown`
- PPID=1 (init owned), no controlling terminal
- VSCode close, SSH close, Claude session close: **process survives**

### 2. Queue intact + serial chaining
- 11 tracks total, sequential `run_track` calls
- bash blocks on each `"$@"` until rc returned
- Auto-skips on failure (FAILED log line) → next track immediately

| # | Track | ETA | Status |
|---|---|---|---|
| 1 | T42 multi-CLIP + church | — | ⊘ FAILED (church port not done; pre-existing v1 result holds) |
| 2 | T43 3rd-order curvature | 24s | ✅ DONE |
| 3 | T44 sample N=256 bedroom | ~5 min | 🔄 RUNNING |
| 4 | T44 sample N=256 ffhq | ~10 min | ⏳ queued |
| 5 | T44 sample N=256 church | ~3 min | ⏳ queued |
| 6 | T45 sample N=512 ffhq | ~20 min | ⏳ queued |
| 7 | T46 SD 9-timesteps | ~2.5h | ⏳ queued |
| 8 | T47 ResNet-18 train + eval | ~4h | ⏳ queued |
| 9 | T48 ResNet-34 train + eval | ~5h | ⏳ queued |
| 10 | inject_meta | <1 min | ⏳ queued |
| 11 | aggregate_results | <30s | ⏳ queued |

Total ETA: ~12 hours from launch (so ~08:54 KST 5/22 finish).

### 3. Reproducible code
- All scripts call `set_deterministic(seed=2027)` (pins cudnn / use_deterministic_algorithms / CUBLAS / numpy / torch / random)
- All scripts emit `run_metadata()` → git commit, torch ver, CUDA ver, GPU, timestamp
- All 11 invocations pass `--seed 2027` explicitly
- JVP determinism: bit-identical 17 decimals (`lib/test_jvp_determinism.py`)

### 4. Cited + cause-effect rationale
| Track | Citation | Cause→Effect |
|---|---|---|
| T43 3rd-order | Lobashev ICML25 (Hessian eigenanalysis) | Extends 2nd-order claim — does non-linearity gap grow at higher orders? Already observed in smoke: pose 3rd/1st=368 vs smile=3.06 |
| T44/T45 sample N | Standard bootstrap CI (Efron 1979) | Larger N → narrower CI → tighter Spearman estimate on N=128 result |
| T46 SD 9-timesteps | DDIM Song ICLR21 (50-step schedule) | Full t-sweep maps curvature emergence across diffusion process; current 3 timesteps (15/25/35) doesn't cover early/late noise regime |
| T47/T48 backbone ablation | He CVPR16 (ResNet family) | T4 sal_corr=0.07 stuck with ResNet-50; smaller backbones (R18/R34) may avoid over-parameterization that causes saliency drift |

### 5. Auto-completion
- `inject_meta.py` runs at end → all metrics.json have `_meta`
- `aggregate_results.py` runs at end → `_aggregate_summary.txt` refreshed
- Logs in `logs/round4/`

---

## When you return (≈ 08:54 KST 5/22)

```bash
# Quick check
tail -20 /mnt/20t/study/HIGAN/paper/logs/round4/wrapper.log

# Refreshed summary
less /mnt/20t/study/HIGAN/paper/experiments/out/_aggregate_summary.txt

# New tracks added: T42-T48
grep "^\[Track 4" /mnt/20t/study/HIGAN/paper/experiments/out/_aggregate_summary.txt
```

If wrapper.log shows `Round 4 COMPLETE`, all done. Otherwise:
- Check `ps -p 659921` to see if still alive
- Check `logs/round4/wrapper.log` last line for current state
- Individual track logs in `logs/round4/t{43,44,45,46,47,48}_*.log`

---

## T42 multi-CLIP+church failure (known, accepted)

- `run_multi_clip_c2.py` only implements bedroom + ffhq sweep
- Adding church requires porting `sweep_one_attr_church` (non-trivial)
- **Result preserved**: existing v1 `multi_clip_c2/metrics.json` has 3 backbones × 2 domains (bedroom+ffhq, Pearson 0.97-0.99)
- Decision: keep v1 result, document church multi-CLIP as future extension

## T43 3rd-order — already complete (24s)

Headline:
- pose: ratio 3rd/1st = **368×**, 3rd/2nd = **26×**
- eyeglasses: ratio 3rd/1st = **151×**
- smile: ratio 3rd/1st = **3.06×**

→ Structural attributes get *more* non-linear at higher orders than textural; supports C2 (curvature = semantic) at the 3rd-derivative level.

---

End of Round 4 status doc.
