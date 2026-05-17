"""Training driver for LatentCLR baseline on FFHQ StyleGAN1.

Outputs: K=100 normalized direction vectors saved to
  experiments/out/latentclr_ffhq/directions.npy.

Schedule:
  100 epochs × 1000 iters each, batch B=16 latents per direction.
  Adam, lr 5e-4, NT-Xent T=0.5.

Feature extractor: synthesis layer 10 mean-pooled activation
  (matches the original LatentCLR config for StyleGAN2).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PAPER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PAPER / "experiments"))
sys.path.insert(0, str(PAPER.parent / "higan_dev"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402
from domains.ffhq.generator import FFHQGenerator                  # noqa: E402
from baselines.latentclr.model import (                            # noqa: E402
    DirectionBank, nt_xent_loss,
)


def install_feature_hook(G: FFHQGenerator, layer_idx: int):
    """Hook to capture mean-pooled activation at the given synthesis layer.

    Stores the *graph-attached* tensor so backward can flow through it.
    """
    cap: dict[str, torch.Tensor | None] = {"feat": None}
    target = getattr(G._net.synthesis, f"layer{layer_idx}")

    def hook(_m, _i, out):
        if isinstance(out, tuple):
            out = out[0]
        cap["feat"] = out.mean(dim=(2, 3))     # (B, C)
    h = target.register_forward_hook(hook)
    return cap, h


def train(K: int = 100, B: int = 16, epochs: int = 100,
          iters_per_epoch: int = 1000, lr: float = 5e-4,
          direction_scale: float = 6.0, feature_layer: int = 10,
          temperature: float = 0.5,
          out: str = "experiments/out/latentclr_ffhq", seed: int = 0,
          chunk: int = 25, lod: float = 2.0):
    """K is the number of directions; chunk controls how many directions
    we forward per minibatch slice (memory).

    lod (level-of-detail): StyleGAN1 supports lod=2 → 256² rendering with
    ~10× less activation memory than lod=0 → 1024². Since LatentCLR only
    uses synthesis-layer-10 features (mid-level), the 256² rendering is
    feature-space equivalent for contrastive training, and fits in 8 GB.
    Set lod=0 for 1024² rendering (will OOM on 8 GB with K=100)."""
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    set_deterministic(seed=seed)
    G = FFHQGenerator(lod_override=lod)
    G_dev = G.device
    print(f"[LatentCLR] FFHQ generator at lod={lod} "
          f"(resolution {G.resolution // (2**int(lod))}²)")

    bank = DirectionBank(K, G.w_dim).to(G_dev)
    opt = torch.optim.Adam(bank.parameters(), lr=lr)

    cap, hook = install_feature_hook(G, feature_layer)

    loss_log = []
    print(f"[{time.strftime('%H:%M:%S')}] LatentCLR train start: "
          f"K={K} B={B} epochs={epochs} iters/epoch={iters_per_epoch} "
          f"lr={lr} α_scale={direction_scale} chunk={chunk}")
    t0 = time.time()
    try:
        for ep in range(epochs):
            ep_losses = []
            for it in range(iters_per_epoch):
                opt.zero_grad()
                # sample one shared batch of base latents
                z = torch.randn(B, G.z_dim, device=G_dev)
                with torch.no_grad():
                    wp_base = G.z_to_wp(z)                  # (B, L, D)
                    _ = G.synthesize(wp_base)
                    base_feat = cap["feat"].detach()        # (B, C)

                directions = bank.unit_dirs                  # (K, w_dim)
                # forward each direction; chunk over K to fit memory
                feats = []
                for k0 in range(0, K, chunk):
                    k1 = min(k0 + chunk, K)
                    chunk_dirs = directions[k0:k1]           # (k1-k0, w_dim)
                    # build batched wp: (chunk*B, L, D)
                    wp_pos = wp_base.unsqueeze(0) + direction_scale * \
                             chunk_dirs.unsqueeze(1).unsqueeze(2)
                    wp_pos = wp_pos.reshape(-1, *wp_base.shape[1:])
                    _ = G.synthesize(wp_pos)
                    feat_pos = cap["feat"]                   # ((k1-k0)*B, C)
                    feat_diff = feat_pos - base_feat.repeat(k1 - k0, 1)
                    feats.append(feat_diff)
                features = torch.cat(feats, dim=0)           # (K*B, C)

                loss = nt_xent_loss(features, K=K, B=B,
                                    temperature=temperature)
                loss.backward()
                opt.step()
                ep_losses.append(loss.item())

                if (it + 1) % 100 == 0:
                    print(f"  ep {ep+1}/{epochs} it {it+1}/{iters_per_epoch} "
                          f"loss {np.mean(ep_losses[-100:]):.4f} "
                          f"({time.time()-t0:.1f}s)")
            loss_log.append(float(np.mean(ep_losses)))
            print(f"  ep {ep+1} mean loss = {loss_log[-1]:.4f}")

            # checkpoint each epoch
            np.save(out_p / "directions.npy",
                    bank.unit_dirs.detach().cpu().numpy())
            with open(out_p / "train_log.json", "w") as fp:
                json.dump({"loss_per_epoch": loss_log,
                           "K": K, "epochs_done": ep + 1,
                           "epochs_total": epochs,
                           "wallclock_s": time.time() - t0}, fp, indent=2)
    finally:
        hook.remove()

    print(f"[{time.strftime('%H:%M:%S')}] done. saved directions to {out_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--B", type=int, default=8,
                    help="batch latents per direction; smaller = less mem")
    ap.add_argument("--chunk", type=int, default=10,
                    help="how many directions to forward per slice")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--direction-scale", type=float, default=6.0)
    ap.add_argument("--feature-layer", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default="experiments/out/latentclr_ffhq")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--lod", type=float, default=2.0,
                    help="StyleGAN1 lod override (2=256², 1=512², 0=1024²); "
                    "lod≥1 needed for K=100 chunk=8+ on 8GB")
    args = ap.parse_args()

    train(K=args.K, B=args.B, epochs=args.epochs,
          iters_per_epoch=args.iters_per_epoch, lr=args.lr,
          direction_scale=args.direction_scale,
          feature_layer=args.feature_layer,
          temperature=args.temperature,
          out=args.out, seed=args.seed, chunk=args.chunk,
          lod=args.lod)
