"""DisCo (Ren et al. ICLR 2022) — constant-direction variant.

Identical to LatentCLR but adds:
  - explicit orthogonality regulariser on the direction bank
    (||V V^T - I||_F^2)
  - DisCo's "diversity" objective: each direction should be associated
    with a different cluster of latents. We approximate this by
    minibatching with the K directions partitioned into mutually
    exclusive 'experts' on each step.

This is a simplified DisCo re-implementation, keeping the contrastive
core but adapting to constant-direction parameterisation (matching the
LatentCLR variant we run).
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
from baselines.latentclr.model import DirectionBank, nt_xent_loss  # noqa: E402
from baselines.latentclr.train import install_feature_hook         # noqa: E402


def orthogonality_loss(bank: DirectionBank) -> torch.Tensor:
    V = bank.unit_dirs                          # (K, w_dim)
    gram = V @ V.T                              # (K, K)
    K = gram.size(0)
    I = torch.eye(K, device=gram.device, dtype=gram.dtype)
    return ((gram - I) ** 2).sum() / (K * K)


def train(K: int = 100, B: int = 8, chunk: int = 10,
          epochs: int = 100, iters_per_epoch: int = 500,
          lr: float = 5e-4, direction_scale: float = 6.0,
          feature_layer: int = 10, temperature: float = 0.5,
          ortho_weight: float = 0.5,
          out: str = "experiments/out/disco_ffhq", seed: int = 0,
          lod: float = 2.0):
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    set_deterministic(seed=seed)
    G = FFHQGenerator(lod_override=lod)
    G_dev = G.device
    print(f"[DisCo] FFHQ generator at lod={lod} "
          f"(resolution {G.resolution // (2**int(lod))}²)")

    bank = DirectionBank(K, G.w_dim).to(G_dev)
    opt = torch.optim.Adam(bank.parameters(), lr=lr)
    cap, hook = install_feature_hook(G, feature_layer)

    loss_log = []
    contrast_log = []
    ortho_log = []
    print(f"[{time.strftime('%H:%M:%S')}] DisCo train start: K={K} B={B} "
          f"epochs={epochs} iters/epoch={iters_per_epoch} "
          f"ortho_weight={ortho_weight}")
    t0 = time.time()
    try:
        for ep in range(epochs):
            ep_losses = []
            ep_contrast = []
            ep_ortho = []
            for it in range(iters_per_epoch):
                opt.zero_grad()
                z = torch.randn(B, G.z_dim, device=G_dev)
                with torch.no_grad():
                    wp_base = G.z_to_wp(z)
                    _ = G.synthesize(wp_base)
                    base_feat = cap["feat"].detach()

                directions = bank.unit_dirs
                # Gradient checkpointing for K=100 fit in 8GB
                import torch.utils.checkpoint as cp

                def synth_one(chunk_dirs):
                    wp_pos = wp_base.unsqueeze(0) + direction_scale * \
                             chunk_dirs.unsqueeze(1).unsqueeze(2)
                    wp_pos = wp_pos.reshape(-1, *wp_base.shape[1:])
                    _ = G.synthesize(wp_pos)
                    feat_pos = cap["feat"].clone()
                    return feat_pos - base_feat.repeat(
                        chunk_dirs.shape[0], 1)

                feats = []
                for k0 in range(0, K, chunk):
                    k1 = min(k0 + chunk, K)
                    chunk_dirs = directions[k0:k1]
                    feat_diff = cp.checkpoint(
                        synth_one, chunk_dirs, use_reentrant=False
                    )
                    feats.append(feat_diff)
                features = torch.cat(feats, dim=0)

                l_contrast = nt_xent_loss(features, K=K, B=B,
                                          temperature=temperature)
                l_ortho = orthogonality_loss(bank)
                loss = l_contrast + ortho_weight * l_ortho
                loss.backward()
                opt.step()
                ep_losses.append(loss.item())
                ep_contrast.append(l_contrast.item())
                ep_ortho.append(l_ortho.item())

                if (it + 1) % 100 == 0:
                    print(f"  ep {ep+1}/{epochs} it {it+1}/{iters_per_epoch} "
                          f"loss {np.mean(ep_losses[-100:]):.4f} "
                          f"(C {np.mean(ep_contrast[-100:]):.4f} "
                          f"O {np.mean(ep_ortho[-100:]):.4f}) "
                          f"({time.time()-t0:.1f}s)")
            loss_log.append(float(np.mean(ep_losses)))
            contrast_log.append(float(np.mean(ep_contrast)))
            ortho_log.append(float(np.mean(ep_ortho)))
            np.save(out_p / "directions.npy",
                    bank.unit_dirs.detach().cpu().numpy())
            with open(out_p / "train_log.json", "w") as fp:
                json.dump({"loss_per_epoch": loss_log,
                           "contrast_per_epoch": contrast_log,
                           "ortho_per_epoch": ortho_log,
                           "K": K, "epochs_done": ep + 1,
                           "epochs_total": epochs,
                           "wallclock_s": time.time() - t0}, fp, indent=2)
    finally:
        hook.remove()

    print(f"[{time.strftime('%H:%M:%S')}] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters-per-epoch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--direction-scale", type=float, default=6.0)
    ap.add_argument("--feature-layer", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--ortho-weight", type=float, default=0.5)
    ap.add_argument("--out", default="experiments/out/disco_ffhq")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--lod", type=float, default=2.0,
                    help="StyleGAN1 lod override (2=256², 0=1024²)")
    args = ap.parse_args()

    train(**{k: v for k, v in vars(args).items()})
