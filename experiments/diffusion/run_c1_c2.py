"""Track 1 main: C1/C2 on Stable Diffusion v1.5 via h-space perturbation.

See designs/01_sd_curvature.md for hypothesis + protocol + citations.

For each attribute prompt P_a (vs a neutral prompt P_0):
  1. Sample 64 noise seeds. For half of them ("training set"),
     compute the SEGA-style h-direction:
        v_a(t) = mean_n[ mid_block(z_t, P_a) - mid_block(z_t, P_0) ]
     averaged at the editing timestep t_edit.
  2. For the other half ("test set"), at each of the three timesteps
     {0.7T, 0.5T, 0.3T}, run two first-order JVPs at alpha = +ε and
     -ε and compute:
        first  = mean alpha-tangent magnitude   (C1)
        second = FD second derivative magnitude (C1)
        rho    = second / first                  (C1 metric)
     Also do the alpha-sweep alpha ∈ [-3, 3] for the CLIP path
     curvature ratio (C2 second measure).

Outputs:
  experiments/out/sd_c1_c2/metrics.json
  experiments/out/sd_c1_c2/sd_c1_c2.png
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
from torch.func import jvp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.reproducibility import set_deterministic, run_metadata     # noqa: E402
from diffusion.generator import SDH, SDConfig                       # noqa: E402

NEUTRAL_PROMPT = "a face, neutral expression, photograph"
ATTR_PROMPTS = {
    "smile":      "a face with a wide smile, photograph",
    "age":        "an old face with many wrinkles, photograph",
    "gender":     "a male face with a beard, photograph",
    "eyeglasses": "a face wearing glasses, photograph",
    "pose":       "a face in side profile, photograph",
}

# Editing timesteps as fractions of T (50-step DDIM schedule).
# Asyrp / Park-NeurIPS23 editing window is [0.5T, 0.7T]; we also probe
# 0.3T (later in sampling) to see how curvature evolves.
TIMESTEP_FRACS = [0.7, 0.5, 0.3]


@torch.no_grad()
def collect_h_direction(sdh: SDH, base_prompt: str, attr_prompt: str,
                        n_train: int, t_edit_idx: int) -> torch.Tensor:
    """SEGA-style h-space direction = mean[ mid_block(z, attr) - mid_block(z, base) ]
    averaged over training seeds at one timestep.
    """
    cond_a, _ = sdh.encode_prompt(attr_prompt, "")
    cond_b, _ = sdh.encode_prompt(base_prompt, "")
    captured = {}

    def grab(_m, _i, out):
        captured.setdefault("h", []).append(out.detach())

    hh = sdh.unet.mid_block.register_forward_hook(grab)
    try:
        delta_acc = None
        for s in range(n_train):
            captured["h"] = []
            H = W = sdh.cfg.resolution // 8
            gen = torch.Generator(device=sdh.cfg.device).manual_seed(7000 + s)
            x = torch.randn(1, 4, H, W, generator=gen,
                            device=sdh.cfg.device, dtype=sdh.cfg.dtype)
            # walk to t_edit
            for i in range(t_edit_idx):
                eps = sdh.epsilon(x, i, cond_b, sdh.encode_prompt("", "")[1])
                x = sdh.ddim_step(x, eps, i)
            captured["h"] = []
            # uncond
            uc = sdh.encode_prompt("", "")[1]
            # one forward at t_edit with attr prompt
            _ = sdh.unet(x, sdh.timesteps[t_edit_idx],
                         encoder_hidden_states=cond_a).sample
            h_a = captured["h"][-1]
            captured["h"] = []
            # one forward at t_edit with base prompt
            _ = sdh.unet(x, sdh.timesteps[t_edit_idx],
                         encoder_hidden_states=cond_b).sample
            h_b = captured["h"][-1]
            delta = h_a - h_b
            delta_acc = delta if delta_acc is None else delta_acc + delta
    finally:
        hh.remove()
    direction = delta_acc / n_train
    direction = direction / direction.norm().clamp_min(1e-8)
    return direction


def jvp_through_chain(sdh: SDH, x_at_edit: torch.Tensor,
                       v: torch.Tensor, alpha_scalar: float,
                       t_edit_idx: int, cond: torch.Tensor,
                       uncond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One first-order JVP through DDIM from t_edit_idx to t=0, evaluated
    at alpha=alpha_scalar with unit tangent.

    Returns (x_final_at_alpha, dx_final/d_alpha).
    """
    def f(alpha):
        x = x_at_edit
        sdh._h_v = v
        sdh._h_alpha = alpha
        for i in range(t_edit_idx, sdh.cfg.num_inference_steps):
            sdh._h_active = (i == t_edit_idx)
            eps = sdh.epsilon(x, i, cond, uncond)
            x = sdh.ddim_step(x, eps, i)
        sdh._h_active = False
        sdh._h_v = None
        sdh._h_alpha = None
        return x

    a_p = torch.tensor(alpha_scalar, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    a_t = torch.tensor(1.0, device=sdh.cfg.device, dtype=sdh.cfg.dtype)
    return jvp(f, (a_p,), (a_t,))


@torch.no_grad()
def latent_to_image(sdh: SDH, x_latent: torch.Tensor) -> torch.Tensor:
    return sdh.decode_image(x_latent)


def latent_dot_to_image_dot(sdh: SDH, x_latent: torch.Tensor,
                            dx_latent: torch.Tensor) -> torch.Tensor:
    """JVP through VAE decoder: image_dot = ∂decode/∂latent · dx_latent."""
    def fn(z):
        return sdh.decode_image(z)
    img, dimg = jvp(fn, (x_latent,), (dx_latent,))
    return img, dimg


def clip_feat(model, mean, std, img):
    x = (img.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.no_grad():
        f = model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=list(ATTR_PROMPTS.keys()))
    ap.add_argument("--n-train", type=int, default=32,
                    help="seeds used to estimate the SEGA-style direction")
    ap.add_argument("--n-test", type=int, default=32,
                    help="seeds for the curvature measurement")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--alpha-sweep-max", type=float, default=3.0,
                    help="α range for CLIP path curvature")
    ap.add_argument("--alpha-sweep-steps", type=int, default=9)
    ap.add_argument("--timestep-fracs", nargs="+", type=float,
                    default=TIMESTEP_FRACS)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out", default="experiments/out/sd_c1_c2")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] starting SD C1/C2 — "
          f"{len(args.attrs)} attrs × {len(args.timestep_fracs)} t × "
          f"{args.n_test} seeds")

    sdh = SDH(SDConfig(resolution=args.resolution))
    print(f"[{time.strftime('%H:%M:%S')}] SDH loaded")

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.eval().to(sdh.cfg.device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=sdh.cfg.device,
                             dtype=sdh.cfg.dtype).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=sdh.cfg.device,
                            dtype=sdh.cfg.dtype).view(1, 3, 1, 1)

    timestep_idxs = [int(round(f * sdh.cfg.num_inference_steps))
                     for f in args.timestep_fracs]
    timestep_idxs = [min(max(0, i), sdh.cfg.num_inference_steps - 2)
                     for i in timestep_idxs]
    print(f"timestep idxs: {timestep_idxs}")

    cond_neu, uncond = sdh.encode_prompt(NEUTRAL_PROMPT, "")

    results = {"per_attr": [], "config": vars(args),
               "timestep_idxs": timestep_idxs}

    for attr in args.attrs:
        attr_prompt = ATTR_PROMPTS[attr]
        cond_a, _ = sdh.encode_prompt(attr_prompt, "")
        print(f"\n=== {attr} === ({attr_prompt})")
        per_t = {}
        for t_idx in timestep_idxs:
            t_start = time.time()
            print(f"  t_idx={t_idx} -- collecting h-direction over "
                  f"{args.n_train} train seeds...")
            v = collect_h_direction(
                sdh, NEUTRAL_PROMPT, attr_prompt,
                args.n_train, t_idx,
            )
            print(f"    direction |v| = {v.norm().item():.4f}, "
                  f"shape {tuple(v.shape)}, took {time.time()-t_start:.1f}s")

            rho_per_seed = []
            clip_path_per_seed = []
            for s in range(args.n_test):
                seed_t0 = time.time()
                # 1. reach t_edit with the neutral prompt (no perturbation)
                H = W = sdh.cfg.resolution // 8
                gen = torch.Generator(device=sdh.cfg.device).manual_seed(
                    20000 + s + 1000 * t_idx
                )
                with torch.no_grad():
                    x = torch.randn(1, 4, H, W, generator=gen,
                                    device=sdh.cfg.device, dtype=sdh.cfg.dtype)
                    for i in range(t_idx):
                        eps = sdh.epsilon(x, i, cond_neu, uncond)
                        x = sdh.ddim_step(x, eps, i)
                x_edit = x.detach()

                # 2. two first-order JVPs at ±ε; the "first derivative" is
                #    their mean tangent magnitude, the second derivative is
                #    the finite difference of tangents.
                _, dx_plus = jvp_through_chain(
                    sdh, x_edit, v, +args.epsilon, t_idx, cond_neu, uncond
                )
                _, dx_minus = jvp_through_chain(
                    sdh, x_edit, v, -args.epsilon, t_idx, cond_neu, uncond
                )
                # decode the tangent to image space too (JVP through VAE)
                _, dimg_plus = latent_dot_to_image_dot(sdh, x_edit, dx_plus)
                _, dimg_minus = latent_dot_to_image_dot(sdh, x_edit, dx_minus)

                first = 0.5 * (dimg_plus.abs() + dimg_minus.abs())
                second = (dimg_plus - dimg_minus) / (2 * args.epsilon)
                rho = (second.abs().mean() / first.mean().clamp_min(1e-8)).item()
                rho_per_seed.append(rho)

                # 3. CLIP path curvature: sweep α ∈ [-A, +A]
                alphas = np.linspace(-args.alpha_sweep_max,
                                     +args.alpha_sweep_max,
                                     args.alpha_sweep_steps)
                feats = []
                for a in alphas:
                    with torch.no_grad():
                        x_loc = x_edit
                        sdh._h_v = v
                        sdh._h_alpha = torch.tensor(float(a),
                                                     device=sdh.cfg.device,
                                                     dtype=sdh.cfg.dtype)
                        for i in range(t_idx, sdh.cfg.num_inference_steps):
                            sdh._h_active = (i == t_idx)
                            eps = sdh.epsilon(x_loc, i, cond_neu, uncond)
                            x_loc = sdh.ddim_step(x_loc, eps, i)
                        sdh._h_active = False
                        sdh._h_v = None
                        sdh._h_alpha = None
                        img = sdh.decode_image(x_loc)
                        feats.append(clip_feat(clip_model, clip_mean,
                                               clip_std, img).squeeze(0))
                F_stack = torch.stack(feats)
                seg = torch.linalg.norm(F_stack[1:] - F_stack[:-1], dim=1)
                path_len = seg.sum().item()
                direct = torch.linalg.norm(F_stack[-1] - F_stack[0]).item()
                clip_ratio = path_len / direct if direct > 1e-8 else 0.0
                clip_path_per_seed.append(clip_ratio)

                torch.cuda.empty_cache()
                print(f"    seed {s+1}/{args.n_test}  "
                      f"ρ={rho:.3f}  CLIP-path/direct={clip_ratio:.3f}  "
                      f"({time.time()-seed_t0:.1f}s)")

            per_t[str(t_idx)] = {
                "rho_mean": float(np.mean(rho_per_seed)),
                "rho_median": float(np.median(rho_per_seed)),
                "rho_std": float(np.std(rho_per_seed)),
                "rho_all": rho_per_seed,
                "clip_path_mean": float(np.mean(clip_path_per_seed)),
                "clip_path_std": float(np.std(clip_path_per_seed)),
                "clip_path_all": clip_path_per_seed,
            }
            print(f"  t_idx={t_idx}  ρ mean={np.mean(rho_per_seed):.3f} "
                  f"clip_path mean={np.mean(clip_path_per_seed):.3f}")

        results["per_attr"].append({"attr": attr, "per_t": per_t})

        # incremental save in case the run is interrupted
        results["_meta"] = run_metadata(seed=args.seed,
                                         extra={"track": "1_sd_c1c2"})
        with open(out / "metrics_partial.json", "w") as fp:
            json.dump(results, fp, indent=2)

    # final save + plot
    results["_meta"] = run_metadata(seed=args.seed,
                                     extra={"track": "1_sd_c1c2"})
    with open(out / "metrics.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
