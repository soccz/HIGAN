"""Track 6 — DAAM saliency baseline on Stable Diffusion.

For each of the 5 face attribute prompts, capture cross-attention
maps for the attribute-targeted token across all U-Net attention
layers and DDIM steps (DAAM-style), then compare to our gradient
saliency from Track 1.

DAAM ref: Tang et al. ACL 2023, arXiv:2210.04885.

This is a CPU-light eval — single forward sampling per seed, with
attention-map hooks. No JVP. Reuses generator from Track 1.
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

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "experiments"))

from lib.reproducibility import set_deterministic, run_metadata    # noqa: E402

from diffusion.generator import SDH, SDConfig                      # noqa: E402


NEUTRAL_PROMPT = "a face, neutral expression, photograph"
ATTR_PROMPTS = {
    "smile":      "a smiling face with teeth, photograph",
    "age":        "an old wrinkled face, photograph",
    "gender":     "a male face with a beard, photograph",
    "eyeglasses": "a face wearing glasses, photograph",
    "pose":       "a face in side profile, photograph",
}
# Target tokens to attribute: pick a content-bearing token from each
# attribute prompt (DAAM eq. 1 sums over the chosen token's attention).
TARGET_TOKENS = {
    "smile":      "smiling",
    "age":        "wrinkled",
    "gender":     "beard",
    "eyeglasses": "glasses",
    "pose":       "profile",
}


def install_attn_hooks(unet):
    """Register a forward hook on every cross-attention layer that
    captures the (Q @ K^T) softmax output. Returns the hook handles
    and a dict where the maps will be accumulated.

    diffusers' AttnProcessor implementation computes the attention
    output via `attn.get_attention_scores(query, key)` -> softmax(QK/√d).
    We capture this intermediate, but since the AttnProcessor doesn't
    expose it directly, we wrap the .__call__ method to also save the
    probabilities.
    """
    from diffusers.models.attention_processor import Attention, AttnProcessor
    captures: dict[str, list[torch.Tensor]] = {"per_layer": []}
    original_call = AttnProcessor.__call__

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, *args, **kwargs):
        # We only care about cross-attention layers (encoder_hidden_states present)
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key,
                                                     attention_mask)
        if is_cross:
            # save: (heads*B, HW, 77) — average over heads later
            captures["per_layer"].append(attention_probs.detach().cpu())
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    AttnProcessor.__call__ = patched_call
    return captures, lambda: setattr(AttnProcessor, "__call__", original_call)


def daam_saliency(sdh: SDH, prompt: str, seed: int, target_token: str,
                  H_out: int = 512) -> np.ndarray:
    """Run a full sampling pass with attention hooks; return per-pixel
    aggregated cross-attention map for the target token."""
    # find token index in the encoded prompt
    tokens = sdh.tokenizer.tokenize(prompt)
    # diffusers tokenizer.encode prepends BOS, so token at i corresponds
    # to encoded position i+1 typically. We search for the substring.
    target_l = target_token.lower()
    token_idx = None
    for i, tok in enumerate(tokens):
        if target_l in tok.lower().replace("</w>", ""):
            token_idx = i + 1  # +1 for BOS
            break
    if token_idx is None:
        raise ValueError(f"target token {target_token!r} not in {tokens}")

    captures, unpatch = install_attn_hooks(sdh.unet)
    try:
        with torch.no_grad():
            cond, uncond = sdh.encode_prompt(prompt, "")
            H = W = sdh.cfg.resolution // 8
            gen = torch.Generator(device=sdh.cfg.device).manual_seed(seed)
            x = torch.randn(1, 4, H, W, generator=gen,
                            device=sdh.cfg.device, dtype=sdh.cfg.dtype)
            for i in range(sdh.cfg.num_inference_steps):
                _ = sdh.epsilon(x, i, cond, uncond)
                # ddim_step:
                eps = sdh.epsilon(x, i, cond, uncond)
                x = sdh.ddim_step(x, eps, i)
    finally:
        unpatch()

    # aggregate: for each captured (heads, HW, 77) tensor, take column
    # = token_idx and upsample to (H_out, H_out). Sum across layers/steps.
    accum = np.zeros((H_out, H_out), dtype=np.float64)
    for attn in captures["per_layer"]:
        try:
            v = attn[..., token_idx].mean(dim=0)  # avg over heads × batch
        except (IndexError, RuntimeError):
            continue
        # v has shape (HW,) for whatever H this layer is at
        side = int(np.sqrt(v.numel()))
        if side * side != v.numel():
            continue
        m = v.reshape(side, side).float().numpy()
        if side != H_out:
            from PIL import Image
            m_img = Image.fromarray(m).resize((H_out, H_out), Image.BILINEAR)
            m = np.asarray(m_img, dtype=np.float64)
        accum += m
    accum -= accum.min()
    if accum.max() > 0:
        accum /= accum.max()
    return accum.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--attrs", nargs="+", default=list(ATTR_PROMPTS.keys()))
    ap.add_argument("--jvp-sal-dir", default="experiments/out/sd_c1_c2",
                    help="dir with Track 1 JVP saliencies (per-seed images)")
    ap.add_argument("--out", default="experiments/out/sd_daam")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    set_deterministic(seed=getattr(args, 'seed', 2027))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sdh = SDH(SDConfig(resolution=512))
    print(f"[{time.strftime('%H:%M:%S')}] SDH loaded for DAAM run")

    results = {}
    for attr in args.attrs:
        prompt = ATTR_PROMPTS[attr]
        tgt = TARGET_TOKENS[attr]
        print(f"\n=== {attr}: prompt={prompt!r} token={tgt!r} ===")
        per_seed_daam = []
        for s in range(args.n_seeds):
            try:
                sal = daam_saliency(sdh, prompt, seed=20000 + s,
                                     target_token=tgt)
                per_seed_daam.append(sal)
                print(f"  seed {s+1}/{args.n_seeds}  daam shape {sal.shape}")
            except Exception as e:
                print(f"  seed {s+1}: failed — {type(e).__name__}: {e}")
        results[attr] = {
            "daam_mean": (np.mean(per_seed_daam, axis=0).tolist()
                          if per_seed_daam else None),
            "n_seeds": len(per_seed_daam),
        }

    with open(out / "daam_maps.json", "w") as fp:
        json.dump({k: {**v, "daam_mean": None} for k, v in results.items()},
                  fp, indent=2)
    np.savez(out / "daam_maps.npz",
             **{a: np.asarray(results[a]["daam_mean"], dtype=np.float32)
                for a in results if results[a]["daam_mean"] is not None})
    print(f"\nsaved {out / 'daam_maps.npz'}")


if __name__ == "__main__":
    main()
