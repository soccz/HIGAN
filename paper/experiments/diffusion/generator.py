"""Stable Diffusion v1.5 wrapper with h-space (mid_block) perturbation +
JVP-safe DDIM sampling.

Design (see designs/01_sd_curvature.md):
- h-space = UNet mid_block output (Kwon et al. ICLR 2023, "Asyrp").
- DDIM, 50 steps, η=0, CFG 7.5 (matches Asyrp / SEGA / Park-NeurIPS23).
- JVP-safe: a custom DDIMStep avoids any .item()/.cpu() calls in the
  differentiable path. The UNet itself is JVP-compatible (diffusers
  0.27 UNet2DConditionModel uses only pure tensor ops in forward).
- h-space perturbation is implemented as a forward-pre-hook that adds
  alpha * v to the mid_block output exactly once per sampling chain
  (at the edit timestep).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch import nn

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel


SD15_REPO = "runwayml/stable-diffusion-v1-5"


@dataclass
class SDConfig:
    model_id: str = SD15_REPO
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    resolution: int = 512
    # fp32 because torch.func.jvp loses dtype consistency with fp16 under
    # the math AttnProcessor (we tested this on torch 2.2.2 / diffusers 0.27.2;
    # JVP introduces fp32 tangents that collide with fp16 weights).
    dtype: torch.dtype = torch.float32
    device: str = "cuda"


def _alphas_for_steps(scheduler: DDIMScheduler, num_steps: int):
    """Return per-step (alpha_bar_t, alpha_bar_prev) as fp32 tensors.

    Pre-extracts the scheduler's discrete schedule so the differentiable
    sampling loop doesn't need any python int -> tensor indexing inside
    a jvp() trace.
    """
    scheduler.set_timesteps(num_steps, device="cpu")
    timesteps = scheduler.timesteps  # descending int64
    ac = scheduler.alphas_cumprod  # 1D fp32, length 1000
    alpha_t = torch.stack([ac[int(t)] for t in timesteps])
    # for the previous timestep
    prev_idx = [int(t) - (scheduler.config.num_train_timesteps
                          // num_steps) for t in timesteps]
    alpha_prev = torch.stack([
        ac[max(0, p)] if p >= 0 else torch.tensor(1.0)
        for p in prev_idx
    ])
    return timesteps, alpha_t, alpha_prev


class SDH:
    """Stable Diffusion h-space wrapper.

    Public API:
      sample(prompt, neg_prompt, seed) -> latents at all 50 steps + x_0 image
      synthesize_with_h_perturb(prompt, seed, t_edit_idx, v, alpha)
          returns the final x_0 image after injecting alpha*v into
          mid_block at the chosen step.
      epsilon_at(latent, t, prompt_embed, neg_embed) — single UNet call
      encode_prompt(prompt) -> (cond, uncond) embeddings cached
    """

    def __init__(self, cfg: SDConfig | None = None):
        cfg = cfg or SDConfig()
        self.cfg = cfg
        print(f"[SDH] loading {cfg.model_id} @ {cfg.resolution}px, dtype={cfg.dtype}")
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id, torch_dtype=cfg.dtype, safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device)
        pipe = pipe.to(cfg.device)
        pipe.set_progress_bar_config(disable=True)
        # Use the math attention processor (default AttnProcessor) so that
        # torch.func.jvp can trace through; the F.scaled_dot_product_attention
        # (AttnProcessor2_0) does not yet support forward AD as of torch 2.2.
        pipe.unet.set_attn_processor(AttnProcessor())
        # VAE also uses scaled_dot_product_attention in its mid_block; same fix.
        try:
            pipe.vae.set_attn_processor(AttnProcessor())
        except Exception:
            pass
        # Attention slicing reduces the attention-matrix peak memory under JVP
        # by processing heads sequentially. Necessary at 512² on 8 GB.
        try:
            pipe.unet.set_attention_slice("auto")
        except Exception:
            pass

        self.pipe = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler

        # disable grads on all params; we only need forward + JVP
        for p in self.unet.parameters(): p.requires_grad_(False)
        for p in self.vae.parameters(): p.requires_grad_(False)
        for p in self.text_encoder.parameters(): p.requires_grad_(False)

        ts, alpha_t, alpha_prev = _alphas_for_steps(
            self.scheduler, cfg.num_inference_steps
        )
        self.timesteps = ts.to(cfg.device)
        self.alpha_t = alpha_t.to(cfg.device).to(cfg.dtype)
        self.alpha_prev = alpha_prev.to(cfg.device).to(cfg.dtype)

        # h-space hook state
        self._hook_handle = None
        self._h_alpha: Optional[torch.Tensor] = None
        self._h_v: Optional[torch.Tensor] = None
        self._h_active: bool = False

        # install permanent hook (no-op unless _h_active = True)
        def mid_hook(_mod, _inp, out):
            if self._h_active and self._h_v is not None and self._h_alpha is not None:
                return out + self._h_alpha * self._h_v
            return out

        self._hook_handle = self.unet.mid_block.register_forward_hook(mid_hook)

    # ---------------------------------------------------------------- prompts
    @torch.no_grad()
    def encode_prompt(self, prompt: str, neg_prompt: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(
            [prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors="pt"
        ).to(self.cfg.device)
        ntoks = self.tokenizer(
            [neg_prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors="pt"
        ).to(self.cfg.device)
        cond = self.text_encoder(toks.input_ids)[0]
        uncond = self.text_encoder(ntoks.input_ids)[0]
        return cond, uncond

    # ---------------------------------------------------------------- single eps
    def epsilon(self, x_t: torch.Tensor, t_idx: int,
                cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """Classifier-free-guided epsilon at step index t_idx.

        Sequential UNet calls (no batch doubling) so attention-matrix peak
        memory under JVP is ~half — important at 512² on 8 GB.
        """
        t = self.timesteps[t_idx]
        e_u = self.unet(x_t, t, encoder_hidden_states=uncond).sample
        e_c = self.unet(x_t, t, encoder_hidden_states=cond).sample
        return e_u + self.cfg.guidance_scale * (e_c - e_u)

    def ddim_step(self, x_t: torch.Tensor, eps: torch.Tensor,
                  t_idx: int) -> torch.Tensor:
        """One η=0 DDIM step. JVP-clean: no .item() / .cpu()."""
        at = self.alpha_t[t_idx]
        ap = self.alpha_prev[t_idx]
        # pred x0
        pred_x0 = (x_t - (1 - at).sqrt() * eps) / at.sqrt()
        # direction
        dirn = (1 - ap).sqrt() * eps
        return ap.sqrt() * pred_x0 + dirn

    # ---------------------------------------------------------------- full sample
    @torch.no_grad()
    def sample(self, prompt: str, seed: int = 0, neg_prompt: str = "") -> dict:
        cond, uncond = self.encode_prompt(prompt, neg_prompt)
        H = W = self.cfg.resolution // 8
        gen = torch.Generator(device=self.cfg.device).manual_seed(seed)
        x = torch.randn(
            (1, 4, H, W), generator=gen,
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        latents = []
        for i in range(self.cfg.num_inference_steps):
            eps = self.epsilon(x, i, cond, uncond)
            x = self.ddim_step(x, eps, i)
            latents.append(x.detach())
        image = self.decode_image(x)
        return {"x_final": x, "image": image, "latents": latents,
                "cond": cond, "uncond": uncond}

    @torch.no_grad()
    def decode_image(self, x_latent: torch.Tensor) -> torch.Tensor:
        """VAE decode → [-1, 1] image."""
        z = x_latent / self.vae.config.scaling_factor
        img = self.vae.decode(z).sample
        return img.clamp(-1, 1)

    # ---------------------------------------------------------------- h-perturb sampler
    def sample_with_h_perturb(
        self, prompt: str, seed: int, t_edit_idx: int,
        v: torch.Tensor, alpha: torch.Tensor,
        neg_prompt: str = "", decode: bool = True,
        start_x: Optional[torch.Tensor] = None,
        cond_uncond: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Run DDIM sampling, injecting alpha*v into mid_block at step t_edit_idx.

        alpha: scalar fp tensor (will be tracked under jvp).
        v:     h-space direction with the same shape as mid_block output.

        Returns x_latent at t=0 (and decoded image if decode=True). The
        chain through DDIM steps preceding t_edit_idx is run without the
        hook active (so alpha doesn't affect those); from t_edit_idx
        onward the hook adds alpha*v at every mid_block call — i.e. the
        perturbation persists. This matches Asyrp's edit-interval
        formulation.
        """
        if cond_uncond is None:
            with torch.no_grad():
                cond, uncond = self.encode_prompt(prompt, neg_prompt)
        else:
            cond, uncond = cond_uncond

        if start_x is None:
            H = W = self.cfg.resolution // 8
            gen = torch.Generator(device=self.cfg.device).manual_seed(seed)
            x = torch.randn(
                (1, 4, H, W), generator=gen,
                device=self.cfg.device, dtype=self.cfg.dtype
            )
        else:
            x = start_x

        self._h_v = v
        self._h_alpha = alpha
        self._h_active = False
        try:
            for i in range(self.cfg.num_inference_steps):
                if i == t_edit_idx:
                    self._h_active = True
                eps = self.epsilon(x, i, cond, uncond)
                x = self.ddim_step(x, eps, i)
        finally:
            self._h_active = False
            self._h_v = None
            self._h_alpha = None

        if decode:
            img = self.decode_image(x)
            return x, img
        return x, None

    # ---------------------------------------------------------------- h-space dim
    def h_space_shape(self) -> tuple[int, ...]:
        """Shape of mid_block output at current resolution.

        Probe by running a single random latent through the UNet down+mid path.
        """
        H = W = self.cfg.resolution // 8
        x = torch.randn(1, 4, H, W, device=self.cfg.device, dtype=self.cfg.dtype)
        cond, uncond = self.encode_prompt("a photo", "")
        captured = {}
        def grab(_m, _i, o):
            captured["h"] = o
        hh = self.unet.mid_block.register_forward_hook(grab)
        try:
            with torch.no_grad():
                _ = self.unet(x, self.timesteps[0], encoder_hidden_states=cond).sample
        finally:
            hh.remove()
        return tuple(captured["h"].shape)
