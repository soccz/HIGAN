from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Paths:
    higan_repo: str
    generator_ckpt: str
    w1k_npy: str
    boundaries_dir: str
    out_dir: str


@dataclass
class GeneratorCfg:
    model_name: str
    resolution: int
    num_layers: int
    latent_dim: int


@dataclass
class EncoderCfg:
    backbone: str
    input_size: int
    num_layers: int
    latent_dim: int


@dataclass
class TrainCfg:
    batch_size: int
    num_iters: int
    lr: float
    warmup_iters: int
    log_every: int
    ckpt_every: int
    vis_every: int
    seed: int
    device: str
    loss_weights: dict[str, float]
    amp: bool


@dataclass
class InvOptimCfg:
    num_steps: int
    lr: float
    num_inits: int
    loss_weights: dict[str, float]


@dataclass
class CamCfg:
    delta: float
    num_samples: int
    steps: int


@dataclass
class Config:
    paths: Paths
    generator: GeneratorCfg
    encoder: EncoderCfg
    train: TrainCfg
    inversion_optim: InvOptimCfg
    cam: CamCfg

    @classmethod
    def load(cls, yaml_path: str | Path) -> "Config":
        with open(yaml_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(
            paths=Paths(**raw["paths"]),
            generator=GeneratorCfg(**raw["generator"]),
            encoder=EncoderCfg(**raw["encoder"]),
            train=TrainCfg(**raw["train"]),
            inversion_optim=InvOptimCfg(**raw["inversion_optim"]),
            cam=CamCfg(**raw["cam"]),
        )


def project_root() -> Path:
    # higan_dev/higan_dev/config.py -> higan_dev/
    return Path(__file__).resolve().parents[1]


def resolve(cfg_path: str) -> Path:
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return project_root() / p
