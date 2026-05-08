"""Train the custom W+ encoder via synthetic supervision.

Example:
    python scripts/03_train_encoder.py --config configs/default.yaml \
        --override train.num_iters=20000 train.batch_size=8
"""
from __future__ import annotations
import argparse

from higan_dev.config import Config, resolve
from higan_dev.encoder.train import train


def _apply_override(cfg: Config, dotted: str) -> None:
    key, val = dotted.split("=", 1)
    obj = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    cur = getattr(obj, parts[-1])
    if isinstance(cur, bool):
        new = val.lower() in ("1", "true", "yes")
    elif isinstance(cur, int):
        new = int(val)
    elif isinstance(cur, float):
        new = float(val)
    else:
        new = val
    setattr(obj, parts[-1], new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted overrides like train.batch_size=4")
    args = ap.parse_args()

    cfg = Config.load(resolve(args.config))
    for o in args.override:
        _apply_override(cfg, o)

    print("=== effective config ===")
    print(f"backbone={cfg.encoder.backbone}  batch={cfg.train.batch_size}  "
          f"iters={cfg.train.num_iters}  lr={cfg.train.lr}  amp={cfg.train.amp}")
    print(f"loss_weights={cfg.train.loss_weights}")
    print()
    train(cfg)


if __name__ == "__main__":
    main()
