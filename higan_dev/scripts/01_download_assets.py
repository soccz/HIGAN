"""Clone genforce/higan and download bedroom256 generator + boundaries + w_1k.

Run from the higan_dev/ directory:
    python scripts/01_download_assets.py
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


HIGAN_REPO = "https://github.com/genforce/higan.git"
GENERATOR_URL = "https://www.dropbox.com/s/h1w7ld4hsvte5zf/stylegan_bedroom256_generator.pth?dl=1"
W1K_URL = "https://www.dropbox.com/s/hwjyclj749qtp89/order_w.npy?dl=1"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} already exists ({dst.stat().st_size:,} bytes)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}\n   ->  {dst}")
    urlretrieve(url, dst)
    print(f"[done] {dst.stat().st_size:,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="root for downloaded assets")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = data_dir / "higan_repo"

    if not repo_dir.exists():
        sh(["git", "clone", "--depth", "1", HIGAN_REPO, str(repo_dir)])
    else:
        print(f"[skip] {repo_dir} already cloned")

    download(
        GENERATOR_URL,
        repo_dir / "models" / "pretrain" / "pytorch" / "stylegan_bedroom256_generator.pth",
    )
    download(W1K_URL, repo_dir / "order_w_1k.npy")

    # Sanity: list boundary files
    bdir = repo_dir / "boundaries" / "stylegan_bedroom"
    if bdir.exists():
        print(f"[info] boundaries available at {bdir}:")
        for p in sorted(bdir.iterdir()):
            print(f"   - {p.name}")
    else:
        print(f"[warn] boundaries dir not found at {bdir}")

    print("\nAll assets ready.")


if __name__ == "__main__":
    main()
