#!/usr/bin/env python3
"""Preflight harness for the TMLR submission package.

This is a local gate: it audits the PDF, source archive, evidence manifest, and
headline numeric claims before upload. It intentionally lives outside the
submission tarball so that the forbidden-pattern checks do not ship their own
blocked strings to reviewers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "note" / "submission"
ARCHIVE = ROOT / "note" / "tmlr_submission.tar.gz"
PDF = SUB / "tmlr_predict.pdf"
TEX = SUB / "tmlr_predict.tex"

EXPECTED_ARCHIVE = {
    "tmlr_predict.tex",
    "tmlr_predict.pdf",
    "tmlr.sty",
    "tmlr.bst",
    "fancyhdr.sty",
    "math_commands.tex",
    "refs.bib",
    "fig1_floor.pdf",
    "fig2_negative.pdf",
    "evidence/README.md",
    "evidence/bedroom_seed32_fd_vs_exact_metrics.json",
    "evidence/bedroom_seed32_trilemma_table.json",
    "evidence/bedroom_seed33_fd_vs_exact_metrics.json",
    "evidence/bedroom_seed33_trilemma_table.json",
    "evidence/compute_trilemma_table.py",
    "evidence/fd_trilemma_robustness_summary.json",
    "evidence/fd_trilemma_split_summary.json",
    "evidence/ffhq_lod0_seed31_fd_vs_exact_metrics.json",
    "evidence/ffhq_lod0_seed31_trilemma_table.json",
    "evidence/ffhq_lod0_seed32_fd_vs_exact_metrics.json",
    "evidence/ffhq_lod0_seed32_trilemma_table.json",
    "evidence/ffhq_lod0_seed33_fd_vs_exact_metrics.json",
    "evidence/ffhq_lod0_seed33_trilemma_table.json",
    "evidence/ffhq_lod1_fd_vs_exact_metrics.json",
    "evidence/ffhq_lod1_trilemma_table.json",
    "evidence/ffhq_lod2_fd_vs_exact_metrics.json",
    "evidence/ffhq_lod2_trilemma_table.json",
    "evidence/ffhq_resolution_metrics.json",
    "evidence/prediction_multianchor_table.json",
    "evidence/prediction_single_anchor_gonogo.json",
    "evidence/run_ffhq_fd_vs_exact.py",
    "evidence/run_ffhq_resolution_invariance.py",
    "evidence/stylegan_fd_vs_exact_table2.json",
    "evidence/stylegan_fp64_precision_control.json",
    "evidence/toy_table_groundtruth.json",
    "evidence/trilemma_table.json",
}

IDENTITY_PATTERNS = [
    "Jihong",
    "ether9073",
    "soccz",
    r"/home/",
    r"/mnt/",
]

INTERNAL_NOTE_PATTERNS = [
    "TODO",
    "FIXME",
    "XXX",
    "HACK",
    "Claude",
    "GPT",
    "handoff",
    "session",
    "내 추천",
]

DEAD_LINE_PATTERNS = [
    "predicts edit-risk",
    "curvature is useless",
    "overturn",
]

TEX_RISK_PATTERNS = [
    r"would.*bury",
    r"bury.*entirely",
    r"sits on (the .45|that floor)",
    r"lies well below",
    r"partial correlations.*floor",
    r"below the finite-difference floor",
    r"inherits the .*magnitude floor",
    r"conservative for milder",
    r"did not test other generators",
    r"Exact higher-order automatic differentiation is not new",
    r"not a new higher-order AD algorithm",
    r"carries no algorithmic novelty",
    r"not a general claim about curvature",
    r"we do not claim",
]

REFERENCE_SENTINELS = {
    "aoshima2023": ["Deep Curvilinear Editing", "CVPR", "2023", "arXiv:2211.14573"],
    "arvanitidis2018": ["Latent Space Oddity", "ICLR", "2018", "arXiv:1710.11379"],
    "baydin2018": ["Automatic Differentiation in Machine Learning", "JMLR", "2018"],
    "chen2018": ["Metrics for Deep Generative Models", "AISTATS", "2018", "arXiv:1711.01204"],
    "cobb2024": ["Second-Order Forward-Mode Automatic Differentiation", "2024", "arXiv:2408.10419"],
    "fike2011": ["Hyper-Dual Numbers", "AIAA", "2011", "AIAA 2011-886"],
    "griewank2008": ["Evaluating Derivatives", "SIAM", "2008"],
    "hewitt2019": ["Designing and Interpreting Probes", "EMNLP", "2019"],
    "higham2002": ["Accuracy and Stability of Numerical Algorithms", "SIAM", "2002"],
    "karras2019": ["A Style-Based Generator Architecture", "CVPR", "2019", "arXiv:1812.04948"],
    "koulakis2026": ["The Data Manifold under the Microscope", "Koulakis", "Seibold", "2026", "arXiv:2606.15760"],
    "lee2023": ["On Explicit Curvature Regularization", "TAG-ML", "2023", "arXiv:2309.10237"],
    "locatello2019": ["Challenging Common Assumptions", "ICML", "2019", "arXiv:1811.12359"],
    "paszke2019": ["PyTorch", "NeurIPS", "2019"],
    "pearlmutter1994": ["Fast Exact Multiplication by the Hessian", "Neural Computation", "1994"],
    "peebles2020": ["The Hessian Penalty", "ECCV", "2020", "arXiv:2008.10599"],
    "press2007": ["Numerical Recipes", "Cambridge University Press", "2007"],
    "shao2018": ["The Riemannian Geometry of Deep Generative Models", "CVPR", "2018", "arXiv:1711.08014"],
    "shen2020": ["Interpreting the Latent Space of GANs", "CVPR", "2020", "arXiv:1907.10786"],
    "yang2018": ["Geodesic Clustering in Deep Generative Models", "2018", "arXiv:1809.04747"],
}


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"[OK] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"[FAIL] {msg}")


def run(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def load_json(path: Path):
    return json.loads(path.read_text())


def approx(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


def archive_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


def archive_all_members(path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(path, "r:gz") as tf:
        return tf.getmembers()


def archive_texts(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    text_suffixes = {".tex", ".sty", ".bst", ".bib", ".md", ".json", ".py"}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if Path(m.name).suffix.lower() not in text_suffixes:
                continue
            data = tf.extractfile(m).read()
            out.append((m.name, data.decode("utf-8", "ignore")))
    return out


def compile_pdf(g: Gate) -> None:
    tectonic = os.environ.get("TECTONIC")
    candidates = [tectonic] if tectonic else []
    candidates += [str(ROOT / "tmp" / "tectonic"), "/home/soccz/22tb/tmp/tectonic", shutil.which("tectonic") or ""]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if not exe:
        g.warn("tectonic not found; skipped compile gate")
        return
    env = os.environ.copy()
    env["TMPDIR"] = str(ROOT / "tmp") if (ROOT / "tmp").exists() else env.get("TMPDIR", "/tmp")
    cp = run([exe, "tmlr_predict.tex"], cwd=SUB, env=env)
    if cp.returncode != 0:
        g.fail("LaTeX compile failed\n" + cp.stdout[-2000:])
    elif "error:" in cp.stdout.lower() or "undefined references" in cp.stdout.lower():
        g.fail("LaTeX compile reported errors/undefined refs\n" + cp.stdout[-2000:])
    else:
        g.ok("LaTeX compile exit 0")


def check_archive(g: Gate) -> None:
    if not ARCHIVE.exists():
        g.fail(f"missing archive: {ARCHIVE}")
        return
    all_members = archive_all_members(ARCHIVE)
    unsafe_members = [
        m.name for m in all_members
        if m.name.startswith("/") or ".." in Path(m.name).parts
    ]
    if unsafe_members:
        g.fail("archive contains unsafe member paths: " + ", ".join(sorted(unsafe_members)))
    else:
        g.ok("archive member paths are relative and traversal-safe")

    non_files = [m.name for m in all_members if not m.isfile()]
    if non_files:
        g.fail("archive contains non-file members: " + ", ".join(sorted(non_files)))
    else:
        g.ok("archive contains regular files only")

    metadata_hits = []
    metadata_patterns = IDENTITY_PATTERNS + [r"\\", r"[A-Za-z]:/"]
    for m in all_members:
        meta = f"{m.uname} {m.gname}"
        for pat in metadata_patterns:
            if re.search(pat, meta, flags=re.IGNORECASE):
                metadata_hits.append(f"{m.name}: {meta}")
                break
    if metadata_hits:
        g.fail("archive owner/group metadata leak patterns found: " + "; ".join(metadata_hits[:20]))
    else:
        g.ok("archive owner/group metadata leak scan clean")

    members = archive_members(ARCHIVE)
    missing = sorted(EXPECTED_ARCHIVE - members)
    extra = sorted(members - EXPECTED_ARCHIVE)
    if missing:
        g.fail("archive missing expected files: " + ", ".join(missing))
    else:
        g.ok("archive contains all expected files")
    if extra:
        g.fail("archive has unexpected files: " + ", ".join(extra))
    else:
        g.ok("archive has no unexpected files")

    forbidden_name_bits = ["__pycache__", ".DS_Store", ".ipynb_checkpoints", "pre_surgery"]
    bad_names = [m for m in members if any(bit in m for bit in forbidden_name_bits)]
    if bad_names:
        g.fail("archive contains generated/internal paths: " + ", ".join(sorted(bad_names)))
    else:
        g.ok("archive has no generated/cache/internal paths")

    hits = []
    patterns = IDENTITY_PATTERNS + INTERNAL_NOTE_PATTERNS + DEAD_LINE_PATTERNS
    for name, text in archive_texts(ARCHIVE):
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(f"{name}: {pat}")
    if hits:
        g.fail("archive text leak patterns found: " + "; ".join(hits[:20]))
    else:
        g.ok("archive text leak scan clean")


def check_pdf(g: Gate) -> None:
    if not PDF.exists():
        g.fail(f"missing PDF: {PDF}")
        return
    info = run(["pdfinfo", str(PDF)])
    if info.returncode != 0:
        g.fail("pdfinfo failed\n" + info.stdout)
        return
    for pat in IDENTITY_PATTERNS:
        if re.search(pat, info.stdout, flags=re.IGNORECASE):
            g.fail(f"PDF metadata leak pattern found: {pat}")
            return
    pages = None
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
    if pages is None:
        g.fail("could not read PDF page count")
    elif pages > 12:
        g.fail(f"PDF is {pages} pages; expected <= 12 for this package")
    else:
        g.ok(f"PDF page count {pages}")

    text_out = run(["pdftotext", str(PDF), "-"])
    if text_out.returncode != 0:
        g.fail("pdftotext failed\n" + text_out.stdout)
        return
    text = text_out.stdout
    for needle in ["[?]", "??", "undefined"]:
        if needle in text:
            g.fail(f"PDF contains unresolved marker: {needle}")
            return
    if "Anonymous authors" not in text:
        g.fail("PDF does not contain expected anonymous-author marker")
    else:
        g.ok("PDF has anonymous-author marker and no unresolved refs")
    for pat in IDENTITY_PATTERNS + DEAD_LINE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            g.fail(f"PDF leak/risk pattern found: {pat}")
            return
    g.ok("PDF leak/risk scan clean")


def check_tex(g: Gate) -> None:
    text = TEX.read_text()
    hits = []
    for pat in TEX_RISK_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)
    if hits:
        g.fail("TeX overclaim/dead-line patterns found: " + ", ".join(hits))
    else:
        g.ok("TeX overclaim/dead-line sweep clean")


def bib_entries() -> dict[str, str]:
    text = (SUB / "refs.bib").read_text()
    matches = list(re.finditer(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))
    entries: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[m.group(1)] = text[start:end]
    return entries


def citation_keys() -> set[str]:
    text = TEX.read_text()
    keys: set[str] = set()
    for m in re.finditer(r"\\cite\w*\{([^}]+)\}", text):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def check_references(g: Gate) -> None:
    entries = bib_entries()
    cites = citation_keys()
    bib_keys = set(entries)

    duplicate_count = len(re.findall(r"^@\w+\{([^,]+),", (SUB / "refs.bib").read_text(), flags=re.MULTILINE))
    if duplicate_count != len(bib_keys):
        g.fail("refs.bib has duplicate keys")
        return

    missing = sorted(cites - bib_keys)
    unused = sorted(bib_keys - cites)
    if missing:
        g.fail("TeX cites missing BibTeX keys: " + ", ".join(missing))
    elif unused:
        g.fail("refs.bib has uncited entries: " + ", ".join(unused))
    else:
        g.ok(f"reference key audit clean ({len(cites)} cited entries)")

    sentinel_missing = sorted(set(REFERENCE_SENTINELS) - bib_keys)
    if sentinel_missing:
        g.fail("reference sentinel keys missing: " + ", ".join(sentinel_missing))
        return

    bad = []
    for key, needles in REFERENCE_SENTINELS.items():
        block = entries[key].lower()
        for needle in needles:
            if needle.lower() not in block:
                bad.append(f"{key}: {needle}")
    if bad:
        g.fail("reference metadata sentinel mismatch: " + "; ".join(bad))
    else:
        g.ok("reference metadata sentinels match expected title/venue/year/arXiv fields")

    broad_authors = []
    for key, block in entries.items():
        if key != "paszke2019" and re.search(r"\band others\b", block, flags=re.IGNORECASE):
            broad_authors.append(key)
    if broad_authors:
        g.fail("non-mega-author references use 'and others': " + ", ".join(sorted(broad_authors)))
    else:
        g.ok("reference author fields have no broad 'and others' except PyTorch")


def check_manifest(g: Gate) -> None:
    readme = SUB / "evidence" / "README.md"
    if not readme.exists():
        g.fail("missing evidence README manifest")
        return
    text = readme.read_text()
    required_phrases = [
        "Table 1",
        "Table 4",
        "full-resolution FFHQ",
        "prediction_multianchor_table.json",
        "run_ffhq_fd_vs_exact.py",
    ]
    missing = [p for p in required_phrases if p not in text]
    if missing:
        g.fail("evidence README missing phrases: " + ", ".join(missing))
    else:
        g.ok("evidence README covers main tables and replication audit")

    listed = re.findall(r"`([^`]+)`", text)
    missing_files = []
    for item in listed:
        if "/" in item or item.endswith((".json", ".py", ".md")):
            # Commands and scalar values also appear in backticks; only validate evidence paths/files.
            p = SUB / "evidence" / item if not item.startswith("evidence/") else SUB / item
            if item.startswith("python "):
                continue
            if item.endswith((".json", ".py", ".md")) and not p.exists():
                missing_files.append(item)
    if missing_files:
        g.fail("evidence README names missing files: " + ", ".join(sorted(set(missing_files))))
    else:
        g.ok("evidence README file references resolve")


def check_numbers(g: Gate) -> None:
    ev = SUB / "evidence"
    tri = load_json(ev / "trilemma_table.json")
    s = tri["summary"]
    if (s["n_records"], s["n_steps"]) != (96, 16):
        g.fail("Table 2 trilemma summary has wrong shape")
    elif (s["second_order_magnitude_best_step"], s["second_order_bias_best_step"], s["second_order_rank_best_step"]) != (3.0, 2.0, 0.2):
        g.fail("Table 2 best steps changed")
    else:
        g.ok("Table 2 trilemma summary matches expected best steps")

    toy = load_json(ev / "toy_table_groundtruth.json")
    if not approx(toy["jvp2_relerr"], 2.5e-17, 1e-17):
        g.fail(f"toy jvp2 relerr unexpected: {toy['jvp2_relerr']}")
    elif not approx(toy["fd2_best_relerr"], 8.6e-9, 5e-10):
        g.fail(f"toy fd2 best relerr unexpected: {toy['fd2_best_relerr']}")
    else:
        g.ok("Table 1 toy numbers match expected values")

    robust = load_json(ev / "fd_trilemma_robustness_summary.json")
    expected_keys = [
        "bedroom_seed31", "bedroom_seed32", "bedroom_seed33",
        "ffhq_lod0_seed31", "ffhq_lod0_seed32", "ffhq_lod0_seed33",
        "ffhq_lod1", "ffhq_lod2",
    ]
    keys = [r["key"] for r in robust]
    if keys != expected_keys:
        g.fail(f"Table 4 robustness keys changed: {keys}")
    else:
        g.ok("Table 4 robustness has expected 8 sweeps")
    by_key = {r["key"]: r for r in robust}
    checks = {
        "bedroom_seed31": (3.0, 2.0, 0.2),
        "bedroom_seed32": (3.0, 1.5, 0.2),
        "bedroom_seed33": (3.0, 2.0, 0.3),
        "ffhq_lod0_seed31": (5.0, 1.5, 0.3),
        "ffhq_lod0_seed32": (5.0, 1.5, 0.1),
        "ffhq_lod0_seed33": (5.0, 1.5, 0.2),
        "ffhq_lod1": (8.0, 2.0, 0.3),
        "ffhq_lod2": (8.0, 3.0, 0.3),
    }
    for key, vals in checks.items():
        row = by_key[key]
        got = (
            row["second_order_magnitude_best_step"],
            row["second_order_bias_best_step"],
            row["second_order_rank_best_step"],
        )
        if got != vals:
            g.fail(f"{key} best steps changed: got {got}, expected {vals}")
            break
    else:
        g.ok("Table 4 best-step assertions pass")

    pred = load_json(ev / "prediction_multianchor_table.json")
    partials = [round(x, 3) for x in pred["partial_given_jvp_distribution"]]
    if partials != [0.098, 0.193, -0.151, 0.373, 0.33]:
        g.fail(f"multi-anchor partials changed: {partials}")
    elif pred["n_anchors_CI_excludes_0"] != 0:
        g.fail("multi-anchor CI exclusion count changed")
    else:
        g.ok("replication-audit multi-anchor numbers match expected values")

    res = load_json(ev / "ffhq_resolution_metrics.json")
    if all(abs(v["r"] - 1.0) < 1e-12 for v in res["rank_stability"].values()):
        g.ok("FFHQ resolution rank-stability assertions pass")
    else:
        g.fail("FFHQ resolution rank-stability changed")


def check_archive_extract_smoke(g: Gate) -> None:
    with tempfile.TemporaryDirectory(prefix="tmlr_preflight_") as td:
        td_path = Path(td)
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            tf.extractall(td_path, filter="data")
        for name in ["tmlr_predict.tex", "refs.bib", "evidence/README.md"]:
            if not (td_path / name).exists():
                g.fail(f"archive extraction smoke missing {name}")
                return
        g.ok("archive extraction smoke passed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    g = Gate()
    if not args.skip_compile:
        compile_pdf(g)
    check_tex(g)
    check_references(g)
    check_pdf(g)
    check_archive(g)
    check_manifest(g)
    check_numbers(g)
    check_archive_extract_smoke(g)

    print()
    if g.failures:
        print("VERDICT: FIX-BEFORE-SUBMIT")
        print("Failures:")
        for item in g.failures:
            print(f"- {item}")
        return 1
    if g.warnings:
        print("VERDICT: SUBMIT-READY with warnings")
        for item in g.warnings:
            print(f"- {item}")
        return 0
    print("VERDICT: SUBMIT-READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
