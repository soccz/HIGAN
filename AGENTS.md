# HIGAN research workflow

- Current research entry point: `INTERACTION.md`; continuity: `MEMORY.md`.
- The active goal is intervention interpretation methodology. Generators are experimental
  objects; image editing product work is outside the current research scope.
- Keep new experiments in `research/interpretation_method/`. Preserve historical
  manuscripts and evidence, including preexisting uncommitted changes.
- Freeze protocol, predictors, tolerances, and fresh evaluation seeds before evaluating.
  Never report directly observed joint outputs as unseen predictions.
- Count independent anchors, not direction-pair rows, for uncertainty estimates.
- Frozen runs include source and input hashes. Add a new runner for new experiments
  rather than invalidating an existing run's measurement source snapshot.
- Tests: `python3 -m unittest discover -s research/interpretation_method -p 'test_*.py' -v`.
- Use each experiment's verifier and an independent saved-case rerender before reporting.
- Prediction v1 tensor audits require the runner's four reduction threads:
  `env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 research/interpretation_method/verify_prediction.py ...`.
  Different reduction thread counts can change final float64 bits; preserve frozen sources.
- GPU access may require the approved execution environment; do not stop other GPU jobs.
