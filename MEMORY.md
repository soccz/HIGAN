# HIGAN Experiment Continuity

## 2026-06-22 TMLR Submission Guide-Fix Pass

- User requested final strict TMLR-guide fixes after the 20-lens audit.
- Applied the non-experimental guide fixes in `note/submission/tmlr_predict.tex`:
  - reduced "first-order displacement is the safe default" to one scoped occurrence in §7;
  - removed abstract/contribution/conclusion self-deflation around the anchor-fragile audit;
  - rewrote §5 and related work so the generator-scale trilemma/measurement decision leads, with classical FD/AD machinery demoted to supporting ancestry;
  - deleted posture/meta language ("lesson for this paper", "reported at the strength...", "in the spirit...");
  - kept the audit as a positive anchor-stability decision enabled by the exact reference.
- Added a verified arXiv HiGAN source citation as `yang2020semantic` in `note/submission/refs.bib` and cited it for the bedroom HiGAN boundaries.
- Did not add the optional FD-substituted §6 partial-correlation result: the submitted multi-anchor JSON stores summary-only anchor results, while the FD sweep records use a different direction/anchor set, so the re-aggregation is not valid from shipped artifacts.
- Rebuilt `note/submission/tmlr_predict.pdf` and regenerated `note/tmlr_submission.tar.gz`.
- Verification: `env TMPDIR=/home/soccz/22tb/tmp /home/soccz/22tb/tmp/tectonic tmlr_predict.tex` exited 0; `python note/tools/tmlr_submission_preflight.py --skip-compile` returned `VERDICT: SUBMIT-READY`; PDF page count 11; citation/unresolved marker sweep clean; archive member list clean.

## 2026-05-24 Control Experiments

- Current claim focus: generator-side curvature/risk is a usable control signal, not just a post-hoc explanation.
- Running queue state at 2026-05-24 10:32 KST:
  - v4 negative controls parent PID `1191403`, currently resuming `control_v4_risk_signal_negatives`.
  - Detached v5 queue PID `1269556` waits for PID `1191403`, then runs `experiments/protocols/control_v5_risk_power_sensitivity.json`.
  - v5 queue logs: `paper/logs/control_v5_after_v4_queue/queue.log`.
  - v4 resume logs: `paper/logs/control_v4_risk_signal_negatives_resume/`.
  - v5 run logs: `paper/logs/control_v5_risk_power_sensitivity/`.
- Newly added reproducibility code:
  - `paper/experiments/control/run_protocol_after_pid.py`
  - `paper/experiments/protocols/control_v5_risk_power_sensitivity.json`
  - `paper/experiments/control/run_risk_aware_repeat_summary.py`
  - `paper/experiments/control/run_risk_signal_negative_summary.py`
  - aggregate blocks in `paper/experiments/aggregate_results.py`
- Next check:
  - Confirm v4 summary exists at `paper/experiments/out/control_risk_signal_negative_controls/metrics.json`.
  - Confirm v5 metrics exist under `paper/experiments/out/control_risk_power_sensitivity/`.
  - Re-run aggregate if needed with `python3 experiments/aggregate_results.py` from `paper/`.

## 2026-05-24 Follow-up Queues

- Reproducibility/code hardening added:
  - `run_control_protocol.py` now errors on unknown `--only` keys, duplicate protocol keys, missing scripts, or empty selections.
  - `run_risk_aware_controller.py` supports actual LPIPS via `--true-lpips`.
  - `run_risk_aware_controller.py` supports the expanded fixed-alpha `editing_head_to_head` candidate-source format.
  - `paper/experiments/control/audit_control_campaign.py` audits protocol output completeness and protocol metadata hashes.
- Additional locked protocols:
  - `control_v6_true_lpips_validation.json`: five true-LPIPS seed runs plus summary.
  - `control_v7_expanded_candidate_universe.json`: five expanded-candidate-universe seed runs plus summary.
  - `control_v8_campaign_audit.json`: final non-GPU audit after queued GPU work.
- Initial detached queue chain at 2026-05-24 10:43 KST was replaced at
  11:12 KST with a guard-based chain because v4 PID `1191403` was still
  under the Codex app-server parent.  Old wait-only queue PIDs
  `1269556`, `1370108`, `1371316`, and `1372147` were stopped.
- Current detached guard queue chain at 2026-05-24 11:12 KST:
  - v4 missing-output guard PID `1642159` waits for current v4 PID `1191403`.
  - v5 queue PID `1643509` waits for guard PID `1642159`.
  - v6 queue PID `1644551` waits for v5 queue PID `1643509`.
  - v7 queue PID `1645491` waits for v6 queue PID `1644551`.
  - v8 audit queue PID `1646393` waits for v7 queue PID `1645491`.
- Queue logs:
  - `paper/logs/control_v4_missing_after_current/queue.log`
  - `paper/logs/control_v5_after_v4_guard/queue.log`
  - `paper/logs/control_v6_after_v5_guard/queue.log`
  - `paper/logs/control_v7_after_v6_guard/queue.log`
  - `paper/logs/control_v8_after_v7_guard/queue.log`

## 2026-05-24 Fixed-Target Negative-Control Queue

- Live audit at about 12:55 KST:
  - v3 robustness complete: 6/6.
  - v4 risk-signal negatives complete: 11/11.
  - v5 risk-power sensitivity: 1/12 complete, 11 pending; current GPU run was `risk_power_0p5_seed_2028`.
  - v6 true-LPIPS validation: 0/6 pending.
  - v7 expanded-candidate-universe validation: 0/6 pending.
  - v9 fixed-target negatives: 0/16 pending.
- Added queue chain after existing v8 audit:
  - v9 queue PID `2582698` waits for v8 queue PID `1646393`, then runs `control_v9_fixed_target_negatives.json` with aggregate.
  - v10 queue PID `2583384` waits for v9 queue PID `2582698`, then runs `control_v10_campaign_audit.json` with aggregate.
- Queue logs:
  - `paper/logs/control_v9_after_v8_guard/queue.log`
  - `paper/logs/control_v10_after_v9_guard/queue.log`
- Rationale:
  - v9 is the stricter causal negative control: target magnitude is estimated from the full candidate universe, so actual-vs-shuffled/inverted risk cannot be explained by target magnitude changing with the selected mode.
  - v10 is the final metadata/output audit for v3/v4/v5/v6/v7/v9.

## 2026-05-24 Extended Control-Validation Queue

- User requested that background experiments keep running and cover all feasible
  main-paper validation axes.
- Added locked protocols:
  - `control_v11_extended_validation.json`: fixed-target true-LPIPS, sample
    budget sensitivity, and k-selection sensitivity.
  - `control_v12_threshold_magnitude_sensitivity.json`: gain-threshold and
    target-magnitude sensitivity.
  - `control_v13_extended_campaign_audit.json`: final audit including v3-v12.
- `aggregate_results.py` now prints the v11/v12 summary outputs when they exist.
- Verification before queueing:
  - `python3 experiments/control/run_control_protocol.py --protocol ... --dry-run`
    passed for v11, v12, and v13.
  - `python3 -m py_compile experiments/aggregate_results.py experiments/control/run_control_protocol.py experiments/control/run_protocol_after_pid.py`
    passed.
- Detached queue chain appended after existing v10:
  - v11 PID `3931878` waits for v10 PID `2583384`.
  - v12 PID `3932366` waits for v11 PID `3931878`.
  - v13 PID `3932885` waits for v12 PID `3932366`.
- Queue logs:
  - `paper/logs/control_v11_after_v10_guard/queue.log`
  - `paper/logs/control_v12_after_v11_guard/queue.log`
  - `paper/logs/control_v13_after_v12_guard/queue.log`

## 2026-05-24 Full Background Extension

- User again requested all feasible background experiments.
- Added more locked protocols:
  - `control_v14_true_lpips_fixed_target_negatives.json`: true-LPIPS,
    fixed-target shuffled/inverted risk negative controls, summarized against
    v11 actual fixed-target true-LPIPS outputs.
  - `control_v15_expanded_universe_true_lpips.json`: true-LPIPS validation on
    the expanded `editing_head_to_head` candidate universe with target source
    fixed to the universe.
  - `control_v16_estimator_stability.json`: larger held-out `n_test=512`
    validation plus probe-alpha 0.5/2.0 sensitivity.
  - `control_v17_full_campaign_audit.json`: final audit including v3-v16.
- `run_risk_signal_negative_summary.py` now conditionally supports
  `mean_lpips_true`; it still skips that metric for older proxy-only runs.
- `aggregate_results.py` now prints v14-v16 summaries when outputs exist.
- Verification before queueing:
  - Dry-run passed for v14, v15, v16, and v17.
  - `python3 -m py_compile experiments/aggregate_results.py experiments/control/run_risk_signal_negative_summary.py experiments/control/run_control_protocol.py experiments/control/run_protocol_after_pid.py`
    passed.
- Detached queue chain appended after v13:
  - v14 PID `3977667` waits for v13 PID `3932885`.
  - v15 PID `3978277` waits for v14 PID `3977667`.
  - v16 PID `3978703` waits for v15 PID `3978277`.
  - v17 PID `3979096` waits for v16 PID `3978703`.
- Queue logs:
  - `paper/logs/control_v14_after_v13_guard/queue.log`
  - `paper/logs/control_v15_after_v14_guard/queue.log`
  - `paper/logs/control_v16_after_v15_guard/queue.log`
  - `paper/logs/control_v17_after_v16_guard/queue.log`

## 2026-05-25 Main-Paper Cross-Domain Upgrade

- User goal: push the work toward CVPR/ICCV/ECCV main by making the central
  claim a reproducible control-signal result rather than a post-hoc
  interpretability result.
- Code hardening added:
  - `audit_control_campaign.py` now fails on missing protocol paths instead of
    silently skipping them.
  - `run_crossdomain_signature.py` and `run_cross_domain_plate.py` now load
    pixel-rho values from `sample_scaling_*` metrics instead of hardcoded
    constants, and plot titles no longer assert unverified significance text.
  - `run_protocol_after_pid.py` accepts `--python-bin` so detached GPU queues
    use the pyenv Python that can see CUDA.
- New cross-domain control experiment:
  - `paper/experiments/control/run_cross_domain_risk_aware_controller.py`
  - `paper/experiments/protocols/control_v18_bedroom_cross_domain_controller.json`
  - `paper/experiments/protocols/control_v19_main_grade_audit.json`
- Verification before queueing:
  - `py_compile` passed for changed Python files.
  - v18/v19 protocol dry-runs passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
  - Tiny bedroom controller smoke run passed, including a true-LPIPS smoke.
  - Existing v3-v17 audit passed after strict missing-protocol patch.
  - Cross-domain signature and plate regenerated without hardcoded rho values.
- Detached queue started at 2026-05-25 22:10 KST:
  - v18 queue PID `1821354`, PPID=1, log:
    `paper/logs/control_v18_bedroom_cross_domain_queue/queue.log`.
  - v19 audit queue PID `1821736`, waits for PID `1821354`, log:
    `paper/logs/control_v19_after_v18_queue/queue.log`.
  - v18 current first run log:
    `paper/logs/control_v18_bedroom_cross_domain/bedroom_actual_seed_2027.log`.
- Next check:
  - Tail the v18 first-run log to estimate wall time.
  - After v18 completes, v19 should audit and aggregate automatically.
  - Final remaining publication-grade issue is still `git_dirty`: `paper/` is
    untracked, so final reproducibility package needs a clean commit/tag or
    archived snapshot.

## 2026-05-26 Church StyleGAN2 Control Extension

- v18/v19 completed at 2026-05-26 01:42 KST and aggregation updated
  `_aggregate_summary.txt`.
- v18 bedroom result summary:
  - risk-aware vs random/high-risk/low-risk is strong on seed means.
  - risk-aware vs gain-only is near-tie; do not claim risk always beats gain.
  - actual risk beats inverted and partly beats shuffled in negative controls,
    supporting the narrower "risk as control signal" claim.
- Code updates:
  - `run_cross_domain_risk_aware_controller.py` now supports `--domain church`
    using `domains.church.generator.ChurchGenerator`, attrs
    `clouds/sunny/vegetation`, and fixed predeclared church layer ranges.
  - `aggregate_results.py` now prints D12/D13 church cross-domain controller
    and negative-control summaries when available.
  - Added locked protocols:
    `control_v20_church_cross_domain_controller.json` and
    `control_v21_main_grade_audit_v2.json`.
- Verification before queueing:
  - Church true-LPIPS smoke run passed at
    `paper/experiments/out/_smoke_church_controller/metrics.json`.
  - `py_compile` passed for changed Python files.
  - v20/v21 JSON structural checks passed: duplicate keys 0, duplicate outs 0.
  - v20/v21 dry-runs passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
- Detached queue started at 2026-05-26 02:07 KST:
  - v20 queue PID `2613736`, PPID=1, log:
    `paper/logs/control_v20_church_cross_domain_queue/queue.log`.
  - v21 audit queue PID `2613988`, waits for PID `2613736`, log:
    `paper/logs/control_v21_after_v20_queue/queue.log`.
  - Current first run log:
    `paper/logs/control_v20_church_cross_domain/church_actual_seed_2027.log`.
- Next check:
  - Monitor `paper/logs/control_v20_church_cross_domain/church_actual_seed_2027.log`
    to estimate per-run wall time.
  - Count completed v20 outputs with:
    `find paper/experiments/out/control_church_cross_domain_true_lpips -maxdepth 2 -name metrics.json`.

## 2026-05-26 Queued Church Risk-Power Sensitivity

- User requested one more predicted experiment after the current queue.
- Added locked protocol:
  - `paper/experiments/protocols/control_v22_church_risk_power_sensitivity.json`
  - Purpose: test whether church StyleGAN2 controller behavior is robust to
    `risk_power=0.5` and `risk_power=2.0`, rather than tuned to the official
    `risk_power=1.0`.
  - Runs: 10 GPU runs total, 5 seeds for each power, plus two summary reducers.
- `aggregate_results.py` now prints D14 church risk-power sensitivity summaries.
- Verification before queueing:
  - `py_compile` passed for changed files.
  - v22 structural check passed: 12 experiments, duplicate keys 0,
    duplicate outs 0.
  - v22 dry-run passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
- Detached queue started at 2026-05-26 02:16 KST:
  - v22 queue PID `2633793`, PPID=1, waits for v21 PID `2613988`.
  - Queue log: `paper/logs/control_v22_after_v21_queue/queue.log`.
  - Run logs will be under:
    `paper/logs/control_v22_church_risk_power_sensitivity/`.

## 2026-05-26 Church Failure-Diagnosis Queue

- v20/v21/v22 completed by 2026-05-26 07:51 KST.  GPU was idle at the next
  check.
- Church StyleGAN2 result interpretation before v23:
  - Actual risk beats gain-only and high-risk.
  - Actual risk loses to random and low-risk baselines.
  - Actual risk still beats shuffled/inverted risk, so the signal is real, but
    the controller/claim must be narrowed.
- Added locked protocols:
  - `control_v23_church_failure_diagnosis.json`
  - `control_v24_main_grade_audit_v3.json`
- v23 tests two predeclared diagnostic axes:
  - `structured_only`: candidate universe is GANSpace+SeFa only, same controller.
  - `high_gain_floor`: full candidate universe, but `min_gain_quantile=0.75`.
- `aggregate_results.py` now prints D15 summaries for these two conditions.
- Verification before queueing:
  - `py_compile` passed for changed Python files.
  - v23/v24 structural checks passed: duplicate keys 0, duplicate outs 0.
  - v23/v24 dry-runs passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
- Detached queue started at 2026-05-26 09:48 KST:
  - v23 queue PID `2689400`, PPID=1, log:
    `paper/logs/control_v23_church_failure_diagnosis_queue/queue.log`.
  - v24 audit queue PID `2689444`, waits for v23 PID `2689400`, log:
    `paper/logs/control_v24_after_v23_queue/queue.log`.
  - First run log:
    `paper/logs/control_v23_church_failure_diagnosis/church_structured_only_seed_2027.log`.

## 2026-05-26 Structured-Only Risk-Signal Negative Queue

- v23/v24 completed by 2026-05-26 11:48 KST.  The next check at 12:04 KST
  showed no running GPU process.
- v23 result interpretation:
  - `structured_only` fixes the church random-baseline failure: risk-aware
    beats random and high-risk 5/5, and is near tie against gain-only/low-risk.
  - `high_gain_floor` does not fix the random/low-risk weakness.
  - This supports a narrower claim: the risk signal is actionable under a
    structured semantic candidate universe, but the broad full-universe
    controller claim remains too strong for main-paper wording.
- Added locked protocols:
  - `control_v25_church_structured_negative_controls.json`
  - `control_v26_main_grade_audit_v4.json`
- v25 purpose:
  - Test whether the v23 structured-only improvement depends on the actual
    curvature/risk signal, not just on restricting candidates to GANSpace+SeFa.
  - Runs shuffled and inverted risk controls for the same five church seeds and
    summarizes against the already completed v23 actual structured-only runs.
- `aggregate_results.py` now prints D16 structured-only church risk-signal
  negative controls when v25 completes.
- Verification before queueing:
  - `py_compile` passed for aggregate, controller, protocol runner, negative
    summary, audit, and queue scripts.
  - v25/v26 structural checks passed: duplicate keys 0, duplicate outs 0.
  - v25/v26 dry-runs passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
- Detached queue started at 2026-05-26 12:06 KST:
  - v25 queue PID `3093649`, PPID=1, log:
    `paper/logs/control_v25_church_structured_negative_controls_queue/queue.log`.
  - v26 audit queue PID `3093691`, PPID=1, waits for PID `3093649`, log:
    `paper/logs/control_v26_after_v25_queue/queue.log`.
  - Current first GPU child PID `3093653`, running:
    `church_structured_shuffled_seed_2027`.
  - Current first run log:
    `paper/logs/control_v25_church_structured_negative_controls/church_structured_shuffled_seed_2027.log`.
- Next check:
  - Monitor v25 first-run log and count outputs with:
    `find paper/experiments/out/control_church_structured_negative_controls -maxdepth 2 -name metrics.json`.
  - After v25 completes, v26 should audit and aggregate automatically.

## 2026-05-26 Main-Grade Follow-up Chain

- User requested all experiments needed for a CVPR/ICCV/ECCV-main attempt.
- Added reproducibility helper:
  - `paper/experiments/control/build_main_grade_followup_protocols.py`
  - It deterministically generates locked follow-up protocols v27-v30 from
    predeclared seed/config grids; it does not inspect results.
- Added locked protocols:
  - `control_v27_church_structured_confirmatory_replication.json`
    - 5 new independent seeds, `2032-2036`.
    - Actual/shuffled/inverted structured-only church runs.
    - `n_test=128`, true LPIPS, fixed target source `universe`.
    - Purpose: confirm the structured-only control result and risk-signal
      negative control on fresh seeds with a larger test set.
  - `control_v28_church_structured_estimator_stability.json`
    - 5 seeds, `2027-2031`.
    - Actual structured-only church runs with larger estimator budgets:
      `n_risk=16`, `n_probe=32`, `n_calib=32`, `n_test=128`.
    - Purpose: show the signal/controller is not an artifact of a tiny
      curvature/probe/calibration estimator.
  - `control_v29_church_structured_source_ablation.json`
    - 5 seeds each for GANSpace-only and SeFa-only candidate universes.
    - Purpose: test whether the structured-only claim is tied to one
      decomposition family.
  - `control_v30_main_grade_audit_v5.json`
    - Final non-GPU audit including v25 and v27-v29.
- `aggregate_results.py` now prints:
  - D17 structured-only confirmatory replication.
  - D17b confirmatory actual-vs-shuffled/inverted negatives.
  - D18 structured-only estimator stability.
  - D19 structured-source ablations.
- Verification before queueing:
  - `py_compile` passed for aggregate, protocol generator, controller,
    protocol runner, repeat summary, negative summary, audit, and queue scripts.
  - v27-v30 structural checks passed: duplicate keys 0, duplicate outs 0,
    unresolved summary inputs 0.
  - v27-v30 dry-runs passed with
    `/home/soccz/.pyenv/versions/3.12.9/bin/python`.
  - `git diff --check` passed for the changed files.
- Detached queue chain appended after v26 at 2026-05-26 12:15 KST:
  - v27 queue PID `3095791`, waits for v26 PID `3093691`.
  - v28 queue PID `3095805`, waits for v27 PID `3095791`.
  - v29 queue PID `3095819`, waits for v28 PID `3095805`.
  - v30 audit queue PID `3095834`, waits for v29 PID `3095819`.
- Queue logs:
  - `paper/logs/control_v27_after_v26_queue/queue.log`
  - `paper/logs/control_v28_after_v27_queue/queue.log`
  - `paper/logs/control_v29_after_v28_queue/queue.log`
  - `paper/logs/control_v30_after_v29_queue/queue.log`
- Current status when queued:
  - v25 first GPU child PID `3093653` was still running
    `church_structured_shuffled_seed_2027`.
  - GPU was active at about 95% util, so no idle gap remained.

## 2026-05-27 Main-Grade Follow-up Completion

- Checked at 2026-05-27 03:33 KST:
  - GPU idle, no running GPU processes.
  - Queue PIDs `3093649`, `3093691`, `3095791`, `3095805`, `3095819`,
    and `3095834` had exited.
  - v30 completed at 2026-05-26 20:03:08 KST.
- Final audit:
  - `control_v25_church_structured_negative_controls`: complete 11, pending 0,
    failed 0.
  - `control_v27_church_structured_confirmatory_replication`: complete 17,
    pending 0, failed 0.
  - `control_v28_church_structured_estimator_stability`: complete 6,
    pending 0, failed 0.
  - `control_v29_church_structured_source_ablation`: complete 12, pending 0,
    failed 0.
  - Remaining audit warnings are only `git_dirty is true or missing`.
- Key v25-v29 result interpretation:
  - v25 structured-only negatives are positive but modest:
    actual vs shuffled `dID=+0.0026`, true-LPIPS `-0.0048`, wins 3/5;
    actual vs inverted `dID=+0.0057`, true-LPIPS `-0.0123`, wins 3/5.
  - v27 fresh-seed confirmatory structured-only:
    beats random (`dID=+0.0096`, true-LPIPS `-0.0230`, LPIPS wins 4/5),
    high-risk (`dID=+0.0158`, true-LPIPS `-0.0308`, LPIPS wins 4/5),
    and low-risk (`dID=+0.0037`, true-LPIPS `-0.0086`, wins 4/5),
    but does not beat gain-only (`dID=-0.0012`, true-LPIPS `+0.0054`).
  - v27 fresh-seed shuffled/inverted negative controls are weak:
    actual vs shuffled true-LPIPS wins 2/5 and actual vs inverted wins 2/5.
  - v28 larger estimator budget:
    beats random and high-risk, but not low-risk; gain-only remains near tie.
  - v29 source ablation:
    GANSpace-only fails the risk-aware advantage.
    SeFa-only beats random/high-risk but not low-risk; gain-only is identical
    because selections collapse/tie in that reduced candidate universe.
- Current paper implication:
  - Strong claim is not "risk-aware dominates all baselines" and not "all
    structured sources".
  - Safer main claim: curvature/risk is a real but conditional control signal;
    it is useful in semantic/structured candidate universes, strongest against
    random/high-risk/corrupted-risk controls, and must be paired with candidate
    source diagnostics and failure characterization.

## 2026-05-27 Decisive Main-Claim Queue

- User requested all remaining CVPR/ICCV/ECCV-main experiments.
- Remaining scientific gaps after v30:
  - Existing controller often ties or loses to gain-only, so the paper needs a
    direct predictive-validity test: does rho predict damage after semantic gain
    is controlled?
  - The original gain/risk ratio is not enough as an actionable controller, so
    test a held-out gain-first/risk-tiebreak rule instead of retuning on results.
- Code added:
  - `run_cross_domain_risk_predictive.py`: evaluates every candidate at matched
    semantic target and reports rho-vs-damage Spearman/regression plus
    gain-matched low-risk vs high-risk pair differences.
  - `run_risk_predictive_summary.py`: aggregates predictive-validity runs.
  - `build_main_grade_decisive_protocols.py`: generates locked v31-v33
    protocols.
  - `run_cross_domain_risk_aware_controller.py` now supports
    `--selection-rule ratio|gain_topk_low_risk` and
    `--gain-pool-multiplier`; default remains original `ratio`.
- Added locked protocols:
  - `control_v31_cross_domain_risk_predictive_validity.json`
    - Church and bedroom structured candidate universes.
    - Seeds `2037-2041`, true LPIPS, matched semantic target.
  - `control_v32_church_gain_first_risk_tiebreak_controller.json`
    - Church structured held-out seeds `2042-2046`.
    - Actual/shuffled/inverted risk with `selection-rule=gain_topk_low_risk`.
  - `control_v33_main_grade_audit_v6.json`
    - Final audit including v31/v32.
- Verification before queueing:
  - `py_compile` passed for changed scripts.
  - v31-v33 dry-runs passed.
  - v31-v33 structural checks passed: duplicate keys 0, duplicate outs 0,
    unresolved summary inputs 0.
  - Predictive-validity smoke run passed at
    `paper/experiments/out/_smoke_risk_predictive/metrics.json`.
  - Gain-first/risk-tiebreak controller smoke run passed at
    `paper/experiments/out/_smoke_gain_first_tiebreak/metrics.json`.
  - `git diff --check` passed for changed files.
- Detached queue started at 2026-05-27 03:41 KST:
  - v31 queue PID `172203`, PPID=1, log:
    `paper/logs/control_v31_cross_domain_risk_predictive_queue/queue.log`.
  - v32 queue PID `172270`, PPID=1, waits for PID `172203`, log:
    `paper/logs/control_v32_after_v31_queue/queue.log`.
  - v33 audit queue PID `172290`, PPID=1, waits for PID `172270`, log:
    `paper/logs/control_v33_after_v32_queue/queue.log`.
  - Current first GPU child PID `172208`, running:
    `church_risk_predictive_seed_2037`.
  - Current first run log:
    `paper/logs/control_v31_cross_domain_risk_predictive_validity/church_risk_predictive_seed_2037.log`.

## 2026-05-27 Main-Claim Readiness Queue

- User emphasized not compromising and requiring completion.
- Added final non-GPU decision/audit layer after v31-v33:
  - `run_main_claim_readiness.py`
  - `control_v34_main_claim_readiness.json`
  - `control_v35_main_grade_audit_v7.json`
- v34 applies predeclared pass/fail rules to v31/v32 summaries:
  - Predictive evidence: rho-vs-LPIPS Spearman, rho beta for LPIPS,
    gain-matched low-risk vs high-risk LPIPS, and gain-matched ID.
  - Controller evidence: gain-first/risk-tiebreak vs gain-only/random/high-risk/
    low-risk plus actual-vs-shuffled/inverted negatives.
  - Outputs one of:
    `strong_main_control_claim_ready`, `borderline_main_control_claim`,
    `main_predictive_claim_only`, or `not_main_ready_without_claim_narrowing`.
- v35 audits the whole campaign including v34.
- Verification:
  - `py_compile` passed for v34 scripts and aggregate.
  - v34/v35 dry-runs passed.
  - v34/v35 structural checks passed: duplicate keys 0, duplicate outs 0.
  - `git diff --check` passed for changed files.
- Detached queue appended at 2026-05-27 03:47 KST:
  - v34 queue PID `173257`, waits for v33 PID `172290`.
  - v35 audit queue PID `173278`, waits for v34 PID `173257`.
  - Queue logs:
    `paper/logs/control_v34_after_v33_queue/queue.log`
    and `paper/logs/control_v35_after_v34_queue/queue.log`.
  - Current live observation at 03:48 KST:
  - v31 first run `church_risk_predictive_seed_2037` still active.
  - Clouds looked strong in the intended direction
    (`rho->LPIPS beta=+0.191`, low-risk vs high-risk `ΔID=+0.0467`,
    `ΔLPIPS=-0.0891`), sunny was mixed, vegetation still running.

## 2026-05-27 Main Evidence Table Queue

- User again requested all experiments/artifacts needed for CVPR/ICCV/ECCV main.
- Added final reviewer-facing evidence layer after v35:
  - `run_main_evidence_table.py`
  - `control_v36_main_evidence_table.json`
  - `control_v37_main_grade_audit_v8.json`
- v36 builds a paper/reviewer table from locked summaries:
  - bedroom controller and negatives
  - church structured controller and negatives
  - church confirmatory controller and negatives
  - v31 church/bedroom predictive validity
  - v32 gain-first/risk-tiebreak controller and negatives
  - v34 readiness result
  - For each evidence row it reports seed-level mean, 95% bootstrap CI, exact
    sign-test p-value, wins/n, and a predeclared pass flag.
  - Outputs:
    `paper/experiments/out/control_main_evidence_table_v1/metrics.json`
    and `paper/experiments/out/control_main_evidence_table_v1/main_evidence_table.md`.
- v37 audits the full campaign including v36.
- Verification:
  - `py_compile` passed for `run_main_evidence_table.py` and aggregate.
  - v36/v37 dry-runs passed.
  - v36/v37 structural checks passed: duplicate keys 0, duplicate outs 0.
  - `git diff --check` passed for changed files.
- Detached queue appended at 2026-05-27 03:52 KST:
  - v36 queue PID `173900`, waits for v35 PID `173278`.
  - v37 audit queue PID `173918`, waits for v36 PID `173900`.
  - Queue logs:
    `paper/logs/control_v36_after_v35_queue/queue.log`
    and `paper/logs/control_v37_after_v36_queue/queue.log`.
- Live progress:
  - v31 first run `church_risk_predictive_seed_2037` completed.
  - First run aggregate was directionally promising overall:
    clouds positive, vegetation positive, sunny mixed.
  - Current GPU child PID `173611` is running
    `church_risk_predictive_seed_2038`.

## 2026-05-27 Semantic-Feasible Control Queue

- v31-v37 completed by about 07:58 KST. Final status before this extension:
  - GPU idle.
  - v34 readiness was `main_predictive_claim_only`.
  - Predictive checks passed 8/8, but gain-first/risk-tiebreak controller
    passed only 5/12 required controller checks.
- Scientific decision:
  - Do not keep claiming an unconstrained controller that beats every baseline.
  - Add a narrower, predeclared controller claim:
    semantic-feasible minimum-risk control. First filter candidates by probe
    gain feasibility, then select lowest curvature/risk within that feasible
    set. The risk-only/low-risk baseline is reported as a boundary, not as a
    required competitor to dominate.
- Code/protocols added or changed:
  - `run_cross_domain_risk_aware_controller.py` now supports
    `--selection-rule gain_feasible_low_risk`.
  - `run_risk_aware_repeat_summary.py` and
    `run_risk_signal_negative_summary.py` now also aggregate
    `mean_probe_gain`, `mean_rho`, `target_hit_rate_calib`, and
    `mean_abs_alpha` when present.
  - Added `run_feasible_control_readiness.py`.
  - Added `run_feasible_control_evidence_table.py`.
  - Added `build_feasible_control_protocols.py`.
  - Added locked protocols v38-v41:
    `control_v38_semantic_feasible_minrisk_controller.json`,
    `control_v39_feasible_control_readiness.json`,
    `control_v40_feasible_control_evidence_table.json`,
    `control_v41_main_grade_audit_v9.json`.
  - `aggregate_results.py` now prints D24-D27 when v38-v40 outputs exist.
- Verification before queueing:
  - `py_compile` passed for all modified/new Python files.
  - v38-v41 protocol dry-runs passed.
  - duplicate protocol keys and duplicate declared outputs were both zero.
  - Tiny church smoke run passed at
    `paper/experiments/out/_smoke_feasible_low_risk/metrics.json`.
  - Smoke summary passed at
    `paper/experiments/out/_smoke_feasible_low_risk_summary/metrics.json`.
  - `git diff --check` passed.
- Detached queue chain started at 2026-05-27 12:14 KST:
  - v38 queue PID `186731`, PPID=1, no TTY.
  - v39 queue PID `186769`, waits for v38 PID `186731`.
  - v40 queue PID `186794`, waits for v39 PID `186769`.
  - v41 queue PID `186806`, waits for v40 PID `186794`.
  - Queue logs:
    `paper/logs/control_v38_semantic_feasible_minrisk_controller_queue/queue.log`,
    `paper/logs/control_v39_after_v38_queue/queue.log`,
    `paper/logs/control_v40_after_v39_queue/queue.log`,
    `paper/logs/control_v41_after_v40_queue/queue.log`.
- Live status at 12:15 KST:
  - GPU child PID `186735` is running first v38 run:
    `church_feasible_low_risk_actual_seed_2047`.
  - First run log:
    `paper/logs/control_v38_semantic_feasible_minrisk_controller/church_feasible_low_risk_actual_seed_2047.log`.
  - `nvidia-smi` showed GPU active with the Python process using about 1.6GB.

## 2026-05-27 Predictive Assumption-Stress Queue

- User asked to keep pushing toward CVPR/ICCV/ECCV main without compromise.
- v38 was still running, so `run_cross_domain_risk_aware_controller.py` was not
  modified again after v38 started. This avoids changing code used by queued
  v38 child runs mid-protocol.
- Added measurement-assumption stress tests for the predictive-validity claim:
  - `run_cross_domain_risk_predictive.py` now supports:
    - `--risk-estimator jvp|fd`
    - `--fd-eps`
    - `--prompt-style default|photo|caption`
  - FD risk estimator uses finite differences on synthesized images:
    central first difference and central second difference, then the same
    second/first ratio form as the JVP metric.
  - Prompt-style robustness uses a held-out `photo` template for the semantic
    gain/target matching prompts.
  - `run_risk_predictive_summary.py` now records `risk_estimator` and
    `prompt_style` in seed rows.
  - Added `run_predictive_assumption_readiness.py`.
  - Added `build_predictive_assumption_protocols.py`.
- Added locked protocols:
  - `control_v42_fd_risk_predictive_validity.json`
    - church+bedroom, seeds `2052-2056`, true LPIPS, FD risk estimator.
  - `control_v43_prompt_template_predictive_validity.json`
    - church+bedroom, seeds `2057-2061`, true LPIPS, JVP risk,
      prompt-style `photo`.
  - `control_v44_predictive_assumption_readiness.json`
  - `control_v45_main_grade_audit_v10.json`
- Verification before queueing:
  - `py_compile` passed for new/modified predictive-assumption scripts.
  - v42-v45 protocol dry-runs passed.
  - duplicate keys and duplicate outs were both zero.
  - `git diff --check` passed.
  - No extra GPU smoke was launched because v38 was already using the GPU.
- Detached queue chain appended after v41 at 2026-05-27 12:24-12:25 KST:
  - v42 queue PID `188048`, waits for v41 PID `186806`.
  - v43 queue PID `188071`, waits for v42 PID `188048`.
  - v44 queue PID `188094`, waits for v43 PID `188071`.
  - v45 queue PID `188165`, waits for v44 PID `188094`.
  - Queue logs:
    `paper/logs/control_v42_after_v41_queue/queue.log`,
    `paper/logs/control_v43_after_v42_queue/queue.log`,
    `paper/logs/control_v44_after_v43_queue/queue.log`,
    `paper/logs/control_v45_after_v44_queue/queue.log`.
- Live status at 12:25 KST:
  - GPU active at about 95% util, current child PID `186735`.
  - v38 first run `church_feasible_low_risk_actual_seed_2047` was in
    vegetation after finishing clouds and sunny.

## 2026-05-27 Extended Predictive Stress Queue

- Checked at 2026-05-27 22:17 KST:
  - v38-v45 had all completed.
  - GPU was idle, with no running GPU processes.
  - v38-v41 completed by 18:29 KST; v42 completed at 19:26, v43 at 20:36,
    v44 at 20:36, and v45 at 20:36.
- Result interpretation at that checkpoint:
  - Semantic-feasible control improved church vs gain-only/high-risk but did
    not beat random/low-risk, and bedroom remained weak.
  - v39 readiness was `predictive_claim_with_feasible_control_boundary`:
    predictive 8/8, controller 12/20, boundary 3/8.
  - v44 predictive assumption readiness was
    `assumption_sensitive_predictive_claim`, 13/16. The main weakness was
    bedroom matched-pair preservation under FD/photo prompt stress.
- Added extended predictive-stress code/protocols:
  - `run_predictive_stress_evidence_table.py`: generic evidence table for
    labeled predictive summaries with predeclared directional checks.
  - `control_v46_caption_prompt_predictive_validity.json`: second held-out
    prompt template (`prompt-style=caption`), seeds 2062-2066, church+bedroom.
  - `control_v47_high_ntest_predictive_validity.json`: doubled held-out
    evaluation set (`n-test=128`), seeds 2067-2071, church+bedroom.
  - `control_v48_strict_gain_match_predictive_validity.json`: stricter
    semantic matching (`gain-match-rel=0.10`), seeds 2072-2076,
    church+bedroom.
  - `control_v49_predictive_stress_evidence_table.json`: combines baseline,
    FD, photo, caption, high-n-test, and strict-gain summaries.
  - `control_v50_main_grade_audit_v11.json`: final audit including v46-v49.
  - `aggregate_results.py` now prints D28 for the added stress roots and D30
    for the extended evidence table.
- Verification before queueing:
  - `py_compile` passed for modified/new Python files.
  - v46-v50 dry-runs passed.
  - v46-v50 structural check passed: duplicate keys 0, duplicate declared
    outputs 0, missing scripts 0.
  - `git diff --check` passed.
  - Tiny `caption` prompt GPU smoke passed at
    `paper/experiments/out/_smoke_prompt_caption/metrics.json`.
- Detached queue chain started at 2026-05-27 22:23-22:24 KST:
  - v46 queue PID `3289107`, PPID=1, started immediately.
  - v47 queue PID `3289990`, waits for v46.
  - v48 queue PID `3291022`, waits for v47.
  - v49 queue PID `3291996`, waits for v48.
  - v50 queue PID `3293241`, waits for v49.
  - Queue logs:
    `paper/logs/control_v46_caption_prompt_predictive_queue/queue.log`,
    `paper/logs/control_v47_after_v46_queue/queue.log`,
    `paper/logs/control_v48_after_v47_queue/queue.log`,
    `paper/logs/control_v49_after_v48_queue/queue.log`,
    `paper/logs/control_v50_after_v49_queue/queue.log`.
- Live status at 22:24 KST:
  - GPU child PID `3289129` was running first v46 run
    `church_control_v46_caption_prompt_predictive_validity_seed_2062`.
  - `nvidia-smi` showed GPU active, about 40% util and 1.1GB VRAM at the
    sampled instant.

## 2026-05-28 Extended Predictive Stress Completion

- Checked at 2026-05-28 02:50 KST:
  - GPU idle, no running GPU processes.
  - v46-v50 queue PIDs had exited.
  - v46 completed at 2026-05-27 23:41:30 KST.
  - v47 completed at 2026-05-28 01:08:25 KST.
  - v48 completed at 2026-05-28 02:25:00 KST.
  - v49 completed at 2026-05-28 02:25:18 KST.
  - v50 completed at 2026-05-28 02:25:26 KST.
- Audit status:
  - v46 caption prompt predictive validity: complete 12, pending 0, failed 0.
  - v47 high-n-test predictive validity: complete 12, pending 0, failed 0.
  - v48 strict-gain-match predictive validity: complete 12, pending 0, failed 0.
  - v49 predictive stress evidence table: complete 1, pending 0, failed 0.
  - v50 audit produced only expected `git_dirty is true or missing` warnings.
- Key results:
  - v49 readiness: `extended_predictive_stress_sensitive`.
  - Required predictive stress checks: 36/48 pass.
  - All evidence-table checks: 57/72 pass.
  - Baseline and FD/photo church remain strong.
  - Caption prompt weakens church matched-pair and LPIPS-beta checks:
    caption_church required failures were LPIPS beta, matched-pair ID, and
    matched-pair LPIPS, each 3/5.
  - Caption bedroom remains weak on matched-pair ID and LPIPS.
  - High `n_test=128` improves much of the picture, but bedroom matched-pair ID
    still fails at 3/5.
  - Strict gain matching preserves rank-correlation strength but still leaves
    matched-pair failures in bedroom and LPIPS-beta weakness in church.
- Paper implication:
  - The honest claim is not "rho universally predicts damage under all semantic
    prompt/matching assumptions."
  - Stronger defensible claim: rho is a robust rank-level damage indicator, but
    matched-pair preservation is prompt/domain sensitive, especially bedroom.
  - Next scientific step should characterize or fix semantic-measurement
    sensitivity, e.g. prompt-ensemble semantic directions or a non-CLIP
    semantic gain estimator, rather than adding more controller variants.

## 2026-05-28 Prompt-Ensemble Diagnostic Queue

- User asked for current status; v46-v50 were complete and GPU was idle.
- Added prompt-ensemble semantic-measurement diagnostic:
  - `run_cross_domain_risk_predictive.py` now supports
    `--prompt-style ensemble`, averaging the default/photo/caption CLIP text
    direction deltas before normalization.
  - `control_v51_prompt_ensemble_predictive_validity.json`: church+bedroom,
    seeds 2077-2081, same predictive-validity setup as v43/v46 but
    `prompt-style=ensemble`.
  - `control_v52_predictive_stress_evidence_table_with_ensemble.json`: evidence
    table including baseline, FD, photo, caption, ensemble, high-n-test, and
    strict-gain summaries.
  - `control_v53_main_grade_audit_v12.json`: audit including v51-v52.
  - `aggregate_results.py` now prints prompt-ensemble D28 rows and D31 evidence.
- Verification before queueing:
  - `py_compile` passed for modified/new scripts.
  - v51-v53 dry-runs passed.
  - Structural check passed: duplicate keys 0, duplicate declared outputs 0,
    missing scripts 0.
  - `git diff --check` passed.
  - Tiny ensemble GPU smoke passed at
    `paper/experiments/out/_smoke_prompt_ensemble/metrics.json`.
- Detached queue chain started at 2026-05-28 02:53 KST:
  - v51 queue PID `1590566`, PPID=1, started immediately.
  - v52 queue PID `1591120`, waits for v51.
  - v53 queue PID `1591632`, waits for v52.
  - Current GPU child at 02:54 KST: PID `1590580`, first v51 run
    `church_control_v51_prompt_ensemble_predictive_validity_seed_2077`.
  - Queue logs:
    `paper/logs/control_v51_prompt_ensemble_predictive_queue/queue.log`,
    `paper/logs/control_v52_after_v51_queue/queue.log`,
    `paper/logs/control_v53_after_v52_queue/queue.log`.

## 2026-05-28 Prompt-Ensemble Follow-up Queue

- User said additional HIGAN experiments are acceptable.
- Constraint while v51 is running:
  - Do not modify `run_cross_domain_risk_predictive.py`, because v51 child
    runs are using that script.
  - Only add new locked protocols and aggregate/evidence plumbing for runs that
    start after v53.
- Added:
  - `build_prompt_ensemble_followup_protocols.py`
  - `control_v54_prompt_ensemble_followup_predictive_validity.json`
    - `prompt-style=ensemble` with `n-test=128`, seeds 2082-2086,
      church+bedroom.
    - `prompt-style=ensemble` with `gain-match-rel=0.10`, seeds 2087-2091,
      church+bedroom.
  - `control_v55_predictive_stress_evidence_table_ensemble_followup.json`
  - `control_v56_main_grade_audit_v13.json`
  - `aggregate_results.py` now prints D32 for the ensemble follow-up table.
- Verification before queueing:
  - `py_compile` passed.
  - v54-v56 dry-runs passed.
  - Structural check passed: duplicate keys 0, duplicate declared outputs 0,
    missing scripts 0.
  - `git diff --check` passed.
- Detached queue chain appended at 2026-05-28 02:58 KST:
  - v54 queue PID `1625357`, waits for v53 PID `1591632`.
  - v55 queue PID `1626303`, waits for v54.
  - v56 queue PID `1627347`, waits for v55.
  - Queue logs:
    `paper/logs/control_v54_after_v53_queue/queue.log`,
    `paper/logs/control_v55_after_v54_queue/queue.log`,
    `paper/logs/control_v56_after_v55_queue/queue.log`.
  - Live status at 02:58 KST:
  - v51 still active, GPU child PID `1590580`, about 94% util.
  - v54-v56 are waiting detached with PPID=1.

## 2026-05-28 Prompt-Ensemble Estimator/Source Queue

- User asked to keep adding meaningful experiments whenever the GPU might go
  idle.
- Added only paper-relevant stress tests, not arbitrary sweeps:
  - `build_prompt_ensemble_extended_protocols.py`
  - `control_v57_prompt_ensemble_fd_predictive_validity.json`
    - prompt ensemble plus finite-difference rho, seeds 2092-2096,
      church+bedroom.
    - Purpose: test whether the prompt-ensemble signal survives an independent
      rho estimator.
  - `control_v58_prompt_ensemble_source_ablation_predictive_validity.json`
    - prompt ensemble with GANSpace-only and SeFa-only candidate universes.
    - Seeds 2097-2101 for GANSpace-only and 2102-2106 for SeFa-only,
      church+bedroom.
    - Purpose: test whether the signal depends on one candidate-source family.
  - `control_v59_prompt_ensemble_extended_evidence_table.json`
  - `control_v60_main_grade_audit_v14.json`
  - `aggregate_results.py` now prints D33 for estimator/source evidence.
- Verification:
  - `py_compile` passed.
  - v57-v60 dry-runs passed.
  - Structural checks passed: duplicate keys 0, duplicate declared outputs 0,
    duplicate audit protocols 0, missing scripts 0.
  - `git diff --check` passed.
- Detached queue chain appended at 2026-05-28 03:02 KST:
  - v57 queue PID `1678120`, waits for v56 PID `1627347`.
  - v58 queue PID `1679364`, waits for v57.
  - v59 queue PID `1680564`, waits for v58.
  - v60 queue PID `1681850`, waits for v59.
  - Queue logs:
    `paper/logs/control_v57_after_v56_queue/queue.log`,
    `paper/logs/control_v58_after_v57_queue/queue.log`,
    `paper/logs/control_v59_after_v58_queue/queue.log`,
    `paper/logs/control_v60_after_v59_queue/queue.log`.
- Live status at 03:02 KST:
  - v51 still running; GPU child PID `1590580`, about 95% util.
  - v57-v60 are waiting detached with PPID=1.

## 2026-05-28 Main-Grade Extension Queue

- User allowed additional GPU experiments if they are paper-relevant and not
  result-fitted.
- Added locked protocols:
  - `control_v61_ffhq_prompt_ensemble_predictive_validity.json`
    - FFHQ prompt-ensemble predictive validity, seeds 2107-2111.
    - Purpose: test whether the predictive rho claim extends beyond
      church/bedroom to a face generator domain.
  - `control_v62_ffhq_prompt_ensemble_followup_predictive_validity.json`
    - FFHQ high `n_test=128` and strict `gain_match_rel=0.10`, five seeds each.
    - Purpose: stat-power and semantic-matching stress for FFHQ extension.
  - `control_v63_random_universe_prompt_ensemble_predictive_validity.json`
    - church/bedroom/FFHQ with GANSpace+SeFa+random candidate universe,
      seeds 2122-2126.
    - Purpose: candidate-universe fairness stress.
  - `control_v64_dino_preservation_predictive_validity.json`
    - church/bedroom/FFHQ with DINOv2 image-preservation metric,
      seeds 2127-2131.
    - Purpose: test whether rho predicts damage outside CLIP-image and LPIPS
      preservation metrics.
  - `control_v65_main_grade_extension_evidence_table.json`
  - `control_v66_dino_preservation_evidence_table.json`
  - `control_v67_main_grade_audit_v15.json`
- Code updates:
  - `run_cross_domain_risk_aware_controller.py` supports optional DINO image
    features inside `evaluate_direction`.
  - `run_cross_domain_risk_predictive.py` supports
    `--dino-preservation`, `--dino-model`, and `--dino-local-files-only`.
  - `run_risk_predictive_summary.py` aggregates optional DINO metrics.
  - `run_predictive_dino_evidence_table.py` builds a DINO-specific evidence
    table.
  - `build_main_grade_extension_protocols.py` generates v61-v67.
  - `aggregate_results.py` prints the new FFHQ/random/DINO stress summaries
    and D34/D35 evidence blocks.
- Verification before queueing:
  - `py_compile` passed for modified/new Python files.
  - v61-v67 dry-runs passed.
  - Structural check passed: duplicate keys 0, duplicate declared outputs 0,
    duplicate audit protocols 0, missing scripts 0.
  - DINOv2 local cache load passed with `local_files_only=True`.
  - Tiny FFHQ+DINO GPU smoke passed at
    `paper/experiments/out/_smoke_ffhq_dino_predictive/metrics.json`, including
    DINO Spearman/regression/matched-pair fields.
  - `git diff --check` passed.
- Detached queue chain started at 2026-05-28 20:38 KST:
  - v61 queue PID `1827121`, PPID=1, started immediately.
  - v62 queue PID `1827883`, waits for v61.
  - v63 queue PID `1828523`, waits for v62.
  - v64 queue PID `1829345`, waits for v63.
  - v65 queue PID `1830200`, waits for v64.
  - v66 queue PID `1830987`, waits for v65.
  - v67 queue PID `1831721`, waits for v66.
  - Queue logs:
    `paper/logs/control_v61_ffhq_prompt_ensemble_queue/queue.log`,
    `paper/logs/control_v62_after_v61_queue/queue.log`,
    `paper/logs/control_v63_after_v62_queue/queue.log`,
    `paper/logs/control_v64_after_v63_queue/queue.log`,
    `paper/logs/control_v65_after_v64_queue/queue.log`,
    `paper/logs/control_v66_after_v65_queue/queue.log`,
    `paper/logs/control_v67_after_v66_queue/queue.log`.
- Live status at 20:38 KST:
  - GPU child PID `1827142` was running first v61 run
    `ffhq_control_v61_ffhq_prompt_ensemble_predictive_validity_ffhq_prompt_ensemble_seed_2107`.
  - `nvidia-smi` showed GPU active, about 41% util and 1.3GB VRAM.
  - First run log:
    `paper/logs/control_v61_ffhq_prompt_ensemble_predictive_validity/ffhq_control_v61_ffhq_prompt_ensemble_predictive_validity_ffhq_prompt_ensemble_seed_2107.log`.

## 2026-05-29 Top-Tier Defense Queue

- User pushed for top-tier readiness rather than stopping at the v67 result.
- Rationale:
  - v67 gave strong main-grade extension evidence, but several key rows still
    had only five seeds, so 5/5 sign-test rows bottom out at p=0.0625.
  - Reviewers can also attack rho as a proxy for semantic gain, alpha, source
    family, or attribute identity.
- Added locked protocols:
  - `control_v68_random_universe_seed_scaling_predictive_validity.json`
    - Adds five new seeds 2132-2136 for church/bedroom/FFHQ random-universe
      prompt-ensemble stress.
    - Writes n=10 summaries combining old v63 seeds 2122-2126 with new seeds.
  - `control_v69_dino_preservation_seed_scaling_predictive_validity.json`
    - Adds five new seeds 2137-2141 for church/bedroom/FFHQ DINO-preservation
      prompt-ensemble stress.
    - Writes n=10 summaries combining old v64 seeds 2127-2131 with new seeds.
  - `control_v70_seed_scaled_extension_evidence_table.json`
  - `control_v71_dino_preservation_seed_scaled_evidence_table.json`
  - `control_v72_rho_incremental_value_table.json`
    - Cross-validated nested regression test of whether rho adds predictive
      value beyond semantic gain, alpha magnitude, candidate source, and attr.
  - `control_v73_main_grade_audit_v16.json`
- Code updates:
  - `run_predictive_incremental_value_table.py`
  - `build_top_tier_defense_protocols.py`
  - `aggregate_results.py` now prints D36/D37/D38.
- Verification before queueing:
  - `py_compile` passed for modified/new Python files.
  - v68-v73 dry-runs passed.
  - Structural check passed: duplicate keys 0, duplicate declared outputs 0,
    duplicate audit protocols 0, missing scripts 0.
  - `git diff --check` passed.
  - Incremental-value smoke on existing v63/v64 church summaries ran and wrote
    `paper/experiments/out/_smoke_rho_incremental_value/metrics.json`.
    The smoke result was sensitive on n=5, which reinforces the n=10 scaling
    motivation.
- Detached queue chain started at 2026-05-29 03:34 KST:
  - v68 queue PID `370434`, PPID=1, started immediately.
  - v69 queue PID `370488`, waits for v68.
  - v70 queue PID `370500`, waits for v69.
  - v71 queue PID `370512`, waits for v70.
  - v72 queue PID `370524`, waits for v71.
  - v73 queue PID `370537`, waits for v72.
  - Queue logs:
    `paper/logs/control_v68_random_universe_seed_scaling_queue/queue.log`,
    `paper/logs/control_v69_after_v68_queue/queue.log`,
    `paper/logs/control_v70_after_v69_queue/queue.log`,
    `paper/logs/control_v71_after_v70_queue/queue.log`,
    `paper/logs/control_v72_after_v71_queue/queue.log`,
    `paper/logs/control_v73_after_v72_queue/queue.log`.
- Live status at 03:34 KST:
  - GPU child PID `370439` was running first v68 run
    `church_control_v68_random_universe_seed_scaling_predictive_validity_random_universe_seed_2132`.
  - `nvidia-smi` showed GPU active, about 97% util and 3.7GB VRAM.

## 2026-05-29 Failure-Boundary Defense Queue

- Added v74-v75 behind the top-tier defense chain.
- Rationale:
  - Top-tier framing should not hide weak rows. The paper needs an explicit
    failure-boundary table so negative rows are explained as mechanism and
    scope, not cherry-picked away.
  - Existing SD 9-timestep evidence was inspected and not forced into the core
    predictive-control claim because several timestep correlations were weak
    or negative. It can remain qualitative or trajectory context, but using it
    as central evidence would weaken the current claim.
- Added locked protocols:
  - `control_v74_failure_boundary_table.json`
    - Consolidates v4/v5/v6 stress evidence, DINO evidence, n=10 evidence,
      and rho incremental-value evidence into a boundary table.
  - `control_v75_main_grade_audit_v17.json`
- Code updates:
  - `run_failure_boundary_table.py`
  - `build_failure_boundary_protocols.py`
  - `aggregate_results.py` now prints D39 failure-boundary evidence.
- Verification before queueing:
  - `py_compile` passed for the new/modified scripts.
  - v74-v75 dry-runs passed.
  - `git diff --check` passed.
- Detached queue chain extension started at 2026-05-29 03:39 KST:
  - v74 queue PID `371227`, PPID=1, waits for v73 PID `370537`.
  - v75 queue PID `371239`, PPID=1, waits for v74 PID `371227`.
  - Queue logs:
    `paper/logs/control_v74_after_v73_queue/queue.log`,
    `paper/logs/control_v75_after_v74_queue/queue.log`.
- Live status at 03:39 KST:
  - v68-v75 queue processes all had PPID=1.
  - GPU child PID `370439` was active.
  - `nvidia-smi` showed GPU active, about 96% util and 3.9GB VRAM.

## 2026-05-29 Exhaustive Top-Tier Defense Queue

- User asked to add all meaningful experiments in advance for CVPR/ICCV/ECCV
  main-readiness.
- Added protocols v76-v84 behind v75.
- Rationale:
  - v68-v75 cover n=10 seed scaling, DINO preservation, incremental value,
    and failure-boundary reporting.
  - Remaining defensible axes were FFHQ estimator/source gaps, estimator
    sampling budget, candidate-universe scale, target-magnitude sensitivity,
    and a permutation-null test against bookkeeping/candidate-pool artifacts.
- Added code:
  - `run_predictive_permutation_null_table.py`
    - Attribute-wise rho shuffle null over raw per-seed predictive metrics.
    - Checks whether observed rho/damage relationships beat shuffled-rho nulls.
  - `build_top_tier_exhaustive_protocols.py`
  - `aggregate_results.py` now prints D40/D41/D42.
- Added locked protocols:
  - `control_v76_ffhq_fd_predictive_validity.json`
    - FFHQ prompt-ensemble finite-difference rho, seeds 2142-2146.
  - `control_v77_ffhq_source_ablation_predictive_validity.json`
    - FFHQ GANSpace-only seeds 2147-2151 and SeFa-only seeds 2152-2156.
  - `control_v78_estimator_budget_predictive_validity.json`
    - Church/bedroom/FFHQ random-universe prompt-ensemble with n-risk 16 and
      n-probe 32, seeds 2157-2161.
  - `control_v79_wide_candidate_universe_predictive_validity.json`
    - Church/bedroom/FFHQ with candidate-k 10 and GANSpace samples 4096,
      seeds 2162-2166.
  - `control_v80_target_magnitude_predictive_validity.json`
    - Church/bedroom/FFHQ target quantile q=.10 seeds 2167-2171 and q=.50
      seeds 2172-2176.
  - `control_v81_exhaustive_defense_evidence_table.json`
  - `control_v82_predictive_permutation_null_table.json`
  - `control_v83_failure_boundary_table_v2.json`
  - `control_v84_main_grade_audit_v18.json`
- Verification before queueing:
  - `py_compile` passed for the new scripts and `aggregate_results.py`.
  - v76-v84 dry-runs passed.
  - Structural check passed: protocols 9, experiments 94, duplicate keys 0,
    duplicate outputs 0, missing scripts 0.
  - `git diff --check` passed.
- Detached queue extension started at 2026-05-29 03:48 KST:
  - v76 queue PID `372615`, waits for v75 PID `371239`.
  - v77 queue PID `372635`, waits for v76.
  - v78 queue PID `372653`, waits for v77.
  - v79 queue PID `372673`, waits for v78.
  - v80 queue PID `372691`, waits for v79.
  - v81 queue PID `372712`, waits for v80.
  - v82 queue PID `372730`, waits for v81.
  - v83 queue PID `372756`, waits for v82.
  - v84 queue PID `372774`, waits for v83.
  - Queue logs:
    `paper/logs/control_v76_after_v75_queue/queue.log`,
    `paper/logs/control_v77_after_v76_queue/queue.log`,
    `paper/logs/control_v78_after_v77_queue/queue.log`,
    `paper/logs/control_v79_after_v78_queue/queue.log`,
    `paper/logs/control_v80_after_v79_queue/queue.log`,
    `paper/logs/control_v81_after_v80_queue/queue.log`,
    `paper/logs/control_v82_after_v81_queue/queue.log`,
    `paper/logs/control_v83_after_v82_queue/queue.log`,
    `paper/logs/control_v84_after_v83_queue/queue.log`.
- Live status at 03:49 KST:
  - v68-v84 queue processes all had PPID=1.
  - GPU child PID `372401` was active.
  - `nvidia-smi` showed GPU active, about 59% util and 1.2GB VRAM.

## 2026-05-29 Exhaustive Defense n=10 Seed-Scaling Queue

- User said long runtime is acceptable and asked to reserve all meaningful
  experiments in advance.
- Added v85-v92 behind v84.
- Rationale:
  - v76-v80 added strong defense axes but still only used five seeds.
  - To avoid selective scaling and weak 5/5 sign-test evidence, every
    predeclared v76-v80 defense axis was extended to n=10.
- Added code:
  - `build_top_tier_seedscale_protocols.py`
  - `aggregate_results.py` now prints D43/D44/D45.
- Added locked protocols:
  - `control_v85_ffhq_estimator_source_seed_scaling_predictive_validity.json`
    - FFHQ FD seeds 2177-2181, FFHQ GANSpace-only seeds 2182-2186, FFHQ
      SeFa-only seeds 2187-2191, each combined with its v76/v77 five seeds
      into n=10 summaries.
  - `control_v86_estimator_budget_seed_scaling_predictive_validity.json`
    - Church/bedroom/FFHQ high-budget seeds 2192-2196, combined with v78
      seeds into n=10 summaries.
  - `control_v87_wide_candidate_universe_seed_scaling_predictive_validity.json`
    - Church/bedroom/FFHQ wide-universe seeds 2197-2201, combined with v79
      seeds into n=10 summaries.
  - `control_v88_target_magnitude_seed_scaling_predictive_validity.json`
    - Church/bedroom/FFHQ target q=.10 seeds 2202-2206 and q=.50 seeds
      2207-2211, combined with v80 seeds into n=10 summaries.
  - `control_v89_seed_scaled_exhaustive_defense_evidence_table.json`
  - `control_v90_seed_scaled_predictive_permutation_null_table.json`
  - `control_v91_failure_boundary_table_v3.json`
  - `control_v92_main_grade_audit_v19.json`
- Verification before queueing:
  - `py_compile` passed.
  - v85-v92 dry-runs passed.
  - Structural check passed: protocols 8, experiments 94, duplicate keys 0,
    duplicate outputs 0, missing scripts 0.
  - `git diff --check` passed.
- Detached queue extension started at 2026-05-29 03:54 KST:
  - v85 queue PID `373308`, waits for v84 PID `372774`.
  - v86 queue PID `373326`, waits for v85.
  - v87 queue PID `373346`, waits for v86.
  - v88 queue PID `373364`, waits for v87.
  - v89 queue PID `373384`, waits for v88.
  - v90 queue PID `373402`, waits for v89.
  - v91 queue PID `373422`, waits for v90.
  - v92 queue PID `373440`, waits for v91.
  - Queue logs:
    `paper/logs/control_v85_after_v84_queue/queue.log`,
    `paper/logs/control_v86_after_v85_queue/queue.log`,
    `paper/logs/control_v87_after_v86_queue/queue.log`,
    `paper/logs/control_v88_after_v87_queue/queue.log`,
    `paper/logs/control_v89_after_v88_queue/queue.log`,
    `paper/logs/control_v90_after_v89_queue/queue.log`,
    `paper/logs/control_v91_after_v90_queue/queue.log`,
    `paper/logs/control_v92_after_v91_queue/queue.log`.
- Live status at 03:54 KST:
  - v85-v92 queue processes all had PPID=1.
  - GPU child PID `372401` was active.
  - `nvidia-smi` showed GPU active, about 95% util and 3.9GB VRAM.

## 2026-05-30 Diagnostic-Utility Queue

- User asked whether more experiments should be added while v86-v92 were still
  running.
- Added v93-v95 behind v92.
- Rationale:
  - GPU stress tests were already saturated through n=10 scaling.
  - The remaining top-tier-relevant weakness is whether rho is practically
    useful as a pre-edit diagnostic score, not only correlated in aggregate.
- Added code:
  - `run_predictive_diagnostic_utility_table.py`
    - Reads raw seed-level candidate rows through summary inputs.
    - Computes within-attribute rho AUROC for high-damage candidates
      (top-quartile damage).
    - Also reports rho AUROC minus the best simple baseline
      (`probe_gain`, `|alpha|`, semantic delta, calibration max delta) as
      non-required context.
  - `build_top_tier_diagnostic_protocols.py`
  - `aggregate_results.py` now prints D46/D47.
- Added locked protocols:
  - `control_v93_predictive_diagnostic_utility_table.json`
  - `control_v94_failure_boundary_table_v4.json`
  - `control_v95_main_grade_audit_v20.json`
- Verification before queueing:
  - `py_compile` passed.
  - v93-v95 dry-runs passed.
  - Structural check passed: protocols 3, experiments 3, duplicate keys 0,
    duplicate outputs 0, missing scripts 0.
  - `git diff --check` passed.
- Detached queue extension started at 2026-05-30 00:19 KST:
  - v93 queue PID `3157274`, waits for v92 PID `373440`.
  - v94 queue PID `3157328`, waits for v93.
  - v95 queue PID `3157401`, waits for v94.
  - Queue logs:
    `paper/logs/control_v93_after_v92_queue/queue.log`,
    `paper/logs/control_v94_after_v93_queue/queue.log`,
    `paper/logs/control_v95_after_v94_queue/queue.log`.
- Live status at 00:19 KST:
  - v93-v95 queue processes all had PPID=1.
  - v86 was active, GPU child PID `3155767`.
  - `nvidia-smi` showed GPU active, about 96% util and 1.45GB VRAM.

## 2026-06-21 TMLR FFHQ Trilemma Probe

- User asked to test the optional FFHQ robustness experiment for the TMLR
  `tmlr_predict.tex` trilemma spine without touching the frozen submission.
- Added isolated script:
  - `paper/experiments/metrics/run_ffhq_fd_vs_exact.py`
  - Uses existing `domains.ffhq.generator.FFHQGenerator` and InterFaceGAN FFHQ
    boundaries.
  - Writes only to `paper/experiments/out/ffhq_fd_vs_exact_trilemma/`; does not
    modify `note/submission` or `note/tmlr_submission.tar.gz`.
- Smoke:
  - `py_compile` passed.
  - 2-record smoke passed at `paper/experiments/out/ffhq_fd_vs_exact_smoke/`.
- Full FFHQ run:
  - Command used pyenv Python 3.12.9, `lod_override=2.0`, 5 FFHQ attributes
    (`smile`, `age`, `pose`, `gender`, `eyeglasses`) + 3 random layered
    W-directions, 12 samples, 16-step grid.
  - Output:
    - `paper/experiments/out/ffhq_fd_vs_exact_trilemma/metrics.json`
    - `paper/experiments/out/ffhq_fd_vs_exact_trilemma/trilemma_table.json`
  - Result: FFHQ reproduces the qualitative trilemma with different optimal
    steps:
    - second-order magnitude-best step `8.0`, mean relative error `72.0%`,
      signed bias `-62.4%`, Spearman `0.917`.
    - second-order signed-bias-best step `3.0`, signed bias `-12.9%`,
      mean relative error `95.0%`, Spearman `0.975`.
    - second-order rank-best step `0.3`, Spearman `0.994`,
      mean relative error `340%`, signed bias `+315%`.
    - first-order has a stable small-step region: e.g. step `0.05` has
      `7.15%` mean relative error and Spearman `0.9997`; step `0.02` has
      `8.73%` mean relative error, `+3.63%` signed bias, Spearman `0.9991`.
- Interpretation:
  - This supports the manuscript's empirical trilemma beyond StyleGAN-bedroom,
    but it should be reported honestly as an FFHQ/InterFaceGAN low-detail
    (`lod=2`) robustness probe.
  - Frozen submission package was not updated yet; next step, if desired, is a
    targeted manuscript/package update adding FFHQ as robustness evidence.

## 2026-06-21 TMLR Robustness Package Finalized

- User asked to do all relevant experiments that increase TMLR probability.
- Added/used robustness evidence for `note/submission/tmlr_predict.tex`:
  - Bedroom seed 32: qualitative trilemma repeats; 2nd-order best steps
    magnitude/bias/rank = `3.0/1.5/0.2`, magnitude-best error `51.3%`,
    rank-best magnitude error `314.8%`.
  - Bedroom seed 33: best steps `3.0/2.0/0.3`, magnitude-best error `42.6%`,
    rank-best magnitude error `258.2%`.
  - FFHQ lod 1: best steps `8.0/2.0/0.3`, magnitude-best error `73.9%`,
    rank-best magnitude error `359.8%`.
  - FFHQ lod 2: best steps `8.0/3.0/0.3`, magnitude-best error `72.0%`,
    rank-best magnitude error `340.3%`.
- Updated manuscript:
  - Abstract/introduction/contributions now state that the qualitative trilemma
    repeats on two additional bedroom seeds and two lower-detail FFHQ sweeps.
  - Added body robustness paragraph and Appendix D robustness table.
  - Fixed remaining overclaim comparing partial correlations to the 45% magnitude
    floor; wording now uses step-selection ambiguity.
  - Scope now says FFHQ robustness was tested for the trilemma, but not for the
    nonlinearity-prediction case study.
- Submission artifacts:
  - `note/submission/tmlr_predict.pdf` rebuilt successfully, 11 pages.
  - `note/tmlr_submission.tar.gz` regenerated with updated source/PDF and new
    evidence files.
  - Tar contents were checked: no `tmlr_paper.*`, no `_excluded_from_submission`,
    no `evidence_unused`, no `sd_sign_inversion`, no known identity/email strings.
- Verification:
  - `tectonic tmlr_predict.tex` exit 0; only underfull vbox warnings.
  - PDF text check found no `[?]` or `??`.
  - `compute_trilemma_table.py` regenerated the body trilemma summary.
  - Robustness summary asserts passed for all five rows.

## 2026-06-21 TMLR Existing-Infra Audit After Robustness

- User cautioned that the repo already has many experiments/installations and
  asked to verify before proceeding.
- Audited existing FD/exact/FFHQ scripts, outputs, checkpoints, and GPU state.
- Findings:
  - Current submission-relevant trilemma evidence remains: bedroom seeds
    `31/32/33` plus FFHQ lod `1/2`.
  - Other existing FFHQ outputs are mostly predictive-validity, encoder, C2
    heatmap, or exact ratio studies; they do not directly replace the trilemma
    evidence.
  - Existing `paper/experiments/out/ffhq_resolution/metrics.json` is useful as a
    narrow support for lower-detail FFHQ: exact-JVP second/first-order curvature
    ratio ranking is preserved across lod `0/1/2` with Spearman `1.0` for all
    pairs.
  - Full-resolution FFHQ trilemma is still blocked by GPU availability:
    process `3294478` from another project holds about `6.45GB` of the 8GB GPU.
    Do not kill it without explicit user approval.
- Updated manuscript/package:
  - Added one narrow sentence in Section 4 and Appendix D noting the FFHQ
    resolution audit, explicitly keeping FFHQ rows as qualitative robustness
    checks rather than full-resolution trilemma estimates.
  - Added evidence files:
    `note/submission/evidence/ffhq_resolution_metrics.json` and
    `note/submission/evidence/run_ffhq_resolution_invariance.py`.
  - Rebuilt `note/submission/tmlr_predict.pdf` (11 pages) and regenerated
    `note/tmlr_submission.tar.gz`.
  - Tar leak sweep remained clean.

## 2026-06-21 Queued Full-Resolution FFHQ Trilemma

- User pointed out that the remaining full-res FFHQ experiment can be queued.
- Added queue wrapper:
  - `paper/experiments/metrics/queue_ffhq_lod0_fd_vs_exact.py`
  - Waits until GPU free memory is at least `7600 MiB` for `2` consecutive
    polls (`60s` apart).
  - Runs a lod0 one-record smoke test first:
    `paper/experiments/out/ffhq_fd_vs_exact_lod0_smoke`.
  - Only if smoke passes, runs the full lod0 FFHQ trilemma sweep:
    `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0`.
  - Writes status/logs to:
    `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0_queue/queue.log`
    and `summary.json`.
  - Does not touch `note/submission` or `note/tmlr_submission.tar.gz`.
- Scheduling:
  - Started via tmux detached session after `nohup`/`setsid` failed to persist.
  - Active scheduler PID at start: `3540544`.
  - Current status at scheduling time: `waiting_for_gpu`; GPU free was about
    `1408 MiB` because process `3294478` from another project held about
    `6.45GB`.
- After completion:
  - If `summary.json` has `"status": "complete"`, inspect lod0 numbers and only
    then decide whether to update `tmlr_predict.tex`/tar.
  - If `"smoke_failed"` or `"full_failed"`, keep the current lower-detail FFHQ
    package unchanged and report the failure honestly.

## 2026-06-22 Full-Resolution FFHQ Integrated

- The queued full-resolution FFHQ lod0 trilemma run completed successfully:
  - `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0/metrics.json`
  - `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0/trilemma_table.json`
  - Queue summary: `status=complete`, `smoke_code=0`, `full_code=0`.
- Ran two additional full-resolution FFHQ lod0 seeds while GPU was idle:
  - seed32:
    `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0_seed32`
  - seed33:
    `paper/experiments/out/ffhq_fd_vs_exact_trilemma_lod0_seed33`
- Full-res FFHQ trilemma results:
  - seed31: magnitude/bias/rank best steps `5.0/1.5/0.3`,
    magnitude-best error `71.4%`, rank-best magnitude error `265.7%`.
  - seed32: best steps `5.0/1.5/0.1`,
    magnitude-best error `70.8%`, rank-best magnitude error `762.7%`.
  - seed33: best steps `5.0/1.5/0.2`,
    magnitude-best error `69.4%`, rank-best magnitude error `358.2%`.
- Updated submission evidence:
  - added `ffhq_lod0_seed31/32/33_*` metrics and trilemma JSONs.
  - regenerated `fd_trilemma_robustness_summary.json` with 8 rows:
    bedroom seeds 31/32/33, FFHQ lod0 seeds 31/32/33, FFHQ lod1/lod2 controls.
- Updated manuscript:
  - Abstract/introduction/contributions now say the qualitative conflict repeats
    across two additional bedroom seeds and three full-resolution FFHQ seeds,
    with lower-detail FFHQ controls.
  - Section 4 robustness paragraph and Appendix D Table 4 now include FFHQ lod0
    seed31/32/33 rows.
  - Scope says trilemma was audited on full-res FFHQ too; prediction case study
    remains bedroom-only.
- Rebuilt artifacts:
  - `note/submission/tmlr_predict.pdf`, 11 pages.
  - `note/tmlr_submission.tar.gz`, updated and leak-checked clean.
- Verification:
  - `tectonic tmlr_predict.tex` exit 0; only underfull warnings.
  - PDF text check found no `[?]` or `??`.
  - Risk/overclaim grep clean.
  - Tar includes new full-res FFHQ evidence and excludes old/dead-line files.

## 2026-06-22 Evidence Manifest Added

- Added reviewer-facing manifest:
  - `note/submission/evidence/README.md`
  - Maps Table 1, Figure 1/Table 2, Tables 2--3, Table 4, and the case study to
    the exact JSON/script artifacts.
  - Explains that Table 4 robustness covers bedroom seeds 31--33 and
    full-resolution FFHQ seeds 31--33, with lower-detail FFHQ lod 1/2 controls.
  - Avoids naming excluded/dead-line files directly so archive leak sweeps stay
    clean.
- Regenerated `note/tmlr_submission.tar.gz` including `evidence/README.md`.
- Verification:
  - Tar listing includes `evidence/README.md`.
  - Archive leak sweep clean for identity strings, old drafts, excluded evidence,
    and dead-line filenames.
  - PDF remains `note/submission/tmlr_predict.pdf`, 11 pages.

## 2026-06-22 TMLR Submission Preflight Harness

- Added local submission gate:
  - `note/tools/tmlr_submission_preflight.py`
  - It lives outside `note/tmlr_submission.tar.gz` so forbidden-pattern strings do
    not ship to reviewers.
- The harness implements the local TMLR package/criteria-auditor standards:
  - rebuilds `tmlr_predict.tex` when `tectonic` is available,
  - checks PDF page count, anonymity, unresolved refs, text and metadata leaks,
  - checks source tar exact membership, cache/internal paths, text leak patterns,
  - checks `evidence/README.md` references,
  - asserts headline numbers for Table 1, Table 2, Table 4, the multi-anchor case
    study, and FFHQ resolution rank stability,
  - extracts the archive as a smoke test.
- Fixed a real source-package issue found by the harness:
  - replaced the unused `\openreview` placeholder `XXXXXXXXXX` in
    `note/submission/tmlr_predict.tex` with an empty definition.
- Rebuilt:
  - `note/submission/tmlr_predict.pdf`
  - `note/tmlr_submission.tar.gz`
- Final verification:
  - `/home/soccz/.pyenv/versions/3.12.9/bin/python note/tools/tmlr_submission_preflight.py`
  - Result: `VERDICT: SUBMIT-READY`.

## 2026-06-22 TMLR Reference Audit Hardened

- User asked for stricter reference verification before TMLR submission.
- Fixed reference issues:
  - Added missing InterFaceGAN citation `shen2020` because FFHQ robustness uses
    InterFaceGAN semantic boundaries/checkpoint.
  - Corrected `koulakis2026` authors from `and others` to `Marios Koulakis and
    Constantin Seibold` based on the arXiv record.
  - Corrected reproducibility text from FFHQ lod `1`/`2` only to lod `0`--`2`.
  - Pruned 10 uncited BibTeX entries from `note/submission/refs.bib`.
- Hardened `note/tools/tmlr_submission_preflight.py`:
  - checks cited key set vs `refs.bib`,
  - fails on missing/unused/duplicate BibTeX keys,
  - checks sentinel title/venue/year/arXiv fields for all 20 cited references,
  - fails on broad `and others` author fields except the PyTorch mega-author
    reference.
- Rebuilt:
  - `note/submission/tmlr_predict.pdf`
  - `note/tmlr_submission.tar.gz`
- Final verification:
  - `/home/soccz/.pyenv/versions/3.12.9/bin/python note/tools/tmlr_submission_preflight.py`
  - Result: `VERDICT: SUBMIT-READY`.

## 2026-06-22 Final 20-Loop TMLR Submission Audit

- User asked for a final strict 20-loop audit against the TMLR harness before
  submission.
- Found and fixed a real anonymity/package issue:
  - `note/tmlr_submission.tar.gz` previously stored tar owner/group metadata as
    the local username.
  - Regenerated the archive with sanitized metadata:
    `tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=2026-06-22T00:00:00Z ...`
  - `tar -tvzf note/tmlr_submission.tar.gz` now shows `0/0` owner/group.
- Hardened `note/tools/tmlr_submission_preflight.py`:
  - checks tar member path safety,
  - fails on non-file members,
  - scans tar owner/group metadata for identity/path leaks.
- Ran a 20-loop adversarial package audit plus an extra standalone compile from
  the uploaded tar contents:
  - Result: `20-LOOP VERDICT: PASS`.
- Ran final full preflight:
  - `/home/soccz/.pyenv/versions/3.12.9/bin/python note/tools/tmlr_submission_preflight.py`
  - Result: `VERDICT: SUBMIT-READY`.
- Final upload targets remain:
  - `note/submission/tmlr_predict.pdf`
  - `note/tmlr_submission.tar.gz`

## 2026-06-22 TMLR Guide Strict Re-Audit

- User explicitly distrusted the previous verdict and asked to check the actual
  TMLR guide.
- Found the guide path from `TMLR_REWRITE_KICKOFF.md`:
  - `/home/soccz/22tb/.claude-packs/tmlr/guide/TMLR_WRITING_GUIDE.md`
  - `/home/soccz/22tb/.claude-packs/tmlr/guide/NEGATIVE_AUDIT_PLAYBOOK.md`
  - `/home/soccz/22tb/.claude-packs/tmlr/guide/DESK_REJECT_AUTOPSY.md`
  - `/home/soccz/22tb/.claude-packs/tmlr/guide/PROCESS_AND_HUMAN_ROLE.md`
- Strict guide-driven fixes applied:
  - Retitled manuscript from `An Exact, Step-Free Instrument for Second-Order
    Generator Geometry` to `Finite Differences Are Unreliable for Second-Order
    Generator Geometry`.
  - Removed first-30-seconds self-shrinking language from abstract/intro:
    `Exact higher-order automatic differentiation is not new`,
    `not a new higher-order AD algorithm`, `carries no algorithmic novelty`,
    `not a general claim about curvature`, `we do not claim`.
  - Replaced `case study` framing with `replication audit` framing.
  - Corrected evidence README: FFHQ runner scripts record GPU protocols and
    require public checkpoints/repository wrappers; fixed JSONs verify submitted
    tables without rerunning generator experiments.
  - Hardened `note/tools/tmlr_submission_preflight.py` to fail on those
    guide-risk phrases.
- Verification:
  - Re-ran `compute_trilemma_table.py`: best steps remain `3.0/2.0/0.2`.
  - Rebuilt `note/submission/tmlr_predict.pdf` successfully, 11 pages.
  - Regenerated `note/tmlr_submission.tar.gz` with sanitized tar metadata.
  - Full preflight: `VERDICT: SUBMIT-READY`.
  - Package preflight after final tar: `VERDICT: SUBMIT-READY`.
  - Guide 20-loop adversarial audit: `GUIDE 20-LOOP VERDICT: PASS`.

## 2026-06-22 Pattern-8 Consequence Reframe

- User surfaced a stricter critique: the manuscript still risked the original
  9732 failure pattern 8, because "measurement is wrong" did not yet close as
  a changed decision strongly enough.
- Accepted the critique as materially correct:
  - New experiments would be needed to fully close the strongest TMLR
    consequence pattern.
  - Without new experiments, the safest guide-compliant move is the
    NEGATIVE_AUDIT_PLAYBOOK "numbered recommendations" route.
- Applied prose-only manuscript changes:
  - Abstract now states the operational recommendation: second-order
    generator-geometry reports should use an exact estimator or validate FD
    step choice for magnitude/sign/rank and replicate direction-level claims
    across anchors.
  - Section 7 renamed to `Recommendations, scope, and limitations`.
  - Added three explicit recommendations:
    1. use the exact instrument for second-order generator geometry,
    2. report the step axis when finite differences are used,
    3. replicate direction-level second-order claims across anchors.
  - Removed the over-strong "drop-in replacement" wording for Hessian Penalty;
    now limited to per-sample second-order measurement at comparable constant
    generator-evaluation cost.
  - Removed remaining defensive "we do not read..." phrasing.
- Rebuilt and repackaged:
  - `note/submission/tmlr_predict.pdf` (11 pages)
  - `note/tmlr_submission.tar.gz`
- Verification:
  - Package preflight: `VERDICT: SUBMIT-READY`.
  - Guide-strict 20-check audit: `GUIDE-STRICT VERDICT: PASS`.

---

## 2026-07-14 — 최종 종결 기록 (폴더 봉인)

이 파일의 위 기록은 "SUBMIT-READY"에서 끝나지만, 그것이 결말이 아니다. 봉인 시점에 결말을 명기한다:

- **TMLR #9732 (tmlr_paper.tex, FD sign-flip 라인): desk-reject** — EiC "TMLR is not a suitable venue for this work". 재도전 없음.
- **리라이트 v2 (tmlr_predict.tex, step-selection trilemma): 제출하지 않고 자발 동결** — 6/19-22 consequence 사냥(워크플로우 6종)이 "다운스트림 consequence 구조적 부재"를 확정. 두 번 리젝된 것이 아님.
- **프로젝트 판정: ABANDONED (2026-06-22)** — 실패가 아니라 검증된 negative로 종결. FD 2차 곡률은 rank를 보존하므로(Spearman 0.815~0.963) magnitude 오차 45%가 어떤 결정도 뒤집지 않는다.
- **산출물**: exact composed-JVP instrument + FD floor/trilemma의 fixed-seed 증거(note/submission/evidence/) — arXiv 기록 가치. 인터랙티브 결과는 soccz.github.io/projects/higan/ 에 공개 유지.
- **계승**: 이 폴더의 tsfm_audit/(5/30 shelved)가 2026-07-07 `22tb/tsfm_starvation/`으로 부활, 사전등록 게이트 7개 + 적대검증을 거쳐 **완성 논문 "Data-Starved Baselines Inflate the Measured Advantage of Time-Series Foundation Models on ETT" (TMLR 제출 대기)**가 됨 — 이 폴더 2년 만의 첫 완성 논문의 씨앗.
- **2026-07-14 정리**: 폐기 라인 산출물 ~64GB 삭제(ffhq_c5*, 중간 ckpt 39개 — 최종 enc_040000.pt는 보존), 죽은 logs/ 삭제, 개인 스크린샷은 repo 밖으로 이동. Google Takeout 백업(etc/higan/*.zip, 13GB)은 개인 데이터라 보존.
