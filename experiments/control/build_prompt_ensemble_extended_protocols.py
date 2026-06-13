"""Generate locked protocols for prompt-ensemble extended stress tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_PREDICTIVE = "experiments/control/run_cross_domain_risk_predictive.py"
PY_PREDICTIVE_SUMMARY = "experiments/control/run_risk_predictive_summary.py"
PY_STRESS_EVIDENCE = "experiments/control/run_predictive_stress_evidence_table.py"
PY_AUDIT = "experiments/control/audit_control_campaign.py"

BASE_AUDIT_PROTOCOLS = [
    "experiments/protocols/control_v3_risk_robustness.json",
    "experiments/protocols/control_v4_risk_signal_negatives.json",
    "experiments/protocols/control_v5_risk_power_sensitivity.json",
    "experiments/protocols/control_v6_true_lpips_validation.json",
    "experiments/protocols/control_v7_expanded_candidate_universe.json",
    "experiments/protocols/control_v9_fixed_target_negatives.json",
    "experiments/protocols/control_v11_extended_validation.json",
    "experiments/protocols/control_v12_threshold_magnitude_sensitivity.json",
    "experiments/protocols/control_v14_true_lpips_fixed_target_negatives.json",
    "experiments/protocols/control_v15_expanded_universe_true_lpips.json",
    "experiments/protocols/control_v16_estimator_stability.json",
    "experiments/protocols/control_v18_bedroom_cross_domain_controller.json",
    "experiments/protocols/control_v20_church_cross_domain_controller.json",
    "experiments/protocols/control_v22_church_risk_power_sensitivity.json",
    "experiments/protocols/control_v23_church_failure_diagnosis.json",
    "experiments/protocols/control_v25_church_structured_negative_controls.json",
    "experiments/protocols/control_v27_church_structured_confirmatory_replication.json",
    "experiments/protocols/control_v28_church_structured_estimator_stability.json",
    "experiments/protocols/control_v29_church_structured_source_ablation.json",
    "experiments/protocols/control_v31_cross_domain_risk_predictive_validity.json",
    "experiments/protocols/control_v32_church_gain_first_risk_tiebreak_controller.json",
    "experiments/protocols/control_v34_main_claim_readiness.json",
    "experiments/protocols/control_v36_main_evidence_table.json",
    "experiments/protocols/control_v38_semantic_feasible_minrisk_controller.json",
    "experiments/protocols/control_v39_feasible_control_readiness.json",
    "experiments/protocols/control_v40_feasible_control_evidence_table.json",
    "experiments/protocols/control_v42_fd_risk_predictive_validity.json",
    "experiments/protocols/control_v43_prompt_template_predictive_validity.json",
    "experiments/protocols/control_v44_predictive_assumption_readiness.json",
    "experiments/protocols/control_v46_caption_prompt_predictive_validity.json",
    "experiments/protocols/control_v47_high_ntest_predictive_validity.json",
    "experiments/protocols/control_v48_strict_gain_match_predictive_validity.json",
    "experiments/protocols/control_v49_predictive_stress_evidence_table.json",
    "experiments/protocols/control_v51_prompt_ensemble_predictive_validity.json",
    "experiments/protocols/control_v52_predictive_stress_evidence_table_with_ensemble.json",
    "experiments/protocols/control_v54_prompt_ensemble_followup_predictive_validity.json",
    "experiments/protocols/control_v55_predictive_stress_evidence_table_ensemble_followup.json",
]


def write_protocol(name: str, payload: dict[str, Any]) -> None:
    path = PROTOCOLS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def attrs_for(domain: str) -> list[str]:
    if domain == "church":
        return ["clouds", "sunny", "vegetation"]
    if domain == "bedroom":
        return ["indoor_lighting", "wood", "view", "carpet"]
    raise ValueError(domain)


def predictive_args(domain: str, seed: int, out: str,
                    overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {
        "domain": domain,
        "attrs": attrs_for(domain),
        "methods": ["ganspace", "sefa"],
        "candidate-k": 6,
        "ganspace-samples": 2048,
        "min-gain-quantile": 0.5,
        "gain-match-rel": 0.25,
        "true-lpips": True,
        "lpips-net": "alex",
        "lpips-size": 256,
        "n-risk": 8,
        "n-probe": 16,
        "probe-alpha": 1.0,
        "n-calib": 16,
        "n-test": 64,
        "alpha-steps": 7,
        "max-alpha": 6,
        "target-quantile": 0.25,
        "batch": 4,
        "risk-estimator": "jvp",
        "prompt-style": "ensemble",
        "out": out,
        "seed": seed,
    }
    if overrides:
        args.update(overrides)
    if args["risk-estimator"] == "fd":
        args["fd-eps"] = 0.25
    return args


def predictive_protocol(version: str, purpose: str, out_root: str,
                        seeds: list[int], overrides: dict[str, Any]) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    order = 1
    for domain in ["church", "bedroom"]:
        for seed in seeds:
            out = f"{out_root}/{domain}_seed_{seed}"
            experiments.append({
                "key": f"{domain}_{version}_seed_{seed}",
                "order": order,
                "script": PY_PREDICTIVE,
                "claim_tested": (
                    "Prompt-ensemble predictive validity should remain "
                    f"directionally stable under this stress setting in {domain}."
                ),
                "args": predictive_args(domain, seed, out, overrides),
            })
            order += 1
    for domain in ["church", "bedroom"]:
        inputs = [f"{out_root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]
        experiments.append({
            "key": f"{domain}_{version}_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": (
                f"Aggregate {domain} prompt-ensemble predictive-validity "
                f"runs for {version}."
            ),
            "args": {
                "inputs": inputs,
                "out": f"{out_root}/{domain}",
                "seed": 4400 + order,
            },
        })
        order += 1
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": purpose,
        "experiments": experiments,
    }


def v57_ensemble_fd() -> dict[str, Any]:
    return predictive_protocol(
        "control_v57_prompt_ensemble_fd_predictive_validity",
        (
            "Test whether prompt-ensemble predictive validity survives replacing "
            "JVP rho with finite-difference rho."
        ),
        "experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_fd",
        list(range(2092, 2097)),
        {"risk-estimator": "fd"},
    )


def source_ablation_protocol() -> dict[str, Any]:
    version = "control_v58_prompt_ensemble_source_ablation_predictive_validity"
    experiments: list[dict[str, Any]] = []
    order = 1
    for label, methods, out_root, seeds in [
        (
            "ganspace_only",
            ["ganspace"],
            "experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_ganspace_only",
            list(range(2097, 2102)),
        ),
        (
            "sefa_only",
            ["sefa"],
            "experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_sefa_only",
            list(range(2102, 2107)),
        ),
    ]:
        for domain in ["church", "bedroom"]:
            for seed in seeds:
                out = f"{out_root}/{domain}_seed_{seed}"
                experiments.append({
                    "key": f"{domain}_{version}_{label}_seed_{seed}",
                    "order": order,
                    "script": PY_PREDICTIVE,
                    "claim_tested": (
                        "Prompt-ensemble predictive validity should not depend "
                        f"entirely on one candidate-source family: {label} in {domain}."
                    ),
                    "args": predictive_args(
                        domain,
                        seed,
                        out,
                        {"methods": methods},
                    ),
                })
                order += 1
        for domain in ["church", "bedroom"]:
            inputs = [f"{out_root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]
            experiments.append({
                "key": f"{domain}_{version}_{label}_summary",
                "order": order,
                "script": PY_PREDICTIVE_SUMMARY,
                "claim_tested": (
                    f"Aggregate {domain} prompt-ensemble {label} predictive "
                    "source-ablation runs."
                ),
                "args": {
                    "inputs": inputs,
                    "out": f"{out_root}/{domain}",
                    "seed": 4500 + order,
                },
            })
            order += 1
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": (
            "Test whether prompt-ensemble predictive validity depends on the "
            "GANSpace or SeFa candidate-source family."
        ),
        "experiments": experiments,
    }


def v59_evidence() -> dict[str, Any]:
    summaries = [
        "baseline_church=experiments/out/control_cross_domain_risk_predictive/church/metrics.json",
        "baseline_bedroom=experiments/out/control_cross_domain_risk_predictive/bedroom/metrics.json",
        "ensemble_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble/church/metrics.json",
        "ensemble_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble/bedroom/metrics.json",
        "ensemble_fd_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_fd/church/metrics.json",
        "ensemble_fd_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_fd/bedroom/metrics.json",
        "ensemble_high_ntest_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_high_ntest128/church/metrics.json",
        "ensemble_high_ntest_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_high_ntest128/bedroom/metrics.json",
        "ensemble_strict_gain_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_strict_gain_match/church/metrics.json",
        "ensemble_strict_gain_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_strict_gain_match/bedroom/metrics.json",
        "ensemble_ganspace_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_ganspace_only/church/metrics.json",
        "ensemble_ganspace_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_ganspace_only/bedroom/metrics.json",
        "ensemble_sefa_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_sefa_only/church/metrics.json",
        "ensemble_sefa_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_sefa_only/bedroom/metrics.json",
    ]
    return {
        "version": "control_v59_prompt_ensemble_extended_evidence_table",
        "locked_on": "2026-05-28",
        "purpose": (
            "Build evidence table for prompt-ensemble estimator and source "
            "ablation stress tests."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v4",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "Prompt-ensemble rho should retain rank-level predictive "
                    "evidence across estimator and candidate-source stresses."
                ),
                "args": {
                    "summaries": summaries,
                    "out": "experiments/out/control_predictive_stress_evidence_table_v4",
                    "seed": 4212,
                },
            }
        ],
    }


def v60_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v57_prompt_ensemble_fd_predictive_validity.json",
        "experiments/protocols/control_v58_prompt_ensemble_source_ablation_predictive_validity.json",
        "experiments/protocols/control_v59_prompt_ensemble_extended_evidence_table.json",
    ]
    return {
        "version": "control_v60_main_grade_audit_v14",
        "locked_on": "2026-05-28",
        "purpose": (
            "Final non-GPU reproducibility audit including prompt-ensemble "
            "estimator and source-ablation stress tests."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v14",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through prompt-ensemble estimator "
                    "and source-ablation stress tests have complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v14",
                    "seed": 3723,
                },
            }
        ],
    }


def main() -> None:
    write_protocol("control_v57_prompt_ensemble_fd_predictive_validity.json", v57_ensemble_fd())
    write_protocol(
        "control_v58_prompt_ensemble_source_ablation_predictive_validity.json",
        source_ablation_protocol(),
    )
    write_protocol("control_v59_prompt_ensemble_extended_evidence_table.json", v59_evidence())
    write_protocol("control_v60_main_grade_audit_v14.json", v60_audit())


if __name__ == "__main__":
    main()
