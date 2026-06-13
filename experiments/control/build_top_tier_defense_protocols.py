"""Generate locked protocols for top-tier reviewer-defense experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_PREDICTIVE = "experiments/control/run_cross_domain_risk_predictive.py"
PY_PREDICTIVE_SUMMARY = "experiments/control/run_risk_predictive_summary.py"
PY_STRESS_EVIDENCE = "experiments/control/run_predictive_stress_evidence_table.py"
PY_DINO_EVIDENCE = "experiments/control/run_predictive_dino_evidence_table.py"
PY_INCREMENTAL = "experiments/control/run_predictive_incremental_value_table.py"
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
    "experiments/protocols/control_v57_prompt_ensemble_fd_predictive_validity.json",
    "experiments/protocols/control_v58_prompt_ensemble_source_ablation_predictive_validity.json",
    "experiments/protocols/control_v59_prompt_ensemble_extended_evidence_table.json",
    "experiments/protocols/control_v61_ffhq_prompt_ensemble_predictive_validity.json",
    "experiments/protocols/control_v62_ffhq_prompt_ensemble_followup_predictive_validity.json",
    "experiments/protocols/control_v63_random_universe_prompt_ensemble_predictive_validity.json",
    "experiments/protocols/control_v64_dino_preservation_predictive_validity.json",
    "experiments/protocols/control_v65_main_grade_extension_evidence_table.json",
    "experiments/protocols/control_v66_dino_preservation_evidence_table.json",
    "experiments/protocols/control_v67_main_grade_audit_v15.json",
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
    if domain == "ffhq":
        return ["smile", "age", "pose", "gender", "eyeglasses"]
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
        "batch": 2 if domain == "ffhq" else 4,
        "risk-estimator": "jvp",
        "prompt-style": "ensemble",
        "out": out,
        "seed": seed,
    }
    if overrides:
        args.update(overrides)
    return args


def old_seed_paths(root: str, domain: str, seeds: list[int]) -> list[str]:
    return [f"{root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]


def add_seed_extension_block(experiments: list[dict[str, Any]], *,
                             version: str, label: str, out_root: str,
                             old_root: str, summary_root: str,
                             domains: list[str], old_seeds: list[int],
                             new_seeds: list[int],
                             overrides: dict[str, Any]) -> None:
    order = len(experiments) + 1
    for domain in domains:
        for seed in new_seeds:
            out = f"{out_root}/{domain}_seed_{seed}"
            experiments.append({
                "key": f"{domain}_{version}_{label}_seed_{seed}",
                "order": order,
                "script": PY_PREDICTIVE,
                "claim_tested": (
                    "Top-tier seed scaling: the rho predictive signal should "
                    f"remain stable when {label} is extended from five to ten "
                    f"seeds in {domain}."
                ),
                "args": predictive_args(domain, seed, out, overrides),
            })
            order += 1
    for domain in domains:
        inputs = [
            *old_seed_paths(old_root, domain, old_seeds),
            *old_seed_paths(out_root, domain, new_seeds),
        ]
        experiments.append({
            "key": f"{domain}_{version}_{label}_n10_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": (
                f"Aggregate {domain} {label} predictive-validity runs at n=10."
            ),
            "args": {
                "inputs": inputs,
                "out": f"{summary_root}/{domain}",
                "seed": 4700 + order,
            },
        })
        order += 1


def v68_random_n10() -> dict[str, Any]:
    version = "control_v68_random_universe_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_seed_extension_block(
        experiments,
        version=version,
        label="random_universe",
        out_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_seed_scaling",
        old_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe",
        summary_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10",
        domains=["church", "bedroom", "ffhq"],
        old_seeds=list(range(2122, 2127)),
        new_seeds=list(range(2132, 2137)),
        overrides={"methods": ["ganspace", "sefa", "random"]},
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Scale the strongest candidate-universe stress test from five to "
            "ten seeds so sign-test evidence can cross conventional thresholds."
        ),
        "experiments": experiments,
    }


def v69_dino_n10() -> dict[str, Any]:
    version = "control_v69_dino_preservation_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_seed_extension_block(
        experiments,
        version=version,
        label="dino_preservation",
        out_root="experiments/out/control_cross_domain_risk_predictive_dino_preservation_seed_scaling",
        old_root="experiments/out/control_cross_domain_risk_predictive_dino_preservation",
        summary_root="experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10",
        domains=["church", "bedroom", "ffhq"],
        old_seeds=list(range(2127, 2132)),
        new_seeds=list(range(2137, 2142)),
        overrides={
            "dino-preservation": True,
            "dino-local-files-only": True,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Scale the non-language DINO-preservation stress test from five to "
            "ten seeds for top-tier statistical robustness."
        ),
        "experiments": experiments,
    }


def v70_n10_evidence() -> dict[str, Any]:
    return {
        "version": "control_v70_seed_scaled_extension_evidence_table",
        "locked_on": "2026-05-29",
        "purpose": (
            "Build the n=10 evidence table for random-universe and "
            "DINO-preservation seed-scaled predictive tests."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v6_n10",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "Seed-scaled random-universe and DINO-preservation "
                    "predictive evidence should remain directionally stable."
                ),
                "args": {
                    "summaries": [
                        "random_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/church/metrics.json",
                        "random_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/bedroom/metrics.json",
                        "random_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/ffhq/metrics.json",
                        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
                        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
                        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
                    ],
                    "out": "experiments/out/control_predictive_stress_evidence_table_v6_n10",
                    "seed": 4216,
                },
            }
        ],
    }


def v71_dino_n10_evidence() -> dict[str, Any]:
    return {
        "version": "control_v71_dino_preservation_seed_scaled_evidence_table",
        "locked_on": "2026-05-29",
        "purpose": (
            "Build a DINO-specific n=10 evidence table for non-language "
            "preservation predictive validity."
        ),
        "experiments": [
            {
                "key": "predictive_dino_evidence_table_v2_n10",
                "order": 1,
                "script": PY_DINO_EVIDENCE,
                "claim_tested": (
                    "The seed-scaled rho signal should predict DINOv2 feature "
                    "damage beyond the original five-seed result."
                ),
                "args": {
                    "summaries": [
                        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
                        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
                        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
                    ],
                    "out": "experiments/out/control_predictive_dino_evidence_table_v2_n10",
                    "seed": 4217,
                },
            }
        ],
    }


def v72_incremental_value() -> dict[str, Any]:
    return {
        "version": "control_v72_rho_incremental_value_table",
        "locked_on": "2026-05-29",
        "purpose": (
            "Test whether rho adds cross-validated predictive value beyond "
            "semantic gain, alpha magnitude, candidate source, and attribute."
        ),
        "experiments": [
            {
                "key": "rho_incremental_value_table_v1",
                "order": 1,
                "script": PY_INCREMENTAL,
                "claim_tested": (
                    "rho should not be merely a proxy for semantic gain, edit "
                    "magnitude, source family, or attribute identity."
                ),
                "args": {
                    "summaries": [
                        "random_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/church/metrics.json",
                        "random_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/bedroom/metrics.json",
                        "random_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/ffhq/metrics.json",
                        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
                        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
                        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
                    ],
                    "out": "experiments/out/control_rho_incremental_value_table_v1",
                    "folds": 5,
                    "seed": 4218,
                },
            }
        ],
    }


def v73_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v68_random_universe_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v69_dino_preservation_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v70_seed_scaled_extension_evidence_table.json",
        "experiments/protocols/control_v71_dino_preservation_seed_scaled_evidence_table.json",
        "experiments/protocols/control_v72_rho_incremental_value_table.json",
    ]
    return {
        "version": "control_v73_main_grade_audit_v16",
        "locked_on": "2026-05-29",
        "purpose": (
            "Final non-GPU reproducibility audit including top-tier seed "
            "scaling and rho incremental-value analyses."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v16",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through top-tier defense tests have "
                    "complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v16",
                    "seed": 3725,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v68_random_universe_seed_scaling_predictive_validity.json",
        v68_random_n10(),
    )
    write_protocol(
        "control_v69_dino_preservation_seed_scaling_predictive_validity.json",
        v69_dino_n10(),
    )
    write_protocol(
        "control_v70_seed_scaled_extension_evidence_table.json",
        v70_n10_evidence(),
    )
    write_protocol(
        "control_v71_dino_preservation_seed_scaled_evidence_table.json",
        v71_dino_n10_evidence(),
    )
    write_protocol("control_v72_rho_incremental_value_table.json", v72_incremental_value())
    write_protocol("control_v73_main_grade_audit_v16.json", v73_audit())


if __name__ == "__main__":
    main()
