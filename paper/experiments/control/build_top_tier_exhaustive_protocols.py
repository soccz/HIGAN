"""Generate locked protocols for additional top-tier defense experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_PREDICTIVE = "experiments/control/run_cross_domain_risk_predictive.py"
PY_PREDICTIVE_SUMMARY = "experiments/control/run_risk_predictive_summary.py"
PY_STRESS_EVIDENCE = "experiments/control/run_predictive_stress_evidence_table.py"
PY_PERMUTATION = "experiments/control/run_predictive_permutation_null_table.py"
PY_BOUNDARY = "experiments/control/run_failure_boundary_table.py"
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
    "experiments/protocols/control_v68_random_universe_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v69_dino_preservation_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v70_seed_scaled_extension_evidence_table.json",
    "experiments/protocols/control_v71_dino_preservation_seed_scaled_evidence_table.json",
    "experiments/protocols/control_v72_rho_incremental_value_table.json",
    "experiments/protocols/control_v73_main_grade_audit_v16.json",
    "experiments/protocols/control_v74_failure_boundary_table.json",
    "experiments/protocols/control_v75_main_grade_audit_v17.json",
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
    if args["risk-estimator"] == "fd":
        args["fd-eps"] = 0.25
    return args


def add_predictive_block(experiments: list[dict[str, Any]], *,
                         version: str, label: str, out_root: str,
                         domains: list[str], seeds: list[int],
                         overrides: dict[str, Any]) -> None:
    order = len(experiments) + 1
    for domain in domains:
        for seed in seeds:
            out = f"{out_root}/{domain}_seed_{seed}"
            experiments.append({
                "key": f"{domain}_{version}_{label}_seed_{seed}",
                "order": order,
                "script": PY_PREDICTIVE,
                "claim_tested": (
                    "Top-tier defense: rho predictive validity should remain "
                    f"auditable under {label} in {domain}; failures are kept "
                    "as boundary evidence."
                ),
                "args": predictive_args(domain, seed, out, overrides),
            })
            order += 1
    for domain in domains:
        inputs = [f"{out_root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]
        experiments.append({
            "key": f"{domain}_{version}_{label}_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": f"Aggregate {domain} predictive runs for {label}.",
            "args": {
                "inputs": inputs,
                "out": f"{out_root}/{domain}",
                "seed": 4800 + order,
            },
        })
        order += 1


def v76_ffhq_fd() -> dict[str, Any]:
    version = "control_v76_ffhq_fd_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="ffhq_fd",
        out_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd",
        domains=["ffhq"],
        seeds=list(range(2142, 2147)),
        overrides={"risk-estimator": "fd"},
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Close the FFHQ estimator gap by replacing JVP rho with "
            "finite-difference rho under the prompt-ensemble protocol."
        ),
        "experiments": experiments,
    }


def v77_ffhq_source_ablation() -> dict[str, Any]:
    version = "control_v77_ffhq_source_ablation_predictive_validity"
    experiments: list[dict[str, Any]] = []
    for label, methods, out_root, seeds in [
        (
            "ffhq_ganspace_only",
            ["ganspace"],
            "experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only",
            list(range(2147, 2152)),
        ),
        (
            "ffhq_sefa_only",
            ["sefa"],
            "experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only",
            list(range(2152, 2157)),
        ),
    ]:
        add_predictive_block(
            experiments,
            version=version,
            label=label,
            out_root=out_root,
            domains=["ffhq"],
            seeds=seeds,
            overrides={"methods": methods},
        )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Close the FFHQ candidate-source gap by testing GANSpace-only and "
            "SeFa-only prompt-ensemble predictive validity."
        ),
        "experiments": experiments,
    }


def v78_estimator_budget() -> dict[str, Any]:
    version = "control_v78_estimator_budget_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="random_universe_high_budget",
        out_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget",
        domains=["church", "bedroom", "ffhq"],
        seeds=list(range(2157, 2162)),
        overrides={
            "methods": ["ganspace", "sefa", "random"],
            "n-risk": 16,
            "n-probe": 32,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Test whether the random-universe prompt-ensemble signal survives "
            "a larger rho/probe sampling budget."
        ),
        "experiments": experiments,
    }


def v79_wide_universe() -> dict[str, Any]:
    version = "control_v79_wide_candidate_universe_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="wide_universe",
        out_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe",
        domains=["church", "bedroom", "ffhq"],
        seeds=list(range(2162, 2167)),
        overrides={
            "methods": ["ganspace", "sefa", "random"],
            "candidate-k": 10,
            "ganspace-samples": 4096,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Test candidate-universe scale sensitivity by expanding the "
            "direction pool and GANSpace PCA sample budget."
        ),
        "experiments": experiments,
    }


def v80_target_magnitude() -> dict[str, Any]:
    version = "control_v80_target_magnitude_predictive_validity"
    experiments: list[dict[str, Any]] = []
    for label, quantile, seeds in [
        ("target_q10", 0.10, list(range(2167, 2172))),
        ("target_q50", 0.50, list(range(2172, 2177))),
    ]:
        add_predictive_block(
            experiments,
            version=version,
            label=label,
            out_root=f"experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_{label}",
            domains=["church", "bedroom", "ffhq"],
            seeds=seeds,
            overrides={
                "methods": ["ganspace", "sefa", "random"],
                "target-quantile": quantile,
            },
        )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Test whether predictive validity is specific to the default "
            "semantic target magnitude or survives lower and higher targets."
        ),
        "experiments": experiments,
    }


def exhaustive_summaries() -> list[str]:
    return [
        "ffhq_fd=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd/ffhq/metrics.json",
        "ffhq_ganspace=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only/ffhq/metrics.json",
        "ffhq_sefa=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only/ffhq/metrics.json",
        "budget_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget/church/metrics.json",
        "budget_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget/bedroom/metrics.json",
        "budget_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget/ffhq/metrics.json",
        "wide_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe/church/metrics.json",
        "wide_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe/bedroom/metrics.json",
        "wide_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe/ffhq/metrics.json",
        "target_q10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10/church/metrics.json",
        "target_q10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10/bedroom/metrics.json",
        "target_q10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10/ffhq/metrics.json",
        "target_q50_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50/church/metrics.json",
        "target_q50_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50/bedroom/metrics.json",
        "target_q50_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50/ffhq/metrics.json",
    ]


def permutation_summaries() -> list[str]:
    return [
        "random_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/church/metrics.json",
        "random_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/bedroom/metrics.json",
        "random_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/ffhq/metrics.json",
        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
        *exhaustive_summaries(),
    ]


def v81_evidence() -> dict[str, Any]:
    return {
        "version": "control_v81_exhaustive_defense_evidence_table",
        "locked_on": "2026-05-29",
        "purpose": (
            "Build the evidence table for the added FFHQ, estimator-budget, "
            "wide-universe, and target-magnitude defense experiments."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v7_exhaustive",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "The rho predictive signal should remain auditable across "
                    "the remaining top-tier defense axes."
                ),
                "args": {
                    "summaries": exhaustive_summaries(),
                    "out": "experiments/out/control_predictive_stress_evidence_table_v7_exhaustive",
                    "seed": 4220,
                },
            }
        ],
    }


def v82_permutation() -> dict[str, Any]:
    return {
        "version": "control_v82_predictive_permutation_null_table",
        "locked_on": "2026-05-29",
        "purpose": (
            "Run an attribute-wise rho permutation null over the top-tier "
            "summary set to test for bookkeeping or candidate-pool artifacts."
        ),
        "experiments": [
            {
                "key": "predictive_permutation_null_table_v1",
                "order": 1,
                "script": PY_PERMUTATION,
                "claim_tested": (
                    "Observed rho/damage relationships should be stronger "
                    "than an attribute-wise shuffled-rho null."
                ),
                "args": {
                    "summaries": permutation_summaries(),
                    "out": "experiments/out/control_predictive_permutation_null_table_v1",
                    "permutations": 500,
                    "alpha": 0.05,
                    "seed": 4221,
                },
            }
        ],
    }


def v83_boundary() -> dict[str, Any]:
    return {
        "version": "control_v83_failure_boundary_table_v2",
        "locked_on": "2026-05-29",
        "purpose": (
            "Update the failure-boundary table after the exhaustive defense "
            "and permutation-null evidence."
        ),
        "experiments": [
            {
                "key": "failure_boundary_table_v2",
                "order": 1,
                "script": PY_BOUNDARY,
                "claim_tested": (
                    "The final paper framing should expose robust and weak "
                    "regimes across all top-tier defense axes."
                ),
                "args": {
                    "tables": [
                        "stress_v4=experiments/out/control_predictive_stress_evidence_table_v4/metrics.json",
                        "extension_v5=experiments/out/control_predictive_stress_evidence_table_v5/metrics.json",
                        "dino_v1=experiments/out/control_predictive_dino_evidence_table_v1/metrics.json",
                        "seed_scaled_v6=experiments/out/control_predictive_stress_evidence_table_v6_n10/metrics.json",
                        "dino_n10_v2=experiments/out/control_predictive_dino_evidence_table_v2_n10/metrics.json",
                        "incremental_v1=experiments/out/control_rho_incremental_value_table_v1/metrics.json",
                        "exhaustive_v7=experiments/out/control_predictive_stress_evidence_table_v7_exhaustive/metrics.json",
                        "permutation_v1=experiments/out/control_predictive_permutation_null_table_v1/metrics.json",
                    ],
                    "out": "experiments/out/control_failure_boundary_table_v2",
                    "seed": 4222,
                },
            }
        ],
    }


def v84_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v76_ffhq_fd_predictive_validity.json",
        "experiments/protocols/control_v77_ffhq_source_ablation_predictive_validity.json",
        "experiments/protocols/control_v78_estimator_budget_predictive_validity.json",
        "experiments/protocols/control_v79_wide_candidate_universe_predictive_validity.json",
        "experiments/protocols/control_v80_target_magnitude_predictive_validity.json",
        "experiments/protocols/control_v81_exhaustive_defense_evidence_table.json",
        "experiments/protocols/control_v82_predictive_permutation_null_table.json",
        "experiments/protocols/control_v83_failure_boundary_table_v2.json",
    ]
    return {
        "version": "control_v84_main_grade_audit_v18",
        "locked_on": "2026-05-29",
        "purpose": (
            "Audit all locked control protocols through the exhaustive "
            "top-tier defense queue."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v18",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through the exhaustive top-tier "
                    "defense queue have traceable metrics outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v18",
                    "seed": 3727,
                },
            }
        ],
    }


def main() -> None:
    write_protocol("control_v76_ffhq_fd_predictive_validity.json", v76_ffhq_fd())
    write_protocol(
        "control_v77_ffhq_source_ablation_predictive_validity.json",
        v77_ffhq_source_ablation(),
    )
    write_protocol(
        "control_v78_estimator_budget_predictive_validity.json",
        v78_estimator_budget(),
    )
    write_protocol(
        "control_v79_wide_candidate_universe_predictive_validity.json",
        v79_wide_universe(),
    )
    write_protocol(
        "control_v80_target_magnitude_predictive_validity.json",
        v80_target_magnitude(),
    )
    write_protocol(
        "control_v81_exhaustive_defense_evidence_table.json",
        v81_evidence(),
    )
    write_protocol(
        "control_v82_predictive_permutation_null_table.json",
        v82_permutation(),
    )
    write_protocol("control_v83_failure_boundary_table_v2.json", v83_boundary())
    write_protocol("control_v84_main_grade_audit_v18.json", v84_audit())


if __name__ == "__main__":
    main()
