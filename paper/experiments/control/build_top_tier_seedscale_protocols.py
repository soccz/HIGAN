"""Generate locked protocols for n=10 scaling of exhaustive defenses."""
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
    "experiments/protocols/control_v76_ffhq_fd_predictive_validity.json",
    "experiments/protocols/control_v77_ffhq_source_ablation_predictive_validity.json",
    "experiments/protocols/control_v78_estimator_budget_predictive_validity.json",
    "experiments/protocols/control_v79_wide_candidate_universe_predictive_validity.json",
    "experiments/protocols/control_v80_target_magnitude_predictive_validity.json",
    "experiments/protocols/control_v81_exhaustive_defense_evidence_table.json",
    "experiments/protocols/control_v82_predictive_permutation_null_table.json",
    "experiments/protocols/control_v83_failure_boundary_table_v2.json",
    "experiments/protocols/control_v84_main_grade_audit_v18.json",
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


def paths(root: str, domain: str, seeds: list[int]) -> list[str]:
    return [f"{root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]


def add_seed_scale_block(experiments: list[dict[str, Any]], *,
                         version: str, label: str, old_root: str,
                         new_root: str, summary_root: str,
                         domains: list[str], old_seeds: list[int],
                         new_seeds: list[int],
                         overrides: dict[str, Any]) -> None:
    order = len(experiments) + 1
    for domain in domains:
        for seed in new_seeds:
            out = f"{new_root}/{domain}_seed_{seed}"
            experiments.append({
                "key": f"{domain}_{version}_{label}_seed_{seed}",
                "order": order,
                "script": PY_PREDICTIVE,
                "claim_tested": (
                    "Top-tier seed scaling: the predeclared defense axis "
                    f"{label} should remain auditable when extended from five "
                    f"to ten seeds in {domain}."
                ),
                "args": predictive_args(domain, seed, out, overrides),
            })
            order += 1
    for domain in domains:
        experiments.append({
            "key": f"{domain}_{version}_{label}_n10_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": f"Aggregate {domain} {label} runs at n=10.",
            "args": {
                "inputs": [
                    *paths(old_root, domain, old_seeds),
                    *paths(new_root, domain, new_seeds),
                ],
                "out": f"{summary_root}/{domain}",
                "seed": 4900 + order,
            },
        })
        order += 1


def v85_ffhq_estimator_source_n10() -> dict[str, Any]:
    version = "control_v85_ffhq_estimator_source_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_seed_scale_block(
        experiments,
        version=version,
        label="ffhq_fd",
        old_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd",
        new_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd_seed_scaling",
        summary_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd_n10",
        domains=["ffhq"],
        old_seeds=list(range(2142, 2147)),
        new_seeds=list(range(2177, 2182)),
        overrides={"risk-estimator": "fd"},
    )
    add_seed_scale_block(
        experiments,
        version=version,
        label="ffhq_ganspace_only",
        old_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only",
        new_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only_seed_scaling",
        summary_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only_n10",
        domains=["ffhq"],
        old_seeds=list(range(2147, 2152)),
        new_seeds=list(range(2182, 2187)),
        overrides={"methods": ["ganspace"]},
    )
    add_seed_scale_block(
        experiments,
        version=version,
        label="ffhq_sefa_only",
        old_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only",
        new_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only_seed_scaling",
        summary_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only_n10",
        domains=["ffhq"],
        old_seeds=list(range(2152, 2157)),
        new_seeds=list(range(2187, 2192)),
        overrides={"methods": ["sefa"]},
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": (
            "Scale FFHQ finite-difference and source-ablation defense axes "
            "from five to ten seeds."
        ),
        "experiments": experiments,
    }


def v86_estimator_budget_n10() -> dict[str, Any]:
    version = "control_v86_estimator_budget_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_seed_scale_block(
        experiments,
        version=version,
        label="random_universe_high_budget",
        old_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget",
        new_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_seed_scaling",
        summary_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_n10",
        domains=["church", "bedroom", "ffhq"],
        old_seeds=list(range(2157, 2162)),
        new_seeds=list(range(2192, 2197)),
        overrides={
            "methods": ["ganspace", "sefa", "random"],
            "n-risk": 16,
            "n-probe": 32,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": "Scale the estimator-budget defense axis from five to ten seeds.",
        "experiments": experiments,
    }


def v87_wide_universe_n10() -> dict[str, Any]:
    version = "control_v87_wide_candidate_universe_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_seed_scale_block(
        experiments,
        version=version,
        label="wide_universe",
        old_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe",
        new_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_seed_scaling",
        summary_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_n10",
        domains=["church", "bedroom", "ffhq"],
        old_seeds=list(range(2162, 2167)),
        new_seeds=list(range(2197, 2202)),
        overrides={
            "methods": ["ganspace", "sefa", "random"],
            "candidate-k": 10,
            "ganspace-samples": 4096,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": "Scale the wide candidate-universe defense axis to n=10.",
        "experiments": experiments,
    }


def v88_target_magnitude_n10() -> dict[str, Any]:
    version = "control_v88_target_magnitude_seed_scaling_predictive_validity"
    experiments: list[dict[str, Any]] = []
    for label, quantile, old_seeds, new_seeds in [
        ("target_q10", 0.10, list(range(2167, 2172)), list(range(2202, 2207))),
        ("target_q50", 0.50, list(range(2172, 2177)), list(range(2207, 2212))),
    ]:
        add_seed_scale_block(
            experiments,
            version=version,
            label=label,
            old_root=f"experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_{label}",
            new_root=f"experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_{label}_seed_scaling",
            summary_root=f"experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_{label}_n10",
            domains=["church", "bedroom", "ffhq"],
            old_seeds=old_seeds,
            new_seeds=new_seeds,
            overrides={
                "methods": ["ganspace", "sefa", "random"],
                "target-quantile": quantile,
            },
        )
    return {
        "version": version,
        "locked_on": "2026-05-29",
        "purpose": "Scale target-magnitude sensitivity axes to n=10.",
        "experiments": experiments,
    }


def exhaustive_n10_summaries() -> list[str]:
    return [
        "ffhq_fd_n10=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_fd_n10/ffhq/metrics.json",
        "ffhq_ganspace_n10=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_ganspace_only_n10/ffhq/metrics.json",
        "ffhq_sefa_n10=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_sefa_only_n10/ffhq/metrics.json",
        "budget_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_n10/church/metrics.json",
        "budget_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_n10/bedroom/metrics.json",
        "budget_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_high_budget_n10/ffhq/metrics.json",
        "wide_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_n10/church/metrics.json",
        "wide_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_n10/bedroom/metrics.json",
        "wide_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_wide_universe_n10/ffhq/metrics.json",
        "target_q10_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10_n10/church/metrics.json",
        "target_q10_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10_n10/bedroom/metrics.json",
        "target_q10_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q10_n10/ffhq/metrics.json",
        "target_q50_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50_n10/church/metrics.json",
        "target_q50_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50_n10/bedroom/metrics.json",
        "target_q50_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_target_q50_n10/ffhq/metrics.json",
    ]


def permutation_n10_summaries() -> list[str]:
    return [
        "random_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/church/metrics.json",
        "random_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/bedroom/metrics.json",
        "random_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/ffhq/metrics.json",
        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
        *exhaustive_n10_summaries(),
    ]


def v89_evidence_n10() -> dict[str, Any]:
    return {
        "version": "control_v89_seed_scaled_exhaustive_defense_evidence_table",
        "locked_on": "2026-05-29",
        "purpose": "Build the n=10 evidence table for exhaustive defense axes.",
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v8_exhaustive_n10",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "The exhaustive top-tier defense axes should remain "
                    "auditable after seed scaling to n=10."
                ),
                "args": {
                    "summaries": exhaustive_n10_summaries(),
                    "out": "experiments/out/control_predictive_stress_evidence_table_v8_exhaustive_n10",
                    "seed": 4223,
                },
            }
        ],
    }


def v90_permutation_n10() -> dict[str, Any]:
    return {
        "version": "control_v90_seed_scaled_predictive_permutation_null_table",
        "locked_on": "2026-05-29",
        "purpose": "Run the permutation-null test on the n=10 defense set.",
        "experiments": [
            {
                "key": "predictive_permutation_null_table_v2_n10",
                "order": 1,
                "script": PY_PERMUTATION,
                "claim_tested": (
                    "The n=10 observed rho/damage relationships should beat "
                    "attribute-wise shuffled-rho nulls."
                ),
                "args": {
                    "summaries": permutation_n10_summaries(),
                    "out": "experiments/out/control_predictive_permutation_null_table_v2_n10",
                    "permutations": 500,
                    "alpha": 0.05,
                    "seed": 4224,
                },
            }
        ],
    }


def v91_boundary_v3() -> dict[str, Any]:
    return {
        "version": "control_v91_failure_boundary_table_v3",
        "locked_on": "2026-05-29",
        "purpose": "Update the final failure-boundary table after n=10 scaling.",
        "experiments": [
            {
                "key": "failure_boundary_table_v3",
                "order": 1,
                "script": PY_BOUNDARY,
                "claim_tested": (
                    "Final paper framing should expose all robust and weak "
                    "regimes after n=10 scaling."
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
                        "exhaustive_n10_v8=experiments/out/control_predictive_stress_evidence_table_v8_exhaustive_n10/metrics.json",
                        "permutation_n10_v2=experiments/out/control_predictive_permutation_null_table_v2_n10/metrics.json",
                    ],
                    "out": "experiments/out/control_failure_boundary_table_v3",
                    "seed": 4225,
                },
            }
        ],
    }


def v92_audit_v19() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v85_ffhq_estimator_source_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v86_estimator_budget_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v87_wide_candidate_universe_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v88_target_magnitude_seed_scaling_predictive_validity.json",
        "experiments/protocols/control_v89_seed_scaled_exhaustive_defense_evidence_table.json",
        "experiments/protocols/control_v90_seed_scaled_predictive_permutation_null_table.json",
        "experiments/protocols/control_v91_failure_boundary_table_v3.json",
    ]
    return {
        "version": "control_v92_main_grade_audit_v19",
        "locked_on": "2026-05-29",
        "purpose": "Audit all locked protocols through n=10 exhaustive scaling.",
        "experiments": [
            {
                "key": "main_grade_audit_v19",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through n=10 exhaustive scaling "
                    "have traceable metrics outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v19",
                    "seed": 3728,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v85_ffhq_estimator_source_seed_scaling_predictive_validity.json",
        v85_ffhq_estimator_source_n10(),
    )
    write_protocol(
        "control_v86_estimator_budget_seed_scaling_predictive_validity.json",
        v86_estimator_budget_n10(),
    )
    write_protocol(
        "control_v87_wide_candidate_universe_seed_scaling_predictive_validity.json",
        v87_wide_universe_n10(),
    )
    write_protocol(
        "control_v88_target_magnitude_seed_scaling_predictive_validity.json",
        v88_target_magnitude_n10(),
    )
    write_protocol(
        "control_v89_seed_scaled_exhaustive_defense_evidence_table.json",
        v89_evidence_n10(),
    )
    write_protocol(
        "control_v90_seed_scaled_predictive_permutation_null_table.json",
        v90_permutation_n10(),
    )
    write_protocol("control_v91_failure_boundary_table_v3.json", v91_boundary_v3())
    write_protocol("control_v92_main_grade_audit_v19.json", v92_audit_v19())


if __name__ == "__main__":
    main()
