"""Generate locked protocols for final diagnostic-utility analyses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_DIAGNOSTIC = "experiments/control/run_predictive_diagnostic_utility_table.py"
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
    "experiments/protocols/control_v85_ffhq_estimator_source_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v86_estimator_budget_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v87_wide_candidate_universe_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v88_target_magnitude_seed_scaling_predictive_validity.json",
    "experiments/protocols/control_v89_seed_scaled_exhaustive_defense_evidence_table.json",
    "experiments/protocols/control_v90_seed_scaled_predictive_permutation_null_table.json",
    "experiments/protocols/control_v91_failure_boundary_table_v3.json",
    "experiments/protocols/control_v92_main_grade_audit_v19.json",
]


def write_protocol(name: str, payload: dict[str, Any]) -> None:
    path = PROTOCOLS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def diagnostic_summaries() -> list[str]:
    return [
        "random_n10_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/church/metrics.json",
        "random_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/bedroom/metrics.json",
        "random_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe_n10/ffhq/metrics.json",
        "dino_n10_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/church/metrics.json",
        "dino_n10_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/bedroom/metrics.json",
        "dino_n10_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation_n10/ffhq/metrics.json",
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


def v93_diagnostic() -> dict[str, Any]:
    return {
        "version": "control_v93_predictive_diagnostic_utility_table",
        "locked_on": "2026-05-30",
        "purpose": (
            "Evaluate whether rho is useful as a pre-edit diagnostic score for "
            "high-damage candidates after the final n=10 defense set."
        ),
        "experiments": [
            {
                "key": "predictive_diagnostic_utility_table_v1",
                "order": 1,
                "script": PY_DIAGNOSTIC,
                "claim_tested": (
                    "rho should separate high-damage edit candidates from "
                    "lower-damage candidates within attributes."
                ),
                "args": {
                    "summaries": diagnostic_summaries(),
                    "out": "experiments/out/control_predictive_diagnostic_utility_table_v1",
                    "high-quantile": 0.75,
                    "rho-auroc-threshold": 0.55,
                    "seed": 4226,
                },
            }
        ],
    }


def v94_boundary_v4() -> dict[str, Any]:
    return {
        "version": "control_v94_failure_boundary_table_v4",
        "locked_on": "2026-05-30",
        "purpose": "Update failure-boundary reporting with diagnostic utility.",
        "experiments": [
            {
                "key": "failure_boundary_table_v4",
                "order": 1,
                "script": PY_BOUNDARY,
                "claim_tested": (
                    "Final framing should expose both stress-test and "
                    "diagnostic-utility boundaries."
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
                        "diagnostic_v1=experiments/out/control_predictive_diagnostic_utility_table_v1/metrics.json",
                    ],
                    "out": "experiments/out/control_failure_boundary_table_v4",
                    "seed": 4227,
                },
            }
        ],
    }


def v95_audit_v20() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v93_predictive_diagnostic_utility_table.json",
        "experiments/protocols/control_v94_failure_boundary_table_v4.json",
    ]
    return {
        "version": "control_v95_main_grade_audit_v20",
        "locked_on": "2026-05-30",
        "purpose": "Audit all locked protocols through diagnostic utility.",
        "experiments": [
            {
                "key": "main_grade_audit_v20",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through diagnostic utility have "
                    "traceable metrics outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v20",
                    "seed": 3729,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v93_predictive_diagnostic_utility_table.json",
        v93_diagnostic(),
    )
    write_protocol("control_v94_failure_boundary_table_v4.json", v94_boundary_v4())
    write_protocol("control_v95_main_grade_audit_v20.json", v95_audit_v20())


if __name__ == "__main__":
    main()
