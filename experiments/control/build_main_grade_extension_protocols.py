"""Generate locked protocols for main-grade extension stress tests."""
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
                    "The rho signal should remain directionally stable under "
                    f"{label} stress in {domain}; failures define boundary "
                    "conditions rather than being filtered out."
                ),
                "args": predictive_args(domain, seed, out, overrides),
            })
            order += 1
    for domain in domains:
        inputs = [f"{out_root}/{domain}_seed_{seed}/metrics.json"
                  for seed in seeds]
        experiments.append({
            "key": f"{domain}_{version}_{label}_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": (
                f"Aggregate {domain} predictive-validity runs for {label}."
            ),
            "args": {
                "inputs": inputs,
                "out": f"{out_root}/{domain}",
                "seed": 4600 + order,
            },
        })
        order += 1


def v61_ffhq_prompt_ensemble() -> dict[str, Any]:
    version = "control_v61_ffhq_prompt_ensemble_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="ffhq_prompt_ensemble",
        out_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble",
        domains=["ffhq"],
        seeds=list(range(2107, 2112)),
        overrides={},
    )
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": (
            "Extend prompt-ensemble predictive validity to FFHQ so the claim "
            "is not only a church/bedroom cross-domain result."
        ),
        "experiments": experiments,
    }


def v62_ffhq_followup() -> dict[str, Any]:
    version = "control_v62_ffhq_prompt_ensemble_followup_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="ffhq_high_ntest128",
        out_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_high_ntest128",
        domains=["ffhq"],
        seeds=list(range(2112, 2117)),
        overrides={"n-test": 128},
    )
    add_predictive_block(
        experiments,
        version=version,
        label="ffhq_strict_gain_match",
        out_root="experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_strict_gain_match",
        domains=["ffhq"],
        seeds=list(range(2117, 2122)),
        overrides={"gain-match-rel": 0.10},
    )
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": (
            "Stress-test the FFHQ extension under higher test-set size and "
            "stricter semantic-gain matching."
        ),
        "experiments": experiments,
    }


def v63_random_universe() -> dict[str, Any]:
    version = "control_v63_random_universe_prompt_ensemble_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="random_universe",
        out_root="experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe",
        domains=["church", "bedroom", "ffhq"],
        seeds=list(range(2122, 2127)),
        overrides={"methods": ["ganspace", "sefa", "random"]},
    )
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": (
            "Test whether prompt-ensemble predictive validity survives adding "
            "unstructured random directions to the candidate universe."
        ),
        "experiments": experiments,
    }


def v64_dino_preservation() -> dict[str, Any]:
    version = "control_v64_dino_preservation_predictive_validity"
    experiments: list[dict[str, Any]] = []
    add_predictive_block(
        experiments,
        version=version,
        label="dino_preservation",
        out_root="experiments/out/control_cross_domain_risk_predictive_dino_preservation",
        domains=["church", "bedroom", "ffhq"],
        seeds=list(range(2127, 2132)),
        overrides={
            "dino-preservation": True,
            "dino-local-files-only": True,
        },
    )
    return {
        "version": version,
        "locked_on": "2026-05-28",
        "purpose": (
            "Test whether rho predicts damage in a non-language DINOv2 image "
            "feature space, not only CLIP-image consistency and LPIPS."
        ),
        "experiments": experiments,
    }


def v65_evidence() -> dict[str, Any]:
    summaries = [
        "ffhq_ensemble=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble/ffhq/metrics.json",
        "ffhq_high_ntest=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_high_ntest128/ffhq/metrics.json",
        "ffhq_strict_gain=experiments/out/control_cross_domain_risk_predictive_ffhq_prompt_ensemble_strict_gain_match/ffhq/metrics.json",
        "random_universe_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe/church/metrics.json",
        "random_universe_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe/bedroom/metrics.json",
        "random_universe_ffhq=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble_random_universe/ffhq/metrics.json",
        "dino_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation/church/metrics.json",
        "dino_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation/bedroom/metrics.json",
        "dino_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation/ffhq/metrics.json",
    ]
    return {
        "version": "control_v65_main_grade_extension_evidence_table",
        "locked_on": "2026-05-28",
        "purpose": (
            "Build a stress evidence table for FFHQ domain extension, random "
            "candidate-universe expansion, and DINO-preservation runs."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v5",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "The predictive signal should survive the main-grade "
                    "extension stresses without hiding boundary failures."
                ),
                "args": {
                    "summaries": summaries,
                    "out": "experiments/out/control_predictive_stress_evidence_table_v5",
                    "seed": 4213,
                },
            }
        ],
    }


def v66_dino_evidence() -> dict[str, Any]:
    return {
        "version": "control_v66_dino_preservation_evidence_table",
        "locked_on": "2026-05-28",
        "purpose": (
            "Build a DINO-specific evidence table for non-language "
            "preservation predictive validity."
        ),
        "experiments": [
            {
                "key": "predictive_dino_evidence_table_v1",
                "order": 1,
                "script": PY_DINO_EVIDENCE,
                "claim_tested": (
                    "Low-rho candidates should preserve DINOv2 image features "
                    "better than high-rho candidates at matched CLIP gain."
                ),
                "args": {
                    "summaries": [
                        "dino_church=experiments/out/control_cross_domain_risk_predictive_dino_preservation/church/metrics.json",
                        "dino_bedroom=experiments/out/control_cross_domain_risk_predictive_dino_preservation/bedroom/metrics.json",
                        "dino_ffhq=experiments/out/control_cross_domain_risk_predictive_dino_preservation/ffhq/metrics.json",
                    ],
                    "out": "experiments/out/control_predictive_dino_evidence_table_v1",
                    "seed": 4214,
                },
            }
        ],
    }


def v67_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v61_ffhq_prompt_ensemble_predictive_validity.json",
        "experiments/protocols/control_v62_ffhq_prompt_ensemble_followup_predictive_validity.json",
        "experiments/protocols/control_v63_random_universe_prompt_ensemble_predictive_validity.json",
        "experiments/protocols/control_v64_dino_preservation_predictive_validity.json",
        "experiments/protocols/control_v65_main_grade_extension_evidence_table.json",
        "experiments/protocols/control_v66_dino_preservation_evidence_table.json",
    ]
    return {
        "version": "control_v67_main_grade_audit_v15",
        "locked_on": "2026-05-28",
        "purpose": (
            "Final non-GPU reproducibility audit including FFHQ, random "
            "candidate-universe, and DINO-preservation extension stresses."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v15",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked protocols through the main-grade extension "
                    "stress tests have complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v15",
                    "seed": 3724,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v61_ffhq_prompt_ensemble_predictive_validity.json",
        v61_ffhq_prompt_ensemble(),
    )
    write_protocol(
        "control_v62_ffhq_prompt_ensemble_followup_predictive_validity.json",
        v62_ffhq_followup(),
    )
    write_protocol(
        "control_v63_random_universe_prompt_ensemble_predictive_validity.json",
        v63_random_universe(),
    )
    write_protocol(
        "control_v64_dino_preservation_predictive_validity.json",
        v64_dino_preservation(),
    )
    write_protocol(
        "control_v65_main_grade_extension_evidence_table.json",
        v65_evidence(),
    )
    write_protocol(
        "control_v66_dino_preservation_evidence_table.json",
        v66_dino_evidence(),
    )
    write_protocol("control_v67_main_grade_audit_v15.json", v67_audit())


if __name__ == "__main__":
    main()
