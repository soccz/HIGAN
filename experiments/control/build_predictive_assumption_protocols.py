"""Generate locked protocols for predictive-validity assumption stress tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_PREDICTIVE = "experiments/control/run_cross_domain_risk_predictive.py"
PY_PREDICTIVE_SUMMARY = "experiments/control/run_risk_predictive_summary.py"
PY_READINESS = "experiments/control/run_predictive_assumption_readiness.py"
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
                    *, risk_estimator: str = "jvp",
                    prompt_style: str = "default",
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
        "risk-estimator": risk_estimator,
        "prompt-style": prompt_style,
        "out": out,
        "seed": seed,
    }
    if risk_estimator == "fd":
        args["fd-eps"] = 0.25
    if overrides:
        args.update(overrides)
    return args


def predictive_protocol(version: str, purpose: str, out_root: str,
                        seeds: list[int], *, risk_estimator: str,
                        prompt_style: str,
                        overrides: dict[str, Any] | None = None) -> dict[str, Any]:
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
                    "Predictive-validity assumption stress test: curvature/"
                    "risk should predict identity/perceptual damage after "
                    f"semantic gain is controlled in {domain}, with "
                    f"risk_estimator={risk_estimator} and "
                    f"prompt_style={prompt_style}."
                ),
                "args": predictive_args(
                    domain,
                    seed,
                    out,
                    risk_estimator=risk_estimator,
                    prompt_style=prompt_style,
                    overrides=overrides,
                ),
            })
            order += 1
    for domain in ["church", "bedroom"]:
        inputs = [f"{out_root}/{domain}_seed_{seed}/metrics.json" for seed in seeds]
        experiments.append({
            "key": f"{domain}_{version}_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": (
                f"Aggregate {domain} predictive-validity assumption stress "
                f"runs for {version}."
            ),
            "args": {
                "inputs": inputs,
                "out": f"{out_root}/{domain}",
                "seed": 4200 + order,
            },
        })
        order += 1
    return {
        "version": version,
        "locked_on": "2026-05-27",
        "purpose": purpose,
        "experiments": experiments,
    }


def v42_fd() -> dict[str, Any]:
    return predictive_protocol(
        "control_v42_fd_risk_predictive_validity",
        (
            "Test whether predictive validity survives replacing the JVP "
            "curvature/risk estimator with an independent finite-difference "
            "second-order estimator."
        ),
        "experiments/out/control_cross_domain_risk_predictive_fd",
        list(range(2052, 2057)),
        risk_estimator="fd",
        prompt_style="default",
    )


def v43_prompt() -> dict[str, Any]:
    return predictive_protocol(
        "control_v43_prompt_template_predictive_validity",
        (
            "Test whether predictive validity survives a held-out prompt "
            "template for semantic gain matching."
        ),
        "experiments/out/control_cross_domain_risk_predictive_prompt_photo",
        list(range(2057, 2062)),
        risk_estimator="jvp",
        prompt_style="photo",
    )


def v46_caption_prompt() -> dict[str, Any]:
    return predictive_protocol(
        "control_v46_caption_prompt_predictive_validity",
        (
            "Test whether predictive validity survives a second held-out "
            "caption-like prompt template, after the photo prompt exposed "
            "bedroom sensitivity."
        ),
        "experiments/out/control_cross_domain_risk_predictive_prompt_caption",
        list(range(2062, 2067)),
        risk_estimator="jvp",
        prompt_style="caption",
    )


def v47_high_ntest() -> dict[str, Any]:
    return predictive_protocol(
        "control_v47_high_ntest_predictive_validity",
        (
            "Test whether predictive validity survives doubling the held-out "
            "evaluation set size from n_test=64 to n_test=128."
        ),
        "experiments/out/control_cross_domain_risk_predictive_high_ntest128",
        list(range(2067, 2072)),
        risk_estimator="jvp",
        prompt_style="default",
        overrides={"n-test": 128},
    )


def v48_strict_gain_match() -> dict[str, Any]:
    return predictive_protocol(
        "control_v48_strict_gain_match_predictive_validity",
        (
            "Test whether predictive validity survives stricter semantic "
            "magnitude matching by reducing gain_match_rel from 0.25 to 0.10."
        ),
        "experiments/out/control_cross_domain_risk_predictive_strict_gain_match",
        list(range(2072, 2077)),
        risk_estimator="jvp",
        prompt_style="default",
        overrides={"gain-match-rel": 0.10},
    )


def v44_readiness() -> dict[str, Any]:
    return {
        "version": "control_v44_predictive_assumption_readiness",
        "locked_on": "2026-05-27",
        "purpose": (
            "Apply predeclared decision rules to FD-estimator and "
            "prompt-template predictive-validity stress tests."
        ),
        "experiments": [
            {
                "key": "predictive_assumption_readiness_v1",
                "order": 1,
                "script": PY_READINESS,
                "claim_tested": (
                    "Decide whether the predictive-validity claim survives "
                    "the two main measurement-assumption stress tests."
                ),
                "args": {
                    "fd-church": (
                        "experiments/out/control_cross_domain_risk_predictive_fd/"
                        "church/metrics.json"
                    ),
                    "fd-bedroom": (
                        "experiments/out/control_cross_domain_risk_predictive_fd/"
                        "bedroom/metrics.json"
                    ),
                    "prompt-church": (
                        "experiments/out/control_cross_domain_risk_predictive_prompt_photo/"
                        "church/metrics.json"
                    ),
                    "prompt-bedroom": (
                        "experiments/out/control_cross_domain_risk_predictive_prompt_photo/"
                        "bedroom/metrics.json"
                    ),
                    "out": "experiments/out/control_predictive_assumption_readiness_v1",
                    "seed": 4191,
                },
            }
        ],
    }


def v49_stress_evidence() -> dict[str, Any]:
    summaries = [
        "baseline_church=experiments/out/control_cross_domain_risk_predictive/church/metrics.json",
        "baseline_bedroom=experiments/out/control_cross_domain_risk_predictive/bedroom/metrics.json",
        "fd_church=experiments/out/control_cross_domain_risk_predictive_fd/church/metrics.json",
        "fd_bedroom=experiments/out/control_cross_domain_risk_predictive_fd/bedroom/metrics.json",
        "photo_church=experiments/out/control_cross_domain_risk_predictive_prompt_photo/church/metrics.json",
        "photo_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_photo/bedroom/metrics.json",
        "caption_church=experiments/out/control_cross_domain_risk_predictive_prompt_caption/church/metrics.json",
        "caption_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_caption/bedroom/metrics.json",
        "high_ntest_church=experiments/out/control_cross_domain_risk_predictive_high_ntest128/church/metrics.json",
        "high_ntest_bedroom=experiments/out/control_cross_domain_risk_predictive_high_ntest128/bedroom/metrics.json",
        "strict_gain_church=experiments/out/control_cross_domain_risk_predictive_strict_gain_match/church/metrics.json",
        "strict_gain_bedroom=experiments/out/control_cross_domain_risk_predictive_strict_gain_match/bedroom/metrics.json",
    ]
    return {
        "version": "control_v49_predictive_stress_evidence_table",
        "locked_on": "2026-05-27",
        "purpose": (
            "Build a generic predeclared evidence table for baseline and "
            "assumption-stress predictive-validity summaries."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v1",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "Curvature/risk predictive validity should have stable "
                    "directional evidence across estimator, prompt, evaluation "
                    "budget, and semantic matching stress tests."
                ),
                "args": {
                    "summaries": summaries,
                    "out": "experiments/out/control_predictive_stress_evidence_table_v1",
                    "seed": 4209,
                },
            }
        ],
    }


def v51_prompt_ensemble() -> dict[str, Any]:
    return predictive_protocol(
        "control_v51_prompt_ensemble_predictive_validity",
        (
            "Test whether predictive validity survives replacing a single "
            "semantic prompt template with an averaged default/photo/caption "
            "prompt ensemble."
        ),
        "experiments/out/control_cross_domain_risk_predictive_prompt_ensemble",
        list(range(2077, 2082)),
        risk_estimator="jvp",
        prompt_style="ensemble",
    )


def v52_stress_evidence_with_ensemble() -> dict[str, Any]:
    summaries = [
        "baseline_church=experiments/out/control_cross_domain_risk_predictive/church/metrics.json",
        "baseline_bedroom=experiments/out/control_cross_domain_risk_predictive/bedroom/metrics.json",
        "fd_church=experiments/out/control_cross_domain_risk_predictive_fd/church/metrics.json",
        "fd_bedroom=experiments/out/control_cross_domain_risk_predictive_fd/bedroom/metrics.json",
        "photo_church=experiments/out/control_cross_domain_risk_predictive_prompt_photo/church/metrics.json",
        "photo_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_photo/bedroom/metrics.json",
        "caption_church=experiments/out/control_cross_domain_risk_predictive_prompt_caption/church/metrics.json",
        "caption_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_caption/bedroom/metrics.json",
        "ensemble_church=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble/church/metrics.json",
        "ensemble_bedroom=experiments/out/control_cross_domain_risk_predictive_prompt_ensemble/bedroom/metrics.json",
        "high_ntest_church=experiments/out/control_cross_domain_risk_predictive_high_ntest128/church/metrics.json",
        "high_ntest_bedroom=experiments/out/control_cross_domain_risk_predictive_high_ntest128/bedroom/metrics.json",
        "strict_gain_church=experiments/out/control_cross_domain_risk_predictive_strict_gain_match/church/metrics.json",
        "strict_gain_bedroom=experiments/out/control_cross_domain_risk_predictive_strict_gain_match/bedroom/metrics.json",
    ]
    return {
        "version": "control_v52_predictive_stress_evidence_table_with_ensemble",
        "locked_on": "2026-05-28",
        "purpose": (
            "Build the extended predictive stress table including a "
            "prompt-ensemble semantic direction diagnostic."
        ),
        "experiments": [
            {
                "key": "predictive_stress_evidence_table_v2",
                "order": 1,
                "script": PY_STRESS_EVIDENCE,
                "claim_tested": (
                    "Prompt-ensemble semantic directions should clarify "
                    "whether prompt-template sensitivity is a measurement "
                    "artifact or a deeper boundary of the rho signal."
                ),
                "args": {
                    "summaries": summaries,
                    "out": "experiments/out/control_predictive_stress_evidence_table_v2",
                    "seed": 4210,
                },
            }
        ],
    }


def v45_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v42_fd_risk_predictive_validity.json",
        "experiments/protocols/control_v43_prompt_template_predictive_validity.json",
        "experiments/protocols/control_v44_predictive_assumption_readiness.json",
    ]
    return {
        "version": "control_v45_main_grade_audit_v10",
        "locked_on": "2026-05-27",
        "purpose": (
            "Final non-GPU reproducibility audit including predictive "
            "assumption-stress protocols."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v10",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through FD and prompt "
                    "assumption stress tests have complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v10",
                    "seed": 3719,
                },
            }
        ],
    }


def v50_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v42_fd_risk_predictive_validity.json",
        "experiments/protocols/control_v43_prompt_template_predictive_validity.json",
        "experiments/protocols/control_v44_predictive_assumption_readiness.json",
        "experiments/protocols/control_v46_caption_prompt_predictive_validity.json",
        "experiments/protocols/control_v47_high_ntest_predictive_validity.json",
        "experiments/protocols/control_v48_strict_gain_match_predictive_validity.json",
        "experiments/protocols/control_v49_predictive_stress_evidence_table.json",
    ]
    return {
        "version": "control_v50_main_grade_audit_v11",
        "locked_on": "2026-05-27",
        "purpose": (
            "Final non-GPU reproducibility audit including extended predictive "
            "stress protocols."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v11",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through extended predictive "
                    "stress tests have complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v11",
                    "seed": 3720,
                },
            }
        ],
    }


def v53_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v42_fd_risk_predictive_validity.json",
        "experiments/protocols/control_v43_prompt_template_predictive_validity.json",
        "experiments/protocols/control_v44_predictive_assumption_readiness.json",
        "experiments/protocols/control_v46_caption_prompt_predictive_validity.json",
        "experiments/protocols/control_v47_high_ntest_predictive_validity.json",
        "experiments/protocols/control_v48_strict_gain_match_predictive_validity.json",
        "experiments/protocols/control_v49_predictive_stress_evidence_table.json",
        "experiments/protocols/control_v51_prompt_ensemble_predictive_validity.json",
        "experiments/protocols/control_v52_predictive_stress_evidence_table_with_ensemble.json",
    ]
    return {
        "version": "control_v53_main_grade_audit_v12",
        "locked_on": "2026-05-28",
        "purpose": (
            "Final non-GPU reproducibility audit including prompt-ensemble "
            "semantic-measurement diagnostics."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v12",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through prompt-ensemble "
                    "predictive stress tests have complete traceable outputs."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v12",
                    "seed": 3721,
                },
            }
        ],
    }


def main() -> None:
    write_protocol("control_v42_fd_risk_predictive_validity.json", v42_fd())
    write_protocol(
        "control_v43_prompt_template_predictive_validity.json",
        v43_prompt(),
    )
    write_protocol(
        "control_v44_predictive_assumption_readiness.json",
        v44_readiness(),
    )
    write_protocol("control_v45_main_grade_audit_v10.json", v45_audit())
    write_protocol(
        "control_v46_caption_prompt_predictive_validity.json",
        v46_caption_prompt(),
    )
    write_protocol(
        "control_v47_high_ntest_predictive_validity.json",
        v47_high_ntest(),
    )
    write_protocol(
        "control_v48_strict_gain_match_predictive_validity.json",
        v48_strict_gain_match(),
    )
    write_protocol(
        "control_v49_predictive_stress_evidence_table.json",
        v49_stress_evidence(),
    )
    write_protocol("control_v50_main_grade_audit_v11.json", v50_audit())
    write_protocol(
        "control_v51_prompt_ensemble_predictive_validity.json",
        v51_prompt_ensemble(),
    )
    write_protocol(
        "control_v52_predictive_stress_evidence_table_with_ensemble.json",
        v52_stress_evidence_with_ensemble(),
    )
    write_protocol("control_v53_main_grade_audit_v12.json", v53_audit())


if __name__ == "__main__":
    main()
