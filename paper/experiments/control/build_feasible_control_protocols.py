"""Generate locked protocols for semantic-feasible minimum-risk control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_CONTROLLER = "experiments/control/run_cross_domain_risk_aware_controller.py"
PY_REPEAT_SUMMARY = "experiments/control/run_risk_aware_repeat_summary.py"
PY_NEGATIVE_SUMMARY = "experiments/control/run_risk_signal_negative_summary.py"
PY_READINESS = "experiments/control/run_feasible_control_readiness.py"
PY_EVIDENCE = "experiments/control/run_feasible_control_evidence_table.py"
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
]

BEDROOM_ATTRS = [
    "indoor_lighting",
    "wood",
    "view",
    "carpet",
    "cluttered_space",
    "glossy",
    "dirt",
    "scary",
]

CHURCH_ATTRS = ["clouds", "sunny", "vegetation"]


def write_protocol(name: str, payload: dict[str, Any]) -> None:
    path = PROTOCOLS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def attrs_for(domain: str) -> list[str]:
    if domain == "church":
        return CHURCH_ATTRS
    if domain == "bedroom":
        return BEDROOM_ATTRS
    raise ValueError(domain)


def controller_args(domain: str, seed: int, mode: str, out: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "attrs": attrs_for(domain),
        "methods": ["ganspace", "sefa"],
        "candidate-k": 6,
        "ganspace-samples": 2048,
        "k-select": 4,
        "min-gain-quantile": 0.5,
        "risk-power": 1.0,
        "selection-rule": "gain_feasible_low_risk",
        "risk-selection-mode": mode,
        "target-source": "universe",
        "true-lpips": True,
        "lpips-net": "alex",
        "lpips-size": 256,
        "n-risk": 8,
        "n-probe": 16,
        "probe-alpha": 1.0,
        "n-calib": 16,
        "n-test": 128,
        "alpha-steps": 7,
        "max-alpha": 6,
        "target-quantile": 0.25,
        "batch": 4,
        "out": out,
        "seed": seed,
    }


def v38_feasible_controller() -> dict[str, Any]:
    seeds = list(range(2047, 2052))
    experiments: list[dict[str, Any]] = []
    order = 1
    for domain in ["church", "bedroom"]:
        for mode in ["actual", "shuffled", "inverted"]:
            for seed in seeds:
                out = (
                    "experiments/out/control_feasible_risk_controller/"
                    f"{domain}/{mode}_seed_{seed}"
                )
                experiments.append({
                    "key": f"{domain}_feasible_low_risk_{mode}_seed_{seed}",
                    "order": order,
                    "script": PY_CONTROLLER,
                    "claim_tested": (
                        "Semantic-feasible minimum-risk controller: after "
                        "a fixed probe-gain feasibility filter, actual "
                        "curvature/risk should choose lower-damage edits."
                        if mode == "actual" else
                        "Negative control for semantic-feasible minimum-risk "
                        f"controller using {mode} risk."
                    ),
                    "args": controller_args(domain, seed, mode, out),
                })
                order += 1
    for domain in ["church", "bedroom"]:
        actual = [
            "experiments/out/control_feasible_risk_controller/"
            f"{domain}/actual_seed_{seed}/metrics.json"
            for seed in seeds
        ]
        shuffled = [
            "experiments/out/control_feasible_risk_controller/"
            f"{domain}/shuffled_seed_{seed}/metrics.json"
            for seed in seeds
        ]
        inverted = [
            "experiments/out/control_feasible_risk_controller/"
            f"{domain}/inverted_seed_{seed}/metrics.json"
            for seed in seeds
        ]
        experiments.append({
            "key": f"{domain}_feasible_low_risk_actual_summary",
            "order": order,
            "script": PY_REPEAT_SUMMARY,
            "claim_tested": (
                f"Aggregate {domain} semantic-feasible minimum-risk "
                "controller runs."
            ),
            "args": {
                "inputs": actual,
                "out": (
                    "experiments/out/control_feasible_risk_controller/"
                    f"{domain}/actual_summary"
                ),
                "seed": 4140 + order,
            },
        })
        order += 1
        experiments.append({
            "key": f"{domain}_feasible_low_risk_negative_summary",
            "order": order,
            "script": PY_NEGATIVE_SUMMARY,
            "claim_tested": (
                f"Aggregate actual-vs-shuffled/inverted controls for the "
                f"{domain} semantic-feasible minimum-risk controller."
            ),
            "args": {
                "actual-inputs": actual,
                "shuffled-inputs": shuffled,
                "inverted-inputs": inverted,
                "out": (
                    "experiments/out/control_feasible_risk_controller/"
                    f"{domain}/negative_summary"
                ),
                "seed": 4150 + order,
            },
        })
        order += 1
    return {
        "version": "control_v38_semantic_feasible_minrisk_controller",
        "locked_on": "2026-05-27",
        "purpose": (
            "Test risk as a control variable under a predeclared semantic "
            "feasibility constraint, rather than as an unconstrained "
            "gain/risk ratio."
        ),
        "experiments": experiments,
    }


def v39_readiness() -> dict[str, Any]:
    return {
        "version": "control_v39_feasible_control_readiness",
        "locked_on": "2026-05-27",
        "purpose": (
            "Apply predeclared main-claim rules to predictive validity plus "
            "semantic-feasible minimum-risk controller outputs."
        ),
        "experiments": [
            {
                "key": "feasible_control_readiness_v1",
                "order": 1,
                "script": PY_READINESS,
                "claim_tested": (
                    "Decide whether the narrowed feasible-control claim is "
                    "supported without requiring domination of a risk-only "
                    "lower-bound baseline."
                ),
                "args": {
                    "church-predictive": (
                        "experiments/out/control_cross_domain_risk_predictive/"
                        "church/metrics.json"
                    ),
                    "bedroom-predictive": (
                        "experiments/out/control_cross_domain_risk_predictive/"
                        "bedroom/metrics.json"
                    ),
                    "church-actual": (
                        "experiments/out/control_feasible_risk_controller/"
                        "church/actual_summary/metrics.json"
                    ),
                    "church-negative": (
                        "experiments/out/control_feasible_risk_controller/"
                        "church/negative_summary/metrics.json"
                    ),
                    "bedroom-actual": (
                        "experiments/out/control_feasible_risk_controller/"
                        "bedroom/actual_summary/metrics.json"
                    ),
                    "bedroom-negative": (
                        "experiments/out/control_feasible_risk_controller/"
                        "bedroom/negative_summary/metrics.json"
                    ),
                    "out": "experiments/out/control_feasible_readiness_v1",
                    "seed": 4121,
                },
            }
        ],
    }


def v40_evidence() -> dict[str, Any]:
    return {
        "version": "control_v40_feasible_control_evidence_table",
        "locked_on": "2026-05-27",
        "purpose": (
            "Build reviewer-facing evidence rows for the narrowed feasible "
            "minimum-risk controller claim."
        ),
        "experiments": [
            {
                "key": "feasible_control_evidence_table_v1",
                "order": 1,
                "script": PY_EVIDENCE,
                "claim_tested": (
                    "Report bootstrap CIs and exact sign tests for predictive "
                    "validity, feasible controller baselines, negative "
                    "controls, and low-risk boundary rows."
                ),
                "args": {
                    "church-predictive": (
                        "experiments/out/control_cross_domain_risk_predictive/"
                        "church/metrics.json"
                    ),
                    "bedroom-predictive": (
                        "experiments/out/control_cross_domain_risk_predictive/"
                        "bedroom/metrics.json"
                    ),
                    "church-actual": (
                        "experiments/out/control_feasible_risk_controller/"
                        "church/actual_summary/metrics.json"
                    ),
                    "church-negative": (
                        "experiments/out/control_feasible_risk_controller/"
                        "church/negative_summary/metrics.json"
                    ),
                    "bedroom-actual": (
                        "experiments/out/control_feasible_risk_controller/"
                        "bedroom/actual_summary/metrics.json"
                    ),
                    "bedroom-negative": (
                        "experiments/out/control_feasible_risk_controller/"
                        "bedroom/negative_summary/metrics.json"
                    ),
                    "readiness": (
                        "experiments/out/control_feasible_readiness_v1/"
                        "metrics.json"
                    ),
                    "out": "experiments/out/control_feasible_evidence_table_v1",
                    "seed": 4131,
                    "n-boot": 10000,
                },
            }
        ],
    }


def v41_audit() -> dict[str, Any]:
    protocols = [
        *BASE_AUDIT_PROTOCOLS,
        "experiments/protocols/control_v38_semantic_feasible_minrisk_controller.json",
        "experiments/protocols/control_v39_feasible_control_readiness.json",
        "experiments/protocols/control_v40_feasible_control_evidence_table.json",
    ]
    return {
        "version": "control_v41_main_grade_audit_v9",
        "locked_on": "2026-05-27",
        "purpose": (
            "Final non-GPU reproducibility audit including the feasible "
            "minimum-risk controller campaign."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v9",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through feasible-control "
                    "readiness and evidence-table outputs have complete, "
                    "traceable metadata."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v9",
                    "seed": 3717,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v38_semantic_feasible_minrisk_controller.json",
        v38_feasible_controller(),
    )
    write_protocol(
        "control_v39_feasible_control_readiness.json",
        v39_readiness(),
    )
    write_protocol(
        "control_v40_feasible_control_evidence_table.json",
        v40_evidence(),
    )
    write_protocol(
        "control_v41_main_grade_audit_v9.json",
        v41_audit(),
    )


if __name__ == "__main__":
    main()
