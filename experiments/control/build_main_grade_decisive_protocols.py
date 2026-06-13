"""Generate final decisive protocols for the main-paper control claim."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

PY_PREDICTIVE = "experiments/control/run_cross_domain_risk_predictive.py"
PY_PREDICTIVE_SUMMARY = "experiments/control/run_risk_predictive_summary.py"
PY_CONTROLLER = "experiments/control/run_cross_domain_risk_aware_controller.py"
PY_REPEAT_SUMMARY = "experiments/control/run_risk_aware_repeat_summary.py"
PY_NEGATIVE_SUMMARY = "experiments/control/run_risk_signal_negative_summary.py"
PY_AUDIT = "experiments/control/audit_control_campaign.py"

BASE_MAIN_PROTOCOLS = [
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
]


def write_protocol(name: str, payload: dict[str, Any]) -> None:
    path = PROTOCOLS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def predictive_args(domain: str, seed: int, out: str) -> dict[str, Any]:
    attrs = (
        ["clouds", "sunny", "vegetation"]
        if domain == "church"
        else ["indoor_lighting", "wood", "view", "carpet"]
    )
    return {
        "domain": domain,
        "attrs": attrs,
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
        "out": out,
        "seed": seed,
    }


def controller_args(seed: int, out: str, mode: str) -> dict[str, Any]:
    return {
        "domain": "church",
        "attrs": ["clouds", "sunny", "vegetation"],
        "methods": ["ganspace", "sefa"],
        "candidate-k": 6,
        "ganspace-samples": 2048,
        "k-select": 4,
        "min-gain-quantile": 0.5,
        "risk-power": 1.0,
        "selection-rule": "gain_topk_low_risk",
        "gain-pool-multiplier": 2.0,
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


def v31_predictive() -> dict[str, Any]:
    seeds = list(range(2037, 2042))
    experiments: list[dict[str, Any]] = []
    order = 1
    for domain in ["church", "bedroom"]:
        for seed in seeds:
            out = f"experiments/out/control_cross_domain_risk_predictive/{domain}_seed_{seed}"
            experiments.append({
                "key": f"{domain}_risk_predictive_seed_{seed}",
                "order": order,
                "script": PY_PREDICTIVE,
                "claim_tested": (
                    "Predictive validity: curvature/risk should predict "
                    "identity/perceptual damage after controlling semantic "
                    f"gain in the {domain} structured candidate universe."
                ),
                "args": predictive_args(domain, seed, out),
            })
            order += 1
    for domain in ["church", "bedroom"]:
        inputs = [
            f"experiments/out/control_cross_domain_risk_predictive/{domain}_seed_{seed}/metrics.json"
            for seed in seeds
        ]
        experiments.append({
            "key": f"{domain}_risk_predictive_summary",
            "order": order,
            "script": PY_PREDICTIVE_SUMMARY,
            "claim_tested": f"Aggregate {domain} risk predictive-validity runs.",
            "args": {
                "inputs": inputs,
                "out": f"experiments/out/control_cross_domain_risk_predictive/{domain}",
                "seed": 4081 + order,
            },
        })
        order += 1
    return {
        "version": "control_v31_cross_domain_risk_predictive_validity",
        "locked_on": "2026-05-27",
        "purpose": (
            "Directly test whether curvature/risk predicts edit damage after "
            "semantic gain is controlled, addressing gain-only ambiguity."
        ),
        "experiments": experiments,
    }


def v32_tiebreak() -> dict[str, Any]:
    seeds = list(range(2042, 2047))
    experiments: list[dict[str, Any]] = []
    order = 1
    for mode in ["actual", "shuffled", "inverted"]:
        for seed in seeds:
            out = f"experiments/out/control_church_gain_first_risk_tiebreak/{mode}_seed_{seed}"
            experiments.append({
                "key": f"church_gain_first_risk_tiebreak_{mode}_seed_{seed}",
                "order": order,
                "script": PY_CONTROLLER,
                "claim_tested": (
                    "Held-out gain-first/risk-tiebreak controller: risk should "
                    "act as a control signal inside a high-gain candidate pool."
                    if mode == "actual" else
                    f"Held-out gain-first/risk-tiebreak negative control using {mode} risk."
                ),
                "args": controller_args(seed, out, mode),
            })
            order += 1
    actual = [
        f"experiments/out/control_church_gain_first_risk_tiebreak/actual_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    shuffled = [
        f"experiments/out/control_church_gain_first_risk_tiebreak/shuffled_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    inverted = [
        f"experiments/out/control_church_gain_first_risk_tiebreak/inverted_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    experiments.append({
        "key": "church_gain_first_risk_tiebreak_actual_summary",
        "order": order,
        "script": PY_REPEAT_SUMMARY,
        "claim_tested": "Aggregate held-out gain-first/risk-tiebreak controller runs.",
        "args": {
            "inputs": actual,
            "out": "experiments/out/control_church_gain_first_risk_tiebreak/actual_summary",
            "seed": 4091,
        },
    })
    order += 1
    experiments.append({
        "key": "church_gain_first_risk_tiebreak_negative_summary",
        "order": order,
        "script": PY_NEGATIVE_SUMMARY,
        "claim_tested": (
            "Aggregate actual-vs-shuffled/inverted risk for the held-out "
            "gain-first/risk-tiebreak controller."
        ),
        "args": {
            "actual-inputs": actual,
            "shuffled-inputs": shuffled,
            "inverted-inputs": inverted,
            "out": "experiments/out/control_church_gain_first_risk_tiebreak/negative_summary",
            "seed": 4093,
        },
    })
    return {
        "version": "control_v32_church_gain_first_risk_tiebreak_controller",
        "locked_on": "2026-05-27",
        "purpose": (
            "Held-out controller test that uses risk as a tiebreaker inside a "
            "high-gain candidate pool, instead of relying on the original "
            "gain/risk ratio."
        ),
        "experiments": experiments,
    }


def v33_audit() -> dict[str, Any]:
    protocols = deepcopy(BASE_MAIN_PROTOCOLS)
    protocols.extend([
        "experiments/protocols/control_v31_cross_domain_risk_predictive_validity.json",
        "experiments/protocols/control_v32_church_gain_first_risk_tiebreak_controller.json",
    ])
    return {
        "version": "control_v33_main_grade_audit_v6",
        "locked_on": "2026-05-27",
        "purpose": (
            "Final non-GPU reproducibility audit after predictive-validity and "
            "gain-first/risk-tiebreak controller protocols."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v6",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through the decisive main "
                    "claim follow-ups have complete outputs and traceable metadata."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v6",
                    "seed": 3711,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v31_cross_domain_risk_predictive_validity.json",
        v31_predictive(),
    )
    write_protocol(
        "control_v32_church_gain_first_risk_tiebreak_controller.json",
        v32_tiebreak(),
    )
    write_protocol("control_v33_main_grade_audit_v6.json", v33_audit())


if __name__ == "__main__":
    main()
