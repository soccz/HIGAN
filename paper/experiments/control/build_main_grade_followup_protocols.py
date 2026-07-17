"""Generate locked follow-up protocols for the main-grade control claim.

The generated protocols are intentionally finite and predeclared.  They test
the strongest remaining paper risks after the church structured-only diagnosis:
independent-seed replication, estimator stability, and decomposition-source
ablation.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[2]
PROTOCOLS = PAPER / "experiments" / "protocols"

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
]


def write_protocol(name: str, payload: dict[str, Any]) -> None:
    path = PROTOCOLS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def controller_args(seed: int, out: str, *,
                    mode: str = "actual",
                    methods: list[str] | None = None,
                    n_risk: int = 8,
                    n_probe: int = 16,
                    n_calib: int = 16,
                    n_test: int = 64,
                    candidate_k: int = 6,
                    k_select: int = 4) -> dict[str, Any]:
    return {
        "domain": "church",
        "attrs": ["clouds", "sunny", "vegetation"],
        "methods": methods or ["ganspace", "sefa"],
        "candidate-k": candidate_k,
        "ganspace-samples": 2048,
        "k-select": k_select,
        "min-gain-quantile": 0.5,
        "risk-power": 1.0,
        "risk-selection-mode": mode,
        "target-source": "universe",
        "true-lpips": True,
        "lpips-net": "alex",
        "lpips-size": 256,
        "n-risk": n_risk,
        "n-probe": n_probe,
        "probe-alpha": 1.0,
        "n-calib": n_calib,
        "n-test": n_test,
        "alpha-steps": 7,
        "max-alpha": 6,
        "target-quantile": 0.25,
        "batch": 4,
        "out": out,
        "seed": seed,
    }


def repeat_summary(key: str, order: int, inputs: list[str], out: str,
                   seed: int, claim: str) -> dict[str, Any]:
    return {
        "key": key,
        "order": order,
        "script": PY_REPEAT_SUMMARY,
        "claim_tested": claim,
        "args": {
            "inputs": inputs,
            "out": out,
            "seed": seed,
        },
    }


def negative_summary(key: str, order: int, actual: list[str],
                     shuffled: list[str], inverted: list[str],
                     out: str, seed: int, claim: str) -> dict[str, Any]:
    return {
        "key": key,
        "order": order,
        "script": PY_NEGATIVE_SUMMARY,
        "claim_tested": claim,
        "args": {
            "actual-inputs": actual,
            "shuffled-inputs": shuffled,
            "inverted-inputs": inverted,
            "out": out,
            "seed": seed,
        },
    }


def v27_confirmatory() -> dict[str, Any]:
    seeds = list(range(2032, 2037))
    experiments: list[dict[str, Any]] = []
    order = 1
    for mode in ["actual", "shuffled", "inverted"]:
        for seed in seeds:
            out = f"experiments/out/control_church_structured_confirmatory/{mode}_seed_{seed}"
            experiments.append({
                "key": f"church_structured_confirmatory_{mode}_seed_{seed}",
                "order": order,
                "script": PY_CONTROLLER,
                "claim_tested": (
                    "Independent-seed church structured-only replication: "
                    "the same predeclared controller should preserve identity "
                    "and reduce true LPIPS without retuning."
                    if mode == "actual" else
                    "Independent-seed structured-only risk-signal negative "
                    f"control using {mode} risk."
                ),
                "args": controller_args(seed, out, mode=mode, n_test=128),
            })
            order += 1

    actual = [
        f"experiments/out/control_church_structured_confirmatory/actual_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    shuffled = [
        f"experiments/out/control_church_structured_confirmatory/shuffled_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    inverted = [
        f"experiments/out/control_church_structured_confirmatory/inverted_seed_{seed}/metrics.json"
        for seed in seeds
    ]
    experiments.append(repeat_summary(
        "church_structured_confirmatory_actual_summary",
        order,
        actual,
        "experiments/out/control_church_structured_confirmatory/actual_summary",
        4041,
        "Aggregate independent-seed structured-only actual controller results.",
    ))
    order += 1
    experiments.append(negative_summary(
        "church_structured_confirmatory_negative_summary",
        order,
        actual,
        shuffled,
        inverted,
        "experiments/out/control_church_structured_confirmatory/negative_summary",
        4043,
        "Aggregate independent-seed actual-vs-shuffled/inverted structured-only risk controls.",
    ))
    return {
        "version": "control_v27_church_structured_confirmatory_replication",
        "locked_on": "2026-05-26",
        "purpose": (
            "Independent-seed, larger-test-set replication of the church "
            "structured-only control-signal result and its shuffled/inverted "
            "risk negative controls."
        ),
        "common": {
            "domain": "church",
            "attrs": ["clouds", "sunny", "vegetation"],
            "methods": ["ganspace", "sefa"],
            "target_source": "universe",
            "lpips_net": "alex",
            "n_test": 128,
        },
        "experiments": experiments,
    }


def v28_estimator_stability() -> dict[str, Any]:
    seeds = list(range(2027, 2032))
    experiments: list[dict[str, Any]] = []
    for order, seed in enumerate(seeds, start=1):
        out = f"experiments/out/control_church_structured_estimator_stability/seed_{seed}"
        experiments.append({
            "key": f"church_structured_estimator_stability_seed_{seed}",
            "order": order,
            "script": PY_CONTROLLER,
            "claim_tested": (
                "Church structured-only estimator stability: increasing "
                "risk/probe/calibration samples should not be required to "
                "change the predeclared controller rule."
            ),
            "args": controller_args(
                seed, out, n_risk=16, n_probe=32, n_calib=32, n_test=128
            ),
        })
    inputs = [
        f"experiments/out/control_church_structured_estimator_stability/seed_{seed}/metrics.json"
        for seed in seeds
    ]
    experiments.append(repeat_summary(
        "church_structured_estimator_stability_summary",
        len(experiments) + 1,
        inputs,
        "experiments/out/control_church_structured_estimator_stability/summary",
        4051,
        "Aggregate larger-estimator church structured-only controller results.",
    ))
    return {
        "version": "control_v28_church_structured_estimator_stability",
        "locked_on": "2026-05-26",
        "purpose": (
            "Test whether the structured-only church controller is stable when "
            "the curvature, probe-gain, and calibration estimates use larger "
            "sample budgets."
        ),
        "common": {
            "domain": "church",
            "attrs": ["clouds", "sunny", "vegetation"],
            "methods": ["ganspace", "sefa"],
            "target_source": "universe",
            "lpips_net": "alex",
            "n_risk": 16,
            "n_probe": 32,
            "n_calib": 32,
            "n_test": 128,
        },
        "experiments": experiments,
    }


def v29_source_ablation() -> dict[str, Any]:
    seeds = list(range(2027, 2032))
    experiments: list[dict[str, Any]] = []
    order = 1
    source_configs = [
        ("ganspace_only", ["ganspace"]),
        ("sefa_only", ["sefa"]),
    ]
    for label, methods in source_configs:
        for seed in seeds:
            out = f"experiments/out/control_church_structured_source_ablation/{label}_seed_{seed}"
            experiments.append({
                "key": f"church_structured_source_ablation_{label}_seed_{seed}",
                "order": order,
                "script": PY_CONTROLLER,
                "claim_tested": (
                    "Structured-source ablation: the control signal should be "
                    f"evaluated within a {label.replace('_', '-')} candidate "
                    "universe without changing the controller rule."
                ),
                "args": controller_args(seed, out, methods=methods, n_test=64),
            })
            order += 1
    for label, _ in source_configs:
        inputs = [
            f"experiments/out/control_church_structured_source_ablation/{label}_seed_{seed}/metrics.json"
            for seed in seeds
        ]
        experiments.append(repeat_summary(
            f"church_structured_source_ablation_{label}_summary",
            order,
            inputs,
            f"experiments/out/control_church_structured_source_ablation/{label}",
            4061 + order,
            f"Aggregate {label} structured-source ablation results.",
        ))
        order += 1
    return {
        "version": "control_v29_church_structured_source_ablation",
        "locked_on": "2026-05-26",
        "purpose": (
            "Test whether the structured-only control claim is tied to one "
            "decomposition family by evaluating GANSpace-only and SeFa-only "
            "candidate universes."
        ),
        "common": {
            "domain": "church",
            "attrs": ["clouds", "sunny", "vegetation"],
            "target_source": "universe",
            "lpips_net": "alex",
        },
        "experiments": experiments,
    }


def v30_audit() -> dict[str, Any]:
    protocols = deepcopy(BASE_MAIN_PROTOCOLS)
    protocols.extend([
        "experiments/protocols/control_v27_church_structured_confirmatory_replication.json",
        "experiments/protocols/control_v28_church_structured_estimator_stability.json",
        "experiments/protocols/control_v29_church_structured_source_ablation.json",
    ])
    return {
        "version": "control_v30_main_grade_audit_v5",
        "locked_on": "2026-05-26",
        "purpose": (
            "Final non-GPU reproducibility audit after the main-grade church "
            "structured-only follow-up protocols."
        ),
        "experiments": [
            {
                "key": "main_grade_audit_v5",
                "order": 1,
                "script": PY_AUDIT,
                "claim_tested": (
                    "All locked control protocols through the main-grade "
                    "structured-only follow-ups have complete outputs and "
                    "traceable protocol metadata."
                ),
                "args": {
                    "protocols": protocols,
                    "out": "experiments/out/control_campaign_audit_main_grade_v5",
                    "seed": 3709,
                },
            }
        ],
    }


def main() -> None:
    write_protocol(
        "control_v27_church_structured_confirmatory_replication.json",
        v27_confirmatory(),
    )
    write_protocol(
        "control_v28_church_structured_estimator_stability.json",
        v28_estimator_stability(),
    )
    write_protocol(
        "control_v29_church_structured_source_ablation.json",
        v29_source_ablation(),
    )
    write_protocol("control_v30_main_grade_audit_v5.json", v30_audit())


if __name__ == "__main__":
    main()
