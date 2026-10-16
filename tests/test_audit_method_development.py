import copy
import unittest
from pathlib import Path

from labelqa.audit_method_development import ExperimentSpec, audit_experiment


def report(delta=0.01, interval=(0.002, 0.02), selected_weight=0.1):
    baseline_dev = {"n": 2393, "all_gold@5": 0.6373}
    candidate_dev = {"n": 2393, "all_gold@5": 0.6373 + delta}
    baseline_validation = {"n": 559, "all_gold@5": 0.6136}
    candidate_validation = {"n": 559, "all_gold@5": 0.62}
    return {
        "status": "nested_grouped_dev_ablation_validation_descriptive_not_holdout",
        "dev_crossfit": {
            "metrics": {"base": baseline_dev, "candidate": candidate_dev},
            "comparison": {
                "candidate_minus_baseline": delta,
                "ci95_label_bootstrap": list(interval),
                "n_questions": 2393,
            },
        },
        "full_dev_selection": {"selected_weight": selected_weight},
        "validation": {
            "metrics": {"base": baseline_validation, "candidate": candidate_validation},
        },
    }


SPEC = ExperimentSpec(
    "synthetic", Path("synthetic.json"), "base", "candidate",
    ("dev_crossfit", "comparison"), True,
)


class MethodDevelopmentAuditTests(unittest.TestCase):
    def test_positive_dev_interval_is_only_eligible_for_future_confirmation(self):
        entry = audit_experiment(SPEC, report())
        self.assertEqual(entry["decision"], "eligible_for_future_confirmation")
        self.assertEqual(entry["decision_reasons"], [])

    def test_zero_interval_boundary_and_zero_final_weight_retain_canonical(self):
        entry = audit_experiment(
            SPEC, report(delta=0.0004, interval=(0.0, 0.0013), selected_weight=0.0))
        self.assertEqual(entry["decision"], "retain_canonical")
        self.assertIn("development_interval_not_strictly_positive", entry["decision_reasons"])
        self.assertIn("full_development_fit_reverts_to_canonical", entry["decision_reasons"])

    def test_inconsistent_reported_delta_fails_closed(self):
        value = report()
        value["dev_crossfit"]["comparison"]["candidate_minus_baseline"] = 0.02
        with self.assertRaises(ValueError):
            audit_experiment(SPEC, value)

    def test_changed_denominator_fails_closed(self):
        value = copy.deepcopy(report())
        value["validation"]["metrics"]["candidate"]["n"] = 558
        with self.assertRaises(ValueError):
            audit_experiment(SPEC, value)


if __name__ == "__main__":
    unittest.main()
