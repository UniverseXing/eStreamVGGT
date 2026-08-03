import csv
import math
import statistics
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENTARY = REPO_ROOT / "supplementary"
BOUNDED_METHODS = ("k4", "k6", "k8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SupplementaryPackageTests(unittest.TestCase):
    def test_package_contains_only_the_two_declared_categories(self):
        expected = {Path("README.md"), Path("CALCULATION_METHODS.md")}
        expected.update(
            path.relative_to(SUPPLEMENTARY)
            for path in (SUPPLEMENTARY / "tables").glob("table_s*.csv")
        )
        self.assertEqual(15, len(expected) - 2)
        actual = {
            path.relative_to(SUPPLEMENTARY)
            for path in SUPPLEMENTARY.rglob("*")
            if path.is_file()
        }
        self.assertEqual(expected, actual)

    def test_public_method_names_are_used(self):
        legacy_names = {
            "anchor_recent_dino_diverse_2old_1recent",
            "anchor_recent_dino_diverse",
            "old_dino_k6",
            "temporal_binned_dino_k8",
        }
        for path in (SUPPLEMENTARY / "tables").glob("*.csv"):
            for row in read_csv(path):
                for field in ("method", "method_id", "cache_policy"):
                    self.assertNotIn(row.get(field), legacy_names, path.name)

    def test_cross_task_macro_matches_documented_calculation(self):
        summary = read_csv(
            SUPPLEMENTARY / "tables" / "table_s13_cross_task_summary.csv"
        )
        regret_rows = read_csv(
            SUPPLEMENTARY / "tables" / "table_s14_cross_task_regret.csv"
        )
        cells = sorted({(row["task"], row["dataset"]) for row in summary})
        self.assertEqual(10, len(cells))

        regrets = {method: [] for method in BOUNDED_METHODS}
        wins = {method: 0 for method in BOUNDED_METHODS}
        for task, dataset in cells:
            rows = {
                row["method_id"]: row
                for row in summary
                if row["task"] == task and row["dataset"] == dataset
            }
            values = {
                method: float(rows[method]["primary_value"])
                for method in BOUNDED_METHODS
            }
            direction = rows["k4"]["primary_direction"]
            oracle = min(values.values()) if direction == "lower" else max(values.values())
            denominator = max(abs(oracle), 1e-12)
            for method, value in values.items():
                regret = (
                    (value - oracle) / denominator
                    if direction == "lower"
                    else (oracle - value) / denominator
                )
                regrets[method].append(regret)
                if math.isclose(value, oracle, rel_tol=1e-8, abs_tol=1e-10):
                    wins[method] += 1

        macro = {
            row["method_id"]: row
            for row in regret_rows
            if row["task"] == "cross_task_macro"
        }
        self.assertEqual(set(BOUNDED_METHODS), set(macro))
        for method in BOUNDED_METHODS:
            self.assertEqual(wins[method], int(macro[method]["oracle_wins"]))
            self.assertTrue(
                math.isclose(
                    statistics.fmean(regrets[method]),
                    float(macro[method]["mean_normalized_regret"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    statistics.median(regrets[method]),
                    float(macro[method]["median_normalized_regret"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    max(regrets[method]),
                    float(macro[method]["max_normalized_regret"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )


if __name__ == "__main__":
    unittest.main()
