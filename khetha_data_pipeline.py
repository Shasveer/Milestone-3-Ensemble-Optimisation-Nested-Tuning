"""Khetha Mutual - Milestone 1: Data Structure & Problem Framing.

Keep this file in the repository root beside claims_data.csv and config.json.
Run: python khetha_data_pipeline.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Kept from the supplied starter code.

ROOT = Path(__file__).resolve().parent


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_config_value(obj: Any, wanted_keys: Iterable[str]) -> Any:
    wanted = {_normalise_name(key) for key in wanted_keys}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if _normalise_name(key) in wanted:
                return value
        for value in obj.values():
            found = _find_config_value(value, wanted_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_config_value(value, wanted_keys)
            if found is not None:
                return found
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


class KhethaDataPipeline:
    """Prevent leakage, split data safely, and validate the resulting datasets."""

    def __init__(self, raw_df, task_type=None, leakage_cols=None):
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame")
        if raw_df.empty:
            raise ValueError("raw_df must contain at least one row")
        if not raw_df.columns.is_unique:
            raise ValueError("raw_df must not contain duplicate column names")

        self.raw_df = raw_df.copy()
        self.task_type = task_type
        self.leakage_cols = list(leakage_cols or [])
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.leakage_columns_removed = []
        self.representativeness_report = {}
        self.class_imbalance_report = {}

    @staticmethod
    def _is_leakage_column(column: str) -> bool:
        name = _normalise_name(column)

        # Record identifiers should not become model features.
        exact_ids = {
            "id", "uuid", "guid", "claimno", "claimnumber", "claimreference",
            "policyno", "policynumber", "policyholdernumber", "vehicleregistration",
            "registrationnumber", "vin", "chassisnumber", "recordid", "referenceid",
        }
        if name in exact_ids or name.endswith(("uuid", "guid")):
            return True
        # Catch identifiers such as incidentid or claimantid even when their names
        # do not contain one of the more common entity prefixes.
        if name.endswith("id") and name not in {"valid", "invalid", "liquid", "solid", "grid", "fluid", "pyramid"}:
            return True

        # Dates/timestamps can encode future information and are not used as raw features.
        date_time_tokens = (
            "date", "timestamp", "datetime", "time", "createdat", "updatedat",
            "settle", "settlement", "closedat", "resolvedat", "reportedat",
            "paidat", "paymentdate", "decisiondate",
        )
        if any(token in name for token in date_time_tokens):
            return True

        # Results known after claim processing are leakage/proxy variables.
        outcome_tokens = (
            "outcome", "result", "decision", "fraud", "confirmed", "investigation",
            "recovered", "settlement", "settled", "payout", "paidamount",
            "paymentamount", "approvedamount", "rejected", "finalcost", "finalamount",
            "claimstatus", "fraudstatus", "label", "target", "avgclaimcost",
        )
        return any(token in name for token in outcome_tokens)

    def detect_task_type(self, target_col):
        """Infer classification or regression from target cardinality and statistics.

        Binary and small discrete targets are classification; high-cardinality numeric
        targets with continuous-like spread are regression. Uses both nunique() and
        describe(), as specified in the milestone instructions.
        """
        if target_col not in self.raw_df.columns:
            raise KeyError(f"Target column {target_col!r} was not found in raw_df")
        target = self.raw_df[target_col]
        assert target.notna().all(), f"Target column {target_col!r} contains nulls"
        unique_count = int(target.nunique(dropna=True))
        summary = target.describe()
        if unique_count < 2:
            raise ValueError(f"Target {target_col!r} must have at least two distinct values")
        if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
            return "classification"
        if unique_count == 2:
            return "classification"

        unique_fraction = unique_count / max(len(target), 1)
        is_float = pd.api.types.is_float_dtype(target.dtype)
        value_range = float(summary.get("max", 0) - summary.get("min", 0))
        target_std = float(summary.get("std", 0) or 0)
        if unique_count > 10 and (is_float or unique_fraction >= 0.05):
            return "regression"
        if unique_count > 20 and value_range > 0 and target_std > 0:
            return "regression"
        return "classification"

    def isolate_features(self, target_col):
        """Separate y from X and remove ID, date, target, and outcome proxy columns."""
        if target_col not in self.raw_df.columns:
            raise KeyError(f"Target column {target_col!r} was not found in raw_df")
        y = self.raw_df[target_col].copy()
        assert not y.isna().any(), f"Target column {target_col!r} contains nulls"

        configured_leakage = {
            col for col in self.leakage_cols
            if col in self.raw_df.columns and col != target_col
        }
        inferred_leakage = {
            col for col in self.raw_df.columns
            if col != target_col and self._is_leakage_column(col)
        }
        drop_columns = configured_leakage | inferred_leakage | {target_col}
        X = self.raw_df.drop(columns=list(drop_columns), errors="ignore").copy()

        # Drop datetime-typed columns even if their names are unconventional.
        datetime_columns = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
        if datetime_columns:
            X = X.drop(columns=datetime_columns)
            drop_columns.update(datetime_columns)

        assert X.shape[0] == len(y), "Feature and target row counts do not match"
        assert X.index.equals(y.index), "Feature and target indexes are not aligned"
        assert target_col not in X.columns, "Target column must not be present in X"
        assert not (set(X.columns) & drop_columns), "At least one leakage column remains in X"
        assert not y.isna().any(), "Target contains nulls"

        self.X, self.y = X, y
        self.leakage_columns_removed = sorted(drop_columns)
        return self.X, self.y

    def split_with_stratification(self, test_size=0.2, random_state=42):
        """Split observations; classification targets are always passed as stratify=y."""
        if self.X is None or self.y is None:
            raise RuntimeError("Call isolate_features() before split_with_stratification()")
        if not 0 < float(test_size) < 1:
            raise ValueError("test_size must be between 0 and 1")
        assert self.X.index.equals(self.y.index), "X/y indexes must align before splitting"
        assert not self.y.isna().any(), "Target contains nulls before splitting"

        stratify = None
        if self.task_type == "classification":
            counts = self.y.value_counts(dropna=False)
            if len(counts) < 2 or (counts < 2).any():
                raise ValueError("Stratified splitting requires at least two observations in each class")
            n_test = int(np.ceil(len(self.y) * float(test_size)))
            n_train = len(self.y) - n_test
            if n_test < len(counts) or n_train < len(counts):
                raise ValueError("Train and test sets must each be large enough to contain every class")
            stratify = self.y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        assert self.X_train.index.equals(self.y_train.index), "Training X/y indexes are not aligned"
        assert self.X_test.index.equals(self.y_test.index), "Test X/y indexes are not aligned"
        assert set(self.X_train.index).isdisjoint(set(self.X_test.index)), "Train/test indexes overlap"
        assert len(self.X_train) + len(self.X_test) == len(self.X), "Split lost or added rows"
        return self.X_train, self.X_test, self.y_train, self.y_test

    def check_class_imbalance(self):
        """Calculate majority/minority ratio and flag minority share under 5%."""
        if self.task_type != "classification" or self.y is None:
            self.class_imbalance_report = {"applicable": False}
            return self.class_imbalance_report
        counts = self.y.value_counts(dropna=False)
        total = int(counts.sum())
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        minority_share = minority_count / total if total else 0.0
        ratio = majority_count / minority_count if minority_count else float("inf")
        below_five_percent = minority_share < 0.05
        self.class_imbalance_report = {
            "applicable": True,
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "majority_minority_ratio": float(ratio),
            "imbalance_ratio": float(ratio),
            "minority_proportion": float(minority_share),
            "minority_percentage": float(minority_share * 100),
            "minority_below_5_percent": bool(below_five_percent),
            "is_imbalanced": bool(below_five_percent),
        }
        print(
            f"Class imbalance | counts={self.class_imbalance_report['class_counts']} | "
            f"majority/minority ratio={ratio:.2f} | minority={minority_share * 100:.2f}%"
        )
        if below_five_percent:
            warnings.warn(
                f"Minority class is {minority_share * 100:.2f}% of the target (<5%).",
                RuntimeWarning,
            )
        return self.class_imbalance_report

    def check_representativeness(self, key_features):
        """Check numeric means/stds and categorical proportions; flag differences over 5%."""
        if self.X is None or self.X_train is None or self.X_test is None:
            raise RuntimeError("Run isolate_features() and split_with_stratification() first")
        report = {}
        threshold = 5.0

        for feature in _as_list(key_features):
            if feature not in self.X.columns:
                report[feature] = {"checked": False, "reason": "not present in model features"}
                warnings.warn(f"Representativeness feature {feature!r} is not in X; skipped.", RuntimeWarning)
                continue

            all_values, train_values, test_values = self.X[feature], self.X_train[feature], self.X_test[feature]
            numeric = pd.api.types.is_numeric_dtype(all_values) and not pd.api.types.is_bool_dtype(all_values)
            if numeric:
                full_mean = float(all_values.mean())
                full_std = float(all_values.std(ddof=1)) if all_values.notna().sum() > 1 else 0.0
                train_mean = float(train_values.mean())
                test_mean = float(test_values.mean())
                train_std = float(train_values.std(ddof=1)) if train_values.notna().sum() > 1 else 0.0
                test_std = float(test_values.std(ddof=1)) if test_values.notna().sum() > 1 else 0.0
                mean_scale = max(abs(full_mean), 1e-12)
                std_scale = max(abs(full_std), 1e-12)
                mean_delta = abs(train_mean - test_mean) / mean_scale * 100
                std_delta = abs(train_std - test_std) / std_scale * 100
                flagged = mean_delta > threshold or std_delta > threshold
                report[feature] = {
                    "type": "numeric", "overall_mean": full_mean,
                    "train_mean": train_mean, "test_mean": test_mean,
                    "overall_std": full_std, "train_std": train_std, "test_std": test_std,
                    "mean_deviation_pct": mean_delta, "std_deviation_pct": std_delta,
                    "threshold_pct": threshold, "flagged_over_5_percent": bool(flagged),
                }
            else:
                def proportions(series):
                    return series.astype("string").fillna("<MISSING>").value_counts(normalize=True)
                full_props, train_props, test_props = map(proportions, (all_values, train_values, test_values))
                categories = full_props.index.union(train_props.index).union(test_props.index)
                full_props = full_props.reindex(categories, fill_value=0.0)
                train_props = train_props.reindex(categories, fill_value=0.0)
                test_props = test_props.reindex(categories, fill_value=0.0)
                max_delta = float((train_props - test_props).abs().max() * 100) if len(categories) else 0.0
                flagged = max_delta > threshold
                report[feature] = {
                    "type": "categorical",
                    "proportions_overall": {str(k): float(v) for k, v in full_props.items()},
                    "proportions_train": {str(k): float(v) for k, v in train_props.items()},
                    "proportions_test": {str(k): float(v) for k, v in test_props.items()},
                    "max_proportion_deviation_percentage_points": max_delta,
                    "threshold_percentage_points": threshold,
                    "flagged_over_5_percent": bool(flagged),
                }

            if report[feature].get("flagged_over_5_percent"):
                warnings.warn(
                    f"Representativeness deviation for {feature!r} exceeds 5%.", RuntimeWarning
                )

        self.representativeness_report = report
        return report

    def run(self, target_col, key_features):
        """Execute the complete preparation and validation workflow."""
        self.task_type = self.detect_task_type(target_col)
        self.isolate_features(target_col)
        self.split_with_stratification()
        self.check_representativeness(key_features)
        self.check_class_imbalance()

        assert self.y_train is not None and self.y_test is not None
        assert not self.y_train.isna().any() and not self.y_test.isna().any()
        assert self.X_train.index.equals(self.y_train.index)
        assert self.X_test.index.equals(self.y_test.index)
        assert all(
            pd.api.types.is_dtype_equal(self.X_train[col].dtype, self.X[col].dtype)
            and pd.api.types.is_dtype_equal(self.X_test[col].dtype, self.X[col].dtype)
            for col in self.X.columns
        ), "Feature dtypes changed during splitting"
        assert pd.api.types.is_dtype_equal(self.y_train.dtype, self.y.dtype)
        assert pd.api.types.is_dtype_equal(self.y_test.dtype, self.y.dtype)

        print(
            f"Pipeline Complete | Task: {self.task_type} | Train: {len(self.X_train)}, "
            f"Test: {len(self.X_test)} | Features: {self.X.shape[1]} | "
            f"Leakage columns removed: {len(self.leakage_columns_removed)}"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


def _get_column(df: pd.DataFrame, requested: Any) -> str | None:
    if not requested:
        return None
    requested_name = _normalise_name(requested)
    for column in df.columns:
        if _normalise_name(column) == requested_name:
            return column
    return None


def main():
    """Run from the repository root using the supplied claims data and config."""
    config_path = ROOT / "config.json"
    config = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        warnings.warn("config.json not found; using default target/feature inference.", RuntimeWarning)

    claims_path = ROOT / "claims_data.csv"
    if not claims_path.exists():
        raise FileNotFoundError("claims_data.csv must be in the same directory as this script")
    raw_df = pd.read_csv(claims_path)

    # The claims table contains both fraud_label and the pre-settlement claim fields.
    # Auxiliary tables are kept separate unless an unambiguous foreign key is available;
    # this dataset's claims table does not include policyholder_id, vehicle_id, or postal_code.
    for filename in (
        "fraud_labels.csv", "vehicles.csv", "policyholders.csv",
        "geographic_risks.csv", "geographic_risk.csv",
    ):
        if not (ROOT / filename).exists():
            warnings.warn(f"Expected companion dataset {filename} is missing.", RuntimeWarning)

    configured_target = _find_config_value(
        config,
        ("target_column", "target_col", "target_variable", "target", "label_column", "label_col", "outcome_column"),
    )
    target_col = _get_column(raw_df, configured_target)
    if target_col is None:
        for candidate in ("fraud_label", "fraud_confirmed", "claim_amount"):
            target_col = _get_column(raw_df, candidate)
            if target_col is not None:
                break
    if target_col is None:
        raise ValueError("Could not identify target; set target_column in config.json")

    missing_targets = int(raw_df[target_col].isna().sum())
    if missing_targets:
        warnings.warn(f"Dropping {missing_targets} row(s) with missing target values.", RuntimeWarning)
        raw_df = raw_df.loc[raw_df[target_col].notna()].copy()
    if raw_df.empty:
        raise ValueError("No rows remain after removing missing target values")

    configured_features = _find_config_value(
        config,
        ("key_features", "representativeness_features", "province_features", "key_provincial_features"),
    )
    key_features = []
    for feature in _as_list(configured_features):
        actual = _get_column(raw_df, feature)
        if actual and actual not in key_features:
            key_features.append(actual)
    if not key_features:
        key_features = [col for col in raw_df.columns if _normalise_name(col) in {"province", "provincename", "provincecode"}]

    configured_leakage = _find_config_value(
        config, ("leakage_cols", "leakage_columns", "columns_to_drop", "exclude_columns")
    )
    leakage_cols = []
    for name in _as_list(configured_leakage):
        actual = _get_column(raw_df, name)
        if actual:
            leakage_cols.append(actual)

    pipeline = KhethaDataPipeline(raw_df, leakage_cols=leakage_cols)
    pipeline.run(target_col, key_features)
    print(f"Target column: {target_col}")
    print(f"Task type: {pipeline.task_type}")
    print(f"Leakage columns removed: {pipeline.leakage_columns_removed}")
    print(f"Representativeness checks completed: {len(pipeline.representativeness_report)}")


if __name__ == "__main__":
    main()
