"""Baseline models and cross-validation for Khetha claims data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

try:
    from khetha_data_pipeline import KhethaDataPipeline
except ImportError:
    KhethaDataPipeline = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Plotting remains available when the optional dependency is installed.
    matplotlib = None
    plt = None


LOGGER = logging.getLogger(__name__)


class KhethaBaselineEngine:
    """Build and validate interpretable baseline models."""

    def __init__(self, X_train, y_train, X_test, y_test, task_type, preprocessor=None):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.task_type = task_type.lower()
        self.preprocessor = preprocessor
        self.best_model = None

    def _fit_features(self):
        if self.preprocessor is None:
            return self.X_train
        self.preprocessor.fit(self.X_train, self.y_train)
        return self.preprocessor.transform(self.X_train)

    def _cv_model(self, model):
        if self.preprocessor is None or isinstance(model, Pipeline):
            return model
        return Pipeline([("preprocessor", clone(self.preprocessor)), ("model", model)])

    def _model_features(self, X):
        if self.preprocessor is None:
            return X
        return self.preprocessor.transform(X)

    @property
    def is_classification(self):
        return self.task_type in {"classification", "class", "fraud", "binary", "multiclass"}

    def build_tree_baseline(self, task_type=None, max_depth=None, min_samples_split=2, ccp_alpha=0.0):
        """Fit a classifier or regressor with explicit complexity controls."""
        selected_task = (task_type or self.task_type).lower()
        if selected_task in {"classification", "class", "fraud", "binary", "multiclass"}:
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                ccp_alpha=ccp_alpha,
                random_state=42,
            )
        elif selected_task in {"regression", "regressor", "claim_cost", "cost"}:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                ccp_alpha=ccp_alpha,
                random_state=42,
            )
        else:
            raise ValueError("task_type must be classification or regression")
        model.fit(self._fit_features(), self.y_train)
        self.best_model = model
        return model

    def build_logistic_baseline(self, multi_class=False):
        """Fit balanced logistic regression for binary or multinomial targets."""
        if not self.is_classification:
            raise ValueError("Logistic regression is only valid for classification tasks")
        kwargs = {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"}
        if multi_class:
            kwargs["multi_class"] = "multinomial"
        model = LogisticRegression(**kwargs)
        model.fit(self._fit_features(), self.y_train)
        self.best_model = model
        return model

    def run_cross_validation(self, model, cv_folds=5):
        """Return cross-validation scores and their mean and standard deviation."""
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.is_classification:
            counts = pd.Series(self.y_train).value_counts()
            folds = min(cv_folds, int(counts.min()))
            if folds < 2:
                raise ValueError("Each class needs at least two samples for stratified cross-validation")
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            scoring = "accuracy"
        else:
            folds = min(cv_folds, len(self.y_train))
            if folds < 2:
                raise ValueError("At least two samples are required for cross-validation")
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
            scoring = "r2"
        scores = cross_val_score(self._cv_model(model), self.X_train, self.y_train, cv=cv, scoring=scoring)
        return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores}

    def control_tree_complexity(self, max_candidates=10):
        """Select tree complexity controls with cross-validation."""
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.is_classification:
            base = DecisionTreeClassifier(random_state=42)
        else:
            base = DecisionTreeRegressor(random_state=42)
        path = base.cost_complexity_pruning_path(self._fit_features(), self.y_train)
        alphas = np.unique(path.ccp_alphas)
        if len(alphas) > max_candidates:
            indexes = np.linspace(0, len(alphas) - 1, max_candidates, dtype=int)
            alphas = alphas[indexes]

        max_depths = [None, 5, 10]
        min_samples_splits = [2, 10, 20]
        candidates = []
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for alpha in alphas:
                    candidate = self.build_tree_baseline(
                        max_depth=max_depth,
                        ccp_alpha=float(alpha),
                        min_samples_split=min_samples_split,
                    )
                    candidates.append(
                        (self.run_cross_validation(candidate)["mean"], max_depth, min_samples_split, float(alpha))
                    )
        _, best_depth, best_split, best_alpha = max(candidates, key=lambda item: item[0])
        best_model = self.build_tree_baseline(
            max_depth=best_depth,
            ccp_alpha=best_alpha,
            min_samples_split=best_split,
        )
        self.best_model = best_model
        return {
            "max_depth": best_depth,
            "min_samples_split": best_split,
            "ccp_alpha": best_alpha,
        }, best_model

    def visualize_top_splits(self, model, max_depth=3):
        """Create and return a plot showing the first three tree levels."""
        if plt is None:
            raise ImportError("visualize_top_splits requires matplotlib")
        if self.preprocessor is not None and hasattr(self.preprocessor, "get_feature_names_out"):
            feature_names = list(self.preprocessor.get_feature_names_out())
        else:
            feature_names = list(getattr(self.X_train, "columns", [])) or None
        fig, axis = plt.subplots(figsize=(18, 9))
        plot_tree(model, max_depth=max_depth, feature_names=feature_names, filled=True, ax=axis)
        axis.set_title("Khetha baseline decision tree: top splits")
        fig.tight_layout()
        return fig

    def diagnose_fit(self, model=None):
        """Log simple train/validation warnings and return the measured scores."""
        model = model or self.best_model
        if model is None:
            raise ValueError("A fitted model is required for diagnostics")
        train_score = float(model.score(self._model_features(self.X_train), self.y_train))
        validation_score = float(model.score(self._model_features(self.X_test), self.y_test))
        if train_score - validation_score > 0.15:
            LOGGER.warning("Potential overfitting: train score %.3f exceeds validation score %.3f by more than 15%%", train_score, validation_score)
        if train_score < 0.7 and validation_score < 0.7:
            LOGGER.warning("Potential underfitting: train and validation scores are both below 0.7 (%.3f, %.3f)", train_score, validation_score)
        return {"train": train_score, "validation": validation_score}


def _load_milestone_one_output(data_dir: Path, target_col: str, key_features):
    """Run the Milestone 1 pipeline and return its prepared train/test outputs."""
    if KhethaDataPipeline is None:
        return None
    claims = pd.read_csv(data_dir / "claims_data.csv")
    pipeline = KhethaDataPipeline(claims, task_type=None)
    return pipeline.run(target_col, key_features)


def _make_preprocessor(X_train):
    categorical = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric = [column for column in X_train.columns if column not in categorical]
    return ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)],
        remainder="drop",
    )


def _claims_features(data_dir: Path):
    claims = pd.read_csv(data_dir / "claims_data.csv")
    risk_paths = [data_dir / "geographic_risks.csv", data_dir / "geographic_risk.csv"]
    risk_path = next((path for path in risk_paths if path.exists()), None)
    if risk_path is not None:
        risk = pd.read_csv(risk_path).groupby("province", as_index=False)[
            ["crime_score", "hijacking_rate_per_10k", "weather_risk_score", "avg_claim_cost_zar", "fraud_rate_historical"]
        ].mean()
        claims = claims.merge(risk, on="province", how="left")
    target = claims["fraud_label"].astype(int)
    features = claims.drop(columns=["claim_id", "fraud_label", "investigation_outcome", "claim_settlement_date", "claim_amount", "claim_date"], errors="ignore")
    categorical = features.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)],
        remainder="drop",
    )
    return features, target, claims["claim_amount"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path(__file__).resolve().parent
    fraud_output = _load_milestone_one_output(data_dir, "fraud_label", ["province", "claim_type", "claim_amount"])
    cost_output = _load_milestone_one_output(data_dir, "claim_amount", ["province", "claim_type"])
    if fraud_output is not None and cost_output is not None:
        LOGGER.info("Using X_train, X_test, y_train, and y_test from khetha_data_pipeline.py")
        fraud_X_train, fraud_X_test, y_train, y_test = fraud_output
        fraud_preprocessor = _make_preprocessor(fraud_X_train)
    else:
        LOGGER.info("Milestone 1 pipeline unavailable; using local claims preparation fallback")
        features, fraud_target, cost_target = _claims_features(data_dir)
        fraud_X_train, fraud_X_test, y_train, y_test = train_test_split(features, fraud_target, test_size=0.2, random_state=42, stratify=fraud_target)
        fraud_preprocessor = _make_preprocessor(fraud_X_train)
    fraud_engine = KhethaBaselineEngine(fraud_X_train, y_train, fraud_X_test, y_test, "classification", fraud_preprocessor)
    fraud_default_tree = fraud_engine.build_tree_baseline()
    LOGGER.info("Fraud default tree CV: %s", fraud_engine.run_cross_validation(fraud_default_tree))
    fraud_engine.diagnose_fit(fraud_default_tree)
    if plt is not None:
        fraud_engine.visualize_top_splits(fraud_default_tree).savefig(data_dir / "khetha_tree_splits.png")
    fraud_logistic = fraud_engine.build_logistic_baseline()
    LOGGER.info("Fraud logistic regression CV: %s", fraud_engine.run_cross_validation(fraud_logistic))
    fraud_config, fraud_tree = fraud_engine.control_tree_complexity()
    LOGGER.info("Selected fraud tree configuration: %s", fraud_config)
    LOGGER.info("Fraud controlled tree CV: %s", fraud_engine.run_cross_validation(fraud_tree))
    fraud_engine.diagnose_fit(fraud_tree)

    if cost_output is not None:
        cost_X_train, cost_X_test, target_train, target_test = cost_output
        cost_preprocessor = _make_preprocessor(cost_X_train)
    else:
        cost_X_train, cost_X_test, target_train, target_test = train_test_split(features, cost_target, test_size=0.2, random_state=42)
        cost_preprocessor = _make_preprocessor(cost_X_train)
    cost_engine = KhethaBaselineEngine(cost_X_train, target_train, cost_X_test, target_test, "regression", cost_preprocessor)
    linear = Pipeline([("preprocessor", clone(cost_preprocessor)), ("model", LinearRegression())])
    linear.fit(cost_X_train, target_train)
    tree = cost_engine.build_tree_baseline()
    LOGGER.info("Claim cost default tree CV: %s", cost_engine.run_cross_validation(tree))
    LOGGER.info("Claim cost linear CV: %s", cost_engine.run_cross_validation(linear))
    cost_engine.diagnose_fit(tree)
    cost_config, controlled_tree = cost_engine.control_tree_complexity()
    LOGGER.info("Selected claim-cost tree configuration: %s", cost_config)
    LOGGER.info("Claim cost controlled tree CV: %s", cost_engine.run_cross_validation(controlled_tree))
    cost_engine.diagnose_fit(controlled_tree)


if __name__ == "__main__":
    main()