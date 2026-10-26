# Milestone 3: Ensemble Optimisation and Nested Tuning

`khetha_ensemble_optimizer.py` provides ensemble learning and hyperparameter
optimisation for the Khetha fraud-detection project. It supports bagging,
boosting, exhaustive and stochastic searches, Bayesian optimisation, and
leakage-safe nested cross-validation.

## Requirements

Python 3.9 or newer is recommended. Install the runtime dependencies with:

```bash
pip install numpy pandas scipy scikit-learn optuna
```

The script can be imported without these packages being installed, and its
entry point prints an installation message. Model construction and tuning
require the dependencies above.

## Repository Contents

| File | Purpose |
| --- | --- |
| `khetha_ensemble_optimizer.py` | Ensemble models, searches, and nested CV |
| `khetha_data_pipeline.py` | Milestone 1 leakage checks and stratified splitting |
| `khetha_baseline_models.py` | Milestone 2 preprocessing and baseline model utilities |
| `claims_data.csv` | Claim-level features and `fraud_label` target |
| `fraud_labels.csv` | Investigation and confirmed-fraud records |
| `vehicles.csv` | Vehicle and policyholder vehicle attributes |
| `policyholders.csv` | Policyholder demographic and policy attributes |
| `geographic_risks.csv` | Postal-code crime, weather, and historical fraud risk |
| `config.json` | Fraud thresholds, deployment values, and domain settings |

The Milestone 3 script imports the included Milestone 1 and Milestone 2
modules and uses their outputs through this preparation path:

1. `KhethaDataPipeline.run("fraud_label", ...)` removes identifiers, dates,
	target proxies, and other leakage columns before creating a stratified
	train/test split.
2. Milestone 2's `_make_preprocessor()` fits categorical encoding on the
	training partition and transforms the holdout without refitting.
3. The encoded matrices and labels are passed to `KhethaEnsembleOptimizer`.

The import helper is also available for inspection:

```python
from khetha_ensemble_optimizer import _import_milestone_outputs

milestone_modules = _import_milestone_outputs()
```

The optimizer also accepts any compatible NumPy array, sparse matrix, pandas
DataFrame, or pandas Series when used programmatically.

## Running the Script

Run the required entry point from the repository root:

```bash
python khetha_ensemble_optimizer.py
```

The demonstration runs the integrated Milestone 1/2 preparation path and then
uses bounded settings to demonstrate bias/variance diagnosis, staged Gradient
Boosting validation, AdaBoost weight tracking, Grid Search, Randomized Search,
Optuna Bayesian optimisation, and three-fold nested CV. The final holdout is
not used by any of these operations.

## Basic Usage

```python
from khetha_ensemble_optimizer import KhethaEnsembleOptimizer

optimizer = KhethaEnsembleOptimizer(X_train, y_train, X_test, y_test)
optimizer.build_ensembles(strategy="random_forest")

grid_result = optimizer.run_grid_search({
		"n_estimators": [100, 200],
		"max_features": ["sqrt", "log2"],
		"min_samples_leaf": [1, 2],
}, cv=5)

nested_result = optimizer.nested_cross_validation(outer_cv=5, inner_cv=3)
print(nested_result["mean_score"])
print(nested_result["ci_95"])
```

`X_train` and `y_train` are used for fitting, tuning, and nested validation.
`X_test` and `y_test` are retained as a final holdout and are never used by
the search methods or nested-CV inner folds.

## Ensemble Strategies

`build_ensembles()` creates all three supported estimators and selects one as
`tuned_model`:

- **Random Forest:** 200 trees, square-root feature sampling, bootstrap
	aggregation, and `oob_score=True` for an out-of-bag estimate.
- **Gradient Boosting:** controlled `learning_rate`, `n_estimators`, and
	`subsample`, with validation-based early stopping through
	`n_iter_no_change`. `monitor_gradient_boosting()` additionally evaluates
	`staged_predict()` predictions and records the best validation stage.
- **AdaBoost:** decision-stump base learners (`max_depth=1`), learning-rate
	control, and per-round learner weights exposed by
	`track_adaboost_weights()`.

Available strategies are `random_forest`, `gradient_boosting`, `adaboost`,
and `auto`. The automatic diagnosis compares shallow-tree, bagging, and
boosting validation performance and prefers Random Forest when the profile
indicates high variance or when boosting does not provide a meaningful gain.

## Hyperparameter Optimisation

### Grid search

`run_grid_search(param_grid, cv=5)` uses `GridSearchCV`, fits only on the
training data, and returns:

- `best_params`
- `best_score`
- `cv_results`
- the fitted `search` object

### Randomized search

`run_randomized_search(param_distributions=None, n_iter=50, cv=5)` uses
`RandomizedSearchCV`. Default distributions use `scipy.stats.randint` for
tree counts and `scipy.stats.loguniform` for Gradient Boosting learning
rates. The cumulative best sampled mean validation score after each candidate
is available as `convergence_trajectory`.

### Bayesian optimisation

`run_bayesian_optimization(n_trials=40)` creates an Optuna study with:

- `direction="maximize"`
- a seeded `TPESampler`
- an objective returning mean cross-validated ROC-AUC

The return value includes `best_params`, `best_score`, `study`, and the
objective function. The resulting Random Forest is stored in `tuned_model`.

## Nested Cross-Validation

`nested_cross_validation(outer_cv=5, inner_cv=3)` performs the following for
each outer fold:

1. Split the training data into an outer training fold and untouched
	 evaluation fold.
2. Run `GridSearchCV` only inside the outer training fold.
3. Fit the best inner model and score it once on the untouched outer fold.
4. Aggregate outer-fold ROC-AUC scores.

The method returns:

```python
{
		"mean_score": 0.0,
		"ci_95": (0.0, 0.0),
		"outer_scores": [],
}
```

The confidence interval uses the normal approximation:
$\bar{x} \mathbin{\pm} 1.96(s / \sqrt{n})$, clipped to the valid ROC-AUC
range of 0 to 1. This keeps tuning separate from unbiased outer evaluation.

## Additional Diagnostics

```python
optimizer.build_ensembles("gradient_boosting")
best_stage, best_score = optimizer.monitor_gradient_boosting()

optimizer.build_ensembles("adaboost")
weights = optimizer.track_adaboost_weights()
```

Diagnostic values are also stored on the optimizer instance as
`gradient_validation_scores`, `best_gradient_stage`, and
`adaboost_weight_trajectory`.

## Data Preparation Notes

Categorical fields must be encoded before fitting the estimators, and missing
values must be imputed in the Milestone 1 pipeline. Keep preprocessing inside
the training workflow when it learns statistics from data; a
`Pipeline`/`ColumnTransformer` is preferred so preprocessing is also fitted
inside each cross-validation fold. Do not use the final holdout while
choosing features, thresholds, or hyperparameters.

The supplied `config.json` contains operational thresholds and South African
domain values. It is configuration rather than a source of model labels.