import numpy as np
import pandas as pd
from typing import Dict, Any
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import mlflow
from datetime import datetime


class AutoMLBuilder:
    """Automated machine learning pipeline builder"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        self.best_model = None
        self.best_score = -np.inf

    def build_optimized_pipeline(
        self, X: pd.DataFrame, y: pd.Series, time_budget: int = 300
    ):
        """Build optimized pipeline with hyperparameter tuning"""

        def objective(trial):
            # Suggest model type
            model_type = trial.suggest_categorical(
                "model_type",
                ["random_forest", "gradient_boosting", "svm", "logistic_regression"],
            )

            # Suggest hyperparameters based on model type
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
                    max_depth=trial.suggest_int("rf_max_depth", 3, 20),
                    min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
                    random_state=42,
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("gb_n_estimators", 50, 300),
                    learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.3),
                    max_depth=trial.suggest_int("gb_max_depth", 3, 15),
                    random_state=42,
                )
            elif model_type == "svm":
                model = SVC(
                    C=trial.suggest_float("svm_c", 0.1, 10.0),
                    kernel=trial.suggest_categorical("svm_kernel", ["linear", "rbf"]),
                    gamma=trial.suggest_categorical("svm_gamma", ["scale", "auto"]),
                )
            else:  # logistic_regression
                model = LogisticRegression(
                    C=trial.suggest_float("lr_c", 0.1, 10.0),
                    solver=trial.suggest_categorical(
                        "lr_solver", ["liblinear", "lbfgs"]
                    ),
                    max_iter=1000,
                    random_state=42,
                )

            # Cross-validation score
            score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
            return score

        # Optimize with Optuna
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=100, timeout=time_budget)

        # Train best model
        best_params = self.study.best_params
        self.best_model = self._create_model_from_params(best_params)
        self.best_model.fit(X, y)
        self.best_score = self.study.best_value

        # Log to MLflow
        with mlflow.start_run(
            run_name=f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_accuracy", self.best_score)
            mlflow.sklearn.log_model(self.best_model, "best_model")

            # Log importance if available
            if hasattr(self.best_model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance": self.best_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                mlflow.log_table(importance_df, "feature_importance.json")

        return self.best_model, self.best_params, self.best_score

    def _create_model_from_params(self, params: Dict) -> Any:
        """Create model instance from optimized parameters"""
        model_type = params["model_type"]

        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=params["rf_n_estimators"],
                max_depth=params["rf_max_depth"],
                min_samples_split=params["rf_min_samples_split"],
                random_state=42,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params["gb_n_estimators"],
                learning_rate=params["gb_learning_rate"],
                max_depth=params["gb_max_depth"],
                random_state=42,
            )
        elif model_type == "svm":
            return SVC(
                C=params["svm_c"],
                kernel=params["svm_kernel"],
                gamma=params["svm_gamma"],
            )
        else:
            return LogisticRegression(
                C=params["lr_c"],
                solver=params["lr_solver"],
                max_iter=1000,
                random_state=42,
            )

    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create ensemble of best models"""
        from sklearn.ensemble import VotingClassifier

        # Get top 3 models from optimization
        top_trials = sorted(self.study.trials, key=lambda x: x.value, reverse=True)[:3]

        estimators = []
        for i, trial in enumerate(top_trials):
            model = self._create_model_from_params(trial.params)
            model.fit(X, y)
            estimators.append((f"model_{i}", model))

        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X, y)

        return ensemble
