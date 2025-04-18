import os
import pickle

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from refactored.source.settings import settings


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, model_path: str
) -> LGBMClassifier:
    def objective(trial, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # Split the data into training and validation sets
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Define the hyperparameter search space
        param = {
            "objective": "binary",
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        # Initialize the model with the chosen set of hyperparameters
        model = LGBMClassifier(**param, verbosity=-1)

        # Train the model
        model.fit(X_train_sub, y_train_sub)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)

        return accuracy

    # Setting the logging level WARNING, the INFO logs are suppressed.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=20,
        show_progress_bar=True,
    )

    # Retrieve the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters: ", best_params)

    # Train the final model with the best hyperparameters
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    with open(settings.MODEL_PATH, "wb") as f:

        pickle.dump(best_model, f)
        print(f"Model saved to {settings.MODEL_PATH}")

    params_file = os.path.join(
        settings.BLOB_STUB_PATH, settings.FINAL_PARAMS_PATH, "best_params.txt"
    )
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, "w") as f:

        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    return best_model


def generate_and_save_report(
    best_model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    report_path: str = os.path.join(settings.BLOB_STUB_PATH, settings.REPORT_PATH),
) -> None:
    """
    Generate and save a report with the model's performance metrics and confusion matrix.
    """
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    cm.figure_.savefig(os.path.join(report_path + "confusion_matrix.png"))
    with open(os.path.join(report_path + "report.txt"), "w") as f:

        for key, value in report.items():
            f.write(f"{key}: {value}\n")
