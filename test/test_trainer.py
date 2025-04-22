import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from source.settings import settings
from source.trainer import generate_and_save_report, train_model


@pytest.fixture
def test_data():
    """Create some test data for the trainer functions."""
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


@pytest.fixture
def mock_model():
    """Create a simple mock model."""
    model = LGBMClassifier(n_estimators=10)
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


class TestTrainer:

    @patch("source.trainer.optuna")
    @patch("source.trainer.LGBMClassifier")
    @patch("builtins.open", create=True)
    @patch("pickle.dump")
    @patch("os.makedirs")
    def test_train_model(
        self,
        mock_makedirs,
        mock_pickle,
        mock_open,
        mock_lgbm,
        mock_optuna,
        test_data,
        temp_dir,
    ):
        X_train, y_train = test_data
        model_path = os.path.join(temp_dir, "test_model.pkl")

        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        mock_optuna.create_study.return_value = mock_study

        mock_model = MagicMock()
        mock_lgbm.return_value = mock_model

        with patch.multiple(
            settings,
            BLOB_STUB_PATH=temp_dir,
            MODEL_NAME="test_model.pkl",
            MODEL_DIR="",
            FINAL_PARAMS_PATH="",
        ):

            with patch("source.trainer.settings.FINAL_PARAMS_PATH", ""):
                train_model(X_train, y_train, model_path)

        # Assertions
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        mock_lgbm.assert_called_with(**mock_study.best_params)
        mock_model.fit.assert_called_with(X_train, y_train)
        mock_makedirs.assert_called()
        mock_open.assert_called()
        mock_pickle.assert_called()

    def test_train_model_with_config(self, test_data, temp_dir):
        # Setup
        X_train, y_train = test_data
        model_path = os.path.join(temp_dir, "test_model.pkl")

        # Import the module to modify it directly
        import source.trainer as trainer

        # Create test config and params data
        hp_config = {
            "objective": "binary",
            "n_jobs": -1,
            "n_estimators": {"type": "int", "low": 10, "high": 100},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.1,
                "log": True,
            },
        }
        best_params = {"n_estimators": 50, "learning_rate": 0.05}

        params_file = os.path.join(temp_dir, "best_params.txt")
        with open(params_file, "w") as f:
            f.write("test params")

        mock_optuna = MagicMock()
        mock_study = MagicMock(best_params=best_params)
        mock_optuna.create_study.return_value = mock_study

        with (
            patch.multiple(
                trainer,
                load_hyperparameter_config=MagicMock(return_value=hp_config),
                optuna=mock_optuna,
            ),
            patch("source.trainer.LGBMClassifier", return_value=MagicMock()),
            patch("builtins.open", create=True),
            patch("pickle.dump"),
            patch("os.makedirs"),
            patch.multiple(
                settings,
                BLOB_STUB_PATH=temp_dir,
                MODEL_NAME="test_model.pkl",
                MODEL_DIR="",
                FINAL_PARAMS_PATH="",
            ),
        ):
            trainer.train_model(X_train, y_train, model_path)

        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()

    def test_generate_and_save_report(self, test_data, mock_model, temp_dir):
        X_test, y_test = test_data

        # Make half of predictions 1 and half 0
        y_pred = np.zeros(len(y_test))
        y_pred[::2] = 1

        mock_model.predict = MagicMock(return_value=y_pred)

        report_path = os.path.join(temp_dir, "")

        generate_and_save_report(mock_model, X_test, y_test, report_path)

        mock_model.predict.assert_called_once()

        assert os.path.exists(os.path.join(report_path, "confusion_matrix.png"))
        assert os.path.exists(os.path.join(report_path, "report.txt"))

        with open(os.path.join(report_path, "report.txt"), "r") as f:
            content = f.read()
            assert "accuracy:" in content
            assert "precision:" in content
            assert "recall:" in content
            assert "f1_score:" in content
