import os
import pickle
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

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
    # Clean up after the test
    shutil.rmtree(tmp_dir)


class TestTrainer:

    @patch("source.trainer.optuna")
    def test_train_model(self, mock_optuna, test_data, temp_dir):
        # Setup
        X_train, y_train = test_data
        model_path = os.path.join(temp_dir, "test_model.pkl")

        # Configure mock study
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

        # Mock the model fitting to avoid actual training
        with patch("source.trainer.LGBMClassifier") as mock_lgbm:
            mock_model = MagicMock()
            mock_lgbm.return_value = mock_model

            # Mock the open and pickle.dump functions to avoid pickling a MagicMock
            with patch("builtins.open", create=True) as mock_open:
                with patch("pickle.dump") as mock_pickle:
                    # Mock the makedirs function to avoid creating directories
                    with patch("os.makedirs") as mock_makedirs:
                        # Call the function under test
                        with patch.object(settings, "BLOB_STUB_PATH", temp_dir):
                            with patch.object(settings, "MODEL_NAME", "test_model.pkl"):
                                with patch.object(settings, "MODEL_DIR", ""):
                                    with patch(
                                        "source.trainer.settings.FINAL_PARAMS_PATH", ""
                                    ):
                                        result = train_model(
                                            X_train, y_train, model_path
                                        )

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

        # Mock the necessary functions and objects
        with patch.object(trainer, "load_hyperparameter_config") as mock_load_config:
            with patch.object(trainer, "optuna") as mock_optuna:
                # Mock config loading
                mock_load_config.return_value = {
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

                # Configure mock study
                mock_study = MagicMock()
                mock_study.best_params = {"n_estimators": 50, "learning_rate": 0.05}
                mock_optuna.create_study.return_value = mock_study

                # Mock the model fitting to avoid actual training
                with patch("source.trainer.LGBMClassifier") as mock_lgbm:
                    mock_model = MagicMock()
                    mock_lgbm.return_value = mock_model

                    # Mock the open and pickle.dump functions to avoid pickling a MagicMock
                    with patch("builtins.open", create=True) as mock_open:
                        with patch("pickle.dump") as mock_pickle:
                            # Mock the makedirs function to avoid creating directories
                            with patch("os.makedirs") as mock_makedirs:
                                # Create a file to simulate the best_params.txt file
                                params_file = os.path.join(temp_dir, "best_params.txt")
                                with open(params_file, "w") as f:
                                    f.write("test params")

                                # Call the function under test
                                with patch.object(settings, "BLOB_STUB_PATH", temp_dir):
                                    with patch.object(
                                        settings, "MODEL_NAME", "test_model.pkl"
                                    ):
                                        with patch.object(settings, "MODEL_DIR", ""):
                                            with patch.object(
                                                settings, "FINAL_PARAMS_PATH", ""
                                            ):
                                                result = trainer.train_model(
                                                    X_train, y_train, model_path
                                                )

                    # Assertions
                    mock_optuna.create_study.assert_called_once()
                    mock_study.optimize.assert_called_once()
                    mock_makedirs.assert_called()
                    mock_open.assert_called()
                    mock_pickle.assert_called()

    def test_generate_and_save_report(self, test_data, mock_model, temp_dir):
        # Setup
        X_test, y_test = test_data

        # Create a model that returns predictable values instead of random
        y_pred = np.zeros(len(y_test))
        y_pred[::2] = 1  # Make half of predictions 1 and half 0

        # Configure the mock model to return our predetermined predictions
        mock_model.predict = MagicMock(return_value=y_pred)

        report_path = os.path.join(temp_dir, "")

        # Call the function under test
        generate_and_save_report(mock_model, X_test, y_test, report_path)

        # Assertions
        mock_model.predict.assert_called_once()

        # Verify that report files were created
        assert os.path.exists(os.path.join(report_path, "confusion_matrix.png"))
        assert os.path.exists(os.path.join(report_path, "report.txt"))

        # Check report contents
        with open(os.path.join(report_path, "report.txt"), "r") as f:
            content = f.read()
            assert "accuracy:" in content
            assert "precision:" in content
            assert "recall:" in content
            assert "f1_score:" in content
