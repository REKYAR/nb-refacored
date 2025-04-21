import json
import pickle
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from source.serve import app


@pytest.fixture
def client():
    """Create a test client for our API."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = mock.MagicMock()
    model.predict.return_value = [0]
    import numpy as np

    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


@mock.patch("source.serve.open")
@mock.patch("source.serve.pickle.load")
def test_reload_model(mock_pickle_load, mock_open, client, mock_model):
    """Test the reload model endpoint."""
    mock_pickle_load.return_value = mock_model

    response = client.get("/reload")

    assert response.status_code == 200
    assert response.json() == {"message": "Model reloaded successfully"}
    mock_open.assert_called_once()
    mock_pickle_load.assert_called_once()


@mock.patch("source.serve.preprocess_data_serving")
def test_predict_success(mock_preprocess, client, mock_model):
    """Test the prediction endpoint with valid data."""
    with mock.patch("source.serve.model", mock_model):
        test_data = [
            {
                "PolNum": "200114978",
                "CalYear": "2009",
                "Gender": "Male",
                "Type": "C",
                "Category": "Large",
                "Occupation": "Employed",
                "Age": "25",
                "Group1": "18",
                "Bonus": "90",
                "Poldur": "3",
                "Value": "15080",
                "Adind": "0",
                "SubGroup2": "L46",
                "Group2": "L",
                "Density": "72.01288299",
                "Exppdays": "365",
                "Numtppd": "1",
                "Numtpbi": "0",
                "Indtppd": "0.0",
                "Indtpbi": "0.0",
            }
        ]

        df = pd.DataFrame(test_data)
        mock_preprocess.return_value = df

        response = client.post("/predict", json=test_data)

        assert response.status_code == 200
        assert response.json() == [0.7]

        mock_preprocess.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()


def test_predict_schema_validation_failure(client):
    """Test prediction endpoint with invalid data that fails schema validation."""
    with mock.patch("source.serve.model", mock.MagicMock()):
        invalid_test_data = [
            {
                "PolNum": "not-a-number",  # Should be an integer
                "CalYear": "1999",  # Out of range (schema requires 2000-2025)
                "Gender": "Unknown",  # Not in allowed values
                "Type": "Z",  # Not in allowed values
                "Category": "XLarge",  # Not in allowed values
                "Occupation": "Student",  # Not in allowed values
                "Age": "-10",  # Negative age not allowed
                "Group1": "18",
                "Bonus": "90",
                "Poldur": "-3",  # Negative duration not allowed
                "Value": "-15080",  # Negative value not allowed
                "Adind": "2",  # Not in allowed values [0,1]
                "SubGroup2": "L46",
                "Group2": "L",
                "Density": "-72.01",  # Negative density not allowed
                "Exppdays": "400",  # Out of range (schema requires 0-366)
                "Numtppd": "1",
                "Numtpbi": "0",
                "Indtppd": "0.0",
                "Indtpbi": "0.0",
            }
        ]

        response = client.post("/predict", json=invalid_test_data)

        assert response.status_code == 200
        assert "schema validation failed" in response.json()
