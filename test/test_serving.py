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
    # Return numpy array instead of list to match expected behavior
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


@mock.patch("source.serve.schema.validate")
@mock.patch("source.serve.preprocess_data_serving")
def test_predict_success(mock_preprocess, mock_validate, client, mock_model):
    """Test the prediction endpoint with valid data."""
    # Replace the global model with our mock
    with mock.patch("source.serve.model", mock_model):
        # Prepare test data with all values as strings
        test_data = [
            {
                "PolNum": "1",
                "CalYear": "2020",
                "Gender": "Male",
                "Type": "A",
                "Category": "Medium",
                "Occupation": "Employed",
                "Age": "30",
                "Group1": "1",
                "Bonus": "1",
                "Poldur": "5",
                "Value": "1000",
                "Adind": "1",
                "SubGroup2": "X",
                "Group2": "Y",
                "Density": "1.5",
                "Exppdays": "365",
            }
        ]

        # Mock the validate and preprocess functions
        df = pd.DataFrame(test_data)
        mock_validate.return_value = df
        mock_preprocess.return_value = df

        # Make the request
        response = client.post("/predict", json=test_data)

        # Check the response
        assert response.status_code == 200
        assert response.json() == [0.7]  # From our mock model's predict_proba

        # Verify the mocks were called correctly
        mock_validate.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()


@mock.patch("source.serve.schema.validate")
def test_predict_schema_validation_failure(mock_validate, client):
    """Test prediction endpoint with invalid data that fails schema validation."""
    # Replace the global model with a mock
    with mock.patch("source.serve.model", mock.MagicMock()):
        # Prepare test data
        test_data = [{"invalid": "data"}]

        # Mock the validate function to raise an exception
        mock_validate.side_effect = Exception("Schema validation failed")

        # Make the request
        response = client.post("/predict", json=test_data)

        # Check the response
        assert response.status_code == 200
        assert "schema validation failed" in response.json()

        # Verify the mock was called
        mock_validate.assert_called_once()
