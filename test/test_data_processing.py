from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from source.data_processing import preprocess_data, preprocess_data_serving
from source.settings import settings


@pytest.fixture
def sample_data():
    """Create sample data for testing preprocessing functions."""
    return pd.DataFrame(
        {
            "PolNum": [123456, 234567, 345678, 456789],
            "Numtppd": [0, 1, 2, 0],
            "Numtpbi": [1, 0, 1, 0],
            "Indtppd": [0.0, 1.0, 1.0, 0.0],
            "Indtpbi": [1.0, 0.0, 0.0, 1.0],
            "CalYear": [2020, 2021, 2020, 2021],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Type": ["A", "B", "C", "D"],
            "Category": ["Medium", "Large", "Small", "Medium"],
            "Occupation": ["Employed", "Self-employed", "Retired", "Housewife"],
            "SubGroup2": ["SG1", "SG2", "SG3", "SG1"],
            "Group2": ["G2A", "G2B", "G2A", "G2C"],
            "Group1": [10, 20, 30, 40],
            "Age": [30, 40, 50, 25],
            "Bonus": [100, 200, 300, 150],
            "Poldur": [3, 5, 7, 2],
            "Value": [10000, 15000, 20000, 12000],
            "Adind": [0, 1, 0, 1],
            "Density": [50.5, 75.3, 60.1, 45.8],
            "Exppdays": [365, 180, 270, 90],
        }
    )


@pytest.fixture
def mock_encoder():
    """Create a mock OneHotEncoder for testing."""
    encoder = MagicMock()
    # Mock the fit_transform method to return a simple array
    encoder.fit_transform.return_value = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1],
        ]
    )
    # Mock the transform method to return the same array
    encoder.transform.return_value = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1],
        ]
    )
    # Mock get_feature_names_out to return feature names
    encoder.get_feature_names_out.return_value = [
        "CalYear_2020",
        "CalYear_2021",
        "Gender_Male",
        "Gender_Female",
        "Type_A",
        "Type_B",
        "Category_Medium",
        "Category_Large",
    ]
    return encoder


class TestDataProcessing:

    @patch("source.data_processing.OneHotEncoder")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_preprocess_data(
        self, mock_pickle_dump, mock_file, mock_encoder_class, sample_data, mock_encoder
    ):
        """Test that preprocess_data correctly transforms the data and saves the encoder."""
        mock_encoder_class.return_value = mock_encoder

        result = preprocess_data(sample_data)

        mock_encoder_class.assert_called_once_with(sparse_output=False)

        # Verify the encoder was fit with the categorical columns
        categorical_columns = [
            "CalYear",
            "Gender",
            "Type",
            "Category",
            "Occupation",
            "SubGroup2",
            "Group2",
            "Group1",
        ]
        mock_encoder.fit_transform.assert_called_once()
        # Check if the first arg to fit_transform is a dataframe with the right columns
        args, _ = mock_encoder.fit_transform.call_args
        pd.testing.assert_index_equal(args[0].columns, pd.Index(categorical_columns))

        # Verify the encoder was saved
        mock_file.assert_called_once_with(settings.ENCODER_PATH, "wb")
        mock_pickle_dump.assert_called_once()

        # Check the result DataFrame
        assert "target" in result.columns
        assert "Numtppd" not in result.columns
        assert "Numtpbi" not in result.columns
        assert "Indtppd" not in result.columns
        assert "Indtpbi" not in result.columns

        # Check target creation logic
        target_values = (
            sample_data["Numtppd"].apply(lambda x: 1 if x != 0 else 0).values
        )
        np.testing.assert_array_equal(result["target"].values, target_values)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_preprocess_data_serving(
        self, mock_pickle_load, mock_file, mock_exists, sample_data, mock_encoder
    ):
        """Test that preprocess_data_serving correctly transforms the data using a saved encoder."""
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_encoder

        # Remove columns that would be dropped during serving
        data_for_serving = sample_data.copy()

        result = preprocess_data_serving(data_for_serving)

        mock_exists.assert_called_once_with(settings.ENCODER_PATH)
        mock_file.assert_called_once_with(settings.ENCODER_PATH, "rb")
        mock_pickle_load.assert_called_once()

        # Verify the encoder transform method was called with the categorical columns
        categorical_columns = [
            "CalYear",
            "Gender",
            "Type",
            "Category",
            "Occupation",
            "SubGroup2",
            "Group2",
            "Group1",
        ]
        mock_encoder.transform.assert_called_once()
        # Check if the first arg to transform is a dataframe with the right columns
        args, _ = mock_encoder.transform.call_args
        pd.testing.assert_index_equal(args[0].columns, pd.Index(categorical_columns))

        # Check the result DataFrame
        for col in ["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"]:
            assert col not in result.columns

        # Verify feature names were retrieved
        mock_encoder.get_feature_names_out.assert_called_once_with(categorical_columns)

    @patch("os.path.exists")
    def test_preprocess_data_serving_no_encoder(self, mock_exists, sample_data):
        """Test that preprocess_data_serving raises an error when the encoder is not found."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError) as excinfo:
            preprocess_data_serving(sample_data)

        assert settings.ENCODER_PATH in str(excinfo.value)
        assert "Encoder not found" in str(excinfo.value)
