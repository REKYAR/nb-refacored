import os
import pickle
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, mock_open, MagicMock

from refactored.source.data_processing import preprocess_data, preprocess_data_serving
from refactored.source.settings import settings


@pytest.fixture
def sample_data():
    """Create sample data for testing preprocessing functions."""
    return pd.DataFrame({
        'Numtppd': [0, 1, 2, 0],
        'Numtpbi': [1, 0, 1, 0],
        'Indtppd': [0, 1, 1, 0],
        'Indtpbi': [1, 0, 0, 1],
        'CalYear': [2020, 2021, 2020, 2021],
        'Gender': ['M', 'F', 'M', 'F'],
        'Type': ['A', 'B', 'A', 'C'],
        'Category': ['X', 'Y', 'Z', 'X'],
        'Occupation': ['Manager', 'Engineer', 'Doctor', 'Teacher'],
        'SubGroup2': ['SG1', 'SG2', 'SG3', 'SG1'],
        'Group2': ['G2A', 'G2B', 'G2A', 'G2C'],
        'Group1': ['G1A', 'G1B', 'G1A', 'G1C'],
        'Age': [30, 40, 50, 25],
        'Height': [175, 165, 180, 160]
    })


@pytest.fixture
def mock_encoder():
    """Create a mock OneHotEncoder for testing."""
    encoder = MagicMock()
    # Mock the fit_transform method to return a simple array
    encoder.fit_transform.return_value = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 1]
    ])
    # Mock the transform method to return the same array
    encoder.transform.return_value = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 1]
    ])
    # Mock get_feature_names_out to return feature names
    encoder.get_feature_names_out.return_value = [
        'CalYear_2020', 'CalYear_2021',
        'Gender_F', 'Gender_M',
        'Type_A', 'Type_B',
        'Category_X', 'Category_Y'
    ]
    return encoder


class TestDataProcessing:
    
    @patch('refactored.source.data_processing.OneHotEncoder')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_preprocess_data(self, mock_pickle_dump, mock_file, mock_encoder_class, sample_data, mock_encoder):
        """Test that preprocess_data correctly transforms the data and saves the encoder."""
        # Set up the mock encoder class to return our mock encoder instance
        mock_encoder_class.return_value = mock_encoder
        
        # Call the function under test
        result = preprocess_data(sample_data)
        
        # Verify the encoder was created with the right parameters
        mock_encoder_class.assert_called_once_with(sparse_output=False)
        
        # Verify the encoder was fit with the categorical columns
        categorical_columns = [
            "CalYear", "Gender", "Type", "Category", "Occupation", 
            "SubGroup2", "Group2", "Group1"
        ]
        mock_encoder.fit_transform.assert_called_once()
        # Check if the first arg to fit_transform is a dataframe with the right columns
        args, _ = mock_encoder.fit_transform.call_args
        pd.testing.assert_index_equal(args[0].columns, pd.Index(categorical_columns))
        
        # Verify the encoder was saved
        mock_file.assert_called_once_with(settings.ENCODER_PATH, 'wb')
        mock_pickle_dump.assert_called_once()
        
        # Check the result DataFrame
        assert 'target' in result.columns
        assert 'Numtppd' not in result.columns
        assert 'Numtpbi' not in result.columns
        assert 'Indtppd' not in result.columns
        assert 'Indtpbi' not in result.columns
        
        # Check target creation logic
        target_values = sample_data['Numtppd'].apply(lambda x: 1 if x != 0 else 0).values
        np.testing.assert_array_equal(result['target'].values, target_values)
        
        # Verify encoded feature columns are present
        for feature_name in mock_encoder.get_feature_names_out():
            assert feature_name in result.columns
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_preprocess_data_serving(self, mock_pickle_load, mock_file, mock_exists, sample_data, mock_encoder):
        """Test that preprocess_data_serving correctly transforms the data using the saved encoder."""
        # Set up the mocks
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_encoder
        
        # Call the function under test
        result = preprocess_data_serving(sample_data)
        
        # Verify encoder was loaded
        mock_exists.assert_called_once_with(settings.ENCODER_PATH)
        mock_file.assert_called_once_with(settings.ENCODER_PATH, 'rb')
        mock_pickle_load.assert_called_once()
        
        # Verify the encoder was used for transformation
        categorical_columns = [
            "CalYear", "Gender", "Type", "Category", "Occupation", 
            "SubGroup2", "Group2", "Group1"
        ]
        mock_encoder.transform.assert_called_once()
        # Check if the first arg to transform is a dataframe with the right columns
        args, _ = mock_encoder.transform.call_args
        pd.testing.assert_index_equal(args[0].columns, pd.Index(categorical_columns))
        
        # Check the result DataFrame
        assert 'Numtppd' not in result.columns
        assert 'Numtpbi' not in result.columns
        assert 'Indtppd' not in result.columns
        assert 'Indtpbi' not in result.columns
        
        # Verify encoded feature columns are present
        for feature_name in mock_encoder.get_feature_names_out():
            assert feature_name in result.columns
    
    @patch('os.path.exists')
    def test_preprocess_data_serving_missing_encoder(self, mock_exists, sample_data):
        """Test that preprocess_data_serving raises FileNotFoundError when encoder is missing."""
        # Set up the mock to return False for encoder file
        mock_exists.return_value = False
        
        # Verify the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            preprocess_data_serving(sample_data)
        
        mock_exists.assert_called_once_with(settings.ENCODER_PATH)