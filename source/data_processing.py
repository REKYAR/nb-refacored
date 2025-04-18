import io
import os
import pickle

import pandas as pd
import rdata
import requests
from sklearn.preprocessing import OneHotEncoder

from refactored.source.settings import settings


def download_data(url: str, file_path: str) -> None:
    """
    Download the data from the given URL and save it to the specified file path.
    """
    if not url:
        url = settings.FILE_URL
    if not file_path:
        file_path = settings.FILE_PATH
    response = requests.get(url)

    f = io.BytesIO(response.content)
    r_data = rdata.read_rda(f)["pg15training"]
    r_data.to_csv(file_path, index=False)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the given file path.
    """
    if not file_path:
        file_path = settings.FILE_PATH
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by dropping unnecessary columns and encoding categorical features.

    Args:
        data: Input DataFrame
        encoder_path: Path to save the fitted OneHotEncoder
    """
    data["target"] = data["Numtppd"].apply(lambda x: 1 if x != 0 else 0)
    data = data.drop(columns=["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])
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

    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_columns])

    with open(settings.ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
    non_categorical = [col for col in data.columns if col not in categorical_columns]
    result = pd.concat([data[non_categorical], encoded_df], axis=1)

    return result


def preprocess_data_serving(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for serving using the same encoder as used during training.

    Args:
        data: Input DataFrame
        encoder_path: Path to the saved OneHotEncoder
    """
    data = data.drop(
        columns=["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"], errors="ignore"
    )
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

    if not os.path.exists(settings.ENCODER_PATH):
        raise FileNotFoundError(
            f"Encoder not found at {settings.ENCODER_PATH}. Run training first."
        )

    with open(settings.ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    encoded_features = encoder.transform(data[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
    non_categorical = [col for col in data.columns if col not in categorical_columns]
    result = pd.concat([data[non_categorical], encoded_df], axis=1)

    return result
