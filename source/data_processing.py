import io
import pandas as pd
import rdata
import requests
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
    r_data = rdata.read_rda(f)['pg15training']
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
    Preprocess the data by dropping unnecessary columns and renaming others.
    """
    data['target'] = data['Numtppd'].apply(lambda x: 1 if x != 0 else 0)
    data = data.drop(columns=["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])
    categorical_columns = ['CalYear', 'Gender', 'Type', 'Category', 'Occupation', 'SubGroup2', 'Group2', 'Group1']
    data = pd.get_dummies(data, columns=categorical_columns)
    return data