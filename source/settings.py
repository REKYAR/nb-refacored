import os
import pathlib

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # data url
    FILE_URL: str = (
        "https://github.com/dutangc/CASdatasets/raw/refs/heads/master/data/pg15training.rda"
    )
    # data path
    FILE_PATH: str = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "data", "pg15training.csv"
    )
    # model path
    MODEL_DIR: str = "models/"
    # model name
    MODEL_NAME: str = "some_model.pkl"
    # encder name
    ENCODER_NAME: str = "encoder.pkl"
    # report path
    REPORT_PATH: str = "reports/"
    # final parameters path
    FINAL_PARAMS_PATH: str = "parameters/"
    # hyperparameter search space path
    HYPERPARAM_CONFIG_PATH: str = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "config", "hyperparams.json"
    )
    # blob stub path
    BLOB_STUB_PATH: str = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "blob_store_stub"
    )

    @property
    def MODEL_PATH(self) -> str:
        """
        Returns the full path to the model file.
        """
        return os.path.join(self.BLOB_STUB_PATH, self.MODEL_DIR, self.MODEL_NAME)

    @property
    def ENCODER_PATH(self) -> str:
        """
        Returns the full path to the encoder file.
        """
        return os.path.join(self.BLOB_STUB_PATH, self.MODEL_DIR, self.ENCODER_NAME)


settings = Settings()
