
from sklearn.model_selection import train_test_split
from refactored.source.trainer import generate_and_save_report, train_model
from settings import settings
import random
import numpy as np
from refactored.source.settings import settings
from refactored.source.data_processing import download_data, load_data, preprocess_data

random.seed(42)
np.random.seed(42)

def main():
    download_data(settings.FILE_URL, settings.FILE_PATH)

    data = load_data(settings.FILE_PATH)
    data = preprocess_data(data)
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = train_model(X_train, y_train, settings.MODEL_PATH)
    generate_and_save_report(best_model, X_test, y_test)

if __name__ == "__main__":
    main()