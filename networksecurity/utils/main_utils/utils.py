import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
import pickle 

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        with open(file_path, 'rb') as file:
            content = yaml.safe_load(file)
        logging.info("YAML file %s read successfully", file_path)
        return content
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
        logging.info("YAML file %s written successfully", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Saves a numpy array to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
        logging.info("Numpy array saved successfully at %s", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a numpy array from a file.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
        with open(file_path, 'rb') as file:
            array = np.load(file)
        logging.info("Numpy array loaded successfully from %s", file_path)
        return array
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.
    """
    try:
        logging.info("Saving object to file %s", file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info("Object saved successfully at %s", file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.
    """
    try:
        logging.info("Loading object from file %s", file_path)
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist.")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info("Object loaded successfully from %s", file_path)
        return obj
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
    Evaluates multiple machine learning models and returns their performance scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)