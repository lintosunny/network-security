import os 
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class NetworkModel:
    def __init__ (self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, x):
        try:
            logging.info("Starting prediction process for %s", x)
            x_transformed = self.preprocessor.transform(x)
            logging.info("Data transformation complete %s -> %s", x, x_transformed)
            y_hat = self.model.predict(x_transformed)
            logging.info("Prediction process complete %s -> %s", x_transformed, y_hat)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)