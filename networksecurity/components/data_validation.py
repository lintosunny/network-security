import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
        def __init__(self, data_validation_config: DataValidationConfig,
                     data_ingestion_artifact: DataIngestionArtifact):
            try:
                logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
                self.data_validation_config = data_validation_config
                self.data_ingestion_artifact = data_ingestion_artifact
                self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
                logging.info("DataValidation initialized with schema at: %s", SCHEMA_FILE_PATH)
            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
        @staticmethod
        def read_data(file_path) -> pd.DataFrame:
            try:
                logging.info("Reading dataset from: %s", file_path)
                return pd.read_csv(file_path)
            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
        def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
            try:
                expected_columns = len(self.schema_config['columns'])
                actual_columns = len(dataframe.columns)

                logging.info("Validating number of columns: expected=%s, actual=%s",
                         expected_columns, actual_columns)
                
                return expected_columns == actual_columns
            except Exception as e:
                raise NetworkSecurityException(e, sys)  
            
        def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
            try:
                schema_numerical_columns = self.schema_config['numerical_columns']
                dataframe_numerical_columns = dataframe.select_dtypes(
                    include=['int64', 'float64']
                ).columns.tolist()

                logging.info("Expected numerical columns: %s", schema_numerical_columns)
                logging.info("Found numerical columns: %s", dataframe_numerical_columns)

                for column in schema_numerical_columns:
                    if column not in dataframe_numerical_columns:
                        logging.error("Missing numerical column: %s", column)
                        return False
                return True
            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
        def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
            try:
                status = False
                drift_report = {}

                logging.info("Detecting data drift with KS-test (threshold=%s)", threshold)

                for column in base_df.columns:
                    d1 = base_df[column]
                    d2 = current_df[column]

                    is_statistically_similar = ks_2samp(d1, d2)
                    p_value = is_statistically_similar.pvalue

                    if p_value >= threshold:
                        is_found = False
                        logging.info("No drift in column '%s' (p-value=%s)", column, p_value)
                    else:
                        is_found = True
                        status = True
                        logging.warning("Data drift detected in column '%s' (p-value=%s)", column, p_value)
                    drift_report.update({column: {
                        "p_value": float(p_value),
                        "drift_status": is_found
                    }})

                drift_report_file_path = self.data_validation_config.drift_report_file_path
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)

                write_yaml_file(file_path=drift_report_file_path, content=drift_report)
                logging.info("Drift report saved at: %s", drift_report_file_path)
                return status

            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
        def initiate_data_validation(self) -> DataValidationArtifact:
             try:
                logging.info("Starting data validation process.")

                train_file_path = self.data_ingestion_artifact.train_file_path
                test_file_path = self.data_ingestion_artifact.test_file_path

                # reading the train and test data 
                train_dataframe = DataValidation.read_data(train_file_path)
                test_dataframe = DataValidation.read_data(test_file_path)

                # validate number of columns
                if not self.validate_number_of_columns(train_dataframe):
                    raise Exception("Train data does not contain all expected columns.")
                if not self.validate_number_of_columns(test_dataframe):
                    raise Exception("Test data does not contain all expected columns.")
                
                # validate numerical columns present in the train and test data
                if not self.validate_numerical_columns(train_dataframe):
                    raise Exception("Train data does not contain all numerical columns.")
                if not self.validate_numerical_columns(test_dataframe):
                    raise Exception("Test data does not contain all numerical columns.")
                
                # check data drift
                status = self.detect_data_drift(base_df=train_dataframe, current_df=test_dataframe)
                dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)

                if not status:
                    logging.info("Saving validated datasets.")
                    train_dataframe.to_csv(
                        self.data_validation_config.valid_train_file_path, index=False, header=True
                    )
                    test_dataframe.to_csv(
                        self.data_validation_config.valid_test_file_path, index=False, header=True
                    )
                else: 
                    raise Exception("Data drift found between train and test data.")
                
                data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

                logging.info("Data validation completed successfully.")
                return data_validation_artifact
    
             except Exception as e:
                raise NetworkSecurityException(e, sys)   