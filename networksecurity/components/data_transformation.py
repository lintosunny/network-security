import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.entity.config_entity import (
    DataTransformationConfig,
    ModelPusherConfig
)
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                model_pusher_config: ModelPusherConfig,
                data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.model_pusher_config = model_pusher_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.
        """
        try:
            logging.info("Reading dataset from: %s", file_path)
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates a scikit-learn Pipeline for imputing missing values using KNNImputer.

        Returns:
            Pipeline: sklearn Pipeline with KNNImputer step.
        """
        try:
            logging.info("Creating data transformation pipeline with KNNImputer")
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("KNNImputer initialized with params: %s", DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor: Pipeline = Pipeline(steps=[('imputer', imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)  
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # Load datasets
            logging.info("Starting data transformation process")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info("Training and testing data loaded successfully")

            # Split features and target
            logging.info("Splitting features and target variable")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            input_feature_test_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = train_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_train_df.replace(-1,0)

            # Create transformation pipeline
            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training features.")
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            logging.info("Transforming training and testing features.")
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Combine features + target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed datasets
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Transformed training and testing data saved successfully")

            # save preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path, obj=preprocessor_object)
            logging.info("Preprocessing object saved at: %s", self.data_transformation_config.transformed_object_file_path)

            final_preprocessor_path = os.path.dirname(self.model_pusher_config.final_preprocessor_file_path)
            os.makedirs(final_preprocessor_path, exist_ok=True)
            save_object(file_path=self.model_pusher_config.final_model_file_path, obj=preprocessor_object)

            # prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data Transformation Artifact created successfully.")
            
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)