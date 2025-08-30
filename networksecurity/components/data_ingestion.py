import os
import sys
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from typing import List
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Data ingestion component responsible for:
        - Fetching data from MongoDB
        - Storing raw data in feature store
        - Splitting data into train/test sets
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion initialized with config: %s", data_ingestion_config.__dict__)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Export MongoDB collection as a Pandas DataFrame.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info("Connecting to MongoDB")
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            logging.info("Fetching data from database: %s, collection: %s", database_name, collection_name)
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
                logging.info("Dropped '_id' column from DataFrame.")

            df.replace({"na": np.nan}, inplace=True)
            logging.info("DataFrame created with shape: %s", df.shape)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_to_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Save the raw DataFrame to the feature store as a CSV.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Data exported to feature store: %s", feature_store_file_path)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split the data into train and test sets and save them as CSV files.
        """
        try:
            logging.info("Splitting data into train and test sets.")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(
                "Data split complete. Train shape: %s, Test shape: %s",
                train_set.shape,
                test_set.shape
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the entire data ingestion process.
        """
        try:
            logging.info("Starting data ingestion process.")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info("Data ingestion completed successfully.")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
