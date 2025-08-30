from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import TraningPipelineConfig, DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys
import os   

if __name__ == "__main__":
    training_pipeline_config = TraningPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    print(data_ingestion_config.feature_store_file_path)
    print(data_ingestion_config.train_file_path)
    print(data_ingestion_config.test_file_path)
    
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    dataingestionartifact = data_ingestion.initiate_data_ingestion()
    print(dataingestionartifact)