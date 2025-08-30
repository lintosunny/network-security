from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import TraningPipelineConfig, DataIngestionConfig, DataValidationConfig
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

    data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = DataValidation(data_validation_config=data_validation_config,
                                     data_ingestion_artifact=dataingestionartifact)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)