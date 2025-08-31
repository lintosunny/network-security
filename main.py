from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys
import os   

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
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

    data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transfromation = DataTransformation(data_transformation_config=data_transformation_config,
                                             data_validation_artifact=data_validation_artifact)
    data_transformation_artifacts=data_transfromation.initiate_data_transformation()
    print(data_transformation_artifacts)

    model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
    model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                 data_transformation_artifact=data_transformation_artifacts)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    print(model_trainer_artifact)