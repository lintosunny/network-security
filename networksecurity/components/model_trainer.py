import os
import sys
import joblib
import dagshub
import mlflow
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact, 
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import (
    ModelTrainerConfig,
    ModelPusherConfig
)
from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object,
    load_object,
    evaluate_models
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

load_dotenv(override=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:  
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classfication_metric):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("Network_Security")
        
        with mlflow.start_run():
            f1_score = classfication_metric.f1_score
            precision_score = classfication_metric.precision_score
            recall_score = classfication_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            logging.info("✅ Metrics saved successfully")

            joblib.dump(best_model, "model.joblib")
            mlflow.log_artifact("model.joblib", artifact_path="model")
            logging.info("✅ Model saved as artifact successfully")

            # Cleanup local files
            os.remove("model.joblib")

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Defining candidate models...")
            models = {
                "Logistic Regression": LogisticRegression(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }

            # Hyperparameter grid for tuning
            params={
                "Decision Tree": {
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['log_loss', 'exponential'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{
                    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    # 'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                "AdaBoost":{
                    # 'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            # Evaluate models
            logging.info("Evaluating models with cross-validation...")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params) 
            logging.info("Model evaluation report: %s", model_report)

            # Select the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info("Best model selected: %s with score: %.4f", best_model_name, best_model_score)

            # Evaluate on training set
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            logging.info(f"Classification metric on training dataset: {classification_train_metric}")

            # Track experiment with MLflow
            self.track_mlflow(best_model, classification_train_metric)

            # Evaluate on testing set
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Classification metric on testing dataset: {classification_test_metric}")

             # Track experiment with MLflow
            self.track_mlflow(best_model, classification_test_metric)
            
            # Load preprocessing object
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Save the trained model
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)
            logging.info(f"Trained model saved at path: {self.model_trainer_config.trained_model_file_path}")

            final_model_path = os.path.dirname(self.model_pusher_config.final_model_file_path)
            os.makedirs(final_model_path, exist_ok=True)
            save_object(file_path=self.model_pusher_config.final_model_file_path, obj=best_model)

            # Create and return artifact 
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:  
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training dataset")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading numpy array
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            logging.info("Splitting training and testing input and target feature")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Training the model")
            model_trainer_artifact = self.train_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
            return model_trainer_artifact 
            
        except Exception as e:  
            raise NetworkSecurityException(e, sys)