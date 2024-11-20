import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import src.utils

# Regression Models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
    )
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

# Evaluation Metrics
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
)

MODELS = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbours": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeRegressor(),
        "AdaBoost Classifier": AdaBoostRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Random Forest": RandomForestRegressor()
    }

@dataclass
class Model_Trainer_Config:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Spliting Training and Test Data...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model_report: dict = src.utils.evaluate_model(
                X_train = X_train, 
                y_train = y_train, 
                X_test = X_test, 
                y_test = y_test, 
                models = MODELS
                )
            
            model_score, model_name = best_model = src.utils.best_model_score(model_report)

            src.utils.save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                object = model_name
            )

            predicted = model_name.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)