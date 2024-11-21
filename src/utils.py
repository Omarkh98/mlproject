import os
import sys
import numpy as Np
import pandas as Pd
from src.exception import CustomException
from src.components import model_training
import dill
from src.logger import logging
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
)
from sklearn.model_selection import GridSearchCV

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open (file_path, "wb") as file_object:
            dill.dump(object, file_object)

        logging.info("PKL Object Saved Successfully!")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for m in range(len(list(models))):
            model = list(models.values())[m]
            parameters = params[list(models.keys())[m]]
            
            logging.info("Tuning HyperParameters...")
            grid_search = GridSearchCV(model, parameters, cv = 3)
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_train_prediction = model.predict(X_train)
            y_test_prediction = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_prediction)
            test_model_score = r2_score(y_test, y_test_prediction)

            # model_train_mae, model_train_mse, model_train_r2

            report[list(models.keys())[m]] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e, sys)
    
def best_model_score(model_report: dict):
    try:
        logging.info("Sorting Model Report In DESC Order...")
        best_score = max(sorted(model_report.values()))
        if best_score < 0.6:
            raise CustomException("No Best Model Found...")
        
        logging.info("Getting Best Model Name From Dictionary...")
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_score)
        ]
        best_model = model_training.MODELS[best_model_name]

        return (
            best_score,
            best_model
        )
    
    except Exception as e:
        raise CustomException(e, sys)