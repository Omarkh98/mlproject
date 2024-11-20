import sys
import os
from dataclasses import dataclass
import numpy as Np
import pandas as Pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

NUMERICAL_FEATURES = ["writing_score", "reading_score"]
CATEGORICAL_FEATURES = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
TARGET_FEATURE = "math_score"

@dataclass
class Data_Transformation_Config:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = Data_Transformation_Config()

    def get_transformer_object(self):
        try:
            logging.info(f"Numerical Features: {NUMERICAL_FEATURES}")
            logging.info(f"Categorical Features: {CATEGORICAL_FEATURES}")

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), # Handling Missing Values using - Median
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )
            logging.info("Numerical Features Standard Scaling Completed!")

            categorical_pipeline = Pipeline(
                steps = [
                    ("impute", SimpleImputer(strategy = "most_frequent")), # Handling Missing Values using - Most Frequent Value
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )
            logging.info("Categorical Features Encoding Completed!")

            preprocessor = ColumnTransformer ([ # Combine Pipelines
                ("numerical_pipeline", numerical_pipeline, NUMERICAL_FEATURES),
                ("categorical_pipeline", categorical_pipeline, CATEGORICAL_FEATURES), 
            ])
            logging.info("Column Transformer Completed!")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = Pd.read_csv(train_path)
            test_df = Pd.read_csv(test_path)
            logging.info("Train and Test Data Read Successfully!")

            logging.info("Obtaining Preprocessing Object...")
            preprocessing_obj = self.get_transformer_object()

            input_feature_train_df = train_df.drop(columns = [TARGET_FEATURE], axis = 1)
            target_feature_train_df = train_df[TARGET_FEATURE]

            input_feature_test_df = test_df.drop(columns = [TARGET_FEATURE], axis = 1)
            target_feature_test_df = test_df[TARGET_FEATURE]

            logging.info("Applying Preprocessing Object On Training & Testing DataFrames...")
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)#

            train_array = Np.c_[
                input_feature_train_array, Np.array(target_feature_train_df)
            ]
            test_array = Np.c_[
                input_feature_test_array, Np.array(target_feature_test_df)
            ]
            logging.info("Saved Processing Object!")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                object = preprocessing_obj
            )

            return (
                train_array, test_array, self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)