import os
import sys
import numpy as Np
import pandas as Pd
from src.exception import CustomException
import dill
from src.logger import logging

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open (file_path, "wb") as file_object:
            dill.dump(object, file_object)

        logging.info("PKL Object Saved Successfully!")

    except Exception as e:
        raise CustomException(e, sys)