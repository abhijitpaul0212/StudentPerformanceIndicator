import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    The save_object function saves an object to a file.
    
    :param file_path: Specify the path to where you want to save the object
    :param obj: Store the object that is being saved
    :return: None
    """
    try:
        logging.info("Object will be saved")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved at {}".format(file_path))
        
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    The evaluate_model function takes in the training and testing datasets, as well as a dictionary of models.
    It then fits each model to the training data, makes predictions on both the train and test sets, 
    and calculates an R2 score for each set. It returns a dictionary containing these scores.
    
    :param X_train: Train the model
    :param y_train: Train the model
    :param X_test: Pass the test data to the model for prediction
    :param y_test: Compare the predicted values with actual values
    :param models: Pass in a dictionary of models to be evaluated
    :return: A dictionary with the r2_score for each model
    """
    try:
        report = {}

        for i in range(len(list(models))):

            name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[name]

            # with hyper-parameters
            rs = RandomizedSearchCV(model, param, cv=5, n_iter=5, n_jobs=-1, verbose=1)
            rs.fit(X_train, y_train)
            logging.info("Best Score {} % is given by Model Param: {}".format(round(rs.best_score_ * 100, 2), rs.best_estimator_))
            model.set_params(**rs.best_params_)

            # without hyper-paramters
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_model_score = r2_score(y_true=y_train, y_pred=train_pred)
            logging.info("R2 score on Training dataset using {} model is {} %".format(name, round(train_model_score * 100, 2)))

            test_model_score = r2_score(y_true=y_test, y_pred=test_pred)
            logging.info("R2 score on Testing dataset using {} model is {} %".format(name, round(test_model_score * 100, 2)))

            report[name] = test_model_score

        logging.info("Model report: \n{}".format(report))
        return report

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    """
    The load_object function loads a pickled object from the file path provided.
    
    :param file_path: Specify the path to the file that will be loaded
    :return: Saved object
    :doc-author: Trelent
    """
    try:
        logging.info("Object will be loaded")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        logging.info("Object loaded from {}".format(file_path))
        
    except Exception as e:
        raise CustomException(e, sys) from e


def load_dataframe(path: str, filename: str):
    try:
        return pd.read_csv(Path(os.path.join(path, filename)))
    except Exception as e:
        raise CustomException(e, sys) from e
