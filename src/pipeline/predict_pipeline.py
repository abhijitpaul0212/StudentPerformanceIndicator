import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass


@dataclass
class PredictPipeline:

    def predict(self, features):
        """
        The predict function takes in a list of features and returns the predicted value.

        :param features: Pass the data to be predicted
        :return: A numpy array
        """
        try:
            logging.info('Prediction Pipeline initiated')
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # preprocessed and scaled data
            scaled_data = preprocessor.transform(features)

            pred = model.predict(scaled_data)
            logging.info('Predicted value: {}'.format(pred[0]))
            
            return pred

        except Exception as e:
            raise CustomException(e, sys) from e
