"""
data_ingestion.py
"""

import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """
        The initiate_data_ingestion function is responsible for ingesting the data from a source.
        The function reads the dataset as a pandas dataframe, splits it into train and test sets, 
        and saves them in csv format.

        :return: The path of the train and test data
        """

        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/abhijitpaul0212/DataSets/main/StudentsPerformance.csv")
            logging.info("Read the dataset as dataframe: \n{}".format(df.head()))

            # Create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving dataframe in csv format
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=41)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e
