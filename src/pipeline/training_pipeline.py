import sys
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


@dataclass
class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def train(self):
        """
        The train function is the main function that will be called by the user. 
        It calls all other functions in order to train a model and return its score.
        
        :return: The score, which is the accuracy of the model
        """
        try: 
            train_data, test_data = self.data_ingestion.initiate_data_ingestion()

            train_arr, test_arr, preprocessor_obj = self.data_transformation.initiate_data_transformation(train_data, test_data)

            score = self.model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_obj)

            print("Training score is: ", score)

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == '__main__':

    training_pipeline = TrainingPipeline()
    training_pipeline.train()
