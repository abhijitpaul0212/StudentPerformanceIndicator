"""
data_transformation.py
"""

import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        The get_data_transformer_object function creates a ColumnTransformer object that will be used to transform the data.
        The function takes no arguments and returns a ColumnTransformer object.

        :return: The preprocessor object
        """
        try:
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            # Creating Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns: {}".format(num_features))
            logging.info("Categorical columns: {}".format(cat_features))

            # Create Transformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        The initiate_data_transformation function reads in the training and testing data, 
        obtains a preprocessing object, applies it to the training and testing dataframes, 
        and returns a tuple containing: (training array with features and target column; 
        testing array with features and target column; file path of saved preprocessor object). 
        
        :param train_path: Read the training data from a csv file
        :param test_path: Read the test data from a csv file
        :return: A tuple of three elements
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "math score"
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e
