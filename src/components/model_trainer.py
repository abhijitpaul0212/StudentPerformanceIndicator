import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException   
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        The initiate_model_trainer function is responsible for training the model.
            It takes in a train_array and test_array, which are numpy arrays containing the features and labels of both datasets.
            The function also takes in a preprocessor path, which is used to load the preprocessor object that was saved during 
            data preparation. This allows us to transform our data using the same parameters as we did during data preparation.
        
        :param train_array: Pass the training dataset to the model trainer
        :param test_array: Test the model
        :param preprocessor_path: Load the preprocessor object, which is used to transform the data
        :return: The r2_squared value
        """
        try:
            logging.info("Splitting training and test dataset")

            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:,-1])

            models = {
                    "LinearRegression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "SGDRegressor": SGDRegressor(),
                    "SVR": SVR(),
                    "KNeighborsRegressor": KNeighborsRegressor(),
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    "RandomForestRegressor": RandomForestRegressor(),
                    "AdaBoostRegressor": AdaBoostRegressor(),
                    "GradientBoostingRegressor": GradientBoostingRegressor(),
                    "CatBoostRegressor": CatBoostRegressor(verbose=False),
                    "XGBoostRegressor": XGBRegressor()
                }
            
            params = {
                "LinearRegression":{},
                "Lasso": {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,50,100]},
                "Ridge": {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,50,100]},
                "SGDRegressor": {
                    # 'loss': ['squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'],
                                # 'penalty': ['l1', 'l2'],
                                'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,50,100],
                                'learning_rate': ['adaptive', 'invscaling', 'optimal', 'constant']
                                },
                "SVR": {'kernel':['linear', 'rbf'], 
                         'C':[0.0001, 1, 10], 
                         'gamma':[1, 10, 100]},
                "KNeighborsRegressor": {},
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGBoostRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },                
                
            }
            
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            # best score
            best_score = max(sorted(model_report.values()))

            # best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model = models[best_model_name]

            logging.info("Best Model: {} | Best Score: {}".format(best_model_name, best_score))

            if best_score < 0.60:
                raise CustomException("No best model found", Exception)

            logging.info("Best model found on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predicted)

            return r2_squared

        except Exception as e:
            raise CustomException(e, sys) from e
