# Creating Data Transformation file code
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Create a config function
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config= DataTransformationConfig()  #initialize the data transformer config
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation 
        """
        try:
            numerical_columns=["writing_score","reading_score"]   # numerical features
            categorical_columns=[
                "gender",
                "race_ethinicity",
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]                                                     # categorical features

            # Create Pipeline ()
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info('Numerical columns Standard scaling completed')
            logging.info(f"Numerical columns: {numerical_columns}")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical columns encoding completed')
            logging.info(f"Categorical columns: {categorical_columns}")
            

            # Club categorical encoding and numerical scaling using ColumnTransformer    
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline, categorical_columns)
                ]
                    
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
# Initiate data tranformation blog to perform actual transformation steps to read data, generate scaled/encode 
# feature, target variable, test train datasets.
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Reading train - test data set completed')
            
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()  # initaite the tranformer object

            target_column_name='math_score'
            numerical_columns=['writing_score', 'reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Transform the features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target variables into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved Preprocessing Object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,    
            )

        except Exception as e:
            raise CustomException(e,sys)