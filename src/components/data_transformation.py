import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Removed unused imports: OneHotEncoder (no categorical columns defined)

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    # Define the alpha value for the EWMA calculation (API_alpha)
    EWMA_ALPHA: float = 0.2 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function defines the preprocessing pipelines for the final numerical features.
        NOTE: The 'Date' column and 'API_alpha' must be handled/created outside this pipeline.
        '''
        try:
            # We only scale the actual measured data and the engineered EWMA feature.
            # 'Date' and 'API_alpha' (EWMA result) are NOT here; they are created/handled separately.
            numerical_columns_to_scale = [
                "Temperature_3PM_C",
                "DewPoint_3PM_C",
                "RH_3PM_%"
                # 'EWMA_Rainfall' (derived from API_alpha logic) will be created externally
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            
            # NOTE: Removed cat_pipeline as no categorical columns are defined.

            logging.info(f"Numerical columns to scale: {numerical_columns_to_scale}")

            # ColumnTransformer is set to ONLY apply the pipeline to the specified columns
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns_to_scale),
                ],
                remainder='drop' # Ensures 'Date', 'API_alpha', etc., are dropped if they weren't used.
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles time-series specific transformations (Date conversion and EWMA calculation).
        """
        try:
            logging.info("Starting time-series data preparation (Date and EWMA).")
            
            # 1. Convert 'Date' column to datetime and sort (CRITICAL for EWMA)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date').reset_index(drop=True)
            
            # 2. Calculate EWMA (Assuming the original column name is 'Rain_fall' for calculation)
            # This creates the engineered feature referred to by the 'API_alpha' column name.
            alpha = self.data_transformation_config.EWMA_ALPHA
            df['API_alpha'] = df['Rain_fall'].ewm(alpha=alpha, adjust=False).mean()
            
            logging.info(f"EWMA_Rainfall (API_alpha) calculated with alpha={alpha}.")
            
            return df
        except Exception as e:
            raise CustomException(e, sys)

            
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # --- PREPARE DATA BEFORE SCALING ---
            train_df = self.prepare_data(train_df)
            test_df = self.prepare_data(test_df)
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Run-off"
            
            # Define final input feature list
            final_input_features = [
                "Temperature_3PM_C", 
                "DewPoint_3PM_C",
                "RH_3PM_%",
                "API_alpha" # The new engineered feature
            ]

            input_feature_train_df=train_df[final_input_features].copy() # Select features to transform
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df[final_input_features].copy()
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)]

            # --- FIX: Ensure the 'artifacts' directory exists ---
            artifacts_dir = os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            # ----------------------------------------------------
            
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)