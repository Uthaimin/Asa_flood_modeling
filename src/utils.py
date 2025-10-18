
import os
import sys
import pandas as pd
from dataclasses import dataclass
# train_test_split is removed as we'll use time-based slicing
# from sklearn.model_selection import train_test_split 

# Assuming these modules exist in your project structure
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # NOTE: Assuming 'stud.csv' now contains the weather/rainfall data
            df = pd.read_csv('ilorin_climate_forecast.csv')
            logging.info('Read the dataset as dataframe')

            # --- START OF TIME-SERIES SPLIT LOGIC ---
            
            # 1. Convert the 'Date' column to datetime objects
            # CRITICAL STEP for time-based splitting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                raise CustomException("DataFrame must contain a 'Date' column for time-series split.", sys)
            
            # 2. Sort the DataFrame by 'Date' (ensuring earlier dates come first)
            df = df.sort_values(by='Date').reset_index(drop=True)
            logging.info("DataFrame sorted by Date.")

            # 3. Calculate the split index for 80/20 division
            split_ratio = 0.8
            split_index = int(len(df) * split_ratio)

            # 4. Split the data based on the calculated index
            # Train set = 0.8 percentile (earlier dates)
            train_set = df.iloc[:split_index]
            # Test set = remaining 0.2 percentile (later dates)
            test_set = df.iloc[split_index:]
            
            logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
            logging.info("Train test split completed based on Date (Time-Series Split).")
            
            # --- END OF TIME-SERIES SPLIT LOGIC ---

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # NOTE: Ensure DataTransformation uses the 'Date' column only for internal reference
    # and drops it before transformation, as implemented in the previous response.
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    # The initiate_model_trainer function from your original code is called here
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))