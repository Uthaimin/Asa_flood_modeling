
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

    SPLIT_RATIO: float = 0.8 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        print("--- DATA INGESTION HAS STARTED! (Confirmation Print) ---") 
        logging.info("Entered the data ingestion method or component")
        logging.info("Entered the data ingestion method or component")
        try:
            # 1. READ DATA (Using the raw string fix from earlier)
            df = pd.read_csv(r'C:\Users\PC\Downloads\ML\Asa_flood_modeling\Ilorin_3PM_Weather_1990_Present.csv')
            logging.info("Successfully read the dataset as DataFrame")
            
            # 2. CREATE ARTIFACTS DIRECTORIES
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # 3. SAVE RAW DATA
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder.")

            # --- REMOVED: Date conversion and sorting. That belongs in DataTransformation. ---

            logging.info("Initiating time-based train/test split...")
            
            # CRITICAL: Sort the data by Date before splitting based on index, 
            # as Data Transformation will rely on the sorted order.
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date').reset_index(drop=True)

            # 4. PERFORM TIME-BASED SPLIT (using row count for exact 80% split)
            split_index = int(len(df) * self.ingestion_config.SPLIT_RATIO)

            train_set = df.iloc[:split_index]
            test_set = df.iloc[split_index:]

            logging.info(f"Train split: {len(train_set)} rows. Test split: {len(test_set)} rows.")

            # 5. SAVE SPLIT DATA
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        


