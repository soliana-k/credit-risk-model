import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessing:
    def __init__(self, df:pd.DataFrame):
        self.df=df
        

    def transaction_aggregate_features(self):
        self.df['TotalTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
        self.df['AverageTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
        self.df['TransactionCount'] = self.df.groupby('CustomerId')['Amount'].transform('count')
        self.df['TransactionVariability'] = self.df.groupby('CustomerId')['Amount'].transform('std')
        logger.info('Successfully engineered aggregate features!')
        return self.df
    
    def time_features(self):
        try:
            time_col = pd.to_datetime(self.df['TransactionStartTime'])
            self.df['TransactionHour']=time_col.dt.hour
            self.df['TransactionDay']=time_col.dt.day
            self.df['TransactionMonth']=time_col.dt.month
            self.df['TransactionYear']=time_col.dt.year
            logger.info('successfully completed')
            return self.df

        except Exception as e:
            logger.warning(f"Unexpected processing error: {e}")
            raise

        


        
