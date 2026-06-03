import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from typing import List, Optional
import os
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


NUM_COLS = [
    'Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount',
    'TransactionCount', 'TransactionVariability', 'TransactionHour',
    'TransactionDay', 'TransactionMonth', 'TransactionYear'
]

CAT_COLS = [
    'ProductCategory', 'ChannelId', 'PricingStrategy',
    'CurrencyCode', 'ProviderId', 'ProductId'
]

WOE_COLS = NUM_COLS + ['PricingStrategy']

HIGH_RISK_N_CLUSTERS = 3
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.70  


#CUSTOM TRANSFORMERS 

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns with high percentage of missing values"""
    
    def __init__(self, threshold: float = MISSING_THRESHOLD):
        self.threshold = threshold
        self.dropped_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        missing_pct = X.isnull().mean()
        self.dropped_cols = missing_pct[missing_pct > self.threshold].index.tolist()
        
        if self.dropped_cols:
            logger.info(f"Dropping columns due to high missingness: {self.dropped_cols}")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.dropped_cols:
            X = X.drop(columns=self.dropped_cols)
        return X


class RFMClusterer(BaseEstimator, TransformerMixin):
    """Task 4: Creates is_high_risk target using RFM + KMeans"""
    
    def __init__(self, n_clusters: int = HIGH_RISK_N_CLUSTERS, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.high_risk_cluster: Optional[int] = None

    def _calculate_rfm(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            rfm = X.groupby('CustomerId').agg({
                'TransactionStartTime': 'max',
                'TransactionId': 'count',
                'Amount': 'sum'
            }).reset_index()
            
            rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
            
            snapshot_date = pd.to_datetime(X['TransactionStartTime']).max()
            rfm['Recency'] = (snapshot_date - pd.to_datetime(rfm['Recency'])).dt.days
            return rfm
        except Exception as e:
            logger.error(f"Error calculating RFM: {e}")
            raise

    def fit(self, X: pd.DataFrame, y=None):
        try:
            rfm = self._calculate_rfm(X)
            rfm_scaled = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
            
            self.kmeans.fit(rfm_scaled)
            
            centers = pd.DataFrame(self.kmeans.cluster_centers_, 
                                   columns=['Recency', 'Frequency', 'Monetary'])
            self.high_risk_cluster = centers['Frequency'].idxmin()
            logger.info(f"High-risk cluster identified: {self.high_risk_cluster}")
            return self
        except Exception as e:
            logger.error(f"Error during RFMClusterer fit: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            rfm = self._calculate_rfm(X)
            rfm_scaled = self.scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
            
            clusters = self.kmeans.predict(rfm_scaled)
            rfm['is_high_risk'] = (clusters == self.high_risk_cluster).astype(int)
            
            return X.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        except Exception as e:
            logger.error(f"Error during RFMClusterer transform: {e}")
            raise


class Aggregator(BaseEstimator, TransformerMixin):
    """Task 3: Aggregate customer-level features"""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            group = 'CustomerId'
            
            X['TotalTransactionAmount'] = X.groupby(group)['Amount'].transform('sum')
            X['AverageTransactionAmount'] = X.groupby(group)['Amount'].transform('mean')
            X['TransactionCount'] = X.groupby(group)['Amount'].transform('count')
            X['TransactionVariability'] = X.groupby(group)['Amount'].transform('std')
            X['Amount_vs_Avg'] = X['Amount'] / X['Amount'].mean()
            
            return X
        except Exception as e:
            logger.error(f"Error in Aggregator: {e}")
            raise


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Task 3: Extract datetime features"""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            dt = pd.to_datetime(X['TransactionStartTime'])
            
            X['TransactionHour'] = dt.dt.hour
            X['TransactionDay'] = dt.dt.day
            X['TransactionMonth'] = dt.dt.month
            X['TransactionYear'] = dt.dt.year
            
            return X.drop(columns=['TransactionStartTime'], errors='ignore')
        except Exception as e:
            logger.error(f"Error in TimeFeatures: {e}")
            raise


class Encoder(BaseEstimator, TransformerMixin):
    """Task 3: One-Hot Encoding"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')

    def fit(self, X: pd.DataFrame, y=None):
        try:
            self.encoder.fit(X[self.columns])
            return self
        except Exception as e:
            logger.error(f"Error fitting Encoder: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            encoded = self.encoder.transform(X[self.columns])
            encoded_df = pd.DataFrame(encoded, 
                                      columns=self.encoder.get_feature_names_out(self.columns),
                                      index=X.index)
            X = X.drop(columns=self.columns)
            return pd.concat([X, encoded_df], axis=1)
        except Exception as e:
            logger.error(f"Error in Encoder transform: {e}")
            raise


class Imputer(BaseEstimator, TransformerMixin):
    """Task 3: Handle missing values"""
    
    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_imp = SimpleImputer(strategy='median')
        self.cat_imp = SimpleImputer(strategy='most_frequent')

    def fit(self, X: pd.DataFrame, y=None):
        try:
            if self.num_cols:
                self.num_imp.fit(X[self.num_cols])
            if self.cat_cols:
                self.cat_imp.fit(X[self.cat_cols])
            return self
        except Exception as e:
            logger.error(f"Error fitting Imputer: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            if self.num_cols:
                X[self.num_cols] = self.num_imp.transform(X[self.num_cols])
            if self.cat_cols:
                X[self.cat_cols] = self.cat_imp.transform(X[self.cat_cols])
            return X
        except Exception as e:
            logger.error(f"Error in Imputer transform: {e}")
            raise


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Task 3: Scale numerical features"""
    
    def __init__(self, num_cols: List[str]):
        self.num_cols = num_cols
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        try:
            if self.num_cols:
                self.scaler.fit(X[self.num_cols])
            return self
        except Exception as e:
            logger.error(f"Error fitting FeatureScaler: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            if self.num_cols:
                X[self.num_cols] = self.scaler.transform(X[self.num_cols])
            return X
        except Exception as e:
            logger.error(f"Error in FeatureScaler transform: {e}")
            raise


class WoETransformer(BaseEstimator, TransformerMixin):
    """Task 3: Weight of Evidence Transformation"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.woe_maps = {}

    def fit(self, X: pd.DataFrame, y=None):
        try:
            y = pd.Series(y)
            total_good = (y == 0).sum()
            total_bad = (y == 1).sum()

            for col in self.columns:
                if col not in X.columns:
                    continue
                data = pd.DataFrame({'val': X[col], 'target': y})
                data['bin'] = pd.qcut(data['val'], q=10, duplicates='drop')
                
                group = data.groupby('bin')['target'].agg(['sum', 'count'])
                group['good'] = group['count'] - group['sum']
                group['woe'] = np.log(
                    ((group['good'] + 0.5) / total_good) /
                    ((group['sum'] + 0.5) / total_bad)
                )
                self.woe_maps[col] = group['woe']
            logger.info(f"WoE fitted for {len(self.woe_maps)} features")
            return self
        except Exception as e:
            logger.error(f"Error in WoE fit: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = X.copy()
            for col in self.columns:
                if col in self.woe_maps:
                    bins = pd.qcut(X[col], q=10, duplicates='drop')
                    X[col] = bins.map(self.woe_maps[col])
            return X
        except Exception as e:
            logger.error(f"Error in WoE transform: {e}")
            raise


#PIPELINE 

def get_data_pipeline() -> Pipeline:
    """Build and return the full feature engineering pipeline"""
    
    pipeline = Pipeline([
        ('column_dropper', ColumnDropper(threshold=MISSING_THRESHOLD)),
        ('aggregator', Aggregator()),
        ('time_features', TimeFeatures()),
        ('imputer', Imputer(num_cols=NUM_COLS, cat_cols=CAT_COLS)),
        ('encoder', Encoder(columns=CAT_COLS)),
        ('scaler', FeatureScaler(num_cols=NUM_COLS)),
        ('woe', WoETransformer(columns=WOE_COLS))
    ])
    
    return pipeline



def process_data(df: pd.DataFrame):
    """
    Main entry point: Processes raw data into model-ready (X, y) pair.
    Includes comprehensive error handling and logging.
    """
    try:
        logger.info(f"Starting data processing. Input shape: {df.shape}")
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logger.info("Creating RFM-based high-risk target...")
        rfm_clusterer = RFMClusterer()
        df_with_target = rfm_clusterer.fit_transform(df)
        
        y = df_with_target['is_high_risk']
        X = df_with_target.drop(columns=['is_high_risk'], errors='ignore')
        
        logger.info("Applying feature engineering pipeline...")
        feature_pipeline = get_data_pipeline()
        X_final = feature_pipeline.fit_transform(X, y)
        os.makedirs('models', exist_ok=True)
        joblib.dump(feature_pipeline, 'models/preprocessor.pkl')
        logger.info(f"Processing completed successfully. Final shape: {X_final.shape}")
        
        return X_final, y
        
    except Exception as e:
        logger.error(f"Critical error in process_data: {e}", exc_info=True)
        raise