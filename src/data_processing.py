import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans




class RFMClusterer(BaseEstimator, TransformerMixin):
    """Task 4: Creates is_high_risk target using RFM + KMeans clustering"""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.high_risk_cluster = None

    def _calculate_rfm(self, X):
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': 'max',
            'TransactionId': 'count',
            'Amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        
        snapshot_date = pd.to_datetime(X['TransactionStartTime']).max()
        rfm['Recency'] = (snapshot_date - pd.to_datetime(rfm['Recency'])).dt.days
        return rfm

    def fit(self, X, y=None):
        rfm = self._calculate_rfm(X)
    
        rfm_scaled = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        self.kmeans.fit(rfm_scaled)
        centers = pd.DataFrame(self.kmeans.cluster_centers_, 
                               columns=['Recency', 'Frequency', 'Monetary'])

        self.high_risk_cluster = centers['Frequency'].idxmin()
        
        return self

    def transform(self, X):
        X = X.copy()
        rfm = self._calculate_rfm(X)
        rfm_scaled = self.scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        clusters = self.kmeans.predict(rfm_scaled)
        rfm['is_high_risk'] = (clusters == self.high_risk_cluster).astype(int)
        return X.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

class Aggregator(BaseEstimator, TransformerMixin):
    """Task 3: Aggregate Features"""
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TotalTransactionAmount'] = X.groupby('CustomerId')['Amount'].transform('sum')
        X['AverageTransactionAmount'] = X.groupby('CustomerId')['Amount'].transform('mean')
        X['TransactionCount'] = X.groupby('CustomerId')['Amount'].transform('count')
        X['TransactionVariability'] = X.groupby('CustomerId')['Amount'].transform('std')
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Task 3: Extract datetime features"""
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = dt.dt.hour
        X['TransactionDay'] = dt.dt.day
        X['TransactionMonth'] = dt.dt.month
        X['TransactionYear'] = dt.dt.year
        X = X.drop(columns=['TransactionStartTime'], errors='ignore')
        return X


class Encoder(BaseEstimator, TransformerMixin):
    """Task 3: One-Hot Encoding"""
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        encoded = self.encoder.transform(X[self.columns])
        encoded_df = pd.DataFrame(encoded, 
                                  columns=self.encoder.get_feature_names_out(self.columns), 
                                  index=X.index)
        X = X.drop(columns=self.columns)
        return pd.concat([X, encoded_df], axis=1)

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.dropped_cols = []

    def fit(self, X, y=None):
        missing_pct = X.isnull().mean()
        self.dropped_cols = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.dropped_cols)
    
class Imputer(BaseEstimator, TransformerMixin):
    """Task 3: Missing Value Handling"""
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_imp = SimpleImputer(strategy='median')
        self.cat_imp = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        if self.num_cols: self.num_imp.fit(X[self.num_cols])
        if self.cat_cols: self.cat_imp.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.num_cols: X[self.num_cols] = self.num_imp.transform(X[self.num_cols])
        if self.cat_cols: X[self.cat_cols] = self.cat_imp.transform(X[self.cat_cols])
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, scaler=StandardScaler()):
        self.num_cols = num_cols
        self.scaler = scaler  

    def fit(self, X, y=None):
        if self.num_cols:
            self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.num_cols:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X


class WoETransformer(BaseEstimator, TransformerMixin):
    """Task 3: WoE Transformation + IV Calculation"""
    def __init__(self, columns):
        self.columns = columns
        self.woe_maps = {}
        self.iv_values = {}

    def fit(self, X, y=None):
        y = pd.Series(y)
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()
        
        for col in self.columns:
            if col not in X.columns: continue
            data = pd.DataFrame({'val': X[col], 'target': y})
            data['bin'] = pd.qcut(data['val'], q=10, duplicates='drop')
            group = data.groupby('bin')['target'].agg(['sum', 'count'])
            group['bad'] = group['sum']
            group['good'] = group['count'] - group['bad']
            group['woe'] = np.log(((group['good'] + 0.5) / total_good) / 
                                  ((group['bad'] + 0.5) / total_bad))
            
            group['iv'] = ((group['good'] / total_good) - (group['bad'] / total_bad)) * group['woe']
            
            self.woe_maps[col] = group[['woe']]
            self.iv_values[col] = group['iv'].sum()
            
        print(f"Feature IVs: {self.iv_values}") 
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in self.woe_maps:
                bins = pd.qcut(X[col], q=10, duplicates='drop')
                X[col] = bins.map(self.woe_maps[col]['woe'])
        return X


def get_full_pipeline():
    """Returns a fitted pipeline that produces model-ready data"""
    
    num_cols = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount',
                'TransactionCount', 'TransactionVariability', 'TransactionHour',
                'TransactionDay', 'TransactionMonth', 'TransactionYear']

    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    woe_cols = num_cols + ['PricingStrategy']

    pipeline = Pipeline([
        ('rfm_cluster', RFMClusterer(n_clusters=3, random_state=42)),   # Task 4
        ('aggregator', Aggregator()),                                   
        ('time_features', TimeFeatures()),                              
        ('imputer', Imputer(num_cols=num_cols, cat_cols=cat_cols)),     
        ('encoder', Encoder(columns=cat_cols)),                         
        ('scaler', FeatureScaler(num_cols=num_cols, scaler=StandardScaler())), 
        ('woe_transformer', WoETransformer(columns=woe_cols))                         
    ])

    return pipeline, woe_cols


def process_data(df: pd.DataFrame):
    """Main function: Returns model-ready X and y"""
    pipeline, woe_cols = get_full_pipeline()
    rfm_preprocessor = RFMClusterer()
    X_rfm = rfm_preprocessor.fit_transform(df)
    y = X_rfm['is_high_risk']
    X_features = X_rfm.drop(columns=['is_high_risk'])
    X_final = pipeline.fit_transform(X_features, y)
    
    return X_final, y