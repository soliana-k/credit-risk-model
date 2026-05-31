import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class Preprocessing:
#     def __init__(self, df:pd.DataFrame):
#         self.df=df
        

#     def transaction_aggregate_features(self):
#         self.df['TotalTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
#         self.df['AverageTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
#         self.df['TransactionCount'] = self.df.groupby('CustomerId')['Amount'].transform('count')
#         self.df['TransactionVariability'] = self.df.groupby('CustomerId')['Amount'].transform('std')
#         logger.info('Successfully engineered aggregate features!')
#         return self.df
    
#     def time_features(self):
#         try:
#             time_col = pd.to_datetime(self.df['TransactionStartTime'])
#             self.df['TransactionHour']=time_col.dt.hour
#             self.df['TransactionDay']=time_col.dt.day
#             self.df['TransactionMonth']=time_col.dt.month
#             self.df['TransactionYear']=time_col.dt.year
#             logger.info('successfully completed')
#             return self.df

#         except Exception as e:
#             logger.warning(f"Unexpected processing error: {e}")
#             raise

#     def category_encoding(self):
#         try:

#             pass
#         except Exception as e:
#             logger.warning(f'the exception {e}')
#             raise




import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE
from sklearn.pipeline import Pipeline

class TransactionAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TotalTransactionAmount'] = X.groupby('CustomerId')['Amount'].transform('sum')
        X['AverageTransactionAmount'] = X.groupby('CustomerId')['Amount'].transform('mean')
        X['TransactionCount'] = X.groupby('CustomerId')['Amount'].transform('count')
        X['TransactionVariability'] = X.groupby('CustomerId')['Amount'].transform('std')
        return X

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        time_col = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = time_col.dt.hour
        X['TransactionDay'] = time_col.dt.day
        X['TransactionMonth'] = time_col.dt.month
        X['TransactionYear'] = time_col.dt.year
        return X
    
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()
        encoded_data = self.encoder.transform(X[self.columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.columns))
        X = X.drop(columns=self.columns).reset_index(drop=True)
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
    
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols, num_strategy='median', cat_strategy='most_frequent'):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_si = SimpleImputer(strategy=self.num_strategy)
        self.cat_si = SimpleImputer(strategy=self.cat_strategy)

    def fit(self, X, y=None):
        if self.num_cols:
            self.num_si.fit(X[self.num_cols])
        if self.cat_cols:
            self.cat_si.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.num_cols:
            X[self.num_cols] = self.num_si.transform(X[self.num_cols])
        if self.cat_cols:
            X[self.cat_cols] = self.cat_si.transform(X[self.cat_cols])
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
    def __init__(self, columns=None):
        self.columns = columns
        self.woe = WOE()

    def fit(self, X, y):
        existing_cols = [c for c in self.columns if c in X.columns]
        X_subset = X[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.woe.fit(X_subset, y)
        return self

    def transform(self, X):
        existing_cols = [c for c in self.columns if c in X.columns]
        X_subset = X[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return self.woe.transform(X_subset)

columns=[
       'CurrencyCode',
       'ProductCategory', 'ChannelId', 'Amount', 'Value',
       'TransactionStartTime', 'PricingStrategy',
       'TotalTransactionAmount', 'AverageTransactionAmount',
       'TransactionCount', 'TransactionVariability', 'TransactionHour',
       'TransactionDay', 'TransactionMonth', 'TransactionYear']

num_features=['AverageTransactionAmount', 'TotalTransactionAmount', 'Amount', 'Value', 'TransactionCount','TransactionVariability',
               'PricingStrategy', ]
cat_features=['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
       'ProductCategory', 'ChannelId' ]

woe_features = [
    'Amount', 'Value', 'PricingStrategy', 
    'TotalTransactionAmount', 'AverageTransactionAmount',
    'TransactionCount', 'TransactionVariability', 'TransactionHour',
    'TransactionDay', 'TransactionMonth', 'TransactionYear'
]

pipeline = Pipeline([
    ('dropper', ColumnDropper(threshold=0.7)),
    ('aggregator', TransactionAggregator()),
    ('time_extractor', TimeFeatureExtractor()),
    ('imputer', MissingValueImputer(num_cols=num_features, cat_cols=cat_features)),
    ('encoder', Encoder(columns=['ProductCategory', 'ChannelId'])), # Run this BEFORE WoE
    ('processor', FeatureScaler(num_cols=num_features, scaler=StandardScaler())),
    ('woe', WoETransformer(columns=woe_features)) # Now X is purely numeric
])