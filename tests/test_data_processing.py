import pytest
import pandas as pd
from src.data_processing import process_data

def test_process_data_output_shape():
    """Verify that process_data returns two objects (X and y) and they have data."""
    # Create a small dummy dataframe
    dummy_data = pd.DataFrame({
       'CustomerId': ['C1', 'C2', 'C3', 'C4'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'BatchId': ['B1', 'B2', 'B3', 'B4'],
        'AccountId': ['A1', 'A2', 'A3', 'A4'],
        'SubscriptionId': ['S1', 'S2', 'S3', 'S4'],
        'CountryCode': [256, 256, 256, 256],
        'Amount': [100.0, 200.0, 300.0, 400.0],
        'Value': [0.1, 0.2, 0.3, 0.4],
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z', 
            '2018-03-20T11:05:30Z', 
            '2018-04-20T11:05:30Z',
            '2018-05-20T11:05:30Z'
        ],
        'FraudResult': [0, 0, 0, 1]
    })
    
    X, y = process_data(dummy_data)
    
    assert len(X) == 4
    assert len(y) == 4
    assert 'is_high_risk' not in X.columns

def test_process_data_returns_numeric():
    """Verify that the output X contains only numeric data (as expected for ML)."""
    dummy_data = pd.DataFrame({
        'CustomerId': ['C1', 'C2'],
        'TransactionId': ['T1', 'T2'],
        'Amount': [100.0, 200.0],
        'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-03-20T11:05:30Z'],
        'FraudResult': [0, 1]
    })
    
    X, y = process_data(dummy_data)
    
    # Check that there are no object/string columns left
    assert X.select_dtypes(include=['object']).shape[1] == 0