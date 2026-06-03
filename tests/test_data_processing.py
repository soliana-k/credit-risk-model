import pytest
import pandas as pd
from src.data_processing import get_data_pipeline, process_data

@pytest.fixture
def sample_data():
    """Create valid sample data with enough customers for KMeans"""
    dates = [
        '2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00',
        '2023-01-04 13:00:00', '2023-01-05 14:00:00', '2023-01-06 15:00:00',
        '2023-01-07 16:00:00', '2023-01-08 17:00:00', '2023-01-09 18:00:00',
        '2023-01-10 19:00:00'
    ]
    
    data = {
        'TransactionId': [f'T{i}' for i in range(1, 11)],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C5'],
        'TransactionStartTime': dates,
        'Amount': [1000, 2000, 1500, 3000, 1200, 1800, 2500, 900, 4000, 1100],
        'Value': [1000, 2000, 1500, 3000, 1200, 1800, 2500, 900, 4000, 1100],
        'ProductCategory': ['airtime', 'data', 'airtime', 'data', 'airtime', 
                           'utility', 'airtime', 'data', 'utility', 'airtime'],
        'ChannelId': ['Channel1', 'Channel2', 'Channel1', 'Channel2', 'Channel1',
                     'Channel3', 'Channel1', 'Channel2', 'Channel3', 'Channel1'],
        'PricingStrategy': [1, 2, 1, 2, 1, 3, 1, 2, 3, 1],
        'ProviderId': ['P1', 'P2', 'P1', 'P2', 'P1', 'P3', 'P1', 'P2', 'P3', 'P1'],
        'ProductId': ['PR1', 'PR2', 'PR1', 'PR2', 'PR1', 'PR3', 'PR1', 'PR2', 'PR3', 'PR1'],
        'CurrencyCode': ['UGX'] * 10,
    }
    return pd.DataFrame(data)


def test_get_data_pipeline_returns_pipeline():
    """Test 1: Verify pipeline is created correctly"""
    pipeline = get_data_pipeline()
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'transform')
    assert len(pipeline.steps) > 3, "Pipeline should contain multiple steps"


def test_process_data_returns_correct_shapes(sample_data):
    """Test 2: Check shapes and basic structure"""
    X, y = process_data(sample_data)
    
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    assert len(X) == len(y), "X and y must have same number of samples"
    assert X.shape[0] > 0, "Processed data should not be empty"
    assert 'is_high_risk' not in X.columns, "Target column should be removed from features"


def test_process_data_contains_key_features(sample_data):
    """Test 3: Check for important engineered features"""
    X, y = process_data(sample_data)
    
    key_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 
                   'TotalTransactionAmount', 'AverageTransactionAmount']
    
    for feature in key_features:
        found = (feature in X.columns) or any(feature in col for col in X.columns)
        assert found, f"Expected feature '{feature}' not found in processed data"