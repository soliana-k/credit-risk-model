import pytest
import pandas as pd
import numpy as np
from src.eda_pipeline import EDAPipeline, EDAConfig

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 100],
        'B': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Category': ['X', 'X', 'Y', 'Y', 'Z']
    })

def test_load_data_validates_input():
    pipeline = EDAPipeline()
    with pytest.raises(ValueError):
        pipeline.load_data(pd.DataFrame())

def test_pipeline_runs_without_error(sample_df):
    config = EDAConfig(sample_size=None,  
        save_plots=False,
        include_plots=False,
        verbose=False)
    pipeline = EDAPipeline(config=config)
    results = pipeline.load_data(sample_df).run()
    
    assert "summary" in results
    assert "outliers" in results
    assert results['overview']['rows'] == 5

def test_distribution_insight_logic():
    pipeline = EDAPipeline()
    insight = pipeline._get_distribution_insight(skew=2.0, kurt=0.0)
    assert "Highly right-skewed" in insight