# tests/test_batch.py

import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import batch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch import prepare_data


def dt(hour, minute, second=0):
    """Helper function to create datetime objects"""
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    """Test the prepare_data function with sample data"""
    
    # Input data from homework
    data = [
        (None, None, dt(1, 1), dt(1, 10)),      # 9 minutes - VALID
        (1, 1, dt(1, 2), dt(1, 10)),            # 8 minutes - VALID  
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),   # 59 seconds = 0.98 min - INVALID (< 1 min)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),       # 60+ minutes - INVALID (> 60 min)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    # Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']
    
    # Call the function we're testing
    actual_df = prepare_data(df, categorical)
    
    # Define expected output
    # Only first 2 rows should remain after duration filtering
    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),    # None becomes '-1'
        ('1', '1', dt(1, 2), dt(1, 10), 8.0),      # 1 becomes '1'
    ]
    
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    
    # Reset index for comparison (prepare_data uses .copy() which preserves original index)
    actual_df = actual_df.reset_index(drop=True)
    expected_df = expected_df.reset_index(drop=True)
    
    # Assertions
    assert len(actual_df) == 2, f"Expected 2 rows, got {len(actual_df)}"
    
    # Check that all expected columns are present
    assert 'duration' in actual_df.columns, "Duration column missing"
    assert 'PULocationID' in actual_df.columns, "PULocationID column missing"
    assert 'DOLocationID' in actual_df.columns, "DOLocationID column missing"
    
    # Check data types
    assert actual_df['PULocationID'].dtype == 'object', "PULocationID should be string"
    assert actual_df['DOLocationID'].dtype == 'object', "DOLocationID should be string"
    
    # Check specific values
    assert actual_df.iloc[0]['PULocationID'] == '-1', "First row PULocationID should be '-1'"
    assert actual_df.iloc[0]['DOLocationID'] == '-1', "First row DOLocationID should be '-1'"
    assert actual_df.iloc[1]['PULocationID'] == '1', "Second row PULocationID should be '1'"
    assert actual_df.iloc[1]['DOLocationID'] == '1', "Second row DOLocationID should be '1'"
    
    # Check durations
    assert abs(actual_df.iloc[0]['duration'] - 9.0) < 0.01, "First row duration should be ~9 minutes"
    assert abs(actual_df.iloc[1]['duration'] - 8.0) < 0.01, "Second row duration should be ~8 minutes"
    
    # Check that duration is within valid range
    assert all(actual_df['duration'] >= 1), "All durations should be >= 1 minute"
    assert all(actual_df['duration'] <= 60), "All durations should be <= 60 minutes"
    
    print("✅ All tests passed!")
    print(f"✅ Expected 2 rows, got {len(actual_df)} rows")
    print(f"✅ Categorical columns converted to strings")
    print(f"✅ Duration filtering working correctly")


if __name__ == "__main__":
    test_prepare_data()