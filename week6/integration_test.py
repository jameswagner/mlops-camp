# integration_test.py (complete version for Q6)

import pandas as pd
from datetime import datetime
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# Set environment variables
os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'
os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

# Create test data (same as Q5)
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

options = {
    'client_kwargs': {
        'endpoint_url': os.getenv('S3_ENDPOINT_URL')
    }
}

# Save test data
input_file = get_input_path(2023, 1)
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

print("✅ Test data saved!")

# Run batch.py for January 2023
print("🚀 Running batch.py...")
os.system('python batch.py 2023 1')

# Read results and check sum
print("📊 Reading results...")
output_file = get_output_path(2023, 1)
df_output = pd.read_parquet(output_file, storage_options=options)

sum_predictions = df_output['predicted_duration'].sum()
print(f"Sum of predicted durations: {sum_predictions}")