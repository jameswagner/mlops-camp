#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df, categorical):
    """
    Transforms the dataframe by:
    1. Calculating trip duration
    2. Filtering by duration (1-60 minutes)
    3. Converting categorical columns to strings
    """
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def read_data(filename, categorical):
    """
    Reads parquet file and applies transformations
    """
    # Check if we need to use custom S3 endpoint (for LocalStack)
    options = {}
    if os.getenv('S3_ENDPOINT_URL'):
        options = {
            'client_kwargs': {
                'endpoint_url': os.getenv('S3_ENDPOINT_URL')
            }
        }
    
    df = pd.read_parquet(filename, storage_options=options)
    return prepare_data(df, categorical)


def save_data(df, filename):
    """
    Saves dataframe to parquet file (supports LocalStack S3)
    """
    options = {}
    if os.getenv('S3_ENDPOINT_URL'):
        options = {
            'client_kwargs': {
                'endpoint_url': os.getenv('S3_ENDPOINT_URL')
            }
        }
    
    df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)


def load_model(model_path='model.bin'):
    """
    Loads the pickled model and vectorizer
    """
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr



def main(year, month):
    """
    Main batch prediction function
    """
    # Define categorical features
    categorical = ['PULocationID', 'DOLocationID']
    
    # Get input and output paths
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    
    # Load model
    dv, lr = load_model()
    
    # Read and prepare data
    df = read_data(input_file, categorical)
    
    # Create ride IDs
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    # Make predictions
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    print('predicted mean duration:', y_pred.mean())
    
    # Prepare results
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    # Save results
    save_data(df_result, output_file)
    
    return df_result


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)