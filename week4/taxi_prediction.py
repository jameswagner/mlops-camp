#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import argparse

print("Loading model...")

# Load the pre-trained model and vectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

print("Model loaded successfully!")

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    """Read and preprocess taxi data"""
    print(f"Reading data from: {filename}")
    df = pd.read_parquet(filename)
    
    # Calculate trip duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter for reasonable trip durations (1-60 minutes)
    print(f"Original data shape: {df.shape}")
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    print(f"After filtering: {df.shape}")

    # Handle missing values and convert to string
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict_duration(df, dv, model):
    """Make predictions on the dataframe"""
    print("Making predictions...")
    
    # Transform features
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    # Predict
    y_pred = model.predict(X_val)
    
    return y_pred

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict taxi trip durations')
    parser.add_argument('--year', type=int, default=2023, help='Year (default: 2023)')
    parser.add_argument('--month', type=int, default=3, help='Month (default: 3)')
    
    args = parser.parse_args()
    
    year = args.year
    month = args.month
    
    # Load data for specified year/month
    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"Processing data for {year}-{month:02d}")
    df = read_data(data_url)
    
    # Make predictions
    y_pred = predict_duration(df, dv, model)
    
    # QUESTION 1: What's the standard deviation of the predicted duration?
    std_deviation = np.std(y_pred)
    mean_duration = np.mean(y_pred)
    
    print(f"\nQUESTION 1 ANSWER:")
    print(f"Standard deviation of predicted duration: {std_deviation:.2f}")
    print(f"Mean predicted duration: {mean_duration:.2f}")
    
    # QUESTION 2: Prepare output and save as parquet
    print(f"\nPreparing output dataframe...")
    
    # Create artificial ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    # Create results dataframe with only ride_id and predictions
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    print(f"Result dataframe shape: {df_result.shape}")
    print(f"Result dataframe columns: {df_result.columns.tolist()}")
    
    # Save as parquet with specified settings
    output_file = f'taxi_predictions_{year:04d}_{month:02d}.parquet'
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    # QUESTION 2: What's the size of the output file?
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)  # Convert to MB
    
    print(f"\nQUESTION 2 ANSWER:")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size_mb:.1f}M ({file_size_bytes:,} bytes)")
    
    # Show sample of results
    print(f"\nSample results:")
    print(df_result.head())
    
    print(f"\nSUMMARY:")
    print(f"- Processed {len(df):,} rides from {year}-{month:02d}")
    print(f"- Mean predicted duration: {mean_duration:.2f}")
    print(f"- Standard deviation of predictions: {std_deviation:.2f}")
    print(f"- Output file size: {file_size_mb:.1f}M")
    print(f"- Results saved to: {output_file}")

if __name__ == "__main__":
    main()



