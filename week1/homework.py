#!/usr/bin/env python
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def download_data(url, output_file):
    """Download parquet file from URL"""
    import requests

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_file, 'wb') as f_out:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f_out.write(chunk)
    print(f"Downloaded to {output_file}")

def q1_download_data():
    """
    Q1. Downloading the data and count columns
    """
    # Define file paths
    jan_parquet = 'data/yellow_tripdata_2023-01.parquet'
    feb_parquet = 'data/yellow_tripdata_2023-02.parquet'

    # Download URLs
    jan_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet'
    feb_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet'

    # Download data
    try:
        download_data(jan_url, jan_parquet)
        download_data(feb_url, feb_parquet)
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

    # Read January data and count columns
    df_jan = pd.read_parquet(jan_parquet)
    num_columns = len(df_jan.columns)
    
    print(f"\nQ1 Answer: Number of columns in January data: {num_columns}")
    return df_jan

def q2_compute_duration(df_jan):
    """
    Q2. Computing duration standard deviation
    """
    # Convert to datetime if needed (should already be datetime in parquet)
    if not pd.api.types.is_datetime64_dtype(df_jan['tpep_pickup_datetime']):
        df_jan['tpep_pickup_datetime'] = pd.to_datetime(df_jan['tpep_pickup_datetime'])
    
    if not pd.api.types.is_datetime64_dtype(df_jan['tpep_dropoff_datetime']):
        df_jan['tpep_dropoff_datetime'] = pd.to_datetime(df_jan['tpep_dropoff_datetime'])
    
    # Compute trip duration in minutes
    df_jan['duration'] = df_jan.tpep_dropoff_datetime - df_jan.tpep_pickup_datetime
    df_jan.duration = df_jan.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Calculate standard deviation
    duration_std = df_jan.duration.std()
    
    print(f"\nQ2 Answer: Standard deviation of trip duration: {duration_std:.2f}")
    return df_jan

def q3_drop_outliers(df_jan):
    """
    Q3. Dropping outliers and calculating fraction of records remaining
    """
    total_records = len(df_jan)
    
    # Keep only records with duration between 1 and 60 minutes (inclusive)
    df_jan_filtered = df_jan[(df_jan.duration >= 1) & (df_jan.duration <= 60)]
    
    # Calculate percentage of records retained
    filtered_records = len(df_jan_filtered)
    fraction_kept = filtered_records / total_records
    
    print(f"\nQ3 Answer: Fraction of records after dropping outliers: {fraction_kept:.2%}")
    return df_jan_filtered

def prepare_dictionaries(df, categorical, numerical):
    """Prepare dictionaries for vectorization"""
    df_dict = df[categorical + numerical].to_dict(orient='records')
    return df_dict

def q4_one_hot_encoding(df_jan):
    """
    Q4. One-hot encoding and dimensionality
    """
    # Define features
    categorical = ['PULocationID', 'DOLocationID']
    numerical = []  # No numerical features for this question
    
    # Convert categorical features to strings
    df_jan[categorical] = df_jan[categorical].astype(str)
    
    # Create dictionary and fit vectorizer
    dicts = prepare_dictionaries(df_jan, categorical, numerical)
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    
    # Get dimensionality (number of columns)
    dimensionality = X.shape[1]
    
    print(f"\nQ4 Answer: Dimensionality of the feature matrix: {dimensionality}")
    return df_jan, dv, X

def q5_train_model(df_jan, X):
    """
    Q5. Train a model and calculate RMSE on training data
    """
    # Prepare target variable
    y = df_jan['duration'].values
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Calculate predictions and RMSE
    y_pred = lr.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\nQ5 Answer: RMSE on training data: {rmse:.2f}")
    return lr

def q6_evaluate_model(lr, dv):
    """
    Q6. Evaluate model on February data
    """
    # Load February data
    df_feb = pd.read_parquet('data/yellow_tripdata_2023-02.parquet')
    
    # Process February data
    if not pd.api.types.is_datetime64_dtype(df_feb['tpep_pickup_datetime']):
        df_feb['tpep_pickup_datetime'] = pd.to_datetime(df_feb['tpep_pickup_datetime'])
    
    if not pd.api.types.is_datetime64_dtype(df_feb['tpep_dropoff_datetime']):
        df_feb['tpep_dropoff_datetime'] = pd.to_datetime(df_feb['tpep_dropoff_datetime'])
    
    # Compute duration
    df_feb['duration'] = df_feb.tpep_dropoff_datetime - df_feb.tpep_pickup_datetime
    df_feb.duration = df_feb.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df_feb = df_feb[(df_feb.duration >= 1) & (df_feb.duration <= 60)]
    
    # Prepare features
    categorical = ['PULocationID', 'DOLocationID']
    numerical = []
    
    df_feb[categorical] = df_feb[categorical].astype(str)
    
    val_dicts = prepare_dictionaries(df_feb, categorical, numerical)
    X_val = dv.transform(val_dicts)
    
    # Calculate predictions and RMSE
    y_val = df_feb['duration'].values
    y_pred = lr.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\nQ6 Answer: RMSE on validation data: {rmse:.2f}")
    
    # Save the model
    with open('models/hw1_model.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)
    print("Model saved to models/hw1_model.bin")

def main():
    print("MLOps Zoomcamp 2025: Homework 1")
    print("="*50)
    
    # Q1: Download data and count columns
    df_jan = q1_download_data()
    
    # Q2: Compute duration and its standard deviation
    df_jan = q2_compute_duration(df_jan)
    
    # Q3: Drop outliers and calculate fraction
    df_jan_filtered = q3_drop_outliers(df_jan)
    
    # Q4: One-hot encoding and dimensionality
    df_jan_filtered, dv, X = q4_one_hot_encoding(df_jan_filtered)
    
    # Q5: Train model and calculate RMSE
    lr = q5_train_model(df_jan_filtered, X)
    
    # Q6: Evaluate model on February data
    q6_evaluate_model(lr, dv)
    
    print("\nAll questions answered!")

if __name__ == "__main__":
    main() 