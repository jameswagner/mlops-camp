import datetime
import time
import logging
import psycopg2
import pandas as pd
import os
from joblib import load
from dotenv import load_dotenv
from prefect import task, flow
from prefect.cache_policies import NONE

# Load environment variables
load_dotenv()

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount, QuantileValue, MaxValue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10

# Get database config from environment
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'example')
DB_NAME = os.getenv('POSTGRES_DB', 'test')

create_table_statement = """
drop table if exists monitoring_metrics;
create table monitoring_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    fare_amount_quantile_50 float
)
"""

@task(cache_policy=NONE)
def prep_db():
    # Create database if not exists
    with psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database="postgres") as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(f"CREATE DATABASE {DB_NAME};")
                print(f"Database '{DB_NAME}' created")
    
    # Create table
    with psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(create_table_statement)
            print("Table created")

@task(cache_policy=NONE)
def calculate_metrics_postgresql(day_offset):
    # Load reference data and model
    reference_data = pd.read_parquet('data/reference.parquet')
    
    with open('models/lin_reg.bin', 'rb') as f_in:
        model = load(f_in)
    
    # Load March 2024 data
    raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')
    
    # Same preprocessing as baseline
    raw_data["duration_min"] = raw_data.lpep_dropoff_datetime - raw_data.lpep_pickup_datetime
    raw_data.duration_min = raw_data.duration_min.apply(lambda td: float(td.total_seconds())/60)
    raw_data = raw_data[(raw_data.duration_min >= 0) & (raw_data.duration_min <= 60)]
    raw_data = raw_data[(raw_data.passenger_count > 0) & (raw_data.passenger_count <= 8)]
    
    # Define features
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]
    
    # New Evidently API: DataDefinition
    data_definition = DataDefinition(
        numerical_columns=num_features + ['prediction'], 
        categorical_columns=cat_features
    )
    
    # Get current day's data
    begin_date = datetime.datetime(2024, 3, 1)
    current_date = begin_date + datetime.timedelta(days=day_offset)
    next_date = current_date + datetime.timedelta(days=1)
    
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= current_date) & 
        (raw_data.lpep_pickup_datetime < next_date)
    ].copy()
    
    if len(current_data) == 0:
        logging.info(f"No data for {current_date}")
        return None
    
    # Generate predictions
    current_data.fillna(0, inplace=True)
    current_data['prediction'] = model.predict(current_data[num_features + cat_features])
    
    # New Evidently API: Create Datasets
    reference_dataset = Dataset.from_pandas(reference_data, data_definition)
    current_dataset = Dataset.from_pandas(current_data, data_definition)
    
    # Q2: Create report with QuantileValue
    report = Report(metrics=[
        ValueDrift(column='prediction'),
        DriftedColumnsCount(),
        MissingValueCount(column='prediction'),
        QuantileValue(column='fare_amount', quantile=0.5),
        MaxValue(column='fare_amount') 
    ])
    
    # Run report with new API
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result = snapshot.dict()
    
    # Extract and convert metrics to Python types (FIX NUMPY ISSUE)
    prediction_drift = float(result['metrics'][0]['value'])
    num_drifted_columns = int(result['metrics'][1]['value']['count'])
    missing_values_count = int(result['metrics'][2]['value']['count'])  
    fare_amount_quantile_50 = float(result['metrics'][3]['value'])
    
    logging.info(f"Date: {current_date}")
    logging.info(f"Prediction drift: {prediction_drift}")
    logging.info(f"Fare amount quantile 50: {fare_amount_quantile_50}")
    
    # Insert into database (NO CURSOR PARAMETER - DIRECT CONNECTION)
    with psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO monitoring_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, fare_amount_quantile_50) VALUES (%s, %s, %s, %s, %s)",
                (current_date, prediction_drift, num_drifted_columns, missing_values_count, fare_amount_quantile_50)
            )
            conn.commit()
    
    return fare_amount_quantile_50

@flow
def batch_monitoring_backfill():
    print("=" * 60)
    print("HOMEWORK 5 MONITORING PIPELINE")
    print("=" * 60)
    
    # Q1: Print March 2024 data shape
    march_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')
    print(f"Q1 ANSWER: March 2024 data shape = {march_data.shape[0]} rows")
    print(f"   (Full shape: {march_data.shape})")
    print()
    
    # Q2: Print added metric
    print("Q2 ANSWER: Added metric = MaxValue(column='fare_amount') - maximum fare amount")
    print()
    
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    
    max_quantile = 0  # Track max value for Q3
    all_quantiles = []  # Store all values
    
    print("Processing March 2024 data day by day...")
    print("-" * 40)
    
    for i in range(31):  # March has 31 days
        try:
            quantile_value = calculate_metrics_postgresql(i)
            if quantile_value is not None:
                all_quantiles.append(quantile_value)
                if quantile_value > max_quantile:
                    max_quantile = quantile_value
                print(f"Day {i+1:2d}: Quantile 0.5 = {quantile_value:.2f}")
                        
        except Exception as e:
            logging.error(f"Error on day {i}: {e}")

        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)
    
    print("-" * 40)
    print(f"Q3 ANSWER: Maximum fare_amount quantile 0.5 value = {max_quantile}")
    print()
    
    # Q4: Dashboard location
    print("Q4 ANSWER: Dashboard config file location = project_folder/dashboards")
    print("   (i.e., 05-monitoring/dashboards/)")
    print()
    
    print("=" * 60)
    print("FINAL HOMEWORK ANSWERS:")
    print("=" * 60)
    print(f"Q1: {march_data.shape[0]} rows")
    print(f"Q2: MaxValue(column='fare_amount') - maximum fare amount")
    print(f"Q3: {max_quantile}")
    print(f"Q4: project_folder/dashboards")
    print("=" * 60)

if __name__ == '__main__':
    batch_monitoring_backfill()