#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import mlflow
from dagster import op, job, Config, get_dagster_logger

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-dagster-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


class DataConfig(Config):
    year: int = 2023
    month: int = 3
    taxi_type: str = "yellow"


@op
def load_taxi_data(config: DataConfig) -> pd.DataFrame:
    """Load Yellow taxi data for March 2023"""
    logger = get_dagster_logger()
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{config.taxi_type}_tripdata_{config.year}-{config.month:02d}.parquet'
    logger.info(f"Starting data load from: {url}")
    
    df = pd.read_parquet(url)
    
    # Answer Question 3: How many records did we load?
    record_count = len(df)
    logger.info(f"QUESTION 3 ANSWER: {record_count:,} records loaded")
    
    logger.info("Data load completed")
    return df


@op 
def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data using the homework logic"""
    logger = get_dagster_logger()
    
    logger.info("Starting data preparation")
    df = raw_data.copy()
    
    # Use tpep_* columns for yellow taxi (instead of lpep_* for green taxi)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Answer Question 4: What's the size after preparation?
    prepared_count = len(df)
    logger.info(f"QUESTION 4 ANSWER: {prepared_count:,} records after preparation")
    
    logger.info("Data preparation completed")
    return df


@op
def train_model(prepared_data: pd.DataFrame) -> tuple[LinearRegression, DictVectorizer]:
    """Train linear regression model and return model + vectorizer"""
    logger = get_dagster_logger()
    
    logger.info("Starting model training")
    
    # Create features - use pickup and dropoff locations separately (no combination)
    categorical = ['PULocationID', 'DOLocationID']  # Separate, not combined

    # Create feature matrix (categorical features only, like HW1)
    dicts = prepared_data[categorical].to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    
    # Target variable
    y = prepared_data['duration'].values
    
    # Train linear regression with default parameters
    model = LinearRegression()
    model.fit(X, y)
    
    # Answer Question 5: What's the intercept?
    intercept = model.intercept_
    logger.info(f"QUESTION 5 ANSWER: Model intercept = {intercept:.2f}")
    
    logger.info("Model training completed")
    return model, dv


@op
def register_model_mlflow(model_and_vectorizer: tuple[LinearRegression, DictVectorizer]) -> str:
    """Register the model with MLflow"""
    logger = get_dagster_logger()
    
    logger.info("Starting MLflow model registration")
    model, dv = model_and_vectorizer
    
    with mlflow.start_run() as run:
        # Log model parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID")
        
        # Log the model intercept as a metric
        mlflow.log_metric("intercept", model.intercept_)
        
        # Save and log the vectorizer
        with open("models/vectorizer.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/vectorizer.pkl", artifact_path="preprocessor")
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name="taxi-duration-linear-regression"
        )
        
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")
        logger.info("For QUESTION 6: Check the MLflow UI to find the model_size_bytes in the MLModel file")
        
        logger.info("MLflow model registration completed")
        return run_id


@job
def taxi_prediction_pipeline():
    """Complete taxi duration prediction pipeline"""
    raw_data = load_taxi_data()
    prepared_data = prepare_data(raw_data)
    model_and_dv = train_model(prepared_data)
    run_id = register_model_mlflow(model_and_dv)


if __name__ == "__main__":
    logger = get_dagster_logger()
    logger.info("Starting taxi prediction pipeline")
    
    # Run the pipeline
    result = taxi_prediction_pipeline.execute_in_process()
    
    logger.info("Pipeline completed successfully") 