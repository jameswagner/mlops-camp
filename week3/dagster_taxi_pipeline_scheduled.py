#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import mlflow
from dagster import op, job, Config, get_dagster_logger, schedule, ScheduleEvaluationContext


# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-scheduled-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


class TrainingConfig(Config):
    """Configuration for the training pipeline"""
    year: Optional[int] = None  # If None, will use previous month's year
    month: Optional[int] = None  # If None, will use previous month
    taxi_type: str = "yellow"
    
    def get_training_period(self) -> tuple[int, int]:
        """Calculate the training period (year, month) - defaults to previous month"""
        if self.year is not None and self.month is not None:
            return self.year, self.month
        
        # Calculate previous month
        today = datetime.now()
        # Go to first day of current month, then subtract 1 day to get last month
        first_day_current_month = today.replace(day=1)
        last_day_previous_month = first_day_current_month - timedelta(days=1)
        
        return last_day_previous_month.year, last_day_previous_month.month


@op
def load_taxi_data(config: TrainingConfig) -> pd.DataFrame:
    """Load taxi data for the specified period"""
    logger = get_dagster_logger()
    
    year, month = config.get_training_period()
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{config.taxi_type}_tripdata_{year}-{month:02d}.parquet'
    
    logger.info(f"Starting data load for {year}-{month:02d} from: {url}")
    
    try:
        df = pd.read_parquet(url)
        record_count = len(df)
        logger.info(f"Successfully loaded {record_count:,} records for {year}-{month:02d}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data for {year}-{month:02d}: {str(e)}")
        raise


@op 
def prepare_data(raw_data: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Prepare the data using standard preprocessing logic"""
    logger = get_dagster_logger()
    
    year, month = config.get_training_period()
    logger.info(f"Starting data preparation for {year}-{month:02d}")
    
    df = raw_data.copy()
    original_count = len(df)
    
    # Use appropriate datetime columns based on taxi type
    if config.taxi_type == "yellow":
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:  # green taxi
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    
    # Calculate duration in minutes
    df['duration'] = df[dropoff_col] - df[pickup_col]
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter for reasonable trip durations (1-60 minutes)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical columns to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    prepared_count = len(df)
    filtered_count = original_count - prepared_count
    
    logger.info(f"Data preparation completed for {year}-{month:02d}. "
               f"Original: {original_count:,}, After filtering: {prepared_count:,}, "
               f"Filtered out: {filtered_count:,} records")
    
    return df


@op
def train_model(prepared_data: pd.DataFrame, config: TrainingConfig) -> tuple[LinearRegression, DictVectorizer]:
    """Train linear regression model and return model + vectorizer"""
    logger = get_dagster_logger()
    
    year, month = config.get_training_period()
    logger.info(f"Starting model training for {year}-{month:02d}")
    
    # Create features - use pickup and dropoff locations separately (like HW1)
    categorical = ['PULocationID', 'DOLocationID']

    # Create feature matrix (categorical features only, like HW1)
    dicts = prepared_data[categorical].to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    
    # Target variable
    y = prepared_data['duration'].values
    
    # Train linear regression with default parameters
    model = LinearRegression()
    model.fit(X, y)
    
    # Log model performance metrics
    intercept = model.intercept_
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    logger.info(f"Model training completed for {year}-{month:02d}. "
               f"Intercept: {intercept:.2f}, Features: {n_features:,}, Samples: {n_samples:,}")
    
    return model, dv


@op
def register_model_mlflow(model_and_vectorizer: tuple[LinearRegression, DictVectorizer], 
                         config: TrainingConfig) -> str:
    """Register the model with MLflow"""
    logger = get_dagster_logger()
    
    year, month = config.get_training_period()
    logger.info(f"Starting MLflow model registration for {year}-{month:02d}")
    
    model, dv = model_and_vectorizer
    
    with mlflow.start_run() as run:
        # Log training period and configuration
        mlflow.log_param("training_year", year)
        mlflow.log_param("training_month", month)
        mlflow.log_param("taxi_type", config.taxi_type)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID")
        
        # Log model metrics
        mlflow.log_metric("intercept", model.intercept_)
        
        # Save and log the vectorizer with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vectorizer_path = f"models/vectorizer_{year}_{month:02d}_{timestamp}.pkl"
        with open(vectorizer_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(vectorizer_path, artifact_path="preprocessor")
        
        # Log the model with descriptive name
        model_name = f"taxi-duration-{config.taxi_type}-{year}-{month:02d}"
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name=model_name
        )
        
        run_id = run.info.run_id
        logger.info(f"MLflow model registration completed for {year}-{month:02d}. "
                   f"Run ID: {run_id}, Model name: {model_name}")
        
        return run_id


@job
def monthly_taxi_prediction_pipeline():
    """Monthly taxi duration prediction pipeline"""
    raw_data = load_taxi_data()
    prepared_data = prepare_data(raw_data)
    model_and_dv = train_model(prepared_data)
    run_id = register_model_mlflow(model_and_dv)


@schedule(
    job=monthly_taxi_prediction_pipeline,
    cron_schedule="0 2 5 * *",  # Run at 2 AM on the 5th of every month
    execution_timezone="US/Eastern",
)
def monthly_training_schedule(context: ScheduleEvaluationContext):
    """Schedule to run monthly training on the 5th of each month"""
    # This ensures we're training on the previous month's data
    # Running on 5th gives time for the previous month's data to be available
    return {}


# Alternative schedule for development/testing - runs daily but only processes previous month
@schedule(
    job=monthly_taxi_prediction_pipeline,
    cron_schedule="0 9 * * *",  # Run daily at 9 AM for testing
    execution_timezone="US/Eastern",
)
def daily_testing_schedule(context: ScheduleEvaluationContext):
    """Daily schedule for testing - still processes previous month's data"""
    return {}


if __name__ == "__main__":
    from dagster import materialize
    
    logger = get_dagster_logger()
    
    # Example 1: Run with default (previous month)
    logger.info("Starting monthly taxi prediction pipeline with default settings (previous month)")
    config = TrainingConfig()
    year, month = config.get_training_period()
    logger.info(f"Will process data for: {year}-{month:02d}")
    
    result = monthly_taxi_prediction_pipeline.execute_in_process(
        run_config={"ops": {"load_taxi_data": {"config": config.dict()},
                           "prepare_data": {"config": config.dict()},
                           "train_model": {"config": config.dict()},
                           "register_model_mlflow": {"config": config.dict()}}}
    )
    
    logger.info("Pipeline completed successfully")
    
    # Example 2: Run with specific month (uncomment to test)
    # logger.info("Running pipeline for specific month (March 2023)")
    # specific_config = TrainingConfig(year=2023, month=3)
    # result = monthly_taxi_prediction_pipeline.execute_in_process(
    #     run_config={"ops": {"load_taxi_data": {"config": specific_config.dict()},
    #                        "prepare_data": {"config": specific_config.dict()},
    #                        "train_model": {"config": specific_config.dict()},
    #                        "register_model_mlflow": {"config": specific_config.dict()}}}
    # ) 