#!/usr/bin/env python
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

def load_data(filename):
    """Load data, downloading if not found"""
    if not os.path.exists(filename):
        print(f"{filename} not found, downloading...")
        import requests
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{os.path.basename(filename)}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f_out:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f_out.write(chunk)
        print(f"Downloaded to {filename}")
    
    return pd.read_parquet(filename)

def prepare_features(df, categorical, numerical):
    """Prepare feature matrix and target"""
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].astype(str)
    
    # Return both features dictionary and target
    return df[categorical + numerical].to_dict(orient='records'), df['duration'].values

def train_model_search(train_dicts, val_dicts, y_train, y_val, model_type='lasso'):
    """
    Train model with hyperparameter search using hyperopt
    Returns the best model and its parameters
    """
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    # Log dataset info
    mlflow.log_params({
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "n_features": X_train.shape[1],
        "model_type": model_type
    })

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            
            if model_type == 'lasso':
                model = Lasso(alpha=params['alpha'], random_state=42)
            else:  # ridge
                model = Ridge(alpha=params['alpha'], random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            mlflow.log_metric("val_rmse", rmse)
            
            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'alpha': hp.loguniform('alpha', np.log(0.01), np.log(10.0))
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

    # Log best parameters in the parent run
    mlflow.log_params({
        "best_alpha": best_result['alpha'],
    })

    return best_result

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-duration-prediction-optimization")

    # Load and prepare data
    train_path = 'data/yellow_tripdata_2023-01.parquet'
    val_path = 'data/yellow_tripdata_2023-02.parquet'
    
    df_train = load_data(train_path)
    df_val = load_data(val_path)

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    # Prepare features and target together
    train_dicts, y_train = prepare_features(df_train, categorical, numerical)
    val_dicts, y_val = prepare_features(df_val, categorical, numerical)

    # Train Lasso model
    with mlflow.start_run(run_name="lasso-optimization"):
        print("Training Lasso model with hyperopt...")
        best_params_lasso = train_model_search(
            train_dicts, val_dicts, y_train, y_val, model_type='lasso'
        )
        print(f"Best Lasso alpha: {best_params_lasso['alpha']:.4f}")

    # Train Ridge model
    with mlflow.start_run(run_name="ridge-optimization"):
        print("Training Ridge model with hyperopt...")
        best_params_ridge = train_model_search(
            train_dicts, val_dicts, y_train, y_val, model_type='ridge'
        )
        print(f"Best Ridge alpha: {best_params_ridge['alpha']:.4f}")

if __name__ == "__main__":
    main() 