# Week 1: NYC Taxi Duration Prediction

This folder contains the homework and scripts for Week 1 of the MLOps Camp, following the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course.

## Contents
- `homework.py` – Main script for downloading data, processing it, training a model, and evaluating results for the NYC Yellow Taxi dataset (Jan/Feb 2023).

## How to Run
1. Make sure you have the required Python packages installed (see project root README for environment setup).
2. From the project root, activate your virtual environment.
3. Run the script:
   ```bash
   python week1/homework.py
   ```

## What the Script Does
- Downloads the January and February 2023 NYC Yellow Taxi trip data.
- Processes the data to compute trip durations and remove outliers.
- Performs one-hot encoding on pickup and dropoff locations.
- Trains a linear regression model to predict trip duration.
- Evaluates the model and prints answers to the homework questions.
- Saves the trained model and vectorizer to the `models/` directory.

## Outputs
- Data files are downloaded to the `data/` directory.
- The trained model is saved as `models/hw1_model.bin`.

## Reference
Based on the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by DataTalks.Club. 