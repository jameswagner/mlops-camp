# MLOps Camp

A hands-on project following the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course by DataTalks.Club.

## Overview
This repository contains code, scripts, and notes for the MLOps Camp, where I learn and apply MLOps best practices using the NYC Taxi dataset.

## Directory Structure
- `week1/` – Homework and scripts for week 1 (and similarly for other weeks)
- `data/` – Data files (populated automatically by scripts; not tracked in git)
- `models/` – Saved models (populated automatically by scripts; not tracked in git)

> **Note:** The `data/` and `models/` folders are created and populated by running the code in each week's homework. For example, running the script in `week1/homework.py` will download the necessary data and save trained models.

## Environment Setup
To set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # (create this file as needed)
```

## Example: Week 1
- Run `python week1/homework.py` to download data, process it, train a model, and save outputs to `data/` and `models/`.

## Credits
Based on the excellent [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by DataTalks.Club. 