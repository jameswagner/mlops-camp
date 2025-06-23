import requests
from tqdm import tqdm
import os

files = [
    'green_tripdata_2022-01.parquet',
    'green_tripdata_2022-02.parquet', 
    'green_tripdata_2024-03.parquet'  # For homework Q1
]

os.makedirs('data', exist_ok=True)

for file in files:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    save_path = f"data/{file}"
    
    print(f"Downloading {file}...")
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('content-length', 0))
    
    with open(save_path, "wb") as handle:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=file) as pbar:
            for data in resp.iter_content(chunk_size=1024):
                handle.write(data)
                pbar.update(len(data))
    print(f"✅ {file} downloaded")