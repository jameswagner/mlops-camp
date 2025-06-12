# Use the provided base image which already contains the model
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set working directory (already set in base image, but being explicit)
WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir pandas pyarrow numpy scikit-learn==1.5.0

# Copy our prediction script
COPY taxi_prediction.py .

# Set the default command to run our script
# Users can override the year and month with docker run arguments
ENTRYPOINT ["python", "taxi_prediction.py"] 