import boto3
import pandas as pd
import io
from dotenv import load_dotenv
import os
import random

load_dotenv()

# Prepare input data (must match model's expected format)
input_data = pd.DataFrame(
    {
        "sensor_1": [random.random()],
        "sensor_2": [random.random()],
        "sensor_3": [random.random()],
    }
)

csv_buffer = io.StringIO()
input_data.to_csv(csv_buffer, header=True, index=False)
payload = csv_buffer.getvalue()

# Create runtime client
runtime = boto3.client("sagemaker-runtime", region_name=os.getenv("REGION"))

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName=os.getenv("PROJECT_NAME") + "-model-endpoint",
    ContentType="text/csv",
    Body=payload,
)

# Parse result
result = response["Body"].read().decode("utf-8")
print("Prediction:", result)
