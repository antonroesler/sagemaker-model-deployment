import numpy as np
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

# Argument parsing
parser = argparse.ArgumentParser(description="Generate and upload synthetic sensor data.")
parser.add_argument("-n", type=int, default=1, help="Suffix number for the dataset file name.")
args = parser.parse_args()

n = args.n

# Check if all required environment variables are set
required_env_vars = ["BUCKET_NAME", "PREFIX", "REGION", "PROJECT_NAME"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


# AWS config
s3_bucket = os.getenv("BUCKET_NAME")
s3_prefix = os.getenv("PREFIX")
s3_key = f"{s3_prefix}/data/sensor_data_{n}.csv"
region = os.getenv("REGION")

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
sensor_1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
sensor_2 = np.random.normal(loc=1.0, scale=0.5, size=n_samples)
sensor_3 = np.random.normal(loc=-1.0, scale=0.8, size=n_samples)
logit = 1.5 * sensor_1 + 2.0 * sensor_2 - 1.0 * sensor_3 + np.random.normal(0, 1, n_samples)
prob = 1 / (1 + np.exp(-logit))
label = (prob > 0.5).astype(int)

df = pd.DataFrame({"sensor_1": sensor_1, "sensor_2": sensor_2, "sensor_3": sensor_3, "label": label})

# Save locally
local_path = f"/tmp/sensor_data_{n}.csv"
df.to_csv(local_path, index=False)
print(f"Data saved locally to {local_path}")

# Upload to S3
s3 = boto3.client("s3", region_name=region)
s3.upload_file(local_path, s3_bucket, s3_key)
print(f"Data uploaded to s3://{s3_bucket}/{s3_key}")
