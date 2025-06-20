import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from dotenv import load_dotenv
import os

load_dotenv()

# Check if all required environment variables are set
required_env_vars = ["SAGEMAKER_ROLE_ARN", "BUCKET_NAME", "PREFIX"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


# Paths
role = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("BUCKET_NAME")
prefix = os.getenv("PREFIX")
code_s3_path = f"s3://{bucket}/{prefix}/code/train.tar.gz"
data_s3_path = f"s3://{bucket}/{prefix}/data/"

session = sagemaker.Session()


# Estimator config
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir=code_s3_path,
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
    base_job_name="sensor-model-training",
    hyperparameters={},
)

# Launch training job
sklearn_estimator.fit({"train": data_s3_path})

# Save model artifact path for next stage (deployment)
model_artifact_uri = sklearn_estimator.model_data
with open("model_artifact_path.txt", "w") as f:
    f.write(model_artifact_uri)
