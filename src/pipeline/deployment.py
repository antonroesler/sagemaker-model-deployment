import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig
from dotenv import load_dotenv
import os

load_dotenv()

# Check if all required environment variables are set
required_env_vars = ["SAGEMAKER_ROLE_ARN", "PROJECT_NAME"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


# Variables
role = os.getenv("SAGEMAKER_ROLE_ARN")
project_name = os.getenv("PROJECT_NAME")

session = sagemaker.Session()

# Read model artifact path
with open("model_artifact_path.txt", "r") as f:
    model_artifact_uri = f.read().strip()

# Register model in Model Registry
model = SKLearnModel(
    model_data=model_artifact_uri,
    role=role,
    entry_point="src/inference.py",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

model_package = model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    model_package_group_name=f"{project_name}-model-group",
    approval_status="Approved",
    description=f"v1 of {project_name} model",
)

print("Model registered:", model_package.model_package_arn)

serverless_config = ServerlessInferenceConfig(memory_size_in_mb=1024, max_concurrency=1)

predictor = model.deploy(
    endpoint_name=f"{project_name}-model-endpoint",
    serverless_inference_config=serverless_config,
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
