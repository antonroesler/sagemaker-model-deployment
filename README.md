# Model Deployment Tutorial

A hands-on tutorial for tracking dataset lineage, training, and deploying models using AWS SageMaker. This repo demonstrates a simple end-to-end ML workflow, including dataset creation, model training, deployment, and inference—all with rollback capability.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Workflow](#workflow)
   - [1. Create Dataset](#1-create-dataset)
   - [2. Package Training Code](#2-package-training-code)
   - [3. Upload to S3](#3-upload-to-s3)
   - [4. Train Model](#4-train-model)
   - [5. Deploy Model](#5-deploy-model)
   - [6. Invoke Endpoint](#6-invoke-endpoint)
6. [Notes](#notes)

---

## Overview

This repository guides you through creating datasets, training models, and deploying them to AWS SageMaker.

---

## Project Structure

```
.
├── src/
│   ├── create-dataset.py      # Script to create datasets
│   ├── train.py               # Local training script
│   ├── inference.py           # Minimal inference script
│   ├── invoke.py              # Script to invoke deployed endpoint
│   └── pipeline/
│       ├── training.py        # SageMaker training pipeline
│       └── deployment.py      # SageMaker deployment pipeline
├── README.md
├── BLOG.md
├── pyproject.toml
```

---

## Prerequisites

- **Python** (3.8+ recommended)
- **uv** ([docs](https://docs.astral.sh/uv/)) for dependency management
- **AWS CLI** configured with credentials
- An **S3 bucket** and prefix for storing datasets and models
- An **IAM Role ARN** (`SAGEMAKER_ROLE_ARN`) with at least `GetObject` and `PutObject` permissions on the S3 bucket

---

## Setup

1. **Install dependencies:**
   ```sh
   uv sync
   ```
2. **Set environment variables:**
   - Copy `.env.template` to `.env` and fill in your values (e.g., `BUCKET_NAME`, `PREFIX`, `SAGEMAKER_ROLE_ARN`).
   - Source your `.env` file:
     ```sh
     source .env
     ```

---

## Workflow

### 1. Create Dataset

Create a dataset and upload it to S3. For example, to create dataset 1:

```sh
uv run src/create-dataset.py -n 1
```

### 2. Package Training Code

Package the training script for SageMaker:

```sh
tar -czvf train.tar.gz -C src train.py
```

### 3. Upload to S3

Upload the training code archive to your S3 bucket:

```sh
aws s3 cp train.tar.gz s3://$BUCKET_NAME/$PREFIX/code/
```

### 4. Train Model

Run the SageMaker training job (should cost less than $0.05):

```sh
uv run src/pipeline/training.py
```

The model artifact will be saved to S3. The path is also saved locally for deployment.

### 5. Deploy Model

Deploy the trained model to a serverless SageMaker endpoint:

```sh
uv run src/pipeline/deployment.py
```

### 6. Invoke Endpoint

Find your endpoint URL in the AWS SageMaker console (under Endpoints). It will look like:

```
https://runtime.sagemaker.<region>.amazonaws.com/endpoints/<your-endpoint-name>/invocations
```

Send data to your model endpoint for inference:

```sh
uv run src/invoke.py
```

---

## Notes

- The inference script (`inference.py`) is referenced in `deployment.py` and uploaded automatically by the SageMaker SDK.
- For more details, see the code and comments in each script.

---

Happy experimenting! 🚀
