import joblib
import io
import os
import pandas as pd


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(input_data, content_type):
    return pd.read_csv(io.StringIO(input_data))


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    return ",".join(str(int(p)) for p in prediction)
