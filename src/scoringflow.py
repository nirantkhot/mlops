from metaflow import FlowSpec, step, Parameter, JSONType
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import os

class RandomForestPredictFlow(FlowSpec):

    vector = Parameter('vector', type=str, required=True)

    @step
    def start(self):
        # Process input
        self.features = np.array([float(x) for x in self.vector.split(',')]).reshape(1, -1)
        print("Received input vector:", self.features)
        self.next(self.load_model)

    @step
    def load_model(self):
        # Load model from registry
        tracking_dir = os.path.abspath("../mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        client = MlflowClient()

        model_name = "best-rf-model"

        # Get all versions and pick the highest version number
        versions = client.get_latest_versions(model_name, stages=[])
        latest_version = max(versions, key=lambda v: int(v.version))

        self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version.version}")
        print(f"Loaded model '{model_name}' version {latest_version.version} from local MLflow registry at {tracking_dir}")
        self.next(self.predict)

    @step
    def predict(self):
        # Generate prediction
        self.prediction = self.model.predict(self.features)[0]
        print(f"Predicted class: {self.prediction}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Successfully performed prediction!")

if __name__ == '__main__':
    RandomForestPredictFlow()