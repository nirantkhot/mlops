from metaflow import FlowSpec, step, Parameter
import os

class RandomForestTrainFlow(FlowSpec):
    data_path = Parameter('path', type=str, required=True)

    @step
    def start(self):
        # Preprocess the data
        from preprocessing import load_data, prepare_data

        data = load_data(self.data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data(data)


        print(f"Data loaded: {len(data)} records")
        self.next(self.train_all)

    @step
    def train_all(self):
        # RUn experiments on ML Flow
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import ParameterSampler
        from sklearn.metrics import accuracy_score
        import numpy as np

        # MLflow setup (local)
        tracking_dir = os.path.abspath("../mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        mlflow.set_experiment("metaflow-local")

        # Define search space
        n_features = self.X_train.shape[1]
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': [2, 5, 8]  # Integer values from 1 to n_features
        }

        # Generate hyperparameter combinations
        param_list = list(ParameterSampler(param_distributions=param_dist, n_iter=100, random_state=42))

        best_score = -1
        best_model = None
        best_params = None
        best_run_id = None

        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f"run_{i}") as run:
                model = RandomForestClassifier(random_state=42, **params)
                model.fit(self.X_train, self.y_train)
                test_pred = model.predict(self.X_test)
                test_score = accuracy_score(self.y_test, test_pred)

                # Log everything
                mlflow.log_params(params)
                mlflow.log_metric("test_accuracy", test_score)
                mlflow.set_tags({
                    "model_type": "RandomForest",
                    "iteration": str(i),
                    "stage": "training"
                })

                print(f"Run {i}: test_score = {test_score:.4f}")

                if test_score > best_score:
                    best_score = test_score
                    best_model = model
                    best_params = params
                    best_run_id = run.info.run_id

        # Save best model to be registered in next step
        self.best_model = best_model
        self.best_params = best_params
        self.best_score = best_score
        self.best_run_id = best_run_id

        print(f"Best test score: {best_score:.4f}")
        self.next(self.register_best)

    @step
    def register_best(self):
        # Register the best model
        import mlflow
        import mlflow.sklearn
        import os

        tracking_dir = os.path.abspath("../mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        mlflow.set_experiment("metaflow-local")

        with mlflow.start_run(run_id=self.best_run_id):
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="best-rf-model"
            )

        print("Best model registered to MLflow")
        self.next(self.end)

    @step
    def end(self):
        # Print Results``
        print("Training and registration completed.")
        print(f"Best Test Accuracy: {self.best_score:.4f}")
        print(f"Registered Run ID: {self.best_run_id}")

if __name__ == '__main__':#
    RandomForestTrainFlow()