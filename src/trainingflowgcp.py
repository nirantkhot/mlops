from metaflow import FlowSpec, step, Parameter, resources, kubernetes, retry, timeout, catch, conda_base
import os

@conda_base(python="3.10", libraries={
    "pandas": "1.5.3",
    "scikit-learn": "1.2.2",
    "mlflow": "2.2.2",
    "numpy": "1.23.5",
    "pip": "23.2.1",                
    "setuptools": "65.6.3",
    "databricks-cli": "0.17.7",
    "gcsfs": "2023.6.0"
})
class RandomForestGCPTrainFlow(FlowSpec):

    data_path = Parameter('data_path', type=str, required=True)
    mlflow_uri = "https://mlflow-run-404670143304.us-west2.run.app"
    experiment_name = "metaflow-gcp-resources-kubernetes"

    @resources(cpu=2, memory=4096)
    @timeout(seconds=300)
    @retry(times=2)
    @step
    def start(self):
        from preprocessing import load_data, prepare_data
        data = load_data(self.data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data(data)
        print(f"Data loaded: {len(data)} records")
        self.next(self.train_all)

    @kubernetes(cpu=4, memory=8192)
    @timeout(seconds=600)
    @retry(times=3)
    @catch(var="train_error")
    @step
    def train_all(self):
        import logging
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        import mlflow
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import ParameterSampler
        from sklearn.metrics import accuracy_score

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': [2, 5, 8]
        }
        param_list = list(ParameterSampler(param_distributions=param_dist, n_iter=5, random_state=42))

        best_score, best_model, best_params, best_run_id = -1, None, None, None

        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f"run_{i}") as run:
                model = RandomForestClassifier(random_state=42, **params)
                model.fit(self.X_train, self.y_train)
                test_pred = model.predict(self.X_test)
                test_score = accuracy_score(self.y_test, test_pred)
                mlflow.log_params(params)
                mlflow.log_metric("test_accuracy", test_score)
                mlflow.set_tags({"model_type": "RandomForest", "iteration": str(i), "stage": "training"})
                print(f"Run {i}: test_score = {test_score:.4f}")
                if test_score > best_score:
                    best_score, best_model, best_params, best_run_id = test_score, model, params, run.info.run_id

        self.best_model, self.best_params, self.best_score, self.best_run_id = best_model, best_params, best_score, best_run_id
        print(f"Best test score: {best_score:.4f}")
        self.next(self.register_best)

    @resources(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=900)
    @catch(var="register_error")
    @step
    def register_best(self):
        import logging
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="register_best_model"):
            mlflow.log_param("source_run_id", self.best_run_id)
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="best-rf-model-gcp"
            )
        print("Best model registered to MLflow")
        self.next(self.end)

    @step
    def end(self):
        print("Training and registration completed.")
        print(f"Best Test Accuracy: {self.best_score:.4f}")
        print(f"Registered Run ID: {self.best_run_id}")

if __name__ == '__main__':
    RandomForestGCPTrainFlow()
