# Goal

Package an H2O Driverless AI time-series scoring pipeline as an MLflow pyfunc model that always runs under Python 3.8, even though Databricks notebooks default to Python 3.12.

## Environment state

The current folder is part of a databricks bundle and everything on ./src can be executed in the databricks environment (the local ./src folder is placed on /Workspace/Users/luiz.santos@h2o.ai/.bundle/h2o_dai_py_scoring_mlflow/dev/files/src). ./src/notebook.ipynb is the main entry point. Folder ./src/h2o_dai_py_scoring_mlflow/mlflow_project contains the MLflow Project that wraps the exported scoring pipeline. Folder ./src/h2o_dai_py_scoring_mlflow/mlflow_driverless contains helper code that can be used both from the notebook and from the MLflow Project. These two folders have outdated code and yaml files who need to be updated, as they were originally written to run directly on the cluster, not from a bundle (so all relative paths are incorrect). Locally, the scoring pipeline is on ../../scoring-pipeline and on the cluster, it lives on /Workspace/Users/luiz.santos@h2o.ai/mlflow_proj/scoring-pipeline.
Code execution instructions: To sync the local and cluster environments, we use the databricks CLI to push/pull files and the databricks bundles feature to run jobs. Use 'databricks bundle deploy' to push local changes to the cluster and 'databricks bundle run h2o_dai_py_scoring_mlflow' to execute the job defined in resources/h2o_dai_py_scoring_mlflow.job.yml. The job runs notebook.ipynb and then calls the main() function from the h2o_dai_py_scoring_mlflow python wheel, which is built from the local ./src folder and installed on the cluster as a library.

## Current status:

### Approach

Wrap the exported scoring-pipeline/ bundle inside an MLflow Project (mlflow_project/) that defines a Python 3.8 virtualenv (python_env.yaml) and two entry points: score for ad-hoc batch scoring and log_model for registering the pipeline as an MLflow model.

### Workflow

From the Databricks notebook we call mlflow.projects.run(..., entry_point="log_model"); this spins up the project’s virtualenv, imports our helper (mlflow_driverless/deployment.py), and logs the model with the correct python_env.yaml, bundled wheels, optional input/output examples, and signature.

### Current state

- mlflow_project/MLproject flattened commands and exposes parameters for input/output examples.
- mlflow_project/scorer_entry.py handles both scoring and logging, loads example files, and forwards them into the helper.
- mlflow_driverless/deployment.py decides whether to log in-place or launch its own Project run, builds signatures from user examples, and ensures artifacts/wheels are attached.
- python_env.yaml pins Py 3.8.12, all required Driverless wheels, mlflow==2.14.3, and importlib-resources==5.12.0 so PySpark imports succeed.

Outstanding verification (to run in main.py): 1. Execute the mlflow.projects.run cell and confirm the managed run finishes without import errors. 2. Inspect the resulting model’s python_env.yaml (should show Python 3.8.x) and artifact list (should contain all wheels plus examples/ input_example / output_example).

# Overall Instructions

Please suggest code changes to fix any issues and achieve the stated goal. Use AGENTS.md to track your progress. Only modify code within the ./src folder.

## Progress

- 2025-09-26: Taught scorer_entry to locate the scoring pipeline in bundle deployments, install its wheels on the fly, and reuse the same logic for logging; trimmed python_env.yaml to pure pip packages and added astunparse; corrected the deployment helper to reuse the mlflow_project template directory.
