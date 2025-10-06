h2o_dai_py_scoring_mlflow – Databricks Importable ZIP

What this ZIP contains
- Python package: h2o_dai_py_scoring_mlflow
  - mlflow_driverless/deployment.py – logs a Driverless AI scoring pipeline as an MLflow pyfunc
  - mlflow_project/* – MLflow Project used by the helper when needed
  - sitecustomize.py – runtime shims for Python 3.8 on Databricks
  - config.py – single source of truth for environment toggles and defaults
  - ZIP_README.md – this file (how to use the ZIP)

How to import on Databricks
- Upload the ZIP to DBFS (e.g., dbfs:/FileStore/h2o_dai_py_scoring_mlflow.zip) and add it to sys.path, OR
- Import the ZIP into the workspace. Databricks unzips it into a folder; add that folder’s src/ to sys.path.

Required license
- Provide a Driverless AI license before logging/predicting:
  - export DRIVERLESS_AI_LICENSE_KEY (or DRIVERLESS_AI_LICENSE) or set DRIVERLESS_AI_LICENSE_FILE

Logging models
- Programmatic API:
  from h2o_dai_py_scoring_mlflow.mlflow_driverless.deployment import log_driverless_scoring_pipeline
  info = log_driverless_scoring_pipeline(scoring_pipeline_dir="/dbfs/.../scoring-pipeline.zip", artifact_path="h2o_dai_scoring_pyfunc")

Important defaults (can be overridden via env):
- H2O_DAI_MLFLOW_FORCE_CONDA=1 (default): include libmagic + libopenblas in conda env for Serving
- H2O_DAI_MLFLOW_DISABLE_LIBMAGIC / H2O_DAI_MLFLOW_DISABLE_OPENBLAS: set to 1 to skip adding those OS libs

Predicting from notebooks (Databricks)
- Recommended: env_manager="conda" (no extra_envs required; conda.yaml injects safe variables)
- If you must use env_manager="virtualenv", set MLFLOW_ENV_ROOT to a writable path (e.g., /databricks/driver/mlflow_envs) and pass extra_envs:
  {"PYTHONPATH":"","SPARK_HOME":"","PYSPARK_PYTHON":"","PYSPARK_DRIVER_PYTHON":""}
- mlflow.models.predict writes to stdout by design; to capture into a variable, use output_path and read the JSON back, or use mlflow.pyfunc.load_model(...).predict(...).

Serving endpoints (Databricks Model Serving)
- Register your logged model version and update the endpoint to that version.
- Set DRIVERLESS_AI_LICENSE_KEY (or FILE) in the endpoint’s environment.

Paths that must be writable (for env creation)
- Prefer: /databricks/driver/mlflow_envs
- Fallbacks: /local_disk0/.ephemeral_nfs/user_tmp_data/mlflow_envs, /local_disk0/tmp/mlflow_envs, /tmp/mlflow_envs_<uid>

Common environment toggles (config.py)
- H2O_DAI_MLFLOW_EXCLUDE_PACKAGES – extra pip packages to drop in addition to defaults (h2o4gpu, pyorc)
- H2O_DAI_MLFLOW_PYSPARK_VERSION / H2O_DAI_MLFLOW_IMPORTLIB_RESOURCES_VERSION – shims for Py3.8
- H2O_DAI_MLFLOW_DISABLE_PROJECT / FORCE_PROJECT – control launching the MLflow Project when not on Py3.8
