"""Entry point that logs the Driverless AI scoring pipeline as an MLflow model.

Configuration is driven via environment variables (see src/h2o_dai_py_scoring_mlflow/config.py):
  - H2O_DAI_MLFLOW_EXPERIMENT: MLflow experiment path (default: /Shared/h2o_dai_py_scoring_mlflow)
  - H2O_DAI_MLFLOW_ARTIFACT_PATH: model artifact path (default: driverless_ts_pyfunc)
  - SCORING_PIPELINE_DIR: absolute path to exported scoring-pipeline directory
"""

from __future__ import annotations

import os
import mlflow

from h2o_dai_py_scoring_mlflow.mlflow_driverless import log_driverless_scoring_pipeline
from h2o_dai_py_scoring_mlflow.config import get_experiment_path, get_artifact_path, get_scoring_dir


def main() -> None:
    scoring_dir = get_scoring_dir()
    # Ensure an experiment exists for wheel tasks as well
    mlflow.set_experiment(get_experiment_path())

    with mlflow.start_run() as active_run:
        model_info = log_driverless_scoring_pipeline(
            scoring_pipeline_dir=scoring_dir,
            artifact_path=get_artifact_path(),
            apply_data_recipes=False,
            scorer_kwargs=None,
            predict_kwargs=None,
            pip_requirements=None,
            python_env=None,
            extra_artifacts=None,
            registered_model_name=None,
            input_example_df=None,
            output_example_df=None,
            input_example_path=None,
            output_example_path=None,
        )

        print("Logged MLflow run:", active_run.info.run_id)
        print("Model URI:", model_info.model_uri)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
