# MLflow packaging for Driverless AI scoring pipeline

This module wraps the Driverless AI Python scoring pipeline as an MLflow ``pyfunc`` model so that the
Linux-only runtime (Python 3.8 and the provided wheels) is recreated automatically when the model is
served. Usage outline:

1. Ensure you are on a Linux x86_64 host (for example inside a container) with Python 3.8 available
   and install ``mlflow`` (the helper reuses the current mlflow version in the target environment).
2. Export the Driverless AI scoring bundle and place it at ``scoring-pipeline`` (the directory already
   tracked in this project).
3. Start an MLflow run (or let the helper create one implicitly) and execute:

   ```bash
   python -m mlflow_ts.mlflow_driverless.deployment scoring-pipeline --artifact-path driverless_ts_model
   ```

   The command copies the scoring pipeline into the logged model, builds a ``python_env.yaml`` that
   pins Python 3.8.12, and inlines every bundled ``.whl`` inside the MLflow model so the runtime can
   recreate the Linux x86_64 / py38 stack without relying on Conda.

   The logger also inspects ``training_data_column_stats.json`` (falling back to column hints inside
   ``example.py``) to materialize a single-row input example and attach it to the MLflow model. On
   macOS/arm64 the scorer binaries cannot be executed, so only the input-side schema is recorded;
   the Linux runtime will still infer outputs when the model is served.

   You can also pass a ``*.zip`` exported scoring bundle directly (for example
   ``python -m mlflow_ts.mlflow_driverless.deployment scorer.zip``); the helper extracts it to a temporary
   directory automatically before logging.

4. Before serving, provide a valid Driverless AI license via ``DRIVERLESS_AI_LICENSE_KEY`` or
   ``DRIVERLESS_AI_LICENSE_FILE`` – this is still required by the scorer at runtime.

5. Serve with MLflow (still on Linux/x86_64):

   ```bash
   mlflow models serve -m runs:/<run_id>/driverless_ts_model --env-manager=virtualenv -p 5000
   ```

   The resulting endpoint will accept pandas-compatible payloads with the columns reported by
   ``scorer.get_column_names()`` (``state``, ``week_start``, ``unweighted_ili`` for the bundled model).

The helper relies on ``scoring-pipeline/requirements.txt`` to populate package pins. If you customize the
scoring export or trim dependencies, adjust that file before logging or pass an explicit list of
requirements to ``log_driverless_scoring_pipeline``.
