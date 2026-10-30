FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/airflow/mlflow && chown airflow /opt/airflow/mlflow

USER airflow
# Separate environment for the ML libraries (avoids conflicts with Airflow)
RUN python -m venv /opt/airflow/ml-venv \
 && /opt/airflow/ml-venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && /opt/airflow/ml-venv/bin/pip install --no-cache-dir tensorflow-cpu ultralytics mlflow \
        pillow scikit-learn pyyaml pandas opencv-python-headless