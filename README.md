# Brain Tumor Analysis: Classification and Detection with Deep Learning

An end-to-end MLOps project that assists the analysis of brain MRI images. It combines **image classification** (InceptionV3) and **tumor detection** (YOLOv8), with experiment tracking (MLflow), pipeline orchestration (Airflow), a REST API (FastAPI) and a web interface (Streamlit).

> **Disclaimer:** this is an educational project. It is not a medical device and must not be used for clinical diagnosis.

---

## Results

| Task | Model | Test set | Result |
|---|---|---|---|
| Classification (4 classes) | InceptionV3, feature extraction | 971 images | 89.8% accuracy |
| Classification (4 classes) | **InceptionV3, fine-tuned** | 971 images | **96.0% accuracy, 0.959 macro F1** |
| Detection | **YOLOv8s** | 255 images | **mAP@0.5 = 0.959**, mAP@0.5:0.95 = 0.749, precision 0.956, recall 0.931 |

Classes: `glioma`, `meningioma`, `notumor`, `pituitary`.

Fine-tuning improved meningioma recall from 0.765 to 0.935. Glioma is the weakest class for detection (recall 0.852).

---

## Architecture

```
                 ┌──────────────┐     HTTP      ┌──────────────────────┐
  MRI image ───► │  Streamlit   │ ────────────► │  FastAPI             │
                 │  (UI)        │ ◄──────────── │  /classify  /detect  │
                 └──────────────┘     JSON      └─────────┬────────────┘
                                                          │ loads at startup
                                          ┌───────────────┴──────────────┐
                                          │ InceptionV3 (.keras)  YOLOv8 (.pt) │
                                          └───────────────▲──────────────┘
                                                          │ trained by
   Airflow DAG ──► check ► validate ► train (cls ‖ det) ► evaluate + MLflow ► save
```

---

## Tech stack

- **Deep learning:** TensorFlow / Keras (InceptionV3), Ultralytics YOLOv8 (PyTorch)
- **Data and evaluation:** NumPy, pandas, scikit-learn, Pillow, OpenCV, Matplotlib, Seaborn
- **Experiment tracking:** MLflow (parameters, metrics, artifacts, Model Registry)
- **Orchestration:** Apache Airflow 2.10 (Docker Compose, PostgreSQL)
- **Serving:** FastAPI, Uvicorn
- **Interface:** Streamlit
- **Training environment:** Google Colab (T4 GPU)

---

## Pipeline

### 1. Classification (InceptionV3)
- File validation (extensions, integrity with `try-except`).
- **Deduplication** with MD5 (exact) and perceptual hash (near duplicates), plus removal of label conflicts, to avoid train/test leakage.
- Stratified split **70 / 15 / 15** (4528 / 970 / 971 images before augmentation).
- Augmentation on the **training set only** (blur, noise, flip) to balance the classes (1218 images per class).
- Transfer learning: frozen InceptionV3 base, new head (`GlobalAveragePooling2D`, `Dropout`, `Dense`), then **fine-tuning** of the last 50 layers (BatchNorm frozen) with a learning rate of 1e-5.
- Evaluation: accuracy, precision, recall, F1, confusion matrix, train vs. test comparison.

### 2. Detection (YOLOv8)
- Annotation checks: image/label matching, YOLO format validation, class consistency (5 invalid files and 15 inconsistent files removed).
- Dataset split: 4722 train / 254 valid / 255 test (test split built from the validation set).
- Configured augmentation (HSV, flip, rotation, scale, mosaic), 50 epochs with early stopping.
- Evaluation with precision, recall, mAP@0.5 and mAP@0.5:0.95 on the test set.

### 3. Experiment tracking (MLflow)
- Experiments for classification and detection, with logged parameters, metrics and artifacts.
- Comparison of feature extraction vs. fine-tuning.
- Best models registered in the **Model Registry** with versions and a `champion` alias.

### 4. Orchestration (Airflow)
DAG `brain_tumor_pipeline`:

```
check_datasets → validate_data → [train_classification ‖ train_detection] → evaluate_log_mlflow → save_artifacts
```

Includes retries, a failure callback, execution logs and task status monitoring. The DAG runs a **light version** of the training (few epochs, CPU) to demonstrate the automation. The final models were trained on a GPU in Colab.

### 5. API and interface
- `POST /classify`: returns the predicted class and confidence scores.
- `POST /detect`: returns bounding boxes, classes and confidences.
- Models are loaded once at startup. Streamlit sends the uploaded image to the API and displays the results and the image with bounding boxes.

---

## Project structure

```
brain-tumor-project/
├── api/                 FastAPI application
├── app/                 Streamlit interface
├── dags/                Airflow DAG
├── docker/              Dockerfile (Airflow + isolated ML environment)
├── notebooks/           01_classification, 02_detection, 03_mlflow
├── src/                 Pipeline script used by Airflow
├── models/              Trained models (not versioned)
├── reports/             Figures and training reports
├── docker-compose.yml
└── requirements.txt
```

---

## Getting started

### Prerequisites
- Python 3.11 or 3.12
- Docker Desktop (for Airflow only)

### Installation

```bash
git clone https://github.com/abderrahmane140/Brain-tumor-project
cd brain-tumor-project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data and models
The datasets and trained weights are not included in the repository. Place them as follows:

```
data/classification/raw/<class_name>/*.jpg     # classification dataset
data/detection/raw/                            # YOLO dataset (images + labels)
models/inceptionv3_ft.keras
models/best.pt
models/class_names.json
```

### Run the API and the interface

```bash
# Terminal 1
uvicorn api.main:app --port 8000

# Terminal 2
streamlit run app/streamlit_app.py
```

- API documentation: http://127.0.0.1:8000/docs
- Interface: http://localhost:8501

### Run the Airflow pipeline

```bash
docker compose build
docker compose up -d
```

- Airflow UI: http://localhost:8080 (default credentials `admin` / `admin`, for local use only)
- Trigger the DAG `brain_tumor_pipeline`, then start the MLflow UI:

```bash
docker compose exec -d airflow /opt/airflow/ml-venv/bin/mlflow ui \
  --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////opt/airflow/mlflow/mlflow.db
```

### View the MLflow experiments from the notebooks

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## Limitations and future work

- Test results come from a single split of public Kaggle datasets and are not clinically validated.
- Glioma is the hardest class for both models.
- Next steps: Grad-CAM explanations, a larger and multi-center dataset, containerized API and UI, CI/CD, and model serving through the MLflow Model Registry.
