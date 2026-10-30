import sys, json, shutil, logging
from pathlib import Path
import yaml
import mlflow
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")

# ---- Settings (light version for CPU; increase on a machine with a GPU) ----
CLS_EPOCHS, YOLO_EPOCHS, YOLO_FRACTION, YOLO_IMGSZ = 1, 1, 0.05, 320

ROOT      = Path("/opt/airflow/project")
CLS_SPLIT = ROOT / "data/classification/split"
DET       = ROOT / "data/detection/dataset"
OUT       = ROOT / "models/pipeline"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES   = ["glioma", "meningioma", "notumor", "pituitary"]


def load_ds(name, shuffle=False):
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    ds = tf.keras.utils.image_dataset_from_directory(
        CLS_SPLIT / name, image_size=(299, 299), batch_size=32,
        label_mode="categorical", shuffle=shuffle, seed=42)
    return ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)


def mlflow_setup(exp):
    mlflow.set_tracking_uri("sqlite:////opt/airflow/mlflow/mlflow.db")
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location="/opt/airflow/mlflow/artifacts")
    mlflow.set_experiment(exp)


# 1. Check that datasets exist
def check_datasets():
    paths = [CLS_SPLIT / s for s in ("train", "val", "test")]
    paths += [DET / "images" / s for s in ("train", "valid", "test")] + [DET / "data.yaml"]
    for p in paths:
        assert p.exists(), f"Missing: {p}"
        log.info("OK  %s", p)


# 2. Validate data quality
def validate_data():
    bad = 0
    for p in CLS_SPLIT.rglob("*"):
        if p.is_file():
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                bad += 1
                log.error("Corrupted image %s: %s", p, e)
    assert bad == 0, f"{bad} corrupted classification images"

    for s in ("train", "valid", "test"):
        imgs = {p.stem for p in (DET / "images" / s).glob("*")}
        lbls = {p.stem for p in (DET / "labels" / s).glob("*.txt")}
        assert imgs == lbls, f"Image/label mismatch in {s}"
        for f in (DET / "labels" / s).glob("*.txt"):
            for line in f.read_text().splitlines():
                v = line.split()
                assert len(v) == 5 and int(v[0]) in range(4) \
                    and all(0 <= float(x) <= 1 for x in v[1:]), f"Bad label: {f}"
        log.info("Detection %s OK (%d images)", s, len(imgs))


# 3. Train the classifier (feature extraction, light version)
def train_classification():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    base.trainable = False
    inp = layers.Input((299, 299, 3))
    x = base(inp, training=False)
    x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(x))
    model = models.Model(inp, layers.Dense(4, activation="softmax")(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(load_ds("train", True), validation_data=load_ds("val"), epochs=CLS_EPOCHS)
    model.save(OUT / "cls_pipeline.keras")
    log.info("Classifier saved")


# 4. Train the detector (light version)
def train_detection():
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(DET / "data.yaml"))
    cfg["path"] = str(DET)
    yaml.dump(cfg, open(OUT / "data_pipeline.yaml", "w"), sort_keys=False)
    YOLO("yolov8n.pt").train(
        data=str(OUT / "data_pipeline.yaml"), epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=16,
        fraction=YOLO_FRACTION, device="cpu", workers=0,
        project=str(OUT), name="yolo", exist_ok=True)
    log.info("Detector saved")


# 5. Evaluate both models and log to MLflow
def evaluate_log_mlflow():
    import tensorflow as tf
    from ultralytics import YOLO
    model = tf.keras.models.load_model(OUT / "cls_pipeline.keras")
    _, acc = model.evaluate(load_ds("test"), verbose=0)
    m = YOLO(OUT / "yolo/weights/best.pt").val(
        data=str(OUT / "data_pipeline.yaml"), split="test", imgsz=YOLO_IMGSZ, device="cpu")
    metrics = {"cls_test_accuracy": float(acc), "det_precision": float(m.box.mp),
               "det_recall": float(m.box.mr), "det_mAP50": float(m.box.map50),
               "det_mAP50_95": float(m.box.map)}

    mlflow_setup("brain-tumor-airflow-pipeline")
    with mlflow.start_run(run_name="pipeline_run"):
        mlflow.log_params({"cls_epochs": CLS_EPOCHS, "yolo_epochs": YOLO_EPOCHS,
                           "yolo_fraction": YOLO_FRACTION, "yolo_imgsz": YOLO_IMGSZ})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(OUT / "cls_pipeline.keras"))
        mlflow.log_artifact(str(OUT / "yolo/weights/best.pt"))
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    log.info("Metrics: %s", metrics)


# 6. Save the final artifacts
def save_artifacts():
    final = OUT / "final"
    final.mkdir(exist_ok=True)
    for src in [OUT / "cls_pipeline.keras", OUT / "yolo/weights/best.pt",
                OUT / "data_pipeline.yaml", OUT / "metrics.json"]:
        shutil.copy(src, final / src.name)
    log.info("Artifacts saved in %s", final)


if __name__ == "__main__":
    globals()[sys.argv[1]]()