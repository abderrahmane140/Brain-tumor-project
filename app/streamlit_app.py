import io
import requests
import streamlit as st
from PIL import Image, ImageDraw

API = "http://127.0.0.1:8000"
COLORS = {"glioma": "red", "meningioma": "lime", "notumor": "blue", "pituitary": "orange"}

st.set_page_config(page_title="Brain Tumor Analysis", layout="wide")
st.title("Brain Tumor Analysis")

file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
mode = st.radio("Task", ["Classification", "Detection"], horizontal=True)

if file:
    image = Image.open(file).convert("RGB")
    payload = {"file": (file.name, file.getvalue(), file.type)}

    if st.button("Run analysis"):
        endpoint = "/classify" if mode == "Classification" else "/detect"
        try:
            r = requests.post(API + endpoint, files=payload, timeout=60)
            r.raise_for_status()
            out = r.json()
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

        left, right = st.columns(2)

        if mode == "Classification":
            left.image(image, caption="Input image", use_container_width=True)
            right.metric("Predicted class", out["class"], f"{out['confidence']:.1%}")
            right.subheader("Confidence scores")
            for name, score in sorted(out["scores"].items(), key=lambda kv: -kv[1]):
                right.write(f"{name}: {score:.1%}")
                right.progress(score)
        else:
            drawn = image.copy()
            d = ImageDraw.Draw(drawn)
            for det in out["detections"]:
                x1, y1, x2, y2 = det["box"]
                color = COLORS.get(det["class"], "white")
                d.rectangle([x1, y1, x2, y2], outline=color, width=4)
                d.text((x1 + 4, max(y1 - 14, 0)), f"{det['class']} {det['confidence']:.2f}", fill=color)
            left.image(drawn, caption="Detections", use_container_width=True)
            right.subheader("Detection details")
            if out["detections"]:
                right.table(out["detections"])
            else:
                right.info("No tumor detected.")