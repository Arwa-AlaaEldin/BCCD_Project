import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from classical_pipeline import ClassicalCellDetector

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

st.set_page_config(page_title="Blood Cell Counter", page_icon="🩸", layout="wide")
st.title("🩸 Blood Cell Detection & Counting")
st.caption("Upload a blood smear image to detect and count Red Blood Cells, White Blood Cells and Platelets.")


@st.cache_resource
def load_classical_detector():
    return ClassicalCellDetector()


@st.cache_resource
def load_yolo_detector(weights_path):
    from yolo_pipeline import YoloCellDetector

    return YoloCellDetector(weights_path)


yolo_available = os.path.exists(WEIGHTS_PATH)

with st.sidebar:
    st.header("Settings")
    method_options = ["Classical (OpenCV)"]
    if yolo_available:
        method_options.append("YOLOv8")
    else:
        st.warning(
            "No YOLO weights found at `models/best.pt`.\n\n"
            "Run the notebook in `notebooks/bccd_train_and_export.ipynb` "
            "(on Kaggle/Colab), download the exported `best.pt`, and place "
            "it in this project's `models/` folder to enable YOLO."
        )
    method = st.radio("Detection method", method_options)

uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read that file as an image.")
    else:
        with st.spinner("Running detection..."):
            if method == "YOLOv8":
                detector = load_yolo_detector(WEIGHTS_PATH)
            else:
                detector = load_classical_detector()
            result = detector.run_pipeline(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Detections")
            st.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Counts")
        c1, c2, c3 = st.columns(3)
        c1.metric("RBC", result["rbc_count"])
        c2.metric("WBC", result["wbc_count"])
        c3.metric("Platelets", result["platelet_count"])
else:
    st.info("Upload an image to get started.")
