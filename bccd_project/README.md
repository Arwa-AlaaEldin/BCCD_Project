# Blood Cell Detection & Counting

Detects and counts Red Blood Cells (RBC), White Blood Cells (WBC) and Platelets in blood smear images, with two swappable detectors:

- **Classical (OpenCV)** — CLAHE + HSV color masking + Watershed. Works out of the box, no training or weights needed.
- **YOLOv8** — fine-tuned object detector. Needs trained weights (see below).

  <img width="952" height="880" alt="image" src="https://github.com/user-attachments/assets/ba0e4b6a-d7da-4fa8-8b40-7a06b287175c" />
  <img width="955" height="876" alt="image" src="https://github.com/user-attachments/assets/0af59d80-0361-4150-8e8a-c4268dca4893" />



## Project layout

```
bccd_project/
├── app.py                                 # Streamlit UI
├── requirements.txt
├── models/
│   └── best.pt                            # YOLO weights (you provide this — see models/README.md)
├── notebooks/
│   └── bccd_train_and_export.ipynb        # Run on Kaggle/Colab to train YOLO and export weights
└── src/
    ├── classical_pipeline.py              # OpenCV detector
    └── yolo_pipeline.py                   # YOLOv8 wrapper
```

## Getting YOLO weights

1. Open `notebooks/bccd_train_and_export.ipynb` in Kaggle (attach the BCCD dataset, turn on a GPU accelerator) or Colab.
2. Run all cells — it converts annotations, trains (or loads) YOLOv8, and copies the result to `best.pt` at the end.
3. Download that `best.pt` and place it at `models/best.pt` in this project.

Without it, the app still runs fine using just the classical pipeline.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a blood smear image, and pick a detection method in the sidebar.
