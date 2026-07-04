# Model weights

Put your trained YOLOv8 weights here as `best.pt`.

To get them:
1. Open `notebooks/bccd_train_and_export.ipynb` in Kaggle or Colab (with the BCCD dataset attached, GPU accelerator on).
2. Run all cells. With `TRAIN_NEW_MODEL = True` it trains from scratch; with `False` it loads an existing checkpoint instead.
3. The last cells copy the weights to `/kaggle/working/best.pt` (or equivalent) — download that file.
4. Place it here as `models/best.pt`.

Without this file, the app still works using the classical OpenCV pipeline only (no training needed).
