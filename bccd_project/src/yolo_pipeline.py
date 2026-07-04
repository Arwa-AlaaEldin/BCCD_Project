"""Thin wrapper around a YOLOv8 model fine-tuned on the BCCD dataset."""

CLASS_NAMES = ["WBC", "RBC", "Platelets"]


class YoloCellDetector:
    def __init__(self, weights_path):
        from ultralytics import YOLO  # imported lazily so classical-only usage doesn't need it installed

        self.model = YOLO(weights_path)

    def run_pipeline(self, image, conf=0.25):
        """image: BGR np.ndarray. Returns counts + detections + annotated image (BGR)."""
        result = self.model.predict(image, conf=conf, verbose=False)[0]

        counts = {name: 0 for name in CLASS_NAMES}
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names.get(cls_id, str(cls_id))
            if label in counts:
                counts[label] += 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"classTitle": label, "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]})

        annotated = result.plot()  # BGR np.ndarray, same convention as the classical pipeline

        return {
            "wbc_count": counts.get("WBC", 0),
            "rbc_count": counts.get("RBC", 0),
            "platelet_count": counts.get("Platelets", 0),
            "detections": detections,
            "annotated_image": annotated,
        }
