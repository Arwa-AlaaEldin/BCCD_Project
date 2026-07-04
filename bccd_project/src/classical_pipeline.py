"""Classical OpenCV pipeline for detecting/counting RBCs, WBCs and Platelets.

No training required. WBCs and platelets are found via CLAHE + HSV color
masking, RBCs via CLAHE on the green channel + adaptive thresholding +
Watershed (to split clustered/overlapping cells).
"""

import cv2
import numpy as np

BOX_COLORS = {"WBC": (0, 0, 255), "RBC": (0, 255, 0), "Platelets": (255, 255, 0)}


def apply_CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def apply_gaussian_blur(image, kernel_size=(7, 7)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def get_hsv_mask(hsv_image, lower_thresh, upper_thresh):
    return cv2.inRange(hsv_image, lower_thresh, upper_thresh)


def clean_wbc_morphology(mask):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    return cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close, iterations=2)


def filter_wbc_by_size(mask, min_area=1500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            final_mask[labels == i] = 255
            count += 1
    return final_mask, count


def get_wbc_mask(hsv_image, lower_thresh, upper_thresh):
    raw_mask = get_hsv_mask(hsv_image, lower_thresh, upper_thresh)
    morph_mask = clean_wbc_morphology(raw_mask)
    return filter_wbc_by_size(morph_mask, min_area=1500)


def platelets_remove_wbc_mask(candidate_mask, wbc_mask):
    kernel = np.ones((9, 9), np.uint8)
    wbc_dilated = cv2.dilate(wbc_mask, kernel, iterations=10)
    wbc_inverted = cv2.bitwise_not(wbc_dilated)
    return cv2.bitwise_and(candidate_mask, candidate_mask, mask=wbc_inverted)


def filter_platelets_by_shape(mask, min_area=50, max_area=3000, min_circularity=0.75):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(cleaned_mask)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > min_circularity:
                cv2.drawContours(final_mask, [c], -1, 255, -1)
                count += 1
    return final_mask, count


def get_platelet_mask(hsv_image, wbc_mask, lower_thresh, upper_thresh):
    color_candidates = get_hsv_mask(hsv_image, lower_thresh, upper_thresh)
    wbc_free_candidates = platelets_remove_wbc_mask(color_candidates, wbc_mask)
    return filter_platelets_by_shape(wbc_free_candidates)


def rbc_extract_and_clahe(image):
    g = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def rbc_apply_thresholding(g_blurred):
    return cv2.adaptiveThreshold(
        g_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=201, C=2
    )


def rbc_remove_wbc_mask(wbc_mask, thresh):
    wbc_dilated = cv2.dilate(wbc_mask, np.ones((5, 5), np.uint8), iterations=2)
    thresh[wbc_dilated > 0] = 0
    return thresh


def rbc_get_sure_bg(thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    return sure_bg, closing


def rbc_get_sure_foreground(closing, sure_bg):
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    dist_transform = cv2.GaussianBlur(dist_transform, (3, 3), 0)
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    return sure_fg, unknown


def rbc_apply_watershed(sure_fg, unknown, image):
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return cv2.watershed(image, markers)


def rbc_size_filtering(typical_rbc_area, markers):
    min_thresh = typical_rbc_area * 0.4
    clump_thresh = typical_rbc_area * 1.8
    final_count = 0
    new_markers = markers.copy()

    for label in np.unique(markers):
        if label <= 1:
            continue
        area = np.sum(markers == label)
        if area < min_thresh:
            new_markers[markers == label] = 1
        elif area > clump_thresh:
            final_count += int(round(area / typical_rbc_area))
        else:
            final_count += 1

    return new_markers, final_count


def get_rbc_watershed(image, wbc_mask, typical_rbc_area=6000):
    g_clahe = rbc_extract_and_clahe(image)
    g_blurred = apply_gaussian_blur(g_clahe, (5, 5))
    thresh = rbc_apply_thresholding(g_blurred)
    thresh = rbc_remove_wbc_mask(wbc_mask, thresh)
    sure_bg, closing = rbc_get_sure_bg(thresh)
    sure_fg, unknown = rbc_get_sure_foreground(closing, sure_bg)
    markers = rbc_apply_watershed(sure_fg, unknown, image)
    return rbc_size_filtering(typical_rbc_area, markers)


def markers_to_detections(markers, wbc_mask, platelet_mask, padding=10):
    """Converts segmentation markers/masks into padded bounding boxes."""
    detections = []
    h_img, w_img = markers.shape[:2]

    def get_padded_box(x, y, w, h, pad, max_w, max_h):
        new_x, new_y = max(0, x - pad), max(0, y - pad)
        new_w = min(max_w - new_x, w + 2 * pad)
        new_h = min(max_h - new_y, h + 2 * pad)
        return [int(new_x), int(new_y), int(new_w), int(new_h)]

    for label in np.unique(markers):
        if label <= 1:
            continue
        cell_mask = np.zeros(markers.shape, dtype="uint8")
        cell_mask[markers == label] = 255
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x, y, w, h = cv2.boundingRect(cnts[0])
            detections.append({"classTitle": "RBC", "box": get_padded_box(x, y, w, h, padding, w_img, h_img)})

    cnts_wbc, _ = cv2.findContours(wbc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_wbc:
        x, y, w, h = cv2.boundingRect(c)
        detections.append({"classTitle": "WBC", "box": get_padded_box(x, y, w, h, padding + 10, w_img, h_img)})

    cnts_plat, _ = cv2.findContours(platelet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_plat:
        x, y, w, h = cv2.boundingRect(c)
        detections.append({"classTitle": "Platelets", "box": get_padded_box(x, y, w, h, padding, w_img, h_img)})

    return detections


def draw_detections(image, detections):
    """Draws boxes+labels on a copy of a BGR image using OpenCV (fast, no matplotlib)."""
    vis = image.copy()
    for obj in detections:
        label = obj["classTitle"]
        x, y, w, h = obj["box"]
        color = BOX_COLORS.get(label, (0, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


class ClassicalCellDetector:
    def __init__(self):
        self.lower_purple = np.array([79, 77, 109])
        self.upper_purple = np.array([151, 212, 212])
        self.typical_rbc_area = 6000

    def detect_wbc(self, image):
        hsv = cv2.cvtColor(apply_gaussian_blur(image), cv2.COLOR_BGR2HSV)
        return get_wbc_mask(hsv, self.lower_purple, self.upper_purple)

    def detect_platelets(self, image, wbc_mask):
        hsv = cv2.cvtColor(apply_gaussian_blur(image), cv2.COLOR_BGR2HSV)
        return get_platelet_mask(hsv, wbc_mask, self.lower_purple, self.upper_purple)

    def detect_rbc(self, image, wbc_mask):
        return get_rbc_watershed(image, wbc_mask, typical_rbc_area=self.typical_rbc_area)

    def run_pipeline(self, image):
        """image: BGR np.ndarray (as read by cv2.imread). Returns counts + detections + annotated image."""
        clahe_image = apply_CLAHE(image)
        wbc_mask, wbc_count = self.detect_wbc(clahe_image.copy())
        platelet_mask, platelet_count = self.detect_platelets(clahe_image.copy(), wbc_mask)
        rbc_markers, rbc_count = self.detect_rbc(image.copy(), wbc_mask)

        detections = markers_to_detections(rbc_markers, wbc_mask, platelet_mask)
        annotated = draw_detections(image, detections)

        return {
            "wbc_count": wbc_count,
            "rbc_count": rbc_count,
            "platelet_count": platelet_count,
            "detections": detections,
            "annotated_image": annotated,
        }
