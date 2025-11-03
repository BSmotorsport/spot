"""
predict_unseen.py
------------------------------------------------------------
Runs hybrid gaze prediction (geometric + learned regression)
on unseen test images, then compares predicted vs ground truth.

Uses:
 - GazeNet (ONNX)
 - XGBoost regressor
 - Mediapipe face detection

Outputs: predicted coordinates & annotated images.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import joblib
from sklearn.metrics import mean_absolute_error

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_ONNX = r"E:\BOTB\codex\models\temp_model.onnx"
REGRESSOR_PATH = r"E:\BOTB\codex\ball_regressor.joblib"
DATASET_DIR = r"E:\BOTB\gazenet\unseen"
OUTPUT_DIR = r"E:\BOTB\gazenet\output_unseen"
IMG_W, IMG_H = 4416, 3336
DEPTH_SCALE = 1100.0      # smaller = flatter intersection lines
W_GEO, W_REG = 0.238, 0.762   # blend weights (geometry vs regressor)
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load models ---
sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
reg = joblib.load(REGRESSOR_PATH)
face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def preprocess_crop(crop):
    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = crop.astype(np.float32) / 255.0
    return crop.reshape(1, 1, 224, 224)

def angles_to_vector(theta, phi):
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    v = np.array([x, y, z])
    return v / np.linalg.norm(v)

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
results = []
gt_coords = []
pred_coords = []

for fname in os.listdir(DATASET_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    path = os.path.join(DATASET_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    if len(parts) < 3:
        continue
    gt_x, gt_y = int(parts[-2]), int(parts[-1])

    # --- Detect faces ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_detector.process(rgb)
    if not res.detections:
        print(f"[!] {fname}: no faces, skipping")
        continue

    faces = []
    for d in res.detections:
        bbox = d.location_data.relative_bounding_box
        x = int(bbox.xmin * IMG_W)
        y = int(bbox.ymin * IMG_H)
        w = int(bbox.width * IMG_W)
        h = int(bbox.height * IMG_H)
        if w > 0 and h > 0:
            faces.append({"box": [x, y, w, h]})

    faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)[:2]
    if len(faces) < 2:
        print(f"[!] {fname}: only {len(faces)} faces, skipping")
        continue

    # --- Get gaze features for both faces ---
    feats = []
    for f in faces:
        x, y, w, h = f["box"]
        cx, cy = x + w/2, y + h/2
        crop = preprocess_crop(img[y:y+h, x:x+w])
        fg = np.zeros((1,1,625,1), np.float32)
        inp = {
            "input_face_images:0": crop,
            "input_left_images:0": crop,
            "input_right_images:0": crop,
            "input_facegrid:0": fg
        }
        out = sess.run(None, inp)[0][0]
        _,_,_,theta,phi = out
        feats.extend([cx, cy, theta, phi])

    # --- Add geometric features for regressor ---
    f1, f2 = faces[0]['box'], faces[1]['box']
    f1_w, f1_h, f2_w, f2_h = f1[2], f1[3], f2[2], f2[3]
    f1_cx, f1_cy = f1[0] + f1_w/2, f1[1] + f1_h/2
    f2_cx, f2_cy = f2[0] + f2_w/2, f2[1] + f2_h/2
    dist_faces = np.hypot(f2_cx - f1_cx, f2_cy - f1_cy)
    angle_between = np.degrees(np.arctan2(f2_cy - f1_cy, f2_cx - f1_cx))
    feats.extend([f1_w, f1_h, f2_w, f2_h, dist_faces, angle_between])

    # --- Normalise features (same as training) ---
    feats[0] /= IMG_W
    feats[1] /= IMG_H
    feats[4] /= IMG_W
    feats[5] /= IMG_H

    # --------------------------------------------------------
    # 1. Geometric intersection
    # --------------------------------------------------------
    try:
        # origins (approx camera-space)
        z1, z2 = DEPTH_SCALE / f1_w, DEPTH_SCALE / f2_w
        O1 = np.array([f1_cx, f1_cy, z1])
        O2 = np.array([f2_cx, f2_cy, z2])
        D1 = angles_to_vector(feats[2], feats[3])
        D2 = angles_to_vector(feats[6], feats[7])

        A = np.stack([D1, -D2], axis=1)
        b = O2 - O1
        t, s = np.linalg.lstsq(A, b, rcond=None)[0]
        P1, P2 = O1 + t * D1, O2 + s * D2
        P = (P1 + P2) / 2.0
        geo_x, geo_y = P[0], P[1]
    except Exception:
        geo_x, geo_y = 0, 0

    # --------------------------------------------------------
    # 2. Regression correction
    # --------------------------------------------------------
    try:
        reg_pred = reg.predict([feats])[0]
        reg_x = reg_pred[0] * IMG_W
        reg_y = reg_pred[1] * IMG_H
    except Exception:
        reg_x, reg_y = 0, 0

    # --------------------------------------------------------
    # 3. Hybrid blend
    # --------------------------------------------------------
    x_pred = W_GEO * geo_x + W_REG * reg_x
    y_pred = W_GEO * geo_y + W_REG * reg_y

    # --------------------------------------------------------
    # Visualisation
    # --------------------------------------------------------
    gt_coords.append((gt_x, gt_y))
    pred_coords.append((x_pred, y_pred))
    cv2.circle(img, (int(gt_x), int(gt_y)), 20, (0, 255, 0), 3)
    cv2.circle(img, (int(x_pred), int(y_pred)), 20, (0, 0, 255), 3)
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, img)
    print(f"[✓] {fname}: GT=({gt_x}, {gt_y}), Pred=({int(x_pred)}, {int(y_pred)})")

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
if gt_coords and pred_coords:
    mae = mean_absolute_error(gt_coords, pred_coords)
    print(f"\n✅ Mean Absolute Error (pixels): {mae:.1f}")
    print(f"Processed {len(pred_coords)} images → Results in {OUTPUT_DIR}")
else:
    print("No valid predictions.")
