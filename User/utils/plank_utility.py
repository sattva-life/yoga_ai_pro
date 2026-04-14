from pathlib import Path
from collections import deque, Counter
import time
import pickle
import warnings

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

# =========================================================
# GLOBAL STATE
# =========================================================
PL_POSE_HISTORY = deque(maxlen=10)
PL_SCORE_HISTORY = deque(maxlen=8)
PL_FEEDBACK_HISTORY = deque(maxlen=8)

PL_HIP_HEIGHT_HISTORY = deque(maxlen=20)
PL_HIP_SHIFT_HISTORY = deque(maxlen=20)
PL_SPINE_LINE_HISTORY = deque(maxlen=20)

PL_VISIBILITY_HISTORY = deque(maxlen=8)
PL_DETECTION_HISTORY = deque(maxlen=8)

PL_HOLD_START = None
PL_BEST_HOLD_TIME = 0.0
PL_PERFECT_HOLD_COUNT = 0

PL_POINT_HISTORY = {}
PL_POINT_HISTORY_SIZE = 6

# =========================================================
# MEDIAPIPE
# =========================================================
mp_pose = mp.solutions.pose
plank_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.62,
    min_tracking_confidence=0.62,
)

# =========================================================
# LANDMARK INDEXES
# =========================================================
PL_NOSE = 0
PL_LEFT_SHOULDER = 11
PL_RIGHT_SHOULDER = 12
PL_LEFT_ELBOW = 13
PL_RIGHT_ELBOW = 14
PL_LEFT_WRIST = 15
PL_RIGHT_WRIST = 16
PL_LEFT_HIP = 23
PL_RIGHT_HIP = 24
PL_LEFT_KNEE = 25
PL_RIGHT_KNEE = 26
PL_LEFT_ANKLE = 27
PL_RIGHT_ANKLE = 28

PL_SELECTED_POINTS = [
    PL_NOSE,
    PL_LEFT_SHOULDER, PL_RIGHT_SHOULDER,
    PL_LEFT_ELBOW, PL_RIGHT_ELBOW,
    PL_LEFT_WRIST, PL_RIGHT_WRIST,
    PL_LEFT_HIP, PL_RIGHT_HIP,
    PL_LEFT_KNEE, PL_RIGHT_KNEE,
    PL_LEFT_ANKLE, PL_RIGHT_ANKLE,
]

PL_POINT_NAME_MAP = {
    PL_NOSE: "nose",
    PL_LEFT_SHOULDER: "left_shoulder",
    PL_RIGHT_SHOULDER: "right_shoulder",
    PL_LEFT_ELBOW: "left_elbow",
    PL_RIGHT_ELBOW: "right_elbow",
    PL_LEFT_WRIST: "left_wrist",
    PL_RIGHT_WRIST: "right_wrist",
    PL_LEFT_HIP: "left_hip",
    PL_RIGHT_HIP: "right_hip",
    PL_LEFT_KNEE: "left_knee",
    PL_RIGHT_KNEE: "right_knee",
    PL_LEFT_ANKLE: "left_ankle",
    PL_RIGHT_ANKLE: "right_ankle",
}

PL_ALL_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
PL_ALL_LANDMARK_NAME_TO_INDEX = {name: idx for idx, name in enumerate(PL_ALL_LANDMARK_NAMES)}

PL_FEATURE_COLUMNS = []
for name in PL_ALL_LANDMARK_NAMES:
    PL_FEATURE_COLUMNS.extend([
        f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility",
    ])

PL_ANGLE_COLUMNS = [
    "left_knee_angle", "right_knee_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_hip_angle", "right_hip_angle",
]

PL_MODEL_COLUMNS = PL_FEATURE_COLUMNS + PL_ANGLE_COLUMNS

# =========================================================
# COLORS
# =========================================================
PL_GREEN = "#00ff66"
PL_RED = "#ff3b30"
PL_YELLOW = "#ffd60a"
PL_CYAN = "#40cfff"

# =========================================================
# MODEL LOAD
# =========================================================
PL_BASE_DIR = Path(settings.BASE_DIR)

def resolve_plank_model_path(filename: str) -> Path:
    candidates = [
        PL_BASE_DIR / "Ml_Models" / filename,
        PL_BASE_DIR / "Ml_models" / filename,
        PL_BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    # If not found, ignore to allow rules-based fallback for now
    return None

PLANK_MODEL_PATH = resolve_plank_model_path("plank_best_model.pkl")
PLANK_SCALER_PATH = resolve_plank_model_path("plank_best_scaler.pkl")

plank_model = None
plank_scaler = None

if PLANK_MODEL_PATH and PLANK_SCALER_PATH:
    try:
        with open(PLANK_MODEL_PATH, "rb") as f:
            plank_model = pickle.load(f)
        with open(PLANK_SCALER_PATH, "rb") as f:
            plank_scaler = pickle.load(f)
    except Exception as e:
        print(f"Failed to load Plank ML models: {e}")

# =========================================================
# RESPONSE HELPERS
# =========================================================
def pl_api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})

def pl_api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)

# =========================================================
# TEXT HELPERS
# =========================================================
def pl_clean_text(text):
    return " ".join(str(text).strip().split())

def pl_normalize_key(text):
    return pl_clean_text(text).lower()

def pl_dedupe_list(items, max_items=None, exclude=None):
    exclude = exclude or []
    exclude_keys = {pl_normalize_key(x) for x in exclude if x}
    output = []
    seen = set()
    for item in items:
        if not item: continue
        text = pl_clean_text(item)
        key = pl_normalize_key(text)
        if not key or key in seen or key in exclude_keys: continue
        seen.add(key)
        output.append(text)
        if max_items and len(output) >= max_items: break
    return output

# =========================================================
# SMOOTHING
# =========================================================
def pl_smooth_label(history, new_label):
    history.append(str(new_label))
    return Counter(history).most_common(1)[0][0]

def pl_smooth_score(new_score):
    PL_SCORE_HISTORY.append(float(new_score))
    return int(round(sum(PL_SCORE_HISTORY) / len(PL_SCORE_HISTORY)))

def pl_smooth_feedback(new_feedback):
    PL_FEEDBACK_HISTORY.append(str(new_feedback))
    return Counter(PL_FEEDBACK_HISTORY).most_common(1)[0][0]

def pl_smooth_boolean(history, value):
    history.append(bool(value))
    return sum(history) >= max(1, len(history) // 2 + 1)

def pl_smooth_point(key, x, y, z):
    if key not in PL_POINT_HISTORY:
        PL_POINT_HISTORY[key] = deque(maxlen=PL_POINT_HISTORY_SIZE)
    PL_POINT_HISTORY[key].append((float(x), float(y), float(z)))
    xs = [p[0] for p in PL_POINT_HISTORY[key]]
    ys = [p[1] for p in PL_POINT_HISTORY[key]]
    zs = [p[2] for p in PL_POINT_HISTORY[key]]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys)), float(sum(zs) / len(zs))

def pl_clear_point_history():
    PL_POINT_HISTORY.clear()

def pl_reset_runtime_state():
    global PL_HOLD_START, PL_PERFECT_HOLD_COUNT
    PL_POSE_HISTORY.clear()
    PL_SCORE_HISTORY.clear()
    PL_FEEDBACK_HISTORY.clear()
    PL_HIP_HEIGHT_HISTORY.clear()
    PL_HIP_SHIFT_HISTORY.clear()
    PL_SPINE_LINE_HISTORY.clear()
    PL_VISIBILITY_HISTORY.clear()
    PL_DETECTION_HISTORY.clear()
    pl_clear_point_history()
    PL_HOLD_START = None
    PL_PERFECT_HOLD_COUNT = 0

# =========================================================
# BASIC HELPERS
# =========================================================
def pl_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def pl_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.06, beta=7)

def pl_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = plank_pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark

def pl_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 60, brightness

def pl_calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def pl_distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def pl_moving_std(values):
    if len(values) < 3: return 0.0
    return float(np.std(list(values)))

# =========================================================
# FEATURES
# =========================================================
def pl_extract_raw_landmark_dict(landmarks):
    lm_dict = {}
    for name, idx in PL_ALL_LANDMARK_NAME_TO_INDEX.items():
        lm = landmarks[idx]
        lm_dict[name] = {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)}
    return lm_dict

def pl_normalize_landmarks_inplace(lm_dict):
    left_hip_x = lm_dict["left_hip"]["x"]
    left_hip_y = lm_dict["left_hip"]["y"]
    for name in lm_dict.keys():
        lm_dict[name]["x"] -= left_hip_x
        lm_dict[name]["y"] -= left_hip_y

def pl_build_feature_dataframe_from_landmarks(landmarks):
    lm_dict = pl_extract_raw_landmark_dict(landmarks)
    pl_normalize_landmarks_inplace(lm_dict)

    pts = {name: [vals["x"], vals["y"]] for name, vals in lm_dict.items()}

    angles = {
        "left_knee_angle": pl_calculate_angle(pts["left_hip"], pts["left_knee"], pts["left_ankle"]),
        "right_knee_angle": pl_calculate_angle(pts["right_hip"], pts["right_knee"], pts["right_ankle"]),
        "left_elbow_angle": pl_calculate_angle(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"]),
        "right_elbow_angle": pl_calculate_angle(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"]),
        "left_hip_angle": pl_calculate_angle(pts["left_shoulder"], pts["left_hip"], pts["left_knee"]),
        "right_hip_angle": pl_calculate_angle(pts["right_shoulder"], pts["right_hip"], pts["right_knee"]),
    }

    row = {}
    for name in PL_ALL_LANDMARK_NAMES:
        row[f"{name}_x"] = lm_dict[name]["x"]
        row[f"{name}_y"] = lm_dict[name]["y"]
        row[f"{name}_z"] = lm_dict[name]["z"]
        row[f"{name}_visibility"] = lm_dict[name]["visibility"]
    for key, value in angles.items():
        row[key] = value

    features_df = pd.DataFrame([row], columns=PL_MODEL_COLUMNS)
    return features_df, lm_dict, angles

def pl_predict_model_label(features_df):
    if plank_model is None or plank_scaler is None:
        return "Unknown", 0.0

    features_df = features_df.astype(np.float32).copy()
    X_array = features_df.to_numpy(dtype=np.float32)

    try:
        scaled_features = plank_scaler.transform(X_array)
        prediction = plank_model.predict(scaled_features)[0]
        confidence = 0.50
        if hasattr(plank_model, "predict_proba"):
            probs = plank_model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probs))
        return str(prediction), confidence
    except Exception as e:
        return "Unknown", 0.0

# =========================================================
# VISIBILITY / SIDE SELECTION
# =========================================================
def pl_side_visibility(landmarks, side):
    idxs = [PL_LEFT_SHOULDER, PL_LEFT_HIP, PL_LEFT_KNEE, PL_LEFT_ANKLE] if side == "left" else [PL_RIGHT_SHOULDER, PL_RIGHT_HIP, PL_RIGHT_KNEE, PL_RIGHT_ANKLE]
    return float(np.mean([float(landmarks[i].visibility) for i in idxs]))

def pl_pick_dominant_side(landmarks):
    left_vis = pl_side_visibility(landmarks, "left")
    right_vis = pl_side_visibility(landmarks, "right")
    if left_vis >= right_vis:
        return "left", left_vis, right_vis
    return "right", right_vis, left_vis

def pl_check_body_visibility(lm_dict):
    core_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    visibilities = [lm_dict[name]["visibility"] for name in core_names]
    visible_count = sum(v > 0.35 for v in visibilities)
    return visible_count >= 5, visible_count, float(np.mean(visibilities))

def pl_check_frame_position(raw_pts):
    xs = [float(p[0]) for p in raw_pts[PL_SELECTED_POINTS]]
    ys = [float(p[1]) for p in raw_pts[PL_SELECTED_POINTS]]
    min_x, max_x = min(xs), max(xs)
    width = max_x - min_x
    feedback = []
    if width > 0.98: feedback.append("Move slightly away to show full length")
    if width < 0.40: feedback.append("Move closer so your body fills the frame horizontally")
    return feedback

# =========================================================
# ANALYSIS
# =========================================================
def analyze_plank_pose(raw_pts, landmarks):
    dom_side, dom_vis, _ = pl_pick_dominant_side(landmarks)

    if dom_side == "left":
        s, e, w = raw_pts[PL_LEFT_SHOULDER], raw_pts[PL_LEFT_ELBOW], raw_pts[PL_LEFT_WRIST]
        h, k, a = raw_pts[PL_LEFT_HIP], raw_pts[PL_LEFT_KNEE], raw_pts[PL_LEFT_ANKLE]
    else:
        s, e, w = raw_pts[PL_RIGHT_SHOULDER], raw_pts[PL_RIGHT_ELBOW], raw_pts[PL_RIGHT_WRIST]
        h, k, a = raw_pts[PL_RIGHT_HIP], raw_pts[PL_RIGHT_KNEE], raw_pts[PL_RIGHT_ANKLE]

    shoulder_center = (raw_pts[PL_LEFT_SHOULDER] + raw_pts[PL_RIGHT_SHOULDER]) / 2.0
    hip_center = (raw_pts[PL_LEFT_HIP] + raw_pts[PL_RIGHT_HIP]) / 2.0
    ankle_center = (raw_pts[PL_LEFT_ANKLE] + raw_pts[PL_RIGHT_ANKLE]) / 2.0
    
    torso_size = pl_distance(shoulder_center[:2], hip_center[:2]) + 1e-6
    
    hip_angle = pl_calculate_angle(s[:2], h[:2], k[:2])
    knee_angle = pl_calculate_angle(h[:2], k[:2], a[:2])
    elbow_angle = pl_calculate_angle(s[:2], e[:2], w[:2])
    shoulder_angle = pl_calculate_angle(h[:2], s[:2], e[:2])

    # Hip Alignment Check (Straight line from Shoulder to Ankle)
    # Estimate ideal hip Y based on shoulder and ankle
    ideal_hip_y = (s[1] + a[1]) / 2.0
    hip_deviation = (h[1] - ideal_hip_y) / torso_size

    hips_too_high = hip_deviation < -0.22 # Hip is significantly higher than the line (negative Y is up)
    hips_sagging = hip_deviation > 0.15 # Hip is dropping too low

    knees_bent = knee_angle < 155
    head_dropped = raw_pts[PL_NOSE][1] > s[1] + 0.15
    
    is_horizontal = abs(s[1] - a[1]) / torso_size < 0.6  # Body roughly horizontal

    is_plank_gate = is_horizontal and (elbow_angle < 120 or elbow_angle > 150) # Forearm or Full plank

    score = 0
    status = "warning"
    pose_label = "Not Plank"
    main_feedback = "Get into a push-up or forearm plank position."
    tips = ["Keep your body in a straight line from head to heels."]

    if not is_horizontal:
        score = 25
        main_feedback = "Lower your body to form a horizontal line."
        tips = ["Bring shoulders and hips parallel to the floor."]
    elif knees_bent:
        score = 55
        main_feedback = "Straighten your legs completely."
        tips = ["Push back through your heels and squeeze your thighs."]
    elif hips_too_high:
        score = 65
        main_feedback = "Lower your hips."
        tips = ["Your hips are peaking up. Bring them down into a straight line."]
    elif hips_sagging:
        score = 65
        main_feedback = "Lift your hips up."
        tips = ["Your hips are dropping. Engage your core to lift them."]
    elif head_dropped:
        score = 80
        main_feedback = "Lift your head slightly."
        tips = ["Look down at the mat slightly ahead of your hands, keep neck neutral."]
    elif is_plank_gate:
        score = 100
        status = "perfect"
        pose_label = "Correct Plank"
        main_feedback = "Perfect Plank! Hold steady."
        tips = ["Excellent straight line.", "Keep your core tight and breathe."]
    else:
        score = 85
        status = "good"
        pose_label = "Plank Pose"
        main_feedback = "Good Plank. Hold steady."
        tips = ["Keep your core engaged.", "Maintain the straight line."]

    checks = {
        "dominant_side": dom_side,
        "is_horizontal": is_horizontal,
        "knees_bent": knees_bent,
        "hips_too_high": hips_too_high,
        "hips_sagging": hips_sagging,
        "head_dropped": head_dropped,
        "is_plank_gate": is_plank_gate
    }

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": tips,
        "angles": {
            "left_knee_angle": round(float(pl_calculate_angle(raw_pts[PL_LEFT_HIP][:2], raw_pts[PL_LEFT_KNEE][:2], raw_pts[PL_LEFT_ANKLE][:2])), 1),
            "right_knee_angle": round(float(pl_calculate_angle(raw_pts[PL_RIGHT_HIP][:2], raw_pts[PL_RIGHT_KNEE][:2], raw_pts[PL_RIGHT_ANKLE][:2])), 1),
            "left_elbow_angle": round(float(pl_calculate_angle(raw_pts[PL_LEFT_SHOULDER][:2], raw_pts[PL_LEFT_ELBOW][:2], raw_pts[PL_LEFT_WRIST][:2])), 1),
            "right_elbow_angle": round(float(pl_calculate_angle(raw_pts[PL_RIGHT_SHOULDER][:2], raw_pts[PL_RIGHT_ELBOW][:2], raw_pts[PL_RIGHT_WRIST][:2])), 1),
            "hip_angle": round(float(hip_angle), 1),
            "shoulder_angle": round(float(shoulder_angle), 1)
        },
        "checks": checks,
    }

def pl_is_plank_like(model_label, model_confidence, analysis):
    if ("plank" in str(model_label).lower()) and model_confidence >= 0.58:
        return True
    
    checks = analysis["checks"]
    if checks.get("is_horizontal") and not checks.get("knees_bent"):
        return True
    return False

# =========================================================
# STABILITY / HOLD / QUALITY
# =========================================================
def pl_update_stability_metrics(raw_pts):
    hip_center = (raw_pts[PL_LEFT_HIP] + raw_pts[PL_RIGHT_HIP]) / 2.0
    shoulder_center = (raw_pts[PL_LEFT_SHOULDER] + raw_pts[PL_RIGHT_SHOULDER]) / 2.0

    PL_HIP_HEIGHT_HISTORY.append(float(hip_center[1]))
    PL_HIP_SHIFT_HISTORY.append(float(hip_center[0]))
    PL_SPINE_LINE_HISTORY.append(abs(float(shoulder_center[1] - hip_center[1])))

def pl_get_stability_feedback():
    hip_wobble = pl_moving_std(PL_HIP_HEIGHT_HISTORY)
    shift_wobble = pl_moving_std(PL_HIP_SHIFT_HISTORY)
    
    feedback = []
    penalty = 0

    if hip_wobble > 0.015:
        feedback.append("Keep your hips steadier")
        penalty += 4
    if shift_wobble > 0.015:
        feedback.append("Stabilize your core")
        penalty += 3

    return feedback, penalty

def pl_update_hold_state(is_pose, full_body_visible, low_light):
    global PL_HOLD_START, PL_BEST_HOLD_TIME

    valid_hold = is_pose and full_body_visible and not low_light
    if valid_hold:
        if PL_HOLD_START is None:
            PL_HOLD_START = time.time()
        hold_time = time.time() - PL_HOLD_START
        PL_BEST_HOLD_TIME = max(PL_BEST_HOLD_TIME, hold_time)
    else:
        hold_time = 0.0
        PL_HOLD_START = None

    return hold_time, PL_BEST_HOLD_TIME

def pl_hold_bonus(hold_time):
    if hold_time >= 15: return 10
    if hold_time >= 10: return 8
    if hold_time >= 5: return 5
    if hold_time >= 3: return 2
    return 0

def pl_quality_from_score(score):
    if score >= 96: return "Perfect_Plank"
    if score >= 85: return "Good_Plank"
    if score >= 70: return "Needs_Correction"
    return "Not_Ready"

def pl_build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    dom_side = checks.get("dominant_side", "left")
    dominant_idxs = (
        [PL_LEFT_SHOULDER, PL_LEFT_ELBOW, PL_LEFT_WRIST, PL_LEFT_HIP, PL_LEFT_KNEE, PL_LEFT_ANKLE]
        if dom_side == "left" else
        [PL_RIGHT_SHOULDER, PL_RIGHT_ELBOW, PL_RIGHT_WRIST, PL_RIGHT_HIP, PL_RIGHT_KNEE, PL_RIGHT_ANKLE]
    )

    points = []
    for idx in PL_SELECTED_POINTS:
        visibility = float(landmarks[idx].visibility)
        if visibility < 0.25: continue

        is_dominant = idx in dominant_idxs
        radius = 7 if is_dominant else 5
        color = PL_GREEN if analysis["score"] >= 90 and is_dominant else PL_YELLOW

        if idx in [PL_LEFT_HIP, PL_RIGHT_HIP] and (checks.get("hips_too_high") or checks.get("hips_sagging")):
            color = PL_RED; radius = 8
        if idx in [PL_LEFT_KNEE, PL_RIGHT_KNEE] and checks.get("knees_bent"):
            color = PL_RED; radius = 8
        if idx == PL_NOSE and checks.get("head_dropped"):
            color = PL_RED; radius = 8

        points.append({
            "name": PL_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
        })
    return points

def pl_build_angle_texts(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    dom_side = checks.get("dominant_side", "left")
    items = []

    if dom_side == "left":
        primary = [(PL_LEFT_HIP, analysis["angles"].get("hip_angle", 0), PL_YELLOW),
                   (PL_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), PL_YELLOW),
                   (PL_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), PL_YELLOW)]
    else:
        primary = [(PL_RIGHT_HIP, analysis["angles"].get("hip_angle", 0), PL_YELLOW),
                   (PL_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), PL_YELLOW),
                   (PL_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), PL_YELLOW)]

    for idx, value, color in primary:
        if float(landmarks[idx].visibility) < 0.30: continue
        items.append({
            "text": f"{int(round(float(value)))}°",
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
        })
    return items

# =========================================================
# MAIN PROCESS
# =========================================================
def process_plank_pose_request(request):
    global PL_PERFECT_HOLD_COUNT, PL_HOLD_START

    try:
        uploaded_file = request.FILES["image"]
        frame = pl_read_uploaded_image(uploaded_file)

        if frame is None: return pl_api_error("Invalid image file", status=400)
        frame = pl_enhance_frame(frame)

        low_light, brightness = pl_check_lighting(frame)
        if low_light:
            pl_reset_runtime_state()
            return pl_api_success(
                pose="Low Light", status="warning", feedback="Room lighting is too low.", coach_text="Improve lighting.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(PL_BEST_HOLD_TIME), 1),
                angles={}, details=["Increase room lighting", "Avoid dark background"], perfect_hold=False, points=[], angle_texts=[]
            )

        landmarks = pl_detect_landmarks(frame)
        has_landmarks = landmarks is not None
        stable_has_landmarks = pl_smooth_boolean(PL_DETECTION_HISTORY, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            pl_reset_runtime_state()
            return pl_api_success(
                pose="Unknown", status="unknown", feedback="No pose detected.", coach_text="Show your full body sideways.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(PL_BEST_HOLD_TIME), 1),
                angles={}, details=["Show full body sideways", "Move slightly back"], perfect_hold=False, points=[], angle_texts=[]
            )

        if landmarks is None:
            pl_clear_point_history()
            return pl_api_success(
                pose="Tracking...", status="warning", feedback="Hold still.", coach_text="Keep steady.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(PL_BEST_HOLD_TIME), 1),
                angles={}, details=[], perfect_hold=False, points=[], angle_texts=[]
            )

        features_df, lm_dict, _ = pl_build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = raw_pts.copy()

        for idx in PL_SELECTED_POINTS:
            sx, sy, sz = pl_smooth_point(f"pl_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])
            smoothed_pts[idx] = [sx, sy, sz]

        full_body_visible, visible_count, avg_visibility = pl_check_body_visibility(lm_dict)
        stable_full_body_visible = pl_smooth_boolean(PL_VISIBILITY_HISTORY, full_body_visible)
        framing_feedback = pl_check_frame_position(smoothed_pts)

        if not full_body_visible and not stable_full_body_visible:
            pl_reset_runtime_state()
            details = ["Show shoulders, hips, and ankles clearly in frame"] + framing_feedback
            return pl_api_success(
                pose="Body Not Visible", status="warning", feedback="Adjust camera for full side profile.", coach_text="Show full body sideways.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(PL_BEST_HOLD_TIME), 1),
                angles={}, details=pl_dedupe_list(details, max_items=3), perfect_hold=False, points=[], angle_texts=[]
            )

        raw_model_label, confidence = pl_predict_model_label(features_df)
        stable_model_label = pl_smooth_label(PL_POSE_HISTORY, raw_model_label)

        analysis = analyze_plank_pose(smoothed_pts, landmarks)
        points = pl_build_points_for_frontend(smoothed_pts, landmarks, analysis)
        angle_texts = pl_build_angle_texts(smoothed_pts, landmarks, analysis)

        is_plank = pl_is_plank_like(stable_model_label, confidence, analysis)

        if not is_plank:
            PL_PERFECT_HOLD_COUNT = 0
            PL_HOLD_START = None
            tips = analysis.get("tips", []) + ["Get into a horizontal push-up or forearm position"] + framing_feedback
            return pl_api_success(
                pose="Not Plank Pose", model_pose=stable_model_label, quality="Not_Ready",
                feedback=analysis.get("main_feedback", "Move into Plank position."),
                coach_text=analysis.get("main_feedback", "Move into Plank position."),
                status="warning", confidence=round(float(confidence), 3), score=max(0, min(65, analysis.get("score", 0))),
                hold_time=0.0, best_hold_time=round(float(PL_BEST_HOLD_TIME), 1), angles=analysis.get("angles", {}),
                details=pl_dedupe_list(tips, max_items=3), perfect_hold=False, points=points, angle_texts=angle_texts
            )

        pl_update_stability_metrics(smoothed_pts)
        stability_tips, stability_penalty = pl_get_stability_feedback()

        base_score = analysis["score"]
        combined_score = max(0, base_score - stability_penalty)

        hold_time, best_hold = pl_update_hold_state(is_plank, stable_full_body_visible, low_light)
        combined_score = min(100, combined_score + pl_hold_bonus(hold_time))
        stable_score = pl_smooth_score(combined_score)

        if analysis["checks"].get("is_plank_gate") and stable_score >= 96 and hold_time >= 3.0:
            pose_name = "Correct Plank"
            status = "perfect"
            feedback_text = "Perfect Plank"
            coach_text = "Excellent. Hold steady and breathe."
        elif stable_score >= 85:
            pose_name = "Plank Pose"
            status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = "Good shape. Refine and hold steady."
        else:
            pose_name = "Plank Needs Correction"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "Fix the highlighted alignment."

        stable_feedback = pl_smooth_feedback(feedback_text)

        if pose_name == "Correct Plank" and hold_time >= 3.2:
            PL_PERFECT_HOLD_COUNT += 1
        else:
            PL_PERFECT_HOLD_COUNT = 0

        tips = []
        if hold_time >= 5: tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        return pl_api_success(
            pose=pose_name, model_pose=stable_model_label, quality=pl_quality_from_score(stable_score),
            feedback=stable_feedback, coach_text=coach_text, status=status, confidence=round(float(confidence), 3),
            score=stable_score, hold_time=round(float(hold_time), 1), best_hold_time=round(float(best_hold), 1),
            angles=analysis["angles"], details=pl_dedupe_list(tips, max_items=3, exclude=[stable_feedback, coach_text]),
            perfect_hold=PL_PERFECT_HOLD_COUNT >= 3, points=points, angle_texts=angle_texts
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e), "error_type": type(e).__name__}, status=500)