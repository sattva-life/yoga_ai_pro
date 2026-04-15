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
WR_POSE_HISTORY = deque(maxlen=10)
WR_SCORE_HISTORY = deque(maxlen=8)
WR_FEEDBACK_HISTORY = deque(maxlen=8)

WR_HIP_HEIGHT_HISTORY = deque(maxlen=20)
WR_HIP_SHIFT_HISTORY = deque(maxlen=20)
WR_SPINE_LINE_HISTORY = deque(maxlen=20)

WR_VISIBILITY_HISTORY = deque(maxlen=8)
WR_DETECTION_HISTORY = deque(maxlen=8)

WR_HOLD_START = None
WR_BEST_HOLD_TIME = 0.0
WR_PERFECT_HOLD_COUNT = 0

WR_POINT_HISTORY = {}
WR_POINT_HISTORY_SIZE = 6

# =========================================================
# MEDIAPIPE
# =========================================================
mp_pose = mp.solutions.pose
warrior_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.62,
    min_tracking_confidence=0.62,
)

# =========================================================
# LANDMARK INDEXES
# =========================================================
WR_NOSE = 0
WR_LEFT_SHOULDER = 11
WR_RIGHT_SHOULDER = 12
WR_LEFT_ELBOW = 13
WR_RIGHT_ELBOW = 14
WR_LEFT_WRIST = 15
WR_RIGHT_WRIST = 16
WR_LEFT_HIP = 23
WR_RIGHT_HIP = 24
WR_LEFT_KNEE = 25
WR_RIGHT_KNEE = 26
WR_LEFT_ANKLE = 27
WR_RIGHT_ANKLE = 28

WR_SELECTED_POINTS = [
    WR_NOSE,
    WR_LEFT_SHOULDER, WR_RIGHT_SHOULDER,
    WR_LEFT_ELBOW, WR_RIGHT_ELBOW,
    WR_LEFT_WRIST, WR_RIGHT_WRIST,
    WR_LEFT_HIP, WR_RIGHT_HIP,
    WR_LEFT_KNEE, WR_RIGHT_KNEE,
    WR_LEFT_ANKLE, WR_RIGHT_ANKLE,
]

WR_POINT_NAME_MAP = {
    WR_NOSE: "nose",
    WR_LEFT_SHOULDER: "left_shoulder",
    WR_RIGHT_SHOULDER: "right_shoulder",
    WR_LEFT_ELBOW: "left_elbow",
    WR_RIGHT_ELBOW: "right_elbow",
    WR_LEFT_WRIST: "left_wrist",
    WR_RIGHT_WRIST: "right_wrist",
    WR_LEFT_HIP: "left_hip",
    WR_RIGHT_HIP: "right_hip",
    WR_LEFT_KNEE: "left_knee",
    WR_RIGHT_KNEE: "right_knee",
    WR_LEFT_ANKLE: "left_ankle",
    WR_RIGHT_ANKLE: "right_ankle",
}

WR_ALL_LANDMARK_NAMES = [
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
WR_ALL_LANDMARK_NAME_TO_INDEX = {name: idx for idx, name in enumerate(WR_ALL_LANDMARK_NAMES)}

WR_FEATURE_COLUMNS = []
for name in WR_ALL_LANDMARK_NAMES:
    WR_FEATURE_COLUMNS.extend([
        f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility",
    ])

WR_ANGLE_COLUMNS = [
    "left_knee_angle", "right_knee_angle",
    "left_elbow_angle", "right_elbow_angle",
]

WR_MODEL_COLUMNS = WR_FEATURE_COLUMNS + WR_ANGLE_COLUMNS

# =========================================================
# COLORS
# =========================================================
WR_GREEN = "#00ff66"
WR_RED = "#ff3b30"
WR_YELLOW = "#ffd60a"
WR_CYAN = "#40cfff"

# =========================================================
# MODEL LOAD
# =========================================================
WR_BASE_DIR = Path(settings.BASE_DIR)

def resolve_warrior_model_path(filename: str) -> Path:
    candidates = [
        WR_BASE_DIR / "Ml_Models" / filename,
        WR_BASE_DIR / "Ml_models" / filename,
        WR_BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

WARRIOR_POSE_MODEL_PATH = resolve_warrior_model_path("warrior_pose_model.pkl")
WARRIOR_POSE_ENCODER_PATH = resolve_warrior_model_path("warrior_label_encoder.pkl")
WARRIOR_DEFECT_MODEL_PATH = resolve_warrior_model_path("warrior_defect_model.pkl")
WARRIOR_DEFECT_ENCODER_PATH = resolve_warrior_model_path("warrior_defect_label_encoder.pkl")

rf_pose, le_pose = None, None
rf_def, le_def = None, None

try:
    if WARRIOR_POSE_MODEL_PATH:
        with open(WARRIOR_POSE_MODEL_PATH, "rb") as f: rf_pose = pickle.load(f)
        with open(WARRIOR_POSE_ENCODER_PATH, "rb") as f: le_pose = pickle.load(f)
    if WARRIOR_DEFECT_MODEL_PATH:
        with open(WARRIOR_DEFECT_MODEL_PATH, "rb") as f: rf_def = pickle.load(f)
        with open(WARRIOR_DEFECT_ENCODER_PATH, "rb") as f: le_def = pickle.load(f)
except Exception as e:
    print(f"Failed to load Warrior ML models: {e}")

# =========================================================
# RESPONSE HELPERS
# =========================================================
def wr_api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})

def wr_api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)

# =========================================================
# TEXT HELPERS
# =========================================================
def wr_clean_text(text):
    return " ".join(str(text).strip().split())

def wr_normalize_key(text):
    return wr_clean_text(text).lower()

def wr_dedupe_list(items, max_items=None, exclude=None):
    exclude = exclude or []
    exclude_keys = {wr_normalize_key(x) for x in exclude if x}
    output = []
    seen = set()
    for item in items:
        if not item: continue
        text = wr_clean_text(item)
        key = wr_normalize_key(text)
        if not key or key in seen or key in exclude_keys: continue
        seen.add(key)
        output.append(text)
        if max_items and len(output) >= max_items: break
    return output

# =========================================================
# SMOOTHING
# =========================================================
def wr_smooth_label(history, new_label):
    history.append(str(new_label))
    return Counter(history).most_common(1)[0][0]

def wr_smooth_score(new_score):
    WR_SCORE_HISTORY.append(float(new_score))
    return int(round(sum(WR_SCORE_HISTORY) / len(WR_SCORE_HISTORY)))

def wr_smooth_feedback(new_feedback):
    WR_FEEDBACK_HISTORY.append(str(new_feedback))
    return Counter(WR_FEEDBACK_HISTORY).most_common(1)[0][0]

def wr_smooth_boolean(history, value):
    history.append(bool(value))
    return sum(history) >= max(1, len(history) // 2 + 1)

def wr_smooth_point(key, x, y, z):
    if key not in WR_POINT_HISTORY:
        WR_POINT_HISTORY[key] = deque(maxlen=WR_POINT_HISTORY_SIZE)
    WR_POINT_HISTORY[key].append((float(x), float(y), float(z)))
    xs = [p[0] for p in WR_POINT_HISTORY[key]]
    ys = [p[1] for p in WR_POINT_HISTORY[key]]
    zs = [p[2] for p in WR_POINT_HISTORY[key]]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys)), float(sum(zs) / len(zs))

def wr_clear_point_history():
    WR_POINT_HISTORY.clear()

def wr_reset_runtime_state():
    global WR_HOLD_START, WR_PERFECT_HOLD_COUNT
    WR_POSE_HISTORY.clear()
    WR_SCORE_HISTORY.clear()
    WR_FEEDBACK_HISTORY.clear()
    WR_HIP_HEIGHT_HISTORY.clear()
    WR_HIP_SHIFT_HISTORY.clear()
    WR_SPINE_LINE_HISTORY.clear()
    WR_VISIBILITY_HISTORY.clear()
    WR_DETECTION_HISTORY.clear()
    wr_clear_point_history()
    WR_HOLD_START = None
    WR_PERFECT_HOLD_COUNT = 0

# =========================================================
# BASIC HELPERS
# =========================================================
def wr_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def wr_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.06, beta=7)

def wr_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = warrior_pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark

def wr_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 60, brightness

def wr_calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def wr_distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def wr_moving_std(values):
    if len(values) < 3: return 0.0
    return float(np.std(list(values)))

# =========================================================
# FEATURES
# =========================================================
def wr_extract_raw_landmark_dict(landmarks):
    lm_dict = {}
    for name, idx in WR_ALL_LANDMARK_NAME_TO_INDEX.items():
        lm = landmarks[idx]
        lm_dict[name] = {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)}
    return lm_dict

def wr_normalize_landmarks_inplace(lm_dict):
    left_hip_x = lm_dict["left_hip"]["x"]
    left_hip_y = lm_dict["left_hip"]["y"]
    for name in lm_dict.keys():
        lm_dict[name]["x"] -= left_hip_x
        lm_dict[name]["y"] -= left_hip_y

def wr_build_feature_dataframe_from_landmarks(landmarks):
    lm_dict = wr_extract_raw_landmark_dict(landmarks)
    wr_normalize_landmarks_inplace(lm_dict)

    pts = {name: [vals["x"], vals["y"]] for name, vals in lm_dict.items()}

    angles = {
        "left_knee_angle": wr_calculate_angle(pts["left_hip"], pts["left_knee"], pts["left_ankle"]),
        "right_knee_angle": wr_calculate_angle(pts["right_hip"], pts["right_knee"], pts["right_ankle"]),
        "left_elbow_angle": wr_calculate_angle(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"]),
        "right_elbow_angle": wr_calculate_angle(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"]),
    }

    row = {}
    for name in WR_ALL_LANDMARK_NAMES:
        row[f"{name}_x"] = lm_dict[name]["x"]
        row[f"{name}_y"] = lm_dict[name]["y"]
        row[f"{name}_z"] = lm_dict[name]["z"]
        row[f"{name}_visibility"] = lm_dict[name]["visibility"]
    for key, value in angles.items():
        row[key] = value

    features_df = pd.DataFrame([row], columns=WR_MODEL_COLUMNS)
    return features_df, lm_dict, angles

def wr_predict_model_label(features_df):
    if rf_pose is None or le_pose is None:
        return "Unknown", 0.0

    # Ensure correct column order
    expected_cols = getattr(rf_pose, "feature_names_in_", WR_MODEL_COLUMNS)
    
    # Fill missing columns with 0 if necessary
    for col in expected_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    temp = features_df[expected_cols].astype(np.float32).copy()

    try:
        prediction = rf_pose.predict(temp)[0]
        label = le_pose.inverse_transform([prediction])[0]
        confidence = 0.50
        if hasattr(rf_pose, "predict_proba"):
            probs = rf_pose.predict_proba(temp)[0]
            confidence = float(np.max(probs))
        return str(label), confidence
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Unknown", 0.0

def wr_predict_defect_label(features_df):
    if rf_def is None or le_def is None:
        return "Unknown", 0.0

    expected_cols = getattr(rf_def, "feature_names_in_", WR_MODEL_COLUMNS)
    for col in expected_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
            
    temp = features_df[expected_cols].astype(np.float32).copy()

    try:
        prediction = rf_def.predict(temp)[0]
        label = le_def.inverse_transform([prediction])[0]
        confidence = 0.50
        if hasattr(rf_def, "predict_proba"):
            probs = rf_def.predict_proba(temp)[0]
            confidence = float(np.max(probs))
        return str(label), confidence
    except Exception as e:
        return "Unknown", 0.0

# =========================================================
# VISIBILITY
# =========================================================
def wr_check_body_visibility(lm_dict):
    core_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist"]
    visibilities = [lm_dict[name]["visibility"] for name in core_names]
    visible_count = sum(v > 0.35 for v in visibilities)
    return visible_count >= 8, visible_count, float(np.mean(visibilities))

def wr_check_frame_position(raw_pts):
    xs = [float(p[0]) for p in raw_pts[WR_SELECTED_POINTS]]
    ys = [float(p[1]) for p in raw_pts[WR_SELECTED_POINTS]]
    min_x, max_x = min(xs), max(xs)
    width = max_x - min_x
    feedback = []
    if width > 0.98: feedback.append("Move slightly away to show full lunge")
    if width < 0.40: feedback.append("Move closer so your body fills the frame")
    return feedback

# =========================================================
# ANALYSIS
# =========================================================
def analyze_warrior_pose(raw_pts, landmarks, angles):
    # Determine the front leg (the one that is bent more)
    l_knee_angle = angles["left_knee_angle"]
    r_knee_angle = angles["right_knee_angle"]
    
    if l_knee_angle < r_knee_angle:
        front_knee_angle = l_knee_angle
        back_knee_angle = r_knee_angle
        front_side = "left"
    else:
        front_knee_angle = r_knee_angle
        back_knee_angle = l_knee_angle
        front_side = "right"

    l_shoulder_y, r_shoulder_y = raw_pts[WR_LEFT_SHOULDER][1], raw_pts[WR_RIGHT_SHOULDER][1]
    l_wrist_y, r_wrist_y = raw_pts[WR_LEFT_WRIST][1], raw_pts[WR_RIGHT_WRIST][1]

    # Rules based on the dataset prep notebook
    back_leg_bent = back_knee_angle < 155
    lunge_too_shallow = front_knee_angle > 125
    arms_not_level = abs(l_wrist_y - l_shoulder_y) > 0.15 or abs(r_wrist_y - r_shoulder_y) > 0.15

    score = 0
    status = "warning"
    pose_label = "Not Warrior"
    main_feedback = "Step wide and bend one knee to enter Warrior Pose."
    tips = ["Extend your arms parallel to the floor.", "Keep your back leg straight."]

    # Heuristic gate if model fails
    is_warrior_gate = not back_leg_bent and front_knee_angle < 145

    if back_leg_bent:
        score = 45
        main_feedback = "Straighten your back leg completely."
        tips = ["Press the outer edge of your back foot into the mat."]
    elif lunge_too_shallow:
        score = 65
        main_feedback = "Bend your front knee deeper."
        tips = ["Aim to get your front thigh parallel to the floor.", "Ensure your knee tracks over your ankle."]
    elif arms_not_level:
        score = 80
        main_feedback = "Level your arms parallel to the floor."
        tips = ["Reach out through your fingertips.", "Relax your shoulders away from your ears."]
    elif is_warrior_gate:
        score = 100
        status = "perfect"
        pose_label = "Correct Warrior"
        main_feedback = "Perfect Warrior! Hold steady."
        tips = ["Gaze over your front fingertips.", "Keep your core engaged."]
    else:
        score = 85
        status = "good"
        pose_label = "Warrior Pose"
        main_feedback = "Good Warrior. Hold steady."
        tips = ["Deepen the lunge slightly.", "Keep your torso centered."]

    checks = {
        "front_side": front_side,
        "back_leg_bent": back_leg_bent,
        "lunge_too_shallow": lunge_too_shallow,
        "arms_not_level": arms_not_level,
        "is_warrior_gate": is_warrior_gate
    }

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": tips,
        "angles": angles,
        "checks": checks,
    }

def wr_is_warrior_like(model_label, model_confidence, analysis):
    if ("warrior" in str(model_label).lower() and "not" not in str(model_label).lower()) and model_confidence >= 0.58:
        return True
    
    checks = analysis["checks"]
    if checks.get("is_warrior_gate"):
        return True
    return False

# =========================================================
# STABILITY / HOLD / QUALITY
# =========================================================
def wr_update_stability_metrics(raw_pts):
    hip_center = (raw_pts[WR_LEFT_HIP] + raw_pts[WR_RIGHT_HIP]) / 2.0
    shoulder_center = (raw_pts[WR_LEFT_SHOULDER] + raw_pts[WR_RIGHT_SHOULDER]) / 2.0

    WR_HIP_HEIGHT_HISTORY.append(float(hip_center[1]))
    WR_HIP_SHIFT_HISTORY.append(float(hip_center[0]))
    WR_SPINE_LINE_HISTORY.append(abs(float(shoulder_center[0] - hip_center[0])))

def wr_get_stability_feedback():
    hip_wobble = wr_moving_std(WR_HIP_HEIGHT_HISTORY)
    shift_wobble = wr_moving_std(WR_HIP_SHIFT_HISTORY)
    
    feedback = []
    penalty = 0

    if hip_wobble > 0.015:
        feedback.append("Keep your lunge depth steady")
        penalty += 4
    if shift_wobble > 0.015:
        feedback.append("Keep your torso centered over your hips")
        penalty += 3

    return feedback, penalty

def wr_update_hold_state(is_pose, full_body_visible, low_light):
    global WR_HOLD_START, WR_BEST_HOLD_TIME

    valid_hold = is_pose and full_body_visible and not low_light
    if valid_hold:
        if WR_HOLD_START is None:
            WR_HOLD_START = time.time()
        hold_time = time.time() - WR_HOLD_START
        WR_BEST_HOLD_TIME = max(WR_BEST_HOLD_TIME, hold_time)
    else:
        hold_time = 0.0
        WR_HOLD_START = None

    return hold_time, WR_BEST_HOLD_TIME

def wr_hold_bonus(hold_time):
    if hold_time >= 15: return 10
    if hold_time >= 10: return 8
    if hold_time >= 5: return 5
    if hold_time >= 3: return 2
    return 0

def wr_quality_from_score(score):
    if score >= 96: return "Perfect_Warrior"
    if score >= 85: return "Good_Warrior"
    if score >= 70: return "Needs_Correction"
    return "Not_Ready"

def wr_build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]

    points = []
    for idx in WR_SELECTED_POINTS:
        visibility = float(landmarks[idx].visibility)
        if visibility < 0.25: continue

        radius = 6
        color = WR_GREEN if analysis["score"] >= 90 else WR_YELLOW

        if idx in [WR_LEFT_KNEE, WR_RIGHT_KNEE] and checks.get("lunge_too_shallow"):
            color = WR_RED; radius = 8
        if idx in [WR_LEFT_KNEE, WR_RIGHT_KNEE] and checks.get("back_leg_bent"):
            color = WR_RED; radius = 8
        if idx in [WR_LEFT_WRIST, WR_RIGHT_WRIST, WR_LEFT_ELBOW, WR_RIGHT_ELBOW] and checks.get("arms_not_level"):
            color = WR_RED; radius = 8

        points.append({
            "name": WR_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
        })
    return points

def wr_build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    primary = [
        (WR_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), WR_YELLOW),
        (WR_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), WR_YELLOW),
        (WR_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), WR_CYAN),
        (WR_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), WR_CYAN)
    ]

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
def process_warrior_pose_request(request):
    global WR_PERFECT_HOLD_COUNT, WR_HOLD_START

    try:
        uploaded_file = request.FILES["image"]
        frame = wr_read_uploaded_image(uploaded_file)

        if frame is None: return wr_api_error("Invalid image file", status=400)
        frame = wr_enhance_frame(frame)

        low_light, brightness = wr_check_lighting(frame)
        if low_light:
            wr_reset_runtime_state()
            return wr_api_success(
                pose="Low Light", status="warning", feedback="Room lighting is too low.", coach_text="Improve lighting.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(WR_BEST_HOLD_TIME), 1),
                angles={}, details=["Increase room lighting", "Avoid dark background"], perfect_hold=False, points=[], angle_texts=[]
            )

        landmarks = wr_detect_landmarks(frame)
        has_landmarks = landmarks is not None
        stable_has_landmarks = wr_smooth_boolean(WR_DETECTION_HISTORY, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            wr_reset_runtime_state()
            return wr_api_success(
                pose="Unknown", status="unknown", feedback="No pose detected.", coach_text="Show your full body.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(WR_BEST_HOLD_TIME), 1),
                angles={}, details=["Show full body", "Move slightly back"], perfect_hold=False, points=[], angle_texts=[]
            )

        if landmarks is None:
            wr_clear_point_history()
            return wr_api_success(
                pose="Tracking...", status="warning", feedback="Hold still.", coach_text="Keep steady.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(WR_BEST_HOLD_TIME), 1),
                angles={}, details=[], perfect_hold=False, points=[], angle_texts=[]
            )

        features_df, lm_dict, angles = wr_build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = raw_pts.copy()

        for idx in WR_SELECTED_POINTS:
            sx, sy, sz = wr_smooth_point(f"wr_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])
            smoothed_pts[idx] = [sx, sy, sz]

        full_body_visible, visible_count, avg_visibility = wr_check_body_visibility(lm_dict)
        stable_full_body_visible = wr_smooth_boolean(WR_VISIBILITY_HISTORY, full_body_visible)
        framing_feedback = wr_check_frame_position(smoothed_pts)

        if not full_body_visible and not stable_full_body_visible:
            wr_reset_runtime_state()
            details = ["Show hands and feet clearly in frame"] + framing_feedback
            return wr_api_success(
                pose="Body Not Visible", status="warning", feedback="Adjust camera for full body view.", coach_text="Show full body.",
                confidence=0.0, score=0, hold_time=0.0, best_hold_time=round(float(WR_BEST_HOLD_TIME), 1),
                angles={}, details=wr_dedupe_list(details, max_items=3), perfect_hold=False, points=[], angle_texts=[]
            )

        raw_model_label, confidence = wr_predict_model_label(features_df)
        stable_model_label = wr_smooth_label(WR_POSE_HISTORY, raw_model_label)

        # Allow ML model or heuristic rules to identify the pose
        analysis = analyze_warrior_pose(smoothed_pts, landmarks, angles)
        points = wr_build_points_for_frontend(smoothed_pts, landmarks, analysis)
        angle_texts = wr_build_angle_texts(smoothed_pts, landmarks, analysis)

        is_warrior = wr_is_warrior_like(stable_model_label, confidence, analysis)

        if not is_warrior:
            WR_PERFECT_HOLD_COUNT = 0
            WR_HOLD_START = None
            tips = analysis.get("tips", []) + ["Step feet wide and bend front knee", "Extend arms parallel to floor"] + framing_feedback
            return wr_api_success(
                pose="Not Warrior Pose", model_pose=stable_model_label, quality="Not_Ready",
                feedback=analysis.get("main_feedback", "Move into Warrior position."),
                coach_text=analysis.get("main_feedback", "Move into Warrior position."),
                status="warning", confidence=round(float(confidence), 3), score=max(0, min(65, analysis.get("score", 0))),
                hold_time=0.0, best_hold_time=round(float(WR_BEST_HOLD_TIME), 1), angles=analysis.get("angles", {}),
                details=wr_dedupe_list(tips, max_items=3), perfect_hold=False, points=points, angle_texts=angle_texts
            )

        # Defect model check if RF model is loaded
        defect_label, defect_confidence = wr_predict_defect_label(features_df)

        wr_update_stability_metrics(smoothed_pts)
        stability_tips, stability_penalty = wr_get_stability_feedback()

        base_score = analysis["score"]
        combined_score = max(0, base_score - stability_penalty)

        hold_time, best_hold = wr_update_hold_state(is_warrior, stable_full_body_visible, low_light)
        combined_score = min(100, combined_score + wr_hold_bonus(hold_time))
        stable_score = wr_smooth_score(combined_score)

        if (analysis["checks"].get("is_warrior_gate") or defect_label == "Perfect_Warrior") and stable_score >= 96 and hold_time >= 3.0:
            pose_name = "Correct Warrior"
            status = "perfect"
            feedback_text = "Perfect Warrior"
            coach_text = "Excellent. Hold steady and breathe."
        elif stable_score >= 85:
            pose_name = "Warrior Pose"
            status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = "Good shape. Refine and hold steady."
        else:
            pose_name = "Warrior Needs Correction"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "Fix the highlighted alignment."

        stable_feedback = wr_smooth_feedback(feedback_text)

        if pose_name == "Correct Warrior" and hold_time >= 3.2:
            WR_PERFECT_HOLD_COUNT += 1
        else:
            WR_PERFECT_HOLD_COUNT = 0

        tips = []
        if hold_time >= 5: tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        return wr_api_success(
            pose=pose_name, model_pose=stable_model_label, quality=wr_quality_from_score(stable_score),
            feedback=stable_feedback, coach_text=coach_text, status=status, confidence=round(float(confidence), 3),
            score=stable_score, hold_time=round(float(hold_time), 1), best_hold_time=round(float(best_hold), 1),
            angles=analysis["angles"], details=wr_dedupe_list(tips, max_items=3, exclude=[stable_feedback, coach_text]),
            perfect_hold=WR_PERFECT_HOLD_COUNT >= 3, points=points, angle_texts=angle_texts
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e), "error_type": type(e).__name__}, status=500)