from pathlib import Path
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from django.conf import settings
from django.http import JsonResponse


# =========================================================
# GLOBAL STATE
# =========================================================
POSE_HISTORY = deque(maxlen=12)
DEFECT_HISTORY = deque(maxlen=12)
SCORE_HISTORY = deque(maxlen=8)
FEEDBACK_HISTORY = deque(maxlen=8)

TORSO_CENTER_HISTORY = deque(maxlen=20)
SHOULDER_TILT_HISTORY = deque(maxlen=20)
TORSO_LEAN_HISTORY = deque(maxlen=20)

TREE_HOLD_START = None
BEST_HOLD_TIME = 0.0
PERFECT_HOLD_COUNT = 0

POINT_HISTORY = {}
POINT_HISTORY_SIZE = 7
BOOLEAN_HISTORY = {}
BOOLEAN_HISTORY_SIZE = 5
TREE_SESSION_RUNTIME_KEY = "tree_runtime_v1"


# =========================================================
# SAFE PATH / MODEL LOAD
# =========================================================
BASE_DIR = Path(settings.BASE_DIR)


def resolve_model_path(filename: str) -> Path:
    candidates = [
        BASE_DIR / "Ml_Models" / filename,
        BASE_DIR / "Ml_models" / filename,
        BASE_DIR / "Ml_Models" / filename,
        BASE_DIR / "Ml_models" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find model file: {filename}\nChecked paths:\n{checked}"
    )


POSE_MODEL_PATH = resolve_model_path("tree_pose_model.pkl")
POSE_LABEL_ENCODER_PATH = resolve_model_path("label_encoder.pkl")
DEFECT_MODEL_PATH = resolve_model_path("tree_defect_model.pkl")
DEFECT_LABEL_ENCODER_PATH = resolve_model_path("defect_label_encoder.pkl")

pose_model = joblib.load(POSE_MODEL_PATH)
pose_label_encoder = joblib.load(POSE_LABEL_ENCODER_PATH)
defect_model = joblib.load(DEFECT_MODEL_PATH)
defect_label_encoder = joblib.load(DEFECT_LABEL_ENCODER_PATH)


# =========================================================
# MEDIAPIPE SETUP
# =========================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)


# =========================================================
# LANDMARK INDEXES
# =========================================================
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SELECTED_POINTS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
]

POINT_NAME_MAP = {
    NOSE: "nose",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee",
    RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle",
    RIGHT_ANKLE: "right_ankle",
}

LANDMARK_NAME_TO_INDEX = {
    "nose": NOSE,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}

FEATURE_LANDMARK_NAMES = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

FEATURE_COLUMNS = []
for name in FEATURE_LANDMARK_NAMES:
    FEATURE_COLUMNS.extend([
        f"{name}_x",
        f"{name}_y",
        f"{name}_z",
        f"{name}_visibility",
    ])

ANGLE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "hip_angle",
    "shoulder_angle",
]

MODEL_COLUMNS = FEATURE_COLUMNS + ANGLE_COLUMNS


# =========================================================
# COLORS
# =========================================================
GREEN = "#00ff66"
RED = "#ff3b30"
YELLOW = "#ffd60a"
GRAY = "#cfcfcf"
TREE_DEGREE_SIGN = chr(176)
TREE_FRONTEND_POINT_VISIBILITY_MIN = 0.18


# =========================================================
# TREE THRESHOLDS
# =========================================================
TREE_STANDING_LEG_MIN_ANGLE = 154.0
TREE_KNEE_OPEN_MAX_ANGLE = 154.0
TREE_KNEE_OPEN_MIN_DISTANCE = 0.18
TREE_FOOT_LIFT_MIN = 0.16
TREE_FOOT_OFF_KNEE_MIN = 0.11
TREE_FOOT_CLOSE_MAX = 0.50
TREE_TORSO_TILT_MAX = 16.0
TREE_WRIST_ABOVE_SHOULDER_MARGIN = 0.08
TREE_HANDS_MAX_WIDTH = 0.42
TREE_PRAYER_HANDS_MAX_WIDTH = 0.16
TREE_ELBOW_STRAIGHT_MIN = 146.0
TREE_ELBOW_SOFT_MIN = 132.0


# =========================================================
# RESPONSE HELPERS
# =========================================================
def api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})


def api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


def reset_tree_runtime_state():
    global TREE_HOLD_START, BEST_HOLD_TIME, PERFECT_HOLD_COUNT

    POSE_HISTORY.clear()
    DEFECT_HISTORY.clear()
    SCORE_HISTORY.clear()
    FEEDBACK_HISTORY.clear()
    TORSO_CENTER_HISTORY.clear()
    SHOULDER_TILT_HISTORY.clear()
    TORSO_LEAN_HISTORY.clear()
    clear_point_history()
    BOOLEAN_HISTORY.clear()

    TREE_HOLD_START = None
    BEST_HOLD_TIME = 0.0
    PERFECT_HOLD_COUNT = 0


def tree_runtime_to_session_data():
    return {
        "pose_history": list(POSE_HISTORY),
        "defect_history": list(DEFECT_HISTORY),
        "score_history": list(SCORE_HISTORY),
        "feedback_history": list(FEEDBACK_HISTORY),
        "torso_center_history": list(TORSO_CENTER_HISTORY),
        "shoulder_tilt_history": list(SHOULDER_TILT_HISTORY),
        "torso_lean_history": list(TORSO_LEAN_HISTORY),
        "tree_hold_start": TREE_HOLD_START,
        "best_hold_time": float(BEST_HOLD_TIME),
        "perfect_hold_count": int(PERFECT_HOLD_COUNT),
        "point_history": {
            str(key): [[float(x), float(y)] for x, y in points]
            for key, points in POINT_HISTORY.items()
        },
        "boolean_history": {
            str(key): [bool(value) for value in values]
            for key, values in BOOLEAN_HISTORY.items()
        },
    }


def load_tree_runtime_state(request):
    global TREE_HOLD_START, BEST_HOLD_TIME, PERFECT_HOLD_COUNT

    if request.session.session_key is None:
        request.session.save()

    data = request.session.get(TREE_SESSION_RUNTIME_KEY, {})
    reset_tree_runtime_state()

    if not isinstance(data, dict):
        return

    POSE_HISTORY.extend(data.get("pose_history", []))
    DEFECT_HISTORY.extend(data.get("defect_history", []))
    SCORE_HISTORY.extend(data.get("score_history", []))
    FEEDBACK_HISTORY.extend(data.get("feedback_history", []))
    TORSO_CENTER_HISTORY.extend(data.get("torso_center_history", []))
    SHOULDER_TILT_HISTORY.extend(data.get("shoulder_tilt_history", []))
    TORSO_LEAN_HISTORY.extend(data.get("torso_lean_history", []))
    TREE_HOLD_START = data.get("tree_hold_start")
    BEST_HOLD_TIME = float(data.get("best_hold_time", 0.0) or 0.0)
    PERFECT_HOLD_COUNT = int(data.get("perfect_hold_count", 0) or 0)

    POINT_HISTORY.update({
        str(key): deque([(float(x), float(y)) for x, y in values], maxlen=POINT_HISTORY_SIZE)
        for key, values in data.get("point_history", {}).items()
    })
    BOOLEAN_HISTORY.update({
        str(key): deque([bool(value) for value in values], maxlen=BOOLEAN_HISTORY_SIZE)
        for key, values in data.get("boolean_history", {}).items()
    })


def save_tree_runtime_state(request):
    if request.session.session_key is None:
        request.session.save()
    request.session[TREE_SESSION_RUNTIME_KEY] = tree_runtime_to_session_data()
    request.session.modified = True


# =========================================================
# TEXT HELPERS
# =========================================================
def clean_text(text):
    return " ".join(str(text).strip().split())


def normalize_text_key(text):
    return clean_text(text).lower()


def dedupe_text_list(items, max_items=None, exclude=None):
    exclude = exclude or []
    exclude_keys = {normalize_text_key(x) for x in exclude if x}

    output = []
    seen = set()

    for item in items:
        if not item:
            continue

        text = clean_text(item)
        key = normalize_text_key(text)

        if not key or key in seen or key in exclude_keys:
            continue

        seen.add(key)
        output.append(text)

        if max_items and len(output) >= max_items:
            break

    return output


# =========================================================
# SMOOTHING HELPERS
# =========================================================
def smooth_label(history, new_label):
    history.append(str(new_label))
    return Counter(history).most_common(1)[0][0]


def smooth_score(new_score):
    SCORE_HISTORY.append(float(new_score))
    return int(round(sum(SCORE_HISTORY) / len(SCORE_HISTORY)))


def smooth_feedback(new_feedback):
    FEEDBACK_HISTORY.append(str(new_feedback))
    return Counter(FEEDBACK_HISTORY).most_common(1)[0][0]


def smooth_boolean(history, value):
    history.append(bool(value))
    true_count = sum(history)
    if len(history) < 3:
        return true_count >= 1
    return true_count >= max(2, len(history) // 2 + 1)


def smooth_runtime_boolean(key, value, maxlen=BOOLEAN_HISTORY_SIZE):
    if key not in BOOLEAN_HISTORY:
        BOOLEAN_HISTORY[key] = deque(maxlen=maxlen)
    return smooth_boolean(BOOLEAN_HISTORY[key], value)


def smooth_point(key, x, y, visibility=1.0):
    if key not in POINT_HISTORY:
        POINT_HISTORY[key] = deque(maxlen=POINT_HISTORY_SIZE)

    history = POINT_HISTORY[key]
    current = np.array([float(x), float(y)], dtype=np.float32)
    visibility = float(np.clip(visibility, 0.0, 1.0))

    if visibility < TREE_FRONTEND_POINT_VISIBILITY_MIN and history:
        last_x, last_y = history[-1]
        return float(last_x), float(last_y)

    if not history:
        history.append((float(current[0]), float(current[1])))
        return float(current[0]), float(current[1])

    previous = np.array(history[-1], dtype=np.float32)
    delta = current - previous
    distance = float(np.linalg.norm(delta))
    max_step = 0.085 if visibility >= 0.65 else 0.045

    if distance > max_step:
        current = previous + (delta / (distance + 1e-6)) * max_step

    alpha = 0.36 if visibility >= 0.65 else 0.22
    smoothed = previous * (1.0 - alpha) + current * alpha
    history.append((float(smoothed[0]), float(smoothed[1])))

    return float(smoothed[0]), float(smoothed[1])


def latest_tree_point(key, fallback_x, fallback_y):
    history = POINT_HISTORY.get(key)
    if history:
        x, y = history[-1]
        return float(x), float(y)
    return float(fallback_x), float(fallback_y)


def smooth_tree_points(raw_pts, landmarks):
    smoothed_pts = raw_pts.copy()

    for idx in SELECTED_POINTS:
        key = f"tree_{idx}"
        visibility = float(landmarks[idx].visibility)
        if visibility >= TREE_FRONTEND_POINT_VISIBILITY_MIN or POINT_HISTORY.get(key):
            sx, sy = smooth_point(key, raw_pts[idx][0], raw_pts[idx][1], visibility=visibility)
            smoothed_pts[idx][0] = sx
            smoothed_pts[idx][1] = sy

    return smoothed_pts


def clear_point_history():
    POINT_HISTORY.clear()


# =========================================================
# MATH HELPERS
# =========================================================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def moving_std(values):
    if len(values) < 3:
        return 0.0
    return float(np.std(list(values)))


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


# =========================================================
# IMAGE HELPERS
# =========================================================
def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def enhance_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.04, beta=5)
    return frame


def detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    return results.pose_landmarks.landmark


def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 65, brightness


# =========================================================
# FEATURE CREATION
# =========================================================
def extract_raw_landmark_dict(landmarks):
    lm_dict = {}
    for name, idx in LANDMARK_NAME_TO_INDEX.items():
        lm = landmarks[idx]
        lm_dict[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
    return lm_dict


def normalize_landmarks_inplace(lm_dict):
    left_hip_x = lm_dict["left_hip"]["x"]
    left_hip_y = lm_dict["left_hip"]["y"]

    for name in lm_dict.keys():
        lm_dict[name]["x"] -= left_hip_x
        lm_dict[name]["y"] -= left_hip_y


def extract_angles_from_dict(lm_dict):
    points = {name: [vals["x"], vals["y"]] for name, vals in lm_dict.items()}

    return {
        "left_knee_angle": calculate_angle(
            points["left_hip"], points["left_knee"], points["left_ankle"]
        ),
        "right_knee_angle": calculate_angle(
            points["right_hip"], points["right_knee"], points["right_ankle"]
        ),
        "hip_angle": calculate_angle(
            points["left_shoulder"], points["left_hip"], points["left_knee"]
        ),
        "shoulder_angle": calculate_angle(
            points["left_elbow"], points["left_shoulder"], points["right_shoulder"]
        ),
    }


def build_feature_dataframe_from_landmarks(landmarks):
    lm_dict = extract_raw_landmark_dict(landmarks)
    normalize_landmarks_inplace(lm_dict)
    angles = extract_angles_from_dict(lm_dict)

    row = {}
    for name in FEATURE_LANDMARK_NAMES:
        row[f"{name}_x"] = lm_dict[name]["x"]
        row[f"{name}_y"] = lm_dict[name]["y"]
        row[f"{name}_z"] = lm_dict[name]["z"]
        row[f"{name}_visibility"] = lm_dict[name]["visibility"]

    for key, value in angles.items():
        row[key] = value

    features_df = pd.DataFrame([row], columns=MODEL_COLUMNS)
    return features_df, lm_dict, angles


def predict_pose_label(features_df):
    prediction = pose_model.predict(features_df)[0]
    predicted_label = pose_label_encoder.inverse_transform([prediction])[0]

    confidence = 0.50
    if hasattr(pose_model, "predict_proba"):
        probs = pose_model.predict_proba(features_df)[0]
        confidence = float(np.max(probs))

    return str(predicted_label), confidence


def predict_defect_label(features_df):
    prediction = defect_model.predict(features_df)[0]
    predicted_label = defect_label_encoder.inverse_transform([prediction])[0]

    confidence = 0.50
    if hasattr(defect_model, "predict_proba"):
        probs = defect_model.predict_proba(features_df)[0]
        confidence = float(np.max(probs))

    return str(predicted_label), confidence


# =========================================================
# VISIBILITY / FRAMING
# =========================================================
def check_body_visibility(lm_dict):
    visibilities = [lm_dict[name]["visibility"] for name in lm_dict]
    visible_count = sum(v > 0.5 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    ankles_visible = (
        lm_dict["left_ankle"]["visibility"] > 0.42 and
        lm_dict["right_ankle"]["visibility"] > 0.42
    )

    shoulders_visible = (
        lm_dict["left_shoulder"]["visibility"] > 0.5 and
        lm_dict["right_shoulder"]["visibility"] > 0.5
    )

    hips_visible = (
        lm_dict["left_hip"]["visibility"] > 0.5 and
        lm_dict["right_hip"]["visibility"] > 0.5
    )

    full_body_visible = visible_count >= 10 and ankles_visible and shoulders_visible and hips_visible
    return full_body_visible, visible_count, avg_visibility


def check_frame_position(raw_pts):
    xs = [float(p[0]) for p in raw_pts[SELECTED_POINTS]]
    ys = [float(p[1]) for p in raw_pts[SELECTED_POINTS]]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    feedback = []

    if width > 0.88 or height > 1.05:
        feedback.append("Move a little away from the camera")

    center_x = (min_x + max_x) / 2
    if center_x < 0.22:
        feedback.append("Move slightly to the right")
    elif center_x > 0.78:
        feedback.append("Move slightly to the left")

    return feedback


# =========================================================
# EMPTY ANALYSIS HELPER
# =========================================================
def create_empty_analysis(msg):
    return {
        "pose_label": "Not Ready",
        "score": 0,
        "status": "warning",
        "main_feedback": msg,
        "coach_text": "Grow tall through the standing leg and build the pose one step at a time.",
        "tips": ["Keep your standing leg completely straight."],
        "standing_side": "none",
        "stand_knee_idx": LEFT_KNEE,
        "bent_knee_idx": RIGHT_KNEE,
        "stand_ankle_idx": LEFT_ANKLE,
        "bent_ankle_idx": RIGHT_ANKLE,
        "angles": {"left_knee_angle": 0, "right_knee_angle": 0, "torso_tilt": 0},
        "checks": {"strict_tree_gate": False}
    }


# =========================================================
# TREE ANALYSIS
# =========================================================
def analyze_tree_pose(raw_pts):
    ls, rs = raw_pts[LEFT_SHOULDER], raw_pts[RIGHT_SHOULDER]
    le, re = raw_pts[LEFT_ELBOW], raw_pts[RIGHT_ELBOW]
    lw, rw = raw_pts[LEFT_WRIST], raw_pts[RIGHT_WRIST]
    lh, rh = raw_pts[LEFT_HIP], raw_pts[RIGHT_HIP]
    lk, rk = raw_pts[LEFT_KNEE], raw_pts[RIGHT_KNEE]
    la, ra = raw_pts[LEFT_ANKLE], raw_pts[RIGHT_ANKLE]

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    torso_size = float(np.linalg.norm(shoulder_center[:2] - hip_center[:2])) + 1e-6

    left_knee_angle = calculate_angle(lh[:2], lk[:2], la[:2])
    right_knee_angle = calculate_angle(rh[:2], rk[:2], ra[:2])

    left_elbow_angle = calculate_angle(ls[:2], le[:2], lw[:2])
    right_elbow_angle = calculate_angle(rs[:2], re[:2], rw[:2])

    is_neutral_standing = left_knee_angle > 158 and right_knee_angle > 158

    if left_knee_angle >= right_knee_angle:
        standing_side = "left"
        stand_knee_idx, bent_knee_idx = LEFT_KNEE, RIGHT_KNEE
        stand_ankle_idx, bent_ankle_idx = LEFT_ANKLE, RIGHT_ANKLE
        stand_angle, bent_angle = left_knee_angle, right_knee_angle
        stand_hip, stand_knee, stand_ankle = lh, lk, la
        bent_hip, bent_knee, bent_ankle = rh, rk, ra
    else:
        standing_side = "right"
        stand_knee_idx, bent_knee_idx = RIGHT_KNEE, LEFT_KNEE
        stand_ankle_idx, bent_ankle_idx = RIGHT_ANKLE, LEFT_ANKLE
        stand_angle, bent_angle = right_knee_angle, left_knee_angle
        stand_hip, stand_knee, stand_ankle = rh, rk, ra
        bent_hip, bent_knee, bent_ankle = lh, lk, la

    foot_lift_height = abs(float(bent_ankle[1] - stand_ankle[1])) / torso_size
    one_foot_lifted = foot_lift_height >= TREE_FOOT_LIFT_MIN

    foot_vs_knee_y = abs(float(bent_ankle[1] - stand_knee[1])) / torso_size
    not_on_knee_joint = foot_vs_knee_y >= TREE_FOOT_OFF_KNEE_MIN

    foot_to_knee_x = abs(float(bent_ankle[0] - stand_knee[0])) / torso_size
    foot_to_hip_x = abs(float(bent_ankle[0] - stand_hip[0])) / torso_size
    foot_close_to_leg = min(foot_to_knee_x, foot_to_hip_x) <= TREE_FOOT_CLOSE_MAX

    ankle_to_stand_ankle_y = float(stand_ankle[1] - bent_ankle[1]) / torso_size
    ankle_to_stand_hip_y = float(stand_hip[1] - bent_ankle[1]) / torso_size

    foot_zone = "unknown"
    if ankle_to_stand_ankle_y < 0.12:
        foot_zone = "low"
    elif 0.12 <= ankle_to_stand_ankle_y <= 0.86:
        foot_zone = "calf"
    elif 0.00 <= ankle_to_stand_hip_y <= 0.76:
        foot_zone = "thigh"

    foot_height_ok = foot_zone in ["calf", "thigh"]

    knee_open_distance = abs(float(bent_knee[0] - stand_hip[0])) / torso_size
    knee_open_ok = knee_open_distance >= TREE_KNEE_OPEN_MIN_DISTANCE and bent_angle <= TREE_KNEE_OPEN_MAX_ANGLE

    standing_leg_ok = stand_angle >= TREE_STANDING_LEG_MIN_ANGLE

    inferred_valid_leg_position = (
        one_foot_lifted and
        not_on_knee_joint and
        knee_open_ok and
        standing_leg_ok
    )

    foot_place_ok = (
        (one_foot_lifted and not_on_knee_joint and foot_height_ok and foot_close_to_leg) or
        inferred_valid_leg_position
    )

    vertical_ref = hip_center + np.array([0.0, -0.30, 0.0], dtype=np.float32)
    torso_tilt = calculate_angle(shoulder_center[:2], hip_center[:2], vertical_ref[:2])
    torso_ok = torso_tilt <= TREE_TORSO_TILT_MAX

    wrist_distance = float(np.linalg.norm(lw[:2] - rw[:2])) / torso_size
    wrist_height_diff = abs(float(lw[1] - rw[1])) / torso_size

    left_wrist_above_left_shoulder = lw[1] < ls[1] - TREE_WRIST_ABOVE_SHOULDER_MARGIN
    right_wrist_above_right_shoulder = rw[1] < rs[1] - TREE_WRIST_ABOVE_SHOULDER_MARGIN

    elbows_straight = left_elbow_angle >= TREE_ELBOW_STRAIGHT_MIN and right_elbow_angle >= TREE_ELBOW_STRAIGHT_MIN
    elbows_soft_ok = left_elbow_angle >= TREE_ELBOW_SOFT_MIN and right_elbow_angle >= TREE_ELBOW_SOFT_MIN

    hands_symmetric = wrist_height_diff < 0.10
    hands_not_too_wide = wrist_distance <= TREE_HANDS_MAX_WIDTH
    hands_close_for_prayer = wrist_distance <= TREE_PRAYER_HANDS_MAX_WIDTH

    left_wrist_near_midline = abs(float(lw[0] - shoulder_center[0])) / torso_size < 0.22
    right_wrist_near_midline = abs(float(rw[0] - shoulder_center[0])) / torso_size < 0.22

    prayer_hands = (
        hands_close_for_prayer and
        wrist_height_diff < 0.08 and
        lw[1] < hip_center[1] and
        rw[1] < hip_center[1] and
        lw[1] > shoulder_center[1] - 0.18 and
        rw[1] > shoulder_center[1] - 0.18 and
        left_wrist_near_midline and
        right_wrist_near_midline
    )

    hands_up = (
        left_wrist_above_left_shoulder and
        right_wrist_above_right_shoulder and
        hands_symmetric and
        hands_not_too_wide and
        elbows_soft_ok
    )

    hands_ready = prayer_hands or hands_up
    left_elbow_ok = prayer_hands or left_elbow_angle >= TREE_ELBOW_SOFT_MIN
    right_elbow_ok = prayer_hands or right_elbow_angle >= TREE_ELBOW_SOFT_MIN
    left_arm_ready = prayer_hands or (left_wrist_above_left_shoulder and left_elbow_ok)
    right_arm_ready = prayer_hands or (right_wrist_above_right_shoulder and right_elbow_ok)

    strict_tree_gate = (
        not is_neutral_standing and
        one_foot_lifted and
        foot_place_ok and
        knee_open_ok and
        standing_leg_ok and
        torso_ok and
        hands_ready
    )

    pose_label = "Not Tree Pose"
    status = "warning"
    score = 0
    main_f = "Lift one foot onto the opposite inner leg."
    coach_f = "Root down through one foot and draw the other foot toward the inner leg."
    tips = ["Stand tall.", "Shift weight to one leg."]

    if is_neutral_standing:
        pose_label = "Standing Still"
        score = 12
        main_f = "Shift your weight onto one leg and start lifting the other foot."
        coach_f = "Press down through the standing foot so the free foot can float up."
        tips = ["Pick one standing leg.", "Lift the other foot off the floor."]

    elif not one_foot_lifted:
        score = 28
        main_f = "Lift the free foot higher toward your inner leg."
        coach_f = "Draw the knee out to the side as the foot rises."
        tips = ["Bring the foot toward the opposite inner leg."]

    elif not not_on_knee_joint:
        score = 46
        main_f = "Move the lifted foot away from the knee joint."
        coach_f = "Place it on the inner calf or upper thigh, never directly on the knee."
        tips = ["Place the foot on inner calf or thigh, never on the knee."]

    elif not foot_place_ok:
        score = 58
        if foot_zone == "low":
            main_f = "Place the lifted foot a little higher on the inner leg."
            coach_f = "Set the foot on the calf or upper thigh for a steadier base."
            tips = ["Your foot is lifted correctly. Now set it on calf or thigh."]
        elif not foot_close_to_leg:
            main_f = "Bring the lifted foot snugly into the standing leg."
            coach_f = "Press the foot and inner leg gently into each other for balance."
            tips = ["Press the foot gently into the inner leg."]
        else:
            main_f = "Keep the lifted foot steady against the inner leg."
            coach_f = "Use light pressure through the foot and leg to stop the slide."
            tips = ["Hold the lifted foot steady and avoid sliding."]

    elif not knee_open_ok:
        score = 70
        main_f = "Open the lifted knee more to the side."
        coach_f = "Rotate from the hip while keeping the torso tall."
        tips = ["Rotate the hip outward.", "Let the knee point sideways."]

    elif not standing_leg_ok:
        score = 79
        main_f = "Straighten the standing leg and root down through the foot."
        coach_f = "Lift up through the thigh as you ground evenly through the heel and toes."
        tips = ["Press down through the grounded foot.", "Keep the support leg long and firm."]

    elif not torso_ok:
        score = 86
        main_f = "Stand taller through your spine."
        coach_f = "Stack your shoulders over your hips and steady your gaze."
        tips = ["Keep shoulders over hips.", "Look straight ahead."]

    elif not hands_ready:
        score = 82
        status = "warning"
        pose_label = "Tree Pose"
        main_f = "Bring your hands to prayer at the chest or reach them overhead."
        coach_f = "Once the leg is steady, finish the pose with calm, balanced arms."
        tips = ["Your leg position is good.", "Now correct the hand and arm position."]

    elif hands_up and not elbows_straight:
        score = 88
        status = "warning"
        pose_label = "Tree Pose"
        main_f = "Lengthen both elbows as the arms reach upward."
        coach_f = "Reach through the fingertips without tightening the shoulders."
        tips = ["Reach upward through both arms.", "Keep both elbows long and balanced."]

    else:
        score = 100
        status = "perfect"
        pose_label = "Correct Tree"
        main_f = "Beautiful Tree pose. Stay tall and breathe steadily."
        coach_f = "Keep rooting down through the standing foot and lifting up through the crown."
        tips = ["Excellent posture.", "Focus on one spot.", "Breathe steadily."]

    checks = {
        "standing_leg": standing_leg_ok,
        "foot_place": foot_place_ok,
        "foot_close_to_leg": foot_close_to_leg,
        "foot_height_ok": foot_height_ok,
        "foot_zone": foot_zone,
        "no_knee_pressure": not_on_knee_joint,
        "knee_open": knee_open_ok,
        "hands_ready": hands_ready,
        "prayer_hands": prayer_hands,
        "hands_up": hands_up,
        "elbows_straight": elbows_straight,
        "elbows_soft_ok": elbows_soft_ok,
        "left_elbow_ok": left_elbow_ok,
        "right_elbow_ok": right_elbow_ok,
        "left_arm_ready": left_arm_ready,
        "right_arm_ready": right_arm_ready,
        "hands_symmetric": hands_symmetric,
        "hands_not_too_wide": hands_not_too_wide,
        "torso": torso_ok,
        "one_foot_lifted": one_foot_lifted,
        "inferred_valid_leg_position": inferred_valid_leg_position,
        "strict_tree_gate": strict_tree_gate,
    }

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_f,
        "coach_text": coach_f,
        "tips": tips,
        "standing_side": standing_side,
        "stand_knee_idx": stand_knee_idx,
        "bent_knee_idx": bent_knee_idx,
        "stand_ankle_idx": stand_ankle_idx,
        "bent_ankle_idx": bent_ankle_idx,
        "angles": {
            "left_knee_angle": round(float(left_knee_angle), 1),
            "right_knee_angle": round(float(right_knee_angle), 1),
            "left_elbow_angle": round(float(left_elbow_angle), 1),
            "right_elbow_angle": round(float(right_elbow_angle), 1),
            "torso_tilt": round(float(torso_tilt), 1),
        },
        "checks": checks,
    }


def is_tree_like(model_label, model_confidence, analysis):
    label = str(model_label).lower()
    checks = analysis["checks"]

    if not checks.get("strict_tree_gate", False):
        return False

    if "tree" in label and model_confidence >= 0.72:
        return True

    if (
        checks.get("standing_leg") and
        checks.get("foot_place") and
        checks.get("no_knee_pressure") and
        checks.get("knee_open") and
        checks.get("torso") and
        checks.get("hands_ready") and
        checks.get("elbows_soft_ok") and
        checks.get("one_foot_lifted")
    ):
        return True

    return False


# =========================================================
# DEFECT / COACHING / STABILITY
# =========================================================
def calculate_defect_score(defect_label):
    score = 100
    penalties = {
        "Perfect_Tree": 0,
        "Bent_Support_Leg": 18,
        "Low_Hands": 12,
        "Torso_Lean": 18,
        "Poor_Balance": 12,
    }
    score -= penalties.get(defect_label, 18)
    return max(score, 45)


def get_defect_tips(defect_label):
    if defect_label == "Perfect_Tree":
        return ["Excellent posture. Maintain balance and breathe steadily."]
    if defect_label == "Bent_Support_Leg":
        return ["Keep your standing leg straighter.", "Press firmly through the grounded foot."]
    if defect_label == "Low_Hands":
        return ["Raise your hands overhead or bring them to prayer.", "Keep both arms balanced."]
    if defect_label == "Torso_Lean":
        return ["Keep your torso upright.", "Avoid leaning sideways."]
    if defect_label == "Poor_Balance":
        return ["Focus on a fixed point.", "Tighten your core and stabilize your balance."]
    return ["Adjust your posture."]


def update_stability_metrics(raw_pts):
    shoulder_center = (raw_pts[LEFT_SHOULDER] + raw_pts[RIGHT_SHOULDER]) / 2.0
    hip_center = (raw_pts[LEFT_HIP] + raw_pts[RIGHT_HIP]) / 2.0

    torso_center_x = float(hip_center[0])
    shoulder_tilt = abs(float(raw_pts[LEFT_SHOULDER][1] - raw_pts[RIGHT_SHOULDER][1]))
    torso_lean = abs(float(shoulder_center[0] - hip_center[0]))

    TORSO_CENTER_HISTORY.append(torso_center_x)
    SHOULDER_TILT_HISTORY.append(shoulder_tilt)
    TORSO_LEAN_HISTORY.append(torso_lean)


def get_stability_feedback():
    shake = moving_std(TORSO_CENTER_HISTORY)
    shoulder_wobble = moving_std(SHOULDER_TILT_HISTORY)
    torso_wobble = moving_std(TORSO_LEAN_HISTORY)

    feedback = []
    penalty = 0

    if shake > 0.020:
        feedback.append("You are shaking - steady your balance")
        penalty += 6

    if shoulder_wobble > 0.012:
        feedback.append("Keep your shoulders more stable")
        penalty += 4

    if torso_wobble > 0.012:
        feedback.append("Stabilize your torso")
        penalty += 5

    return feedback, penalty


def update_hold_state(is_tree, defect_label, full_body_visible, low_light):
    return 0.0, 0.0


def hold_bonus(hold_time):
    return 0


def choose_quality_label(analysis, defect_label, defect_confidence):
    checks = analysis["checks"]

    if (
        analysis["score"] >= 90 and
        checks.get("torso") and
        checks.get("strict_tree_gate") and
        checks.get("hands_ready")
    ):
        return "Perfect_Tree"

    if defect_confidence >= 0.65 and defect_label != "N/A":
        return defect_label

    if not checks.get("standing_leg"):
        return "Bent_Support_Leg"

    if not checks.get("hands_ready") or (checks.get("hands_up") and not checks.get("elbows_soft_ok")):
        return "Low_Hands"

    if not checks.get("torso"):
        return "Torso_Lean"

    return "Perfect_Tree"


# =========================================================
# FRONTEND OVERLAY HELPERS
# =========================================================
def build_tree_joint_states(analysis):
    checks = analysis["checks"]
    angles = analysis["angles"]
    standing_side = analysis.get("standing_side", "left")
    lifted_side = "right" if standing_side == "left" else "left"

    keys = [
        "standing_leg",
        "foot_place",
        "no_knee_pressure",
        "knee_open",
        "hands_ready",
        "prayer_hands",
        "hands_up",
        "elbows_straight",
        "elbows_soft_ok",
        "left_elbow_ok",
        "right_elbow_ok",
        "left_arm_ready",
        "right_arm_ready",
        "hands_symmetric",
        "hands_not_too_wide",
        "torso",
        "one_foot_lifted",
        "strict_tree_gate",
    ]
    smoothed = {key: smooth_runtime_boolean(key, checks.get(key, False)) for key in keys}

    left_is_standing = standing_side == "left"
    right_is_standing = standing_side == "right"
    right_lifted = lifted_side == "right"

    standing_leg_ok = smoothed["standing_leg"]
    lifted_leg_ok = smoothed["one_foot_lifted"] and smoothed["foot_place"] and smoothed["no_knee_pressure"]
    hands_ok = smoothed["hands_ready"]

    joint_states = {
        "standing_side": standing_side,
        "lifted_side": lifted_side,
        "standing_leg": {
            "ok": standing_leg_ok,
            "side": standing_side,
            "angle": angles["left_knee_angle"] if left_is_standing else angles["right_knee_angle"],
            "threshold": TREE_STANDING_LEG_MIN_ANGLE,
        },
        "lifted_leg": {
            "ok": lifted_leg_ok,
            "side": lifted_side,
            "angle": angles["right_knee_angle"] if right_lifted else angles["left_knee_angle"],
            "off_knee": smoothed["no_knee_pressure"],
            "foot_placed": smoothed["foot_place"],
        },
        "torso": {
            "ok": smoothed["torso"],
            "angle": angles["torso_tilt"],
            "threshold": TREE_TORSO_TILT_MAX,
        },
        "hands": {
            "ok": hands_ok,
            "mode": "prayer" if smoothed["prayer_hands"] else ("overhead" if smoothed["hands_up"] else "transition"),
        },
        "left_knee": {
            "ok": standing_leg_ok if left_is_standing else smoothed["knee_open"],
            "angle": angles["left_knee_angle"],
            "role": "standing" if left_is_standing else "lifted",
        },
        "right_knee": {
            "ok": standing_leg_ok if right_is_standing else smoothed["knee_open"],
            "angle": angles["right_knee_angle"],
            "role": "standing" if right_is_standing else "lifted",
        },
        "left_ankle": {
            "ok": standing_leg_ok if left_is_standing else lifted_leg_ok,
            "role": "standing" if left_is_standing else "lifted",
        },
        "right_ankle": {
            "ok": standing_leg_ok if right_is_standing else lifted_leg_ok,
            "role": "standing" if right_is_standing else "lifted",
        },
        "left_hip": {
            "ok": smoothed["torso"] and (standing_leg_ok if left_is_standing else smoothed["knee_open"]),
            "role": "standing" if left_is_standing else "lifted",
        },
        "right_hip": {
            "ok": smoothed["torso"] and (standing_leg_ok if right_is_standing else smoothed["knee_open"]),
            "role": "standing" if right_is_standing else "lifted",
        },
        "left_shoulder": {
            "ok": smoothed["torso"] and hands_ok,
        },
        "right_shoulder": {
            "ok": smoothed["torso"] and hands_ok,
        },
        "left_elbow": {
            "ok": smoothed["left_elbow_ok"],
            "angle": angles["left_elbow_angle"],
            "threshold": TREE_ELBOW_SOFT_MIN,
        },
        "right_elbow": {
            "ok": smoothed["right_elbow_ok"],
            "angle": angles["right_elbow_angle"],
            "threshold": TREE_ELBOW_SOFT_MIN,
        },
        "left_wrist": {
            "ok": smoothed["left_arm_ready"],
        },
        "right_wrist": {
            "ok": smoothed["right_arm_ready"],
        },
        "nose": {
            "ok": smoothed["torso"],
        },
    }

    return joint_states, smoothed


def build_points_for_frontend(raw_pts, landmarks, analysis):
    base_color = GREEN if analysis["score"] >= 86 else YELLOW

    points = []
    for idx in SELECTED_POINTS:
        lm = landmarks[idx]
        visibility = float(lm.visibility)
        key = f"tree_{idx}"
        has_recent_point = bool(POINT_HISTORY.get(key))

        if visibility < TREE_FRONTEND_POINT_VISIBILITY_MIN and not has_recent_point:
            continue

        sx, sy = latest_tree_point(key, raw_pts[idx][0], raw_pts[idx][1])
        current_visible = visibility >= TREE_FRONTEND_POINT_VISIBILITY_MIN

        points.append({
            "name": POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": clip01(sx),
            "y": clip01(sy),
            "color": base_color,
            "radius": 6 if current_visible else 5,
            "visible": True,
            "visibility": round(visibility, 3),
        })

    return points


def build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    mapping = [
        (LEFT_KNEE, analysis.get("angles", {}).get("left_knee_angle", 0), "left_knee"),
        (RIGHT_KNEE, analysis.get("angles", {}).get("right_knee_angle", 0), "right_knee"),
        (LEFT_ELBOW, analysis.get("angles", {}).get("left_elbow_angle", 0), "left_elbow"),
        (RIGHT_ELBOW, analysis.get("angles", {}).get("right_elbow_angle", 0), "right_elbow"),
    ]

    for idx, value, joint_key in mapping:
        lm = landmarks[idx]
        key = f"tree_{idx}"
        if float(lm.visibility) < TREE_FRONTEND_POINT_VISIBILITY_MIN and not POINT_HISTORY.get(key):
            continue
        sx, sy = latest_tree_point(key, raw_pts[idx][0], raw_pts[idx][1])

        items.append({
            "text": f"{int(round(float(value)))}{TREE_DEGREE_SIGN}",
            "x": clip01(sx),
            "y": clip01(sy),
            "color": YELLOW,
            "joint_key": joint_key,
        })

    return items


# =========================================================
# MAIN PROCESS FUNCTION
# =========================================================
def process_yoga_pose_request(request):
    global TREE_HOLD_START, PERFECT_HOLD_COUNT

    try:
        load_tree_runtime_state(request)

        if request.POST.get("reset") == "true":
            reset_tree_runtime_state()

        uploaded_file = request.FILES["image"]
        frame = read_uploaded_image(uploaded_file)

        if frame is None:
            return api_error("Invalid image file", status=400)

        frame = enhance_frame(frame)

        low_light, brightness = check_lighting(frame)
        if low_light:
            reset_tree_runtime_state()

            return api_success(
                pose="Low Light",
                model_pose="Unknown",
                quality="N/A",
                feedback="Room lighting is too low. Increase light for accurate pose detection.",
                coach_text="Improve room lighting.",
                status="warning",
                confidence=0.0,
                defect_confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=0.0,
                angles={},
                details=["Increase room lighting", "Face the light source", "Avoid dark background"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
                joint_states={},
            )

        landmarks = detect_landmarks(frame)
        if not landmarks:
            reset_tree_runtime_state()

            return api_success(
                pose="Unknown",
                model_pose="Unknown",
                quality="N/A",
                feedback="No human pose detected. Move back slightly and show full body clearly.",
                coach_text="Stand where your full body is visible.",
                status="unknown",
                confidence=0.0,
                defect_confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=0.0,
                angles={},
                details=["Show full body", "Improve room lighting", "Stay centered in frame"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
                joint_states={},
            )

        features_df, _, _ = build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = smooth_tree_points(raw_pts, landmarks)

        lm_dict = extract_raw_landmark_dict(landmarks)
        full_body_visible, visible_count, avg_visibility = check_body_visibility(lm_dict)
        framing_feedback = check_frame_position(smoothed_pts)

        if not full_body_visible:
            reset_tree_runtime_state()

            details = ["Show full body clearly", "Keep both feet visible", "Stand in the center of the camera"]
            details.extend(framing_feedback)

            return api_success(
                pose="Body Not Visible",
                model_pose="Unknown",
                quality="N/A",
                feedback="Move a little back. Full body should be visible.",
                coach_text="Move back until your full body is visible.",
                status="warning",
                confidence=0.0,
                defect_confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=0.0,
                angles={},
                details=dedupe_text_list(details, max_items=3),
                perfect_hold=False,
                points=[],
                angle_texts=[],
                joint_states={},
            )

        predicted_label, confidence = predict_pose_label(features_df)
        stable_predicted_label = smooth_label(POSE_HISTORY, predicted_label)

        analysis = analyze_tree_pose(smoothed_pts)
        joint_states, _ = build_tree_joint_states(analysis)
        points = build_points_for_frontend(smoothed_pts, landmarks, analysis)
        angle_texts = build_angle_texts(smoothed_pts, landmarks, analysis)

        tree_like = is_tree_like(stable_predicted_label, confidence, analysis)

        if not tree_like:
            DEFECT_HISTORY.clear()
            TREE_HOLD_START = None
            PERFECT_HOLD_COUNT = 0

            tips = []
            tips.extend(analysis.get("tips", []))
            tips.extend([
                "Lift one foot and place it on the opposite inner leg",
                "Keep the standing leg straight",
                "Open the bent knee outward",
            ])
            tips.extend(framing_feedback)

            return api_success(
                pose="Not Tree Pose",
                model_pose=stable_predicted_label,
                quality="N/A",
                feedback=analysis.get("main_feedback", "This is not Tree pose yet."),
                coach_text=analysis.get("coach_text", "Lift one foot onto the opposite inner leg to enter Tree pose."),
                status="warning",
                confidence=round(float(confidence), 3),
                defect_confidence=0.0,
                score=max(0, min(45, analysis.get("score", 0))),
                hold_time=0.0,
                best_hold_time=0.0,
                angles=analysis.get("angles", {}),
                details=dedupe_text_list(tips, max_items=3),
                perfect_hold=False,
                points=points,
                angle_texts=angle_texts,
                joint_states=joint_states,
            )

        raw_defect_label, raw_defect_confidence = predict_defect_label(features_df)

        if raw_defect_confidence >= 0.58:
            stable_defect = smooth_label(DEFECT_HISTORY, raw_defect_label)
            defect_label = choose_quality_label(analysis, stable_defect, raw_defect_confidence)
            defect_confidence = raw_defect_confidence
        else:
            defect_label = choose_quality_label(analysis, "N/A", 0.0)
            defect_confidence = 0.0

        update_stability_metrics(smoothed_pts)
        stability_tips, stability_penalty = get_stability_feedback()

        rule_score = analysis["score"]
        defect_score = calculate_defect_score(defect_label)
        combined_score = int(round((rule_score * 0.82) + (defect_score * 0.18)))
        combined_score = max(0, combined_score - stability_penalty)
        stable_score = smooth_score(combined_score)

        if stable_score >= 94 and analysis["checks"].get("strict_tree_gate") and defect_label in ["Perfect_Tree", "N/A"]:
            pose_name = "Correct Tree"
            stable_status = "perfect"
            feedback_text = "Beautiful Tree pose. Stay tall and hold steady."
            coach_text = analysis.get("coach_text", "Keep breathing and stay steady.")
        elif stable_score >= 80:
            pose_name = "Tree Pose"
            stable_status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = analysis.get("coach_text", "Refine the pose and stay steady.")
        elif stable_score >= 58:
            pose_name = "Tree Needs Correction"
            stable_status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = analysis.get("coach_text", "Make one calm correction at a time.")
        else:
            pose_name = "Not Ready Yet"
            stable_status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = analysis.get("coach_text", "Build the pose from the ground up.")

        stable_feedback = smooth_feedback(feedback_text)

        tips = []
        tips.extend(get_defect_tips(defect_label))
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        cleaned_tips = dedupe_text_list(
            tips,
            max_items=3,
            exclude=[stable_feedback, coach_text]
        )

        return api_success(
            pose=pose_name,
            model_pose=stable_predicted_label,
            quality=defect_label,
            feedback=stable_feedback,
            coach_text=coach_text,
            status=stable_status,
            confidence=round(float(confidence), 3),
            defect_confidence=round(float(defect_confidence), 3),
            score=stable_score,
            hold_time=0.0,
            best_hold_time=0.0,
            angles=analysis.get("angles", {}),
            details=cleaned_tips,
            perfect_hold=False,
            points=points,
            angle_texts=angle_texts,
            joint_states=joint_states,
        )

    except Exception as e:
        print("predict_yoga_pose error:", str(e))
        return api_error(str(e), status=500)
    finally:
        save_tree_runtime_state(request)
