from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from django.conf import settings
from django.http import JsonResponse


# =========================================================
# RUNTIME / THRESHOLDS
# =========================================================
WR_POSE_HISTORY_SIZE = 10
WR_SCORE_HISTORY_SIZE = 8
WR_FEEDBACK_HISTORY_SIZE = 8
WR_STABILITY_HISTORY_SIZE = 20
WR_VISIBILITY_HISTORY_SIZE = 8
WR_DETECTION_HISTORY_SIZE = 8
WR_POINT_HISTORY_SIZE = 4
WR_FRONTEND_POINT_VISIBILITY_MIN = 0.18
WR_SESSION_RUNTIME_KEY = "warrior_runtime_v2"
WR_DEGREE_SIGN = chr(176)

WR_STANCE_RATIO_MIN = 1.30
WR_STANCE_RATIO_IDEAL = 1.52
WR_FRONT_KNEE_BENT_MAX = 138.0
WR_FRONT_KNEE_IDEAL_MIN = 88.0
WR_FRONT_KNEE_IDEAL_MAX = 124.0
WR_BACK_KNEE_SOFT_MIN = 150.0
WR_BACK_KNEE_STRAIGHT_MIN = 158.0
WR_ARM_LEVEL_MAX_DIFF = 0.11
WR_TORSO_CENTER_MAX_OFFSET = 0.18
WR_FRONT_KNEE_OVER_ANKLE_MAX = 0.16


@dataclass
class WarriorRuntime:
    pose_history: deque = field(default_factory=lambda: deque(maxlen=WR_POSE_HISTORY_SIZE))
    score_history: deque = field(default_factory=lambda: deque(maxlen=WR_SCORE_HISTORY_SIZE))
    feedback_history: deque = field(default_factory=lambda: deque(maxlen=WR_FEEDBACK_HISTORY_SIZE))
    hip_height_history: deque = field(default_factory=lambda: deque(maxlen=WR_STABILITY_HISTORY_SIZE))
    hip_shift_history: deque = field(default_factory=lambda: deque(maxlen=WR_STABILITY_HISTORY_SIZE))
    spine_line_history: deque = field(default_factory=lambda: deque(maxlen=WR_STABILITY_HISTORY_SIZE))
    visibility_history: deque = field(default_factory=lambda: deque(maxlen=WR_VISIBILITY_HISTORY_SIZE))
    detection_history: deque = field(default_factory=lambda: deque(maxlen=WR_DETECTION_HISTORY_SIZE))
    point_history: dict = field(default_factory=dict)
    hold_start: float | None = None
    best_hold_time: float = 0.0
    perfect_hold_count: int = 0


def wr_runtime_to_session_data(runtime):
    return {
        "pose_history": list(runtime.pose_history),
        "score_history": list(runtime.score_history),
        "feedback_history": list(runtime.feedback_history),
        "hip_height_history": list(runtime.hip_height_history),
        "hip_shift_history": list(runtime.hip_shift_history),
        "spine_line_history": list(runtime.spine_line_history),
        "visibility_history": list(runtime.visibility_history),
        "detection_history": list(runtime.detection_history),
        "point_history": {
            str(key): [[float(x), float(y), float(z)] for x, y, z in values]
            for key, values in runtime.point_history.items()
        },
        "hold_start": runtime.hold_start,
        "best_hold_time": float(runtime.best_hold_time),
        "perfect_hold_count": int(runtime.perfect_hold_count),
    }


def wr_runtime_from_session_data(data):
    runtime = WarriorRuntime()
    if not isinstance(data, dict):
        return runtime

    runtime.pose_history = deque(data.get("pose_history", []), maxlen=WR_POSE_HISTORY_SIZE)
    runtime.score_history = deque(data.get("score_history", []), maxlen=WR_SCORE_HISTORY_SIZE)
    runtime.feedback_history = deque(data.get("feedback_history", []), maxlen=WR_FEEDBACK_HISTORY_SIZE)
    runtime.hip_height_history = deque(data.get("hip_height_history", []), maxlen=WR_STABILITY_HISTORY_SIZE)
    runtime.hip_shift_history = deque(data.get("hip_shift_history", []), maxlen=WR_STABILITY_HISTORY_SIZE)
    runtime.spine_line_history = deque(data.get("spine_line_history", []), maxlen=WR_STABILITY_HISTORY_SIZE)
    runtime.visibility_history = deque(data.get("visibility_history", []), maxlen=WR_VISIBILITY_HISTORY_SIZE)
    runtime.detection_history = deque(data.get("detection_history", []), maxlen=WR_DETECTION_HISTORY_SIZE)
    runtime.point_history = {
        str(key): deque(
            [(float(x), float(y), float(z)) for x, y, z in values],
            maxlen=WR_POINT_HISTORY_SIZE,
        )
        for key, values in data.get("point_history", {}).items()
    }
    runtime.hold_start = data.get("hold_start")
    runtime.best_hold_time = float(data.get("best_hold_time", 0.0) or 0.0)
    runtime.perfect_hold_count = int(data.get("perfect_hold_count", 0) or 0)
    return runtime


def wr_get_runtime(request):
    if request.session.session_key is None:
        request.session.save()
    return wr_runtime_from_session_data(request.session.get(WR_SESSION_RUNTIME_KEY))


def wr_store_runtime(request, runtime):
    if request.session.session_key is None:
        request.session.save()
    request.session[WR_SESSION_RUNTIME_KEY] = wr_runtime_to_session_data(runtime)
    request.session.modified = True


def wr_reset_runtime_state(runtime):
    runtime.pose_history.clear()
    runtime.score_history.clear()
    runtime.feedback_history.clear()
    runtime.hip_height_history.clear()
    runtime.hip_shift_history.clear()
    runtime.spine_line_history.clear()
    runtime.visibility_history.clear()
    runtime.detection_history.clear()
    runtime.point_history.clear()
    runtime.hold_start = None
    runtime.best_hold_time = 0.0
    runtime.perfect_hold_count = 0


# =========================================================
# MEDIAPIPE
# =========================================================
mp_pose = mp.solutions.pose
warrior_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.52,
    min_tracking_confidence=0.52,
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
        f"{name}_x",
        f"{name}_y",
        f"{name}_z",
        f"{name}_visibility",
    ])

WR_ANGLE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
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


def resolve_warrior_model_path(filename: str) -> Path | None:
    candidates = [
        WR_BASE_DIR / "Ml_Models" / filename,
        WR_BASE_DIR / "Ml_models" / filename,
        WR_BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def wr_load_serialized_object(path: Path | None, label: str):
    if path is None:
        return None

    try:
        return joblib.load(path)
    except Exception as joblib_error:
        try:
            with open(path, "rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as pickle_error:
            warnings.warn(
                f"Failed to load {label} from {path}: "
                f"joblib={joblib_error}; pickle={pickle_error}"
            )
            return None


WARRIOR_POSE_MODEL_PATH = resolve_warrior_model_path("warrior_pose_model.pkl")
WARRIOR_POSE_ENCODER_PATH = resolve_warrior_model_path("warrior_label_encoder.pkl")
WARRIOR_DEFECT_MODEL_PATH = resolve_warrior_model_path("warrior_defect_model.pkl")
WARRIOR_DEFECT_ENCODER_PATH = resolve_warrior_model_path("warrior_defect_label_encoder.pkl")

rf_pose = wr_load_serialized_object(WARRIOR_POSE_MODEL_PATH, "Warrior pose model")
le_pose = wr_load_serialized_object(WARRIOR_POSE_ENCODER_PATH, "Warrior pose label encoder")
rf_def = wr_load_serialized_object(WARRIOR_DEFECT_MODEL_PATH, "Warrior defect model")
le_def = wr_load_serialized_object(WARRIOR_DEFECT_ENCODER_PATH, "Warrior defect label encoder")


# =========================================================
# RESPONSE / TEXT HELPERS
# =========================================================
def wr_pose_success(**kwargs):
    payload = {
        "pose": "Unknown",
        "model_pose": "Unknown",
        "quality": "Not_Ready",
        "feedback": "Waiting for pose.",
        "coach_text": "Show your full body in the frame.",
        "status": "unknown",
        "confidence": 0.0,
        "score": 0,
        "hold_time": 0.0,
        "best_hold_time": 0.0,
        "angles": {},
        "details": [],
        "perfect_hold": False,
        "points": [],
        "angle_texts": [],
        "joint_states": {},
        "pose_ready": False,
        "hold_ready": False,
    }
    payload.update(kwargs)
    return JsonResponse({"success": True, **payload})


def wr_api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


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
        if not item:
            continue
        text = wr_clean_text(item)
        key = wr_normalize_key(text)
        if not key or key in seen or key in exclude_keys:
            continue
        seen.add(key)
        output.append(text)
        if max_items and len(output) >= max_items:
            break
    return output


# =========================================================
# BASIC HELPERS
# =========================================================
def wr_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def wr_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.08, beta=8)


def wr_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = warrior_pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark


def wr_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 58, brightness


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
    if len(values) < 3:
        return 0.0
    return float(np.std(list(values)))


# =========================================================
# RUNTIME SMOOTHING
# =========================================================
def wr_smooth_label(runtime, new_label):
    runtime.pose_history.append(str(new_label))
    return Counter(runtime.pose_history).most_common(1)[0][0]


def wr_smooth_score(runtime, new_score):
    runtime.score_history.append(float(new_score))
    return int(round(sum(runtime.score_history) / len(runtime.score_history)))


def wr_smooth_feedback(runtime, new_feedback):
    runtime.feedback_history.append(str(new_feedback))
    return Counter(runtime.feedback_history).most_common(1)[0][0]


def wr_smooth_boolean(history, value):
    history.append(bool(value))
    return sum(history) >= max(1, len(history) // 2 + 1)


def wr_smooth_point(runtime, key, x, y, z, visibility=1.0):
    if key not in runtime.point_history:
        runtime.point_history[key] = deque(maxlen=WR_POINT_HISTORY_SIZE)

    history = runtime.point_history[key]
    current = np.array([float(x), float(y), float(z)], dtype=np.float32)
    visibility = float(np.clip(visibility, 0.0, 1.0))

    if visibility < WR_FRONTEND_POINT_VISIBILITY_MIN and history:
        previous = history[-1]
        return float(previous[0]), float(previous[1]), float(previous[2])

    if not history:
        history.append(tuple(current))
        return float(current[0]), float(current[1]), float(current[2])

    prev = np.array(history[-1], dtype=np.float32)
    delta = current - prev
    distance = float(np.linalg.norm(delta[:2]))

    if visibility >= 0.70 and distance > 0.28:
        history.append(tuple(current))
        return float(current[0]), float(current[1]), float(current[2])

    max_step = 0.18 if visibility >= 0.65 else 0.10

    if distance > max_step:
        current = prev + (delta / (float(np.linalg.norm(delta)) + 1e-6)) * max_step

    alpha = 0.72 if visibility >= 0.65 else 0.48

    smoothed = prev * (1.0 - alpha) + current * alpha
    history.append(tuple(smoothed))
    return float(smoothed[0]), float(smoothed[1]), float(smoothed[2])


# =========================================================
# FEATURES
# =========================================================
def wr_extract_raw_landmark_dict(landmarks):
    lm_dict = {}
    for name, idx in WR_ALL_LANDMARK_NAME_TO_INDEX.items():
        lm = landmarks[idx]
        lm_dict[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
    return lm_dict


def wr_normalize_landmarks_inplace(lm_dict):
    hip_center_x = (lm_dict["left_hip"]["x"] + lm_dict["right_hip"]["x"]) / 2.0
    hip_center_y = (lm_dict["left_hip"]["y"] + lm_dict["right_hip"]["y"]) / 2.0
    for name in lm_dict.keys():
        lm_dict[name]["x"] -= hip_center_x
        lm_dict[name]["y"] -= hip_center_y


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


def wr_angles_from_points(raw_pts):
    return {
        "left_knee_angle": wr_calculate_angle(raw_pts[WR_LEFT_HIP][:2], raw_pts[WR_LEFT_KNEE][:2], raw_pts[WR_LEFT_ANKLE][:2]),
        "right_knee_angle": wr_calculate_angle(raw_pts[WR_RIGHT_HIP][:2], raw_pts[WR_RIGHT_KNEE][:2], raw_pts[WR_RIGHT_ANKLE][:2]),
        "left_elbow_angle": wr_calculate_angle(raw_pts[WR_LEFT_SHOULDER][:2], raw_pts[WR_LEFT_ELBOW][:2], raw_pts[WR_LEFT_WRIST][:2]),
        "right_elbow_angle": wr_calculate_angle(raw_pts[WR_RIGHT_SHOULDER][:2], raw_pts[WR_RIGHT_ELBOW][:2], raw_pts[WR_RIGHT_WRIST][:2]),
    }


def wr_predict_model_label(features_df):
    if rf_pose is None or le_pose is None:
        return "Unknown", 0.0

    expected_cols = getattr(rf_pose, "feature_names_in_", WR_MODEL_COLUMNS)
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
    except Exception:
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
    except Exception:
        return "Unknown", 0.0


# =========================================================
# VISIBILITY / FRAMING
# =========================================================
def wr_check_body_visibility(lm_dict):
    core_names = [
        "left_shoulder", "right_shoulder",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    visibilities = [lm_dict[name]["visibility"] for name in core_names]
    visible_count = sum(v > 0.22 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    shoulders_ok = (
        lm_dict["left_shoulder"]["visibility"] > 0.28 or
        lm_dict["right_shoulder"]["visibility"] > 0.28
    )
    hips_ok = (
        lm_dict["left_hip"]["visibility"] > 0.30 or
        lm_dict["right_hip"]["visibility"] > 0.30
    )
    knees_ok = (
        lm_dict["left_knee"]["visibility"] > 0.24 or
        lm_dict["right_knee"]["visibility"] > 0.24
    )
    ankles_ok = (
        lm_dict["left_ankle"]["visibility"] > 0.20 or
        lm_dict["right_ankle"]["visibility"] > 0.20
    )
    wrists_ok = (
        lm_dict["left_wrist"]["visibility"] > 0.18 or
        lm_dict["right_wrist"]["visibility"] > 0.18
    )

    full_body_visible = visible_count >= 6 and shoulders_ok and hips_ok and knees_ok and (ankles_ok or wrists_ok)
    return full_body_visible, visible_count, avg_visibility


def wr_check_frame_position(raw_pts, landmarks):
    visible_pts = [raw_pts[i] for i in WR_SELECTED_POINTS if float(landmarks[i].visibility) > 0.22]
    if len(visible_pts) < 4:
        return []

    xs = [float(p[0]) for p in visible_pts]
    ys = [float(p[1]) for p in visible_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2.0

    feedback = []
    if width > 0.97:
        feedback.append("Move slightly away to show your full lunge")
    if width < 0.36:
        feedback.append("Move a little closer so your full body is easier to track")
    if center_x < 0.28:
        feedback.append("Move slightly to your right")
    elif center_x > 0.72:
        feedback.append("Move slightly to your left")
    if min_y < 0.02 or height > 0.97:
        feedback.append("Keep a little more space above your head")
    return feedback


# =========================================================
# ANALYSIS
# =========================================================
def analyze_warrior_pose(raw_pts, landmarks, angles):
    left_knee_angle = float(angles["left_knee_angle"])
    right_knee_angle = float(angles["right_knee_angle"])

    if left_knee_angle <= right_knee_angle:
        front_side = "left"
        front_knee_angle = left_knee_angle
        back_knee_angle = right_knee_angle
        front_knee = raw_pts[WR_LEFT_KNEE]
        front_ankle = raw_pts[WR_LEFT_ANKLE]
    else:
        front_side = "right"
        front_knee_angle = right_knee_angle
        back_knee_angle = left_knee_angle
        front_knee = raw_pts[WR_RIGHT_KNEE]
        front_ankle = raw_pts[WR_RIGHT_ANKLE]

    left_shoulder = raw_pts[WR_LEFT_SHOULDER]
    right_shoulder = raw_pts[WR_RIGHT_SHOULDER]
    left_wrist = raw_pts[WR_LEFT_WRIST]
    right_wrist = raw_pts[WR_RIGHT_WRIST]
    left_hip = raw_pts[WR_LEFT_HIP]
    right_hip = raw_pts[WR_RIGHT_HIP]
    left_ankle = raw_pts[WR_LEFT_ANKLE]
    right_ankle = raw_pts[WR_RIGHT_ANKLE]

    shoulder_width = max(wr_distance(left_shoulder[:2], right_shoulder[:2]), 1e-6)
    ankle_width = wr_distance(left_ankle[:2], right_ankle[:2])
    stance_ratio = ankle_width / shoulder_width

    left_arm_level = abs(float(left_wrist[1] - left_shoulder[1])) <= WR_ARM_LEVEL_MAX_DIFF
    right_arm_level = abs(float(right_wrist[1] - right_shoulder[1])) <= WR_ARM_LEVEL_MAX_DIFF
    elbows_straight = (
        float(angles["left_elbow_angle"]) >= 148.0 and
        float(angles["right_elbow_angle"]) >= 148.0
    )
    arms_level = left_arm_level and right_arm_level and elbows_straight

    shoulder_center_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
    hip_center_x = float((left_hip[0] + right_hip[0]) / 2.0)
    torso_centered = abs(shoulder_center_x - hip_center_x) <= WR_TORSO_CENTER_MAX_OFFSET
    front_knee_over_ankle = abs(float(front_knee[0] - front_ankle[0])) <= WR_FRONT_KNEE_OVER_ANKLE_MAX

    front_knee_bent = front_knee_angle <= WR_FRONT_KNEE_BENT_MAX
    front_knee_ideal = WR_FRONT_KNEE_IDEAL_MIN <= front_knee_angle <= WR_FRONT_KNEE_IDEAL_MAX
    back_leg_soft = back_knee_angle >= WR_BACK_KNEE_SOFT_MIN
    back_leg_straight = back_knee_angle >= WR_BACK_KNEE_STRAIGHT_MIN
    stance_wide_enough = stance_ratio >= WR_STANCE_RATIO_MIN
    stance_ideal = stance_ratio >= WR_STANCE_RATIO_IDEAL
    arms_reaching = (
        float(angles["left_elbow_angle"]) >= 144.0 and
        float(angles["right_elbow_angle"]) >= 144.0
    )

    score = 0
    score += 18 if stance_wide_enough else 6
    score += 20 if front_knee_bent else 4
    score += 8 if front_knee_ideal else 0
    score += 18 if back_leg_soft else 4
    score += 8 if back_leg_straight else 0
    score += 12 if arms_reaching else 4
    score += 10 if arms_level else 0
    score += 4 if stance_ideal else 0
    score += 6 if torso_centered else 0
    score += 6 if front_knee_over_ankle else 0
    score = int(max(0, min(100, score)))

    pose_label = "Not Warrior"
    status = "warning"
    main_feedback = "Step wide and bend your front knee to enter Warrior II."

    if not stance_wide_enough:
        main_feedback = "Step your feet wider for a stronger Warrior II base."
    elif not back_leg_soft:
        main_feedback = "Straighten your back leg more."
    elif not front_knee_bent:
        main_feedback = "Bend your front knee deeper."
    elif not arms_reaching:
        main_feedback = "Reach strongly through both arms."
    elif not arms_level:
        main_feedback = "Level your arms parallel to the floor."
    elif not torso_centered:
        main_feedback = "Keep your torso centered over your hips."
    elif not front_knee_over_ankle:
        main_feedback = "Stack your front knee more directly over your ankle."

    tips = []
    if not stance_wide_enough:
        tips.append("Widen your stance so your legs can create a stronger base")
    if not front_knee_bent:
        tips.append("Bend the front knee until it tracks closer over the ankle")
    if not back_leg_soft:
        tips.append("Press through the back leg and keep it straighter")
    if not arms_level:
        tips.append("Lift or lower your arms until both wrists line up with the shoulders")
    if not torso_centered:
        tips.append("Keep your shoulders stacked more evenly above the hips")
    if not front_knee_over_ankle:
        tips.append("Adjust the front leg so the knee stays over the ankle")

    core_warrior_gate = stance_wide_enough and front_knee_bent and back_leg_soft and arms_reaching
    balanced_warrior_gate = core_warrior_gate and arms_level and torso_centered and front_knee_over_ankle

    if balanced_warrior_gate and score >= 92:
        pose_label = "Correct Warrior"
        status = "perfect" if front_knee_ideal and back_leg_straight else "good"
        main_feedback = "Perfect Warrior! Hold steady." if status == "perfect" else "Strong Warrior. Hold steady."
        tips = ["Keep your gaze soft over the front fingertips", "Stay grounded through both feet"]
    elif core_warrior_gate and score >= 78:
        pose_label = "Warrior Pose"
        status = "good"
        if not tips:
            tips = ["Deepen the lunge a little more", "Keep the chest open and steady"]
    elif score >= 62 and (front_knee_bent or back_leg_soft):
        pose_label = "Warrior Needs Correction"
        status = "warning"

    checks = {
        "front_side": front_side,
        "stance_ratio": stance_ratio,
        "stance_wide_enough": stance_wide_enough,
        "stance_ideal": stance_ideal,
        "front_knee_bent": front_knee_bent,
        "front_knee_ideal": front_knee_ideal,
        "back_leg_soft": back_leg_soft,
        "back_leg_straight": back_leg_straight,
        "arms_reaching": arms_reaching,
        "arms_level": arms_level,
        "torso_centered": torso_centered,
        "front_knee_over_ankle": front_knee_over_ankle,
        "core_warrior_gate": core_warrior_gate,
        "balanced_warrior_gate": balanced_warrior_gate,
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


def wr_pose_flags(checks, stable_score, pose_status):
    pose_ready = bool(
        checks.get("core_warrior_gate") or
        (
            checks.get("stance_wide_enough") and
            checks.get("front_knee_bent") and
            checks.get("back_leg_soft")
        )
    )
    good_pose_ready = bool(
        pose_ready and
        stable_score >= 80 and
        checks.get("arms_reaching")
    )
    hold_ready = bool(
        good_pose_ready and
        checks.get("balanced_warrior_gate") and
        stable_score >= 92 and
        str(pose_status).lower() in {"good", "perfect"}
    )
    return {
        "pose_ready": pose_ready,
        "good_pose_ready": good_pose_ready,
        "hold_ready": hold_ready,
    }


def wr_is_warrior_like(model_label, model_confidence, analysis, defect_label="Unknown", defect_confidence=0.0):
    label_text = str(model_label).lower()
    defect_text = str(defect_label).lower()

    if "warrior" in label_text and "not" not in label_text and model_confidence >= 0.42:
        return True
    if "warrior" in defect_text and "not" not in defect_text and defect_confidence >= 0.35:
        return True

    checks = analysis["checks"]
    return bool(
        checks.get("core_warrior_gate") or
        (
            checks.get("stance_wide_enough") and
            checks.get("front_knee_bent") and
            checks.get("back_leg_soft")
        )
    )


def wr_display_model_label(model_label, pose_name, pose_ready=False):
    label_text = str(model_label or "Unknown")
    pose_text = str(pose_name or "").lower()
    label_lower = label_text.lower()

    if "warrior" in pose_text and pose_ready and "not" in label_lower:
        return "Warrior"
    if pose_text == "correct warrior" and ("warrior" not in label_lower or "not" in label_lower):
        return "Warrior"
    return label_text


# =========================================================
# STABILITY / HOLD / QUALITY
# =========================================================
def wr_update_stability_metrics(runtime, raw_pts):
    hip_center = (raw_pts[WR_LEFT_HIP] + raw_pts[WR_RIGHT_HIP]) / 2.0
    shoulder_center = (raw_pts[WR_LEFT_SHOULDER] + raw_pts[WR_RIGHT_SHOULDER]) / 2.0

    runtime.hip_height_history.append(float(hip_center[1]))
    runtime.hip_shift_history.append(float(hip_center[0]))
    runtime.spine_line_history.append(abs(float(shoulder_center[0] - hip_center[0])))


def wr_get_stability_feedback(runtime):
    hip_wobble = wr_moving_std(runtime.hip_height_history)
    shift_wobble = wr_moving_std(runtime.hip_shift_history)

    feedback = []
    penalty = 0

    if hip_wobble > 0.015:
        feedback.append("Keep your lunge depth steady")
        penalty += 4
    if shift_wobble > 0.015:
        feedback.append("Keep your torso centered over your hips")
        penalty += 3

    return feedback, penalty


def wr_update_hold_state(runtime, is_pose, full_body_visible, low_light):
    valid_hold = is_pose and full_body_visible and not low_light
    if valid_hold:
        if runtime.hold_start is None:
            runtime.hold_start = time.time()
        hold_time = time.time() - runtime.hold_start
        runtime.best_hold_time = max(runtime.best_hold_time, hold_time)
    else:
        hold_time = 0.0
        runtime.hold_start = None

    return hold_time, runtime.best_hold_time


def wr_hold_bonus(hold_time):
    if hold_time >= 15:
        return 10
    if hold_time >= 10:
        return 8
    if hold_time >= 5:
        return 5
    if hold_time >= 3:
        return 2
    return 0


def wr_quality_from_score(score):
    if score >= 96:
        return "Perfect_Warrior"
    if score >= 85:
        return "Good_Warrior"
    if score >= 70:
        return "Needs_Correction"
    return "Not_Ready"


# =========================================================
# FRONTEND HELPERS
# =========================================================
def wr_build_joint_states(analysis):
    checks = analysis["checks"]
    angles = analysis["angles"]
    front_side = checks.get("front_side", "left")
    back_side = "right" if front_side == "left" else "left"

    def leg_state(side):
        is_front = side == front_side
        return {
            "ok": bool(checks.get("front_knee_bent") if is_front else checks.get("back_leg_soft")),
            "angle": angles.get(f"{side}_knee_angle", 0),
            "role": "front" if is_front else "back",
            "threshold": WR_FRONT_KNEE_BENT_MAX if is_front else WR_BACK_KNEE_SOFT_MIN,
        }

    return {
        "front_side": front_side,
        "back_side": back_side,
        "stance": {
            "ok": bool(checks.get("stance_wide_enough")),
            "ratio": round(float(checks.get("stance_ratio", 0.0)), 2),
            "threshold": WR_STANCE_RATIO_MIN,
        },
        "left_knee": leg_state("left"),
        "right_knee": leg_state("right"),
        "left_ankle": {
            "ok": bool(checks.get("front_knee_over_ankle") if front_side == "left" else checks.get("back_leg_soft")),
            "role": "front" if front_side == "left" else "back",
        },
        "right_ankle": {
            "ok": bool(checks.get("front_knee_over_ankle") if front_side == "right" else checks.get("back_leg_soft")),
            "role": "front" if front_side == "right" else "back",
        },
        "left_elbow": {
            "ok": bool(checks.get("arms_level") and checks.get("arms_reaching")),
            "angle": angles.get("left_elbow_angle", 0),
            "threshold": 148.0,
        },
        "right_elbow": {
            "ok": bool(checks.get("arms_level") and checks.get("arms_reaching")),
            "angle": angles.get("right_elbow_angle", 0),
            "threshold": 148.0,
        },
        "left_wrist": {"ok": bool(checks.get("arms_level"))},
        "right_wrist": {"ok": bool(checks.get("arms_level"))},
        "left_shoulder": {"ok": bool(checks.get("torso_centered") and checks.get("arms_level"))},
        "right_shoulder": {"ok": bool(checks.get("torso_centered") and checks.get("arms_level"))},
        "torso": {"ok": bool(checks.get("torso_centered")), "threshold": WR_TORSO_CENTER_MAX_OFFSET},
    }


def wr_latest_point(runtime, key, fallback):
    history = runtime.point_history.get(key)
    if history:
        return history[-1]
    return fallback


def wr_build_points_for_frontend(runtime, raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    front_side = checks.get("front_side")

    points = []
    for idx in WR_SELECTED_POINTS:
        visibility = float(landmarks[idx].visibility)
        point_key = f"wr_{idx}"
        recently_visible = visibility >= WR_FRONTEND_POINT_VISIBILITY_MIN or bool(runtime.point_history.get(point_key))
        if not recently_visible:
            continue
        display_point = raw_pts[idx] if visibility >= WR_FRONTEND_POINT_VISIBILITY_MIN else wr_latest_point(runtime, point_key, raw_pts[idx])
        current_visible = visibility >= WR_FRONTEND_POINT_VISIBILITY_MIN

        radius = 6 if current_visible else 5
        color = WR_GREEN if analysis["score"] >= 88 else WR_YELLOW

        if idx == (WR_LEFT_KNEE if front_side == "left" else WR_RIGHT_KNEE) and not checks.get("front_knee_bent"):
            color = WR_RED
            radius = 8
        if idx == (WR_RIGHT_KNEE if front_side == "left" else WR_LEFT_KNEE) and not checks.get("back_leg_soft"):
            color = WR_RED
            radius = 8
        if idx in [WR_LEFT_WRIST, WR_RIGHT_WRIST, WR_LEFT_ELBOW, WR_RIGHT_ELBOW] and not checks.get("arms_level"):
            color = WR_RED
            radius = 8

        points.append({
            "name": WR_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": float(np.clip(display_point[0], 0.0, 1.0)),
            "y": float(np.clip(display_point[1], 0.0, 1.0)),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
            "role": "front" if idx in [
                WR_LEFT_SHOULDER if front_side == "left" else WR_RIGHT_SHOULDER,
                WR_LEFT_ELBOW if front_side == "left" else WR_RIGHT_ELBOW,
                WR_LEFT_WRIST if front_side == "left" else WR_RIGHT_WRIST,
                WR_LEFT_HIP if front_side == "left" else WR_RIGHT_HIP,
                WR_LEFT_KNEE if front_side == "left" else WR_RIGHT_KNEE,
                WR_LEFT_ANKLE if front_side == "left" else WR_RIGHT_ANKLE,
            ] else "back",
        })
    return points


def wr_build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    primary = [
        (WR_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), WR_YELLOW, "left_knee"),
        (WR_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), WR_YELLOW, "right_knee"),
        (WR_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), WR_CYAN, "left_elbow"),
        (WR_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), WR_CYAN, "right_elbow"),
    ]

    for idx, value, color, joint_key in primary:
        if float(landmarks[idx].visibility) < 0.30:
            continue
        items.append({
            "text": f"{int(round(float(value)))}{WR_DEGREE_SIGN}",
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
            "joint_key": joint_key,
        })
    return items


# =========================================================
# MAIN PROCESS
# =========================================================
def process_warrior_pose_request(request):
    runtime = wr_get_runtime(request)

    try:
        if request.POST.get("reset") == "true":
            wr_reset_runtime_state(runtime)

        uploaded_file = request.FILES["image"]
        frame = wr_read_uploaded_image(uploaded_file)
        if frame is None:
            return wr_api_error("Invalid image file", status=400)

        frame = wr_enhance_frame(frame)
        low_light, _brightness = wr_check_lighting(frame)
        if low_light:
            wr_reset_runtime_state(runtime)
            wr_store_runtime(request, runtime)
            return wr_pose_success(
                pose="Low Light",
                status="warning",
                feedback="Room lighting is too low for accurate tracking.",
                coach_text="Improve your lighting and keep your full body visible.",
                details=["Increase room lighting", "Avoid standing in front of a dark background"],
            )

        landmarks = wr_detect_landmarks(frame)
        has_landmarks = landmarks is not None
        stable_has_landmarks = wr_smooth_boolean(runtime.detection_history, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            wr_reset_runtime_state(runtime)
            wr_store_runtime(request, runtime)
            return wr_pose_success(
                pose="Unknown",
                status="unknown",
                feedback="No human pose detected.",
                coach_text="Step back and show your full body.",
                details=["Show your whole body", "Move slightly back from the camera"],
            )

        if landmarks is None:
            wr_store_runtime(request, runtime)
            return wr_pose_success(
                pose="Tracking...",
                status="warning",
                feedback="Hold still while tracking stabilizes.",
                coach_text="Keep the pose steady for a moment.",
                best_hold_time=round(float(runtime.best_hold_time), 1),
            )

        features_df, lm_dict, angles = wr_build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = raw_pts.copy()

        for idx in WR_SELECTED_POINTS:
            sx, sy, sz = wr_smooth_point(
                runtime,
                f"wr_{idx}",
                raw_pts[idx][0],
                raw_pts[idx][1],
                raw_pts[idx][2],
                visibility=float(landmarks[idx].visibility),
            )
            smoothed_pts[idx] = [sx, sy, sz]

        smoothed_angles = wr_angles_from_points(smoothed_pts)
        analysis = analyze_warrior_pose(smoothed_pts, landmarks, smoothed_angles)
        joint_states = wr_build_joint_states(analysis)
        points = wr_build_points_for_frontend(runtime, smoothed_pts, landmarks, analysis)
        angle_texts = wr_build_angle_texts(smoothed_pts, landmarks, analysis)

        full_body_visible, _visible_count, _avg_visibility = wr_check_body_visibility(lm_dict)
        stable_full_body_visible = wr_smooth_boolean(runtime.visibility_history, full_body_visible)
        framing_feedback = wr_check_frame_position(smoothed_pts, landmarks)

        if not full_body_visible and not stable_full_body_visible:
            visible_points = points
            wr_reset_runtime_state(runtime)
            wr_store_runtime(request, runtime)
            details = ["Show hands and feet clearly in the frame"] + framing_feedback
            return wr_pose_success(
                pose="Body Not Visible",
                status="warning",
                feedback="Adjust the camera so your full Warrior II shape is visible.",
                coach_text="Show your whole body before starting the hold.",
                details=wr_dedupe_list(details, max_items=3),
                points=visible_points,
                angle_texts=angle_texts,
                joint_states=joint_states,
            )

        raw_model_label, confidence = wr_predict_model_label(features_df)
        stable_model_label = wr_smooth_label(runtime, raw_model_label)
        defect_label, defect_confidence = wr_predict_defect_label(features_df)

        is_warrior = wr_is_warrior_like(
            stable_model_label,
            confidence,
            analysis,
            defect_label=defect_label,
            defect_confidence=defect_confidence,
        )

        wr_update_stability_metrics(runtime, smoothed_pts)
        stability_tips, stability_penalty = wr_get_stability_feedback(runtime)

        combined_score = max(0, analysis["score"] - stability_penalty)
        if "warrior" in str(stable_model_label).lower() and confidence >= 0.65:
            combined_score = min(100, combined_score + 4)
        if "perfect" in str(defect_label).lower() and defect_confidence >= 0.50:
            combined_score = min(100, combined_score + 3)

        pose_flags = wr_pose_flags(analysis["checks"], combined_score, analysis["status"])

        if not pose_flags["pose_ready"] and not is_warrior and combined_score < 72:
            runtime.hold_start = None
            runtime.perfect_hold_count = 0
            wr_store_runtime(request, runtime)
            tips = analysis.get("tips", [])
            tips.extend(["Step feet wider and bend the front knee", "Reach both arms long and level"])
            tips.extend(framing_feedback)
            return wr_pose_success(
                pose="Not Warrior Pose",
                model_pose=stable_model_label,
                raw_model_pose=stable_model_label,
                quality="Not_Ready",
                feedback=analysis.get("main_feedback", "Move into Warrior II."),
                coach_text=analysis.get("main_feedback", "Move into Warrior II."),
                status="warning",
                confidence=round(float(confidence), 3),
                score=max(0, min(70, combined_score)),
                best_hold_time=round(float(runtime.best_hold_time), 1),
                angles=analysis.get("angles", {}),
                details=wr_dedupe_list(tips, max_items=3),
                points=points,
                angle_texts=angle_texts,
                joint_states=joint_states,
            )

        hold_time, best_hold = wr_update_hold_state(
            runtime,
            pose_flags["hold_ready"],
            stable_full_body_visible,
            low_light,
        )
        combined_score = min(100, combined_score + wr_hold_bonus(hold_time))
        stable_score = wr_smooth_score(runtime, combined_score)
        pose_flags = wr_pose_flags(analysis["checks"], stable_score, analysis["status"])

        if pose_flags["hold_ready"] and stable_score >= 95:
            pose_name = "Correct Warrior"
            status = "perfect"
            feedback_text = "Perfect Warrior"
            coach_text = "Excellent alignment. Hold steady and breathe."
        elif pose_flags["good_pose_ready"] or (is_warrior and stable_score >= 80):
            pose_name = "Warrior Pose"
            status = "good"
            feedback_text = analysis["main_feedback"] if analysis["status"] != "perfect" else "Strong Warrior. Hold steady."
            coach_text = "Good shape. Refine the details and hold steady."
        else:
            pose_name = "Warrior Needs Correction"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "Fix the highlighted alignment and keep your body steady."

        stable_feedback = wr_smooth_feedback(runtime, feedback_text)

        if pose_name == "Correct Warrior" and hold_time >= 3.0:
            runtime.perfect_hold_count += 1
        else:
            runtime.perfect_hold_count = 0

        tips = []
        if hold_time >= 5:
            tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        display_model_label = wr_display_model_label(
            stable_model_label,
            pose_name,
            pose_ready=pose_flags["pose_ready"] or is_warrior,
        )

        wr_store_runtime(request, runtime)
        return wr_pose_success(
            pose=pose_name,
            model_pose=display_model_label,
            raw_model_pose=stable_model_label,
            quality=wr_quality_from_score(stable_score),
            feedback=stable_feedback,
            coach_text=coach_text,
            status=status,
            confidence=round(float(confidence), 3),
            score=stable_score,
            hold_time=round(float(hold_time), 1),
            best_hold_time=round(float(best_hold), 1),
            angles=analysis["angles"],
            details=wr_dedupe_list(tips, max_items=3, exclude=[stable_feedback, coach_text]),
            perfect_hold=runtime.perfect_hold_count >= 3,
            points=points,
            angle_texts=angle_texts,
            joint_states=joint_states,
            pose_ready=pose_flags["pose_ready"] or is_warrior,
            hold_ready=pose_flags["hold_ready"],
        )

    except Exception as exc:
        return JsonResponse(
            {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            status=500,
        )
