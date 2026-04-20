from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass, field
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
# RUNTIME / THRESHOLDS
# =========================================================
DD_POSE_HISTORY_SIZE = 10
DD_SCORE_HISTORY_SIZE = 8
DD_FEEDBACK_HISTORY_SIZE = 8
DD_STABILITY_HISTORY_SIZE = 20
DD_VISIBILITY_HISTORY_SIZE = 8
DD_DETECTION_HISTORY_SIZE = 8
DD_POINT_HISTORY_SIZE = 6
DD_BOOLEAN_HISTORY_SIZE = 4

DD_DOMINANT_ELBOW_STRAIGHT_ANGLE = 146.0
DD_SUPPORT_ELBOW_STRAIGHT_ANGLE = 142.0
DD_DOMINANT_KNEE_STRAIGHT_ANGLE = 142.0
DD_SUPPORT_KNEE_STRAIGHT_ANGLE = 136.0
DD_DOMINANT_KNEE_STRONG_ANGLE = 148.0
DD_SHOULDER_OPEN_MIN_ANGLE = 124.0
DD_HIP_FOLD_MAX_ANGLE = 132.0
DD_SUPPORT_VISIBILITY_MIN = 0.40
DD_DOMINANT_VISIBILITY_MIN = 0.46
DD_POINT_VISIBILITY_MIN = 0.18
DD_POSE_READY_SCORE = 72
DD_HOLD_READY_SCORE = 86
DD_CORE_SHAPE_READY_MIN = 4
DD_DETAIL_SHAPE_READY_MIN = 2
DD_DEGREE_SIGN = chr(176)

DD_SESSION_RUNTIME_KEY = "down_dog_runtime_v1"


@dataclass
class DownDogRuntime:
    pose_history: deque = field(default_factory=lambda: deque(maxlen=DD_POSE_HISTORY_SIZE))
    score_history: deque = field(default_factory=lambda: deque(maxlen=DD_SCORE_HISTORY_SIZE))
    feedback_history: deque = field(default_factory=lambda: deque(maxlen=DD_FEEDBACK_HISTORY_SIZE))
    hip_center_history: deque = field(default_factory=lambda: deque(maxlen=DD_STABILITY_HISTORY_SIZE))
    shoulder_center_history: deque = field(default_factory=lambda: deque(maxlen=DD_STABILITY_HISTORY_SIZE))
    hip_height_history: deque = field(default_factory=lambda: deque(maxlen=DD_STABILITY_HISTORY_SIZE))
    spine_line_history: deque = field(default_factory=lambda: deque(maxlen=DD_STABILITY_HISTORY_SIZE))
    visibility_history: deque = field(default_factory=lambda: deque(maxlen=DD_VISIBILITY_HISTORY_SIZE))
    detection_history: deque = field(default_factory=lambda: deque(maxlen=DD_DETECTION_HISTORY_SIZE))
    point_history: dict = field(default_factory=dict)
    boolean_histories: dict = field(default_factory=dict)


def dd_runtime_to_session_data(runtime):
    return {
        "pose_history": list(runtime.pose_history),
        "score_history": list(runtime.score_history),
        "feedback_history": list(runtime.feedback_history),
        "hip_center_history": list(runtime.hip_center_history),
        "shoulder_center_history": list(runtime.shoulder_center_history),
        "hip_height_history": list(runtime.hip_height_history),
        "spine_line_history": list(runtime.spine_line_history),
        "visibility_history": list(runtime.visibility_history),
        "detection_history": list(runtime.detection_history),
        "point_history": {
            str(key): [[float(x), float(y), float(z)] for x, y, z in points]
            for key, points in runtime.point_history.items()
        },
        "boolean_histories": {
            str(key): [bool(value) for value in values]
            for key, values in runtime.boolean_histories.items()
        },
    }


def dd_runtime_from_session_data(data):
    runtime = DownDogRuntime()
    if not isinstance(data, dict):
        return runtime

    runtime.pose_history = deque(data.get("pose_history", []), maxlen=DD_POSE_HISTORY_SIZE)
    runtime.score_history = deque(data.get("score_history", []), maxlen=DD_SCORE_HISTORY_SIZE)
    runtime.feedback_history = deque(data.get("feedback_history", []), maxlen=DD_FEEDBACK_HISTORY_SIZE)
    runtime.hip_center_history = deque(data.get("hip_center_history", []), maxlen=DD_STABILITY_HISTORY_SIZE)
    runtime.shoulder_center_history = deque(data.get("shoulder_center_history", []), maxlen=DD_STABILITY_HISTORY_SIZE)
    runtime.hip_height_history = deque(data.get("hip_height_history", []), maxlen=DD_STABILITY_HISTORY_SIZE)
    runtime.spine_line_history = deque(data.get("spine_line_history", []), maxlen=DD_STABILITY_HISTORY_SIZE)
    runtime.visibility_history = deque(data.get("visibility_history", []), maxlen=DD_VISIBILITY_HISTORY_SIZE)
    runtime.detection_history = deque(data.get("detection_history", []), maxlen=DD_DETECTION_HISTORY_SIZE)
    runtime.point_history = {
        str(key): deque(
            [(float(x), float(y), float(z)) for x, y, z in values],
            maxlen=DD_POINT_HISTORY_SIZE,
        )
        for key, values in data.get("point_history", {}).items()
    }
    runtime.boolean_histories = {
        str(key): deque([bool(value) for value in values], maxlen=DD_BOOLEAN_HISTORY_SIZE)
        for key, values in data.get("boolean_histories", {}).items()
    }
    return runtime


# =========================================================
# MEDIAPIPE
# =========================================================
mp_pose = mp.solutions.pose
down_dog_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.52,
    min_tracking_confidence=0.52,
)


# =========================================================
# LANDMARK INDEXES
# =========================================================
DD_NOSE = 0
DD_LEFT_SHOULDER = 11
DD_RIGHT_SHOULDER = 12
DD_LEFT_ELBOW = 13
DD_RIGHT_ELBOW = 14
DD_LEFT_WRIST = 15
DD_RIGHT_WRIST = 16
DD_LEFT_HIP = 23
DD_RIGHT_HIP = 24
DD_LEFT_KNEE = 25
DD_RIGHT_KNEE = 26
DD_LEFT_ANKLE = 27
DD_RIGHT_ANKLE = 28
DD_LEFT_HEEL = 29
DD_RIGHT_HEEL = 30
DD_LEFT_FOOT_INDEX = 31
DD_RIGHT_FOOT_INDEX = 32

DD_SELECTED_POINTS = [
    DD_NOSE,
    DD_LEFT_SHOULDER, DD_RIGHT_SHOULDER,
    DD_LEFT_ELBOW, DD_RIGHT_ELBOW,
    DD_LEFT_WRIST, DD_RIGHT_WRIST,
    DD_LEFT_HIP, DD_RIGHT_HIP,
    DD_LEFT_KNEE, DD_RIGHT_KNEE,
    DD_LEFT_ANKLE, DD_RIGHT_ANKLE,
]

DD_EXTRA_POINTS = [
    DD_LEFT_HEEL, DD_RIGHT_HEEL,
    DD_LEFT_FOOT_INDEX, DD_RIGHT_FOOT_INDEX,
]

DD_POINT_NAME_MAP = {
    DD_NOSE: "nose",
    DD_LEFT_SHOULDER: "left_shoulder",
    DD_RIGHT_SHOULDER: "right_shoulder",
    DD_LEFT_ELBOW: "left_elbow",
    DD_RIGHT_ELBOW: "right_elbow",
    DD_LEFT_WRIST: "left_wrist",
    DD_RIGHT_WRIST: "right_wrist",
    DD_LEFT_HIP: "left_hip",
    DD_RIGHT_HIP: "right_hip",
    DD_LEFT_KNEE: "left_knee",
    DD_RIGHT_KNEE: "right_knee",
    DD_LEFT_ANKLE: "left_ankle",
    DD_RIGHT_ANKLE: "right_ankle",
    DD_LEFT_HEEL: "left_heel",
    DD_RIGHT_HEEL: "right_heel",
    DD_LEFT_FOOT_INDEX: "left_foot_index",
    DD_RIGHT_FOOT_INDEX: "right_foot_index",
}

DD_ALL_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
DD_ALL_LANDMARK_NAME_TO_INDEX = {name: idx for idx, name in enumerate(DD_ALL_LANDMARK_NAMES)}

DD_FEATURE_COLUMNS = []
for name in DD_ALL_LANDMARK_NAMES:
    DD_FEATURE_COLUMNS.extend([
        f"{name}_x",
        f"{name}_y",
        f"{name}_z",
        f"{name}_visibility",
    ])

DD_ANGLE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_hip_angle",
    "right_hip_angle",
]

DD_MODEL_COLUMNS = DD_FEATURE_COLUMNS + DD_ANGLE_COLUMNS


# =========================================================
# COLORS
# =========================================================
DD_GREEN = "#00ff66"
DD_RED = "#ff3b30"
DD_YELLOW = "#ffd60a"
DD_GRAY = "#cfcfcf"
DD_CYAN = "#40cfff"


# =========================================================
# MODEL LOAD
# =========================================================
DD_BASE_DIR = Path(settings.BASE_DIR)


def resolve_down_dog_model_path(filename: str) -> Path:
    candidates = [
        DD_BASE_DIR / "Ml_Models" / filename,
        DD_BASE_DIR / "Ml_models" / filename,
        DD_BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find model file: {filename}\nChecked:\n{checked}")


DOWNDOG_MODEL_PATH = resolve_down_dog_model_path("downdog_best_model.pkl")
DOWNDOG_SCALER_PATH = resolve_down_dog_model_path("downdog_best_scaler.pkl")

with open(DOWNDOG_MODEL_PATH, "rb") as f:
    downdog_model = pickle.load(f)

with open(DOWNDOG_SCALER_PATH, "rb") as f:
    downdog_scaler = pickle.load(f)


# =========================================================
# PAGE
# =========================================================
def down_dog_live_page(request):
    return render(request, "User/downdog_camera.html")


# =========================================================
# RESPONSE HELPERS
# =========================================================
def dd_api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})


def dd_api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


# =========================================================
# RUNTIME HELPERS
# =========================================================
def dd_get_runtime(request):
    if request.session.session_key is None:
        request.session.save()

    return dd_runtime_from_session_data(request.session.get(DD_SESSION_RUNTIME_KEY))


def dd_store_runtime(request, runtime):
    if request.session.session_key is None:
        request.session.save()
    request.session[DD_SESSION_RUNTIME_KEY] = dd_runtime_to_session_data(runtime)
    request.session.modified = True


# =========================================================
# TEXT HELPERS
# =========================================================
def dd_clean_text(text):
    return " ".join(str(text).strip().split())


def dd_normalize_key(text):
    return dd_clean_text(text).lower()


def dd_dedupe_list(items, max_items=None, exclude=None):
    exclude = exclude or []
    exclude_keys = {dd_normalize_key(x) for x in exclude if x}

    output = []
    seen = set()

    for item in items:
        if not item:
            continue

        text = dd_clean_text(item)
        key = dd_normalize_key(text)

        if not key or key in seen or key in exclude_keys:
            continue

        seen.add(key)
        output.append(text)

        if max_items and len(output) >= max_items:
            break

    return output


# =========================================================
# SMOOTHING
# =========================================================
def dd_smooth_label(history, new_label):
    history.append(str(new_label))
    return Counter(history).most_common(1)[0][0]


def dd_smooth_score(runtime, new_score):
    runtime.score_history.append(float(new_score))
    return int(round(sum(runtime.score_history) / len(runtime.score_history)))


def dd_smooth_feedback(runtime, new_feedback):
    runtime.feedback_history.append(str(new_feedback))
    return Counter(runtime.feedback_history).most_common(1)[0][0]


def dd_smooth_boolean(history, value):
    history.append(bool(value))
    true_count = sum(history)
    return true_count >= max(1, len(history) // 2 + 1)


def dd_smooth_runtime_boolean(runtime, key, value, maxlen=DD_BOOLEAN_HISTORY_SIZE):
    if key not in runtime.boolean_histories:
        runtime.boolean_histories[key] = deque(maxlen=maxlen)
    history = runtime.boolean_histories[key]
    history.append(bool(value))
    true_count = sum(history)

    if len(history) < 3:
        return true_count >= 1

    return true_count >= max(2, len(history) // 2 + 1)


def dd_smooth_point(runtime, key, x, y, z):
    if key not in runtime.point_history:
        runtime.point_history[key] = deque(maxlen=DD_POINT_HISTORY_SIZE)

    runtime.point_history[key].append((float(x), float(y), float(z)))
    xs = [p[0] for p in runtime.point_history[key]]
    ys = [p[1] for p in runtime.point_history[key]]
    zs = [p[2] for p in runtime.point_history[key]]

    return (
        float(sum(xs) / len(xs)),
        float(sum(ys) / len(ys)),
        float(sum(zs) / len(zs)),
    )


def dd_clear_point_history(runtime):
    runtime.point_history.clear()


def dd_reset_runtime_state(runtime):
    runtime.pose_history.clear()
    runtime.score_history.clear()
    runtime.feedback_history.clear()
    runtime.hip_center_history.clear()
    runtime.shoulder_center_history.clear()
    runtime.hip_height_history.clear()
    runtime.spine_line_history.clear()
    runtime.visibility_history.clear()
    runtime.detection_history.clear()
    runtime.boolean_histories.clear()
    dd_clear_point_history(runtime)


# =========================================================
# BASIC HELPERS
# =========================================================
def dd_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def dd_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.06, beta=7)


def dd_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = down_dog_pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark


def dd_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 60, brightness


def dd_clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def dd_moving_std(values):
    if len(values) < 3:
        return 0.0
    return float(np.std(list(values)))


def dd_calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def dd_distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def dd_angle_at_least(angle, threshold):
    return float(angle) >= float(threshold)


def dd_angle_at_most(angle, threshold):
    return float(angle) <= float(threshold)


# =========================================================
# FEATURES
# =========================================================
def dd_extract_raw_landmark_dict(landmarks):
    lm_dict = {}
    for name, idx in DD_ALL_LANDMARK_NAME_TO_INDEX.items():
        lm = landmarks[idx]
        lm_dict[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
    return lm_dict


def dd_normalize_landmarks_inplace(lm_dict):
    left_hip_x = lm_dict["left_hip"]["x"]
    left_hip_y = lm_dict["left_hip"]["y"]

    for name in lm_dict.keys():
        lm_dict[name]["x"] -= left_hip_x
        lm_dict[name]["y"] -= left_hip_y


def dd_build_feature_dataframe_from_landmarks(landmarks):
    lm_dict = dd_extract_raw_landmark_dict(landmarks)
    dd_normalize_landmarks_inplace(lm_dict)

    pts = {name: [vals["x"], vals["y"]] for name, vals in lm_dict.items()}

    angles = {
        "left_knee_angle": dd_calculate_angle(pts["left_hip"], pts["left_knee"], pts["left_ankle"]),
        "right_knee_angle": dd_calculate_angle(pts["right_hip"], pts["right_knee"], pts["right_ankle"]),
        "left_elbow_angle": dd_calculate_angle(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"]),
        "right_elbow_angle": dd_calculate_angle(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"]),
        "left_hip_angle": dd_calculate_angle(pts["left_shoulder"], pts["left_hip"], pts["left_knee"]),
        "right_hip_angle": dd_calculate_angle(pts["right_shoulder"], pts["right_hip"], pts["right_knee"]),
    }

    row = {}
    for name in DD_ALL_LANDMARK_NAMES:
        row[f"{name}_x"] = lm_dict[name]["x"]
        row[f"{name}_y"] = lm_dict[name]["y"]
        row[f"{name}_z"] = lm_dict[name]["z"]
        row[f"{name}_visibility"] = lm_dict[name]["visibility"]

    for key, value in angles.items():
        row[key] = value

    features_df = pd.DataFrame([row], columns=DD_MODEL_COLUMNS)
    return features_df, lm_dict, angles


def dd_predict_model_label(features_df):
    features_df = features_df.astype(np.float32).copy()
    expected_names = getattr(downdog_scaler, "feature_names_in_", None)
    expected_count = getattr(downdog_scaler, "n_features_in_", features_df.shape[1])

    X_for_scaler = None

    if expected_names is not None:
        expected_names_list = list(expected_names)
        expected_names_str = [str(x) for x in expected_names_list]
        current_names_str = [str(x) for x in features_df.columns]

        if set(expected_names_str).issubset(set(current_names_str)):
            rename_map = {str(col): col for col in features_df.columns}
            X_for_scaler = features_df[[rename_map[name] for name in expected_names_str]]

    if X_for_scaler is None:
        X_array = features_df.to_numpy(dtype=np.float32)
        if X_array.shape[1] != expected_count:
            raise ValueError(
                f"Feature count mismatch: scaler expects {expected_count}, "
                f"but runtime produced {X_array.shape[1]} features."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but StandardScaler was fitted with feature names"
            )
            scaled_features = downdog_scaler.transform(X_array)
    else:
        scaled_features = downdog_scaler.transform(X_for_scaler)

    scaled_features = np.asarray(scaled_features, dtype=np.float32)

    prediction = downdog_model.predict(scaled_features)[0]

    confidence = 0.50
    if hasattr(downdog_model, "predict_proba"):
        probs = downdog_model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probs))

    return str(prediction), confidence


# =========================================================
# VISIBILITY / SIDE SELECTION
# =========================================================
def dd_side_visibility(landmarks, side):
    if side == "left":
        idxs = [DD_LEFT_SHOULDER, DD_LEFT_ELBOW, DD_LEFT_WRIST, DD_LEFT_HIP, DD_LEFT_KNEE, DD_LEFT_ANKLE]
    else:
        idxs = [DD_RIGHT_SHOULDER, DD_RIGHT_ELBOW, DD_RIGHT_WRIST, DD_RIGHT_HIP, DD_RIGHT_KNEE, DD_RIGHT_ANKLE]

    vals = [float(landmarks[i].visibility) for i in idxs]
    return float(np.mean(vals))


def dd_pick_dominant_side(landmarks):
    left_vis = dd_side_visibility(landmarks, "left")
    right_vis = dd_side_visibility(landmarks, "right")
    if left_vis >= right_vis:
        return "left", left_vis, right_vis
    return "right", right_vis, left_vis


def dd_check_body_visibility(lm_dict):
    core_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    visibilities = [lm_dict[name]["visibility"] for name in core_names]
    visible_count = sum(v > 0.20 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    shoulders_ok = (
        lm_dict["left_shoulder"]["visibility"] > 0.28 or
        lm_dict["right_shoulder"]["visibility"] > 0.28
    )
    hips_ok = (
        lm_dict["left_hip"]["visibility"] > 0.30 or
        lm_dict["right_hip"]["visibility"] > 0.30
    )
    wrists_ok = (
        lm_dict["left_wrist"]["visibility"] > 0.22 or
        lm_dict["right_wrist"]["visibility"] > 0.22
    )
    ankles_ok = (
        lm_dict["left_ankle"]["visibility"] > 0.22 or
        lm_dict["right_ankle"]["visibility"] > 0.22
    )

    full_body_visible = visible_count >= 5 and shoulders_ok and hips_ok and (wrists_ok or ankles_ok)
    return full_body_visible, visible_count, avg_visibility


def dd_check_frame_position(raw_pts):
    xs = [float(p[0]) for p in raw_pts[DD_SELECTED_POINTS]]
    ys = [float(p[1]) for p in raw_pts[DD_SELECTED_POINTS]]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    feedback = []

    # CHANGED: Relaxed bounding boxes to prevent constant "Move Away" errors 
    # since Downward Dog requires extending across the entire camera view
    if width > 0.98:
        feedback.append("Move slightly away from the camera")
    if height > 0.99:
        feedback.append("Show a little more of your full body")
    if width < 0.34:
        feedback.append("Move a little closer so your body fills the frame better")

    center_x = (min_x + max_x) / 2
    if center_x < 0.18:
        feedback.append("Move slightly to the right")
    elif center_x > 0.82:
        feedback.append("Move slightly to the left")

    return feedback


# =========================================================
# ANALYSIS
# =========================================================
def analyze_down_dog_pose(raw_pts, landmarks):
    dominant_side, dominant_vis, support_vis = dd_pick_dominant_side(landmarks)
    support_visible = support_vis >= DD_SUPPORT_VISIBILITY_MIN
    dominant_visible = dominant_vis >= DD_DOMINANT_VISIBILITY_MIN

    if dominant_side == "left":
        s, e, w = raw_pts[DD_LEFT_SHOULDER], raw_pts[DD_LEFT_ELBOW], raw_pts[DD_LEFT_WRIST]
        h, k, a = raw_pts[DD_LEFT_HIP], raw_pts[DD_LEFT_KNEE], raw_pts[DD_LEFT_ANKLE]
        os, oe, ow = raw_pts[DD_RIGHT_SHOULDER], raw_pts[DD_RIGHT_ELBOW], raw_pts[DD_RIGHT_WRIST]
        oh, ok, oa = raw_pts[DD_RIGHT_HIP], raw_pts[DD_RIGHT_KNEE], raw_pts[DD_RIGHT_ANKLE]
        heel = raw_pts[DD_LEFT_HEEL]
        other_heel = raw_pts[DD_RIGHT_HEEL]
        side_name = "Left"
    else:
        s, e, w = raw_pts[DD_RIGHT_SHOULDER], raw_pts[DD_RIGHT_ELBOW], raw_pts[DD_RIGHT_WRIST]
        h, k, a = raw_pts[DD_RIGHT_HIP], raw_pts[DD_RIGHT_KNEE], raw_pts[DD_RIGHT_ANKLE]
        os, oe, ow = raw_pts[DD_LEFT_SHOULDER], raw_pts[DD_LEFT_ELBOW], raw_pts[DD_LEFT_WRIST]
        oh, ok, oa = raw_pts[DD_LEFT_HIP], raw_pts[DD_LEFT_KNEE], raw_pts[DD_LEFT_ANKLE]
        heel = raw_pts[DD_RIGHT_HEEL]
        other_heel = raw_pts[DD_LEFT_HEEL]
        side_name = "Right"

    ls, rs = raw_pts[DD_LEFT_SHOULDER], raw_pts[DD_RIGHT_SHOULDER]
    lh, rh = raw_pts[DD_LEFT_HIP], raw_pts[DD_RIGHT_HIP]
    lw, rw = raw_pts[DD_LEFT_WRIST], raw_pts[DD_RIGHT_WRIST]
    la, ra = raw_pts[DD_LEFT_ANKLE], raw_pts[DD_RIGHT_ANKLE]
    nose = raw_pts[DD_NOSE]

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    wrist_center = (lw + rw) / 2.0
    ankle_center = (la + ra) / 2.0

    torso_size = dd_distance(shoulder_center[:2], hip_center[:2]) + 1e-6
    base_length = dd_distance(wrist_center[:2], ankle_center[:2]) / torso_size

    dom_elbow_angle = dd_calculate_angle(s[:2], e[:2], w[:2])
    dom_knee_angle = dd_calculate_angle(h[:2], k[:2], a[:2])
    dom_shoulder_open = dd_calculate_angle(e[:2], s[:2], h[:2])
    dom_hip_fold = dd_calculate_angle(s[:2], h[:2], k[:2])

    sup_elbow_angle = dd_calculate_angle(os[:2], oe[:2], ow[:2])
    sup_knee_angle = dd_calculate_angle(oh[:2], ok[:2], oa[:2])

    arm_line_length = dd_distance(w[:2], h[:2]) / torso_size
    leg_line_length = dd_distance(h[:2], a[:2]) / torso_size
    shoulder_to_wrist = dd_distance(s[:2], w[:2]) / torso_size
    hip_to_ankle = dd_distance(h[:2], a[:2]) / torso_size

    # Down Dog often comes from a side or diagonal webcam angle, so these gates
    # allow small perspective errors while still requiring a clear inverted V.
    hips_above_shoulders = hip_center[1] <= shoulder_center[1] + (0.08 * torso_size)
    hips_high_enough = (wrist_center[1] - hip_center[1]) / torso_size > 0.24
    hips_peak_good = (ankle_center[1] - hip_center[1]) / torso_size > 0.20

    spine_long = base_length > 1.18
    side_profile_clear = arm_line_length > 0.82 and leg_line_length > 0.88
    arm_length_ok = shoulder_to_wrist > 0.78
    leg_length_ok = hip_to_ankle > 0.82

    dominant_arm_straight = dd_angle_at_least(dom_elbow_angle, DD_DOMINANT_ELBOW_STRAIGHT_ANGLE)
    support_arm_straight = dd_angle_at_least(sup_elbow_angle, DD_SUPPORT_ELBOW_STRAIGHT_ANGLE) if support_visible else True
    dominant_leg_straight = dd_angle_at_least(dom_knee_angle, DD_DOMINANT_KNEE_STRAIGHT_ANGLE)
    dominant_leg_strict = dd_angle_at_least(dom_knee_angle, DD_DOMINANT_KNEE_STRONG_ANGLE)
    support_leg_straight = dd_angle_at_least(sup_knee_angle, DD_SUPPORT_KNEE_STRAIGHT_ANGLE) if support_visible else True

    shoulder_open_ok = dd_angle_at_least(dom_shoulder_open, DD_SHOULDER_OPEN_MIN_ANGLE)
    hip_fold_ok = dd_angle_at_most(dom_hip_fold, DD_HIP_FOLD_MAX_ANGLE)

    hands_width_ratio = dd_distance(lw[:2], rw[:2]) / (dd_distance(ls[:2], rs[:2]) + 1e-6)
    feet_width_ratio = dd_distance(la[:2], ra[:2]) / (dd_distance(lh[:2], rh[:2]) + 1e-6)

    hands_width_ok = 0.72 <= hands_width_ratio <= 1.75
    feet_width_ok = 0.62 <= feet_width_ratio <= 1.75

    head_between_arms = nose[1] > shoulder_center[1] - (0.10 * torso_size)

    shoulder_symmetry = abs(float(ls[1] - rs[1])) / torso_size < 0.18
    hip_symmetry = abs(float(lh[1] - rh[1])) / torso_size < 0.18
    balanced_shape = shoulder_symmetry and hip_symmetry

    heel_lift = abs(float(heel[1] - a[1])) / torso_size
    heel_lift_other = abs(float(other_heel[1] - oa[1])) / torso_size
    heels_reasonable = heel_lift < 0.24
    both_heels_reasonable = heel_lift < 0.24 and heel_lift_other < 0.28

    core_shape_count = sum([
        hips_above_shoulders,
        hips_high_enough,
        dominant_arm_straight,
        dominant_leg_straight,
        shoulder_open_ok,
        hip_fold_ok,
        spine_long,
    ])
    detail_shape_count = sum([
        head_between_arms,
        side_profile_clear,
        arm_length_ok,
        leg_length_ok,
        hands_width_ok,
        feet_width_ok,
    ])
    core_shape_ready = core_shape_count >= DD_CORE_SHAPE_READY_MIN
    detail_shape_ready = detail_shape_count >= DD_DETAIL_SHAPE_READY_MIN
    soft_down_dog_gate = (
        hips_above_shoulders and
        hips_high_enough and
        hip_fold_ok and
        core_shape_ready and
        detail_shape_ready
    )

    dominant_side_gate = (
        hips_above_shoulders and
        hips_high_enough and
        dominant_arm_straight and
        hip_fold_ok and
        core_shape_count >= 5 and
        detail_shape_count >= 3
    )

    both_side_bonus_gate = (
        support_visible and
        support_arm_straight and
        support_leg_straight and
        balanced_shape and
        hands_width_ok and
        feet_width_ok and
        both_heels_reasonable
    )

    strict_down_dog_gate = (
        dominant_side_gate and
        dominant_leg_straight and
        shoulder_open_ok and
        spine_long and
        (support_vis < DD_SUPPORT_VISIBILITY_MIN or both_side_bonus_gate)
    )

    checks = {
        "dominant_visible": dominant_visible,
        "support_visible": support_visible,
        "dominant_arm_straight": dominant_arm_straight,
        "dominant_leg_strict": dominant_leg_strict,
        "dominant_leg_straight": dominant_leg_straight,
        "support_arm_straight": support_arm_straight,
        "support_leg_straight": support_leg_straight,
        "hips_above_shoulders": hips_above_shoulders,
        "hips_high": hips_high_enough,
        "hips_peak": hips_peak_good,
        "head_between_arms": head_between_arms,
        "spine_long": spine_long,
        "shoulder_open": shoulder_open_ok,
        "hip_fold_ok": hip_fold_ok,
        "side_profile_clear": side_profile_clear,
        "arm_length_ok": arm_length_ok,
        "leg_length_ok": leg_length_ok,
        "core_shape_count": core_shape_count,
        "detail_shape_count": detail_shape_count,
        "core_shape_ready": core_shape_ready,
        "detail_shape_ready": detail_shape_ready,
        "soft_down_dog_gate": soft_down_dog_gate,
        "balanced_shape": balanced_shape,
        "hands_width_ok": hands_width_ok,
        "feet_width_ok": feet_width_ok,
        "heels_reasonable": heels_reasonable,
        "both_heels_reasonable": both_heels_reasonable,
        "dominant_side_gate": dominant_side_gate,
        "both_side_bonus_gate": both_side_bonus_gate,
        "strict_down_dog_gate": strict_down_dog_gate,
        "left_elbow_ok": dominant_arm_straight if dominant_side == "left" else support_arm_straight,
        "right_elbow_ok": dominant_arm_straight if dominant_side == "right" else support_arm_straight,
        "left_knee_ok": dominant_leg_straight if dominant_side == "left" else support_leg_straight,
        "right_knee_ok": dominant_leg_straight if dominant_side == "right" else support_leg_straight,
    }

    return {
        "dominant_side": dominant_side,
        "dominant_side_name": side_name,
        "dominant_vis": round(float(dominant_vis), 3),
        "support_vis": round(float(support_vis), 3),
        "angles": {
            "dominant_elbow_angle": round(float(dom_elbow_angle), 1),
            "dominant_knee_angle": round(float(dom_knee_angle), 1),
            "dominant_shoulder_open_angle": round(float(dom_shoulder_open), 1),
            "dominant_hip_fold_angle": round(float(dom_hip_fold), 1),
            "support_elbow_angle": round(float(sup_elbow_angle), 1),
            "support_knee_angle": round(float(sup_knee_angle), 1),
            "left_elbow_angle": round(dd_calculate_angle(ls[:2], raw_pts[DD_LEFT_ELBOW][:2], lw[:2]), 1),
            "right_elbow_angle": round(dd_calculate_angle(rs[:2], raw_pts[DD_RIGHT_ELBOW][:2], rw[:2]), 1),
            "left_knee_angle": round(dd_calculate_angle(lh[:2], raw_pts[DD_LEFT_KNEE][:2], la[:2]), 1),
            "right_knee_angle": round(dd_calculate_angle(rh[:2], raw_pts[DD_RIGHT_KNEE][:2], ra[:2]), 1),
        },
        "measures": {
            "hips_height_ratio": round(float((wrist_center[1] - hip_center[1]) / torso_size), 3),
            "hips_peak_ratio": round(float((ankle_center[1] - hip_center[1]) / torso_size), 3),
            "hands_width_ratio": round(float(hands_width_ratio), 3),
            "feet_width_ratio": round(float(feet_width_ratio), 3),
            "core_shape_count": int(core_shape_count),
            "detail_shape_count": int(detail_shape_count),
        },
        "checks": checks,
    }

def dd_build_joint_states(runtime, analysis):
    checks = analysis["checks"]
    angles = analysis["angles"]
    measures = analysis["measures"]

    smoothed_checks = {
        "dominant_visible": dd_smooth_runtime_boolean(runtime, "dominant_visible", checks["dominant_visible"]),
        "support_visible": dd_smooth_runtime_boolean(runtime, "support_visible", checks["support_visible"]),
        "dominant_arm_straight": dd_smooth_runtime_boolean(runtime, "dominant_arm_straight", checks["dominant_arm_straight"]),
        "support_arm_straight": dd_smooth_runtime_boolean(runtime, "support_arm_straight", checks["support_arm_straight"]),
        "dominant_leg_straight": dd_smooth_runtime_boolean(runtime, "dominant_leg_straight", checks["dominant_leg_straight"]),
        "dominant_leg_strict": dd_smooth_runtime_boolean(runtime, "dominant_leg_strict", checks["dominant_leg_strict"]),
        "support_leg_straight": dd_smooth_runtime_boolean(runtime, "support_leg_straight", checks["support_leg_straight"]),
        "hips_above_shoulders": dd_smooth_runtime_boolean(runtime, "hips_above_shoulders", checks["hips_above_shoulders"]),
        "hips_high": dd_smooth_runtime_boolean(runtime, "hips_high", checks["hips_high"]),
        "hips_peak": dd_smooth_runtime_boolean(runtime, "hips_peak", checks["hips_peak"]),
        "head_between_arms": dd_smooth_runtime_boolean(runtime, "head_between_arms", checks["head_between_arms"]),
        "spine_long": dd_smooth_runtime_boolean(runtime, "spine_long", checks["spine_long"]),
        "shoulder_open": dd_smooth_runtime_boolean(runtime, "shoulder_open", checks["shoulder_open"]),
        "hip_fold_ok": dd_smooth_runtime_boolean(runtime, "hip_fold_ok", checks["hip_fold_ok"]),
        "side_profile_clear": dd_smooth_runtime_boolean(runtime, "side_profile_clear", checks["side_profile_clear"]),
        "arm_length_ok": dd_smooth_runtime_boolean(runtime, "arm_length_ok", checks["arm_length_ok"]),
        "leg_length_ok": dd_smooth_runtime_boolean(runtime, "leg_length_ok", checks["leg_length_ok"]),
        "core_shape_ready": dd_smooth_runtime_boolean(runtime, "core_shape_ready", checks["core_shape_ready"]),
        "detail_shape_ready": dd_smooth_runtime_boolean(runtime, "detail_shape_ready", checks["detail_shape_ready"]),
        "soft_down_dog_gate": dd_smooth_runtime_boolean(runtime, "soft_down_dog_gate", checks["soft_down_dog_gate"]),
        "balanced_shape": dd_smooth_runtime_boolean(runtime, "balanced_shape", checks["balanced_shape"]),
        "hands_width_ok": dd_smooth_runtime_boolean(runtime, "hands_width_ok", checks["hands_width_ok"]),
        "feet_width_ok": dd_smooth_runtime_boolean(runtime, "feet_width_ok", checks["feet_width_ok"]),
        "heels_reasonable": dd_smooth_runtime_boolean(runtime, "heels_reasonable", checks["heels_reasonable"]),
        "both_heels_reasonable": dd_smooth_runtime_boolean(runtime, "both_heels_reasonable", checks["both_heels_reasonable"]),
        "dominant_side_gate": dd_smooth_runtime_boolean(runtime, "dominant_side_gate", checks["dominant_side_gate"]),
        "both_side_bonus_gate": dd_smooth_runtime_boolean(runtime, "both_side_bonus_gate", checks["both_side_bonus_gate"]),
        "strict_down_dog_gate": dd_smooth_runtime_boolean(runtime, "strict_down_dog_gate", checks["strict_down_dog_gate"]),
        "left_elbow_ok": dd_smooth_runtime_boolean(runtime, "left_elbow_ok", checks["left_elbow_ok"]),
        "right_elbow_ok": dd_smooth_runtime_boolean(runtime, "right_elbow_ok", checks["right_elbow_ok"]),
        "left_knee_ok": dd_smooth_runtime_boolean(runtime, "left_knee_ok", checks["left_knee_ok"]),
        "right_knee_ok": dd_smooth_runtime_boolean(runtime, "right_knee_ok", checks["right_knee_ok"]),
    }

    joint_states = {
        "dominant_side": analysis["dominant_side"],
        "dominant_visible": {
            "ok": smoothed_checks["dominant_visible"],
            "value": analysis["dominant_vis"],
        },
        "support_visible": {
            "ok": smoothed_checks["support_visible"],
            "value": analysis["support_vis"],
        },
        "left_elbow": {
            "ok": smoothed_checks["left_elbow_ok"],
            "angle": angles["left_elbow_angle"],
            "threshold": DD_DOMINANT_ELBOW_STRAIGHT_ANGLE if analysis["dominant_side"] == "left" else DD_SUPPORT_ELBOW_STRAIGHT_ANGLE,
        },
        "right_elbow": {
            "ok": smoothed_checks["right_elbow_ok"],
            "angle": angles["right_elbow_angle"],
            "threshold": DD_DOMINANT_ELBOW_STRAIGHT_ANGLE if analysis["dominant_side"] == "right" else DD_SUPPORT_ELBOW_STRAIGHT_ANGLE,
        },
        "left_knee": {
            "ok": smoothed_checks["left_knee_ok"],
            "angle": angles["left_knee_angle"],
            "threshold": DD_DOMINANT_KNEE_STRAIGHT_ANGLE if analysis["dominant_side"] == "left" else DD_SUPPORT_KNEE_STRAIGHT_ANGLE,
        },
        "right_knee": {
            "ok": smoothed_checks["right_knee_ok"],
            "angle": angles["right_knee_angle"],
            "threshold": DD_DOMINANT_KNEE_STRAIGHT_ANGLE if analysis["dominant_side"] == "right" else DD_SUPPORT_KNEE_STRAIGHT_ANGLE,
        },
        "hips_high": {
            "ok": smoothed_checks["hips_high"],
            "value": measures["hips_height_ratio"],
        },
        "spine_long": {
            "ok": smoothed_checks["spine_long"],
        },
        "shoulders_open": {
            "ok": smoothed_checks["shoulder_open"],
            "angle": angles["dominant_shoulder_open_angle"],
            "threshold": DD_SHOULDER_OPEN_MIN_ANGLE,
            "side": analysis["dominant_side_name"],
        },
        "hip_fold": {
            "ok": smoothed_checks["hip_fold_ok"],
            "angle": angles["dominant_hip_fold_angle"],
            "threshold": DD_HIP_FOLD_MAX_ANGLE,
            "side": analysis["dominant_side_name"],
        },
        "head_between_arms": {
            "ok": smoothed_checks["head_between_arms"],
        },
        "hands_width": {
            "ok": smoothed_checks["hands_width_ok"],
            "value": measures["hands_width_ratio"],
        },
        "feet_width": {
            "ok": smoothed_checks["feet_width_ok"],
            "value": measures["feet_width_ratio"],
        },
    }

    return joint_states, smoothed_checks


def dd_score_down_dog_pose(analysis, checks):
    score = 0
    status = "warning"
    pose_label = "Not Down Dog"
    feedback = "Move into Downward Dog."
    coach_text = "Press the floor away and lift the hips up."
    tips = ["Place both hands and feet firmly on the floor."]

    if not checks["hips_above_shoulders"]:
        score = 35
        feedback = "Lift your hips higher."
        coach_text = "Lift your hips higher."
        tips = ["Push the floor away and send the hips up."]
    elif not checks["hips_high"]:
        score = 48
        feedback = "Lift your hips more to build the inverted V."
        coach_text = "Lift your hips more."
        tips = ["Press strongly through your hands and feet."]
    elif not (checks["dominant_arm_straight"] and checks["support_arm_straight"]):
        score = 65
        feedback = "Straighten your elbows."
        coach_text = "Straighten your elbows."
        tips = ["Press firmly through the hands.", "Lengthen through both arms."]
    elif not checks["dominant_leg_straight"]:
        score = 72
        feedback = "Lengthen your main visible leg more."
        coach_text = "Lengthen your visible leg more."
        tips = ["Straighten the knee as much as comfortable."]
    elif not checks["shoulder_open"]:
        score = 78
        feedback = "Push back through your hands more."
        coach_text = "Open the shoulders more."
        tips = ["Reach your chest back toward your thighs.", "Open the shoulders."]
    elif not checks["hip_fold_ok"]:
        score = 82
        feedback = "Fold more from the hips."
        coach_text = "Fold more from the hips."
        tips = ["Lift the hips and draw the ribs back."]
    elif not checks["spine_long"]:
        score = 86
        feedback = "Lengthen your spine more."
        coach_text = "Lengthen your spine more."
        tips = ["Reach the hips up and the chest back."]
    elif not checks["head_between_arms"]:
        score = 88
        feedback = "Relax your head more between the arms."
        coach_text = "Relax your head between the arms."
        tips = ["Let the neck stay soft and natural.", "Look toward your toes."]
    elif not checks["hands_width_ok"]:
        score = 90
        feedback = "Adjust your hands slightly."
        coach_text = "Adjust your hands slightly."
        tips = ["Keep the hands around shoulder width."]
    elif not checks["feet_width_ok"]:
        score = 90
        feedback = "Adjust your feet slightly."
        coach_text = "Adjust your feet slightly."
        tips = ["Keep the feet around hip width."]
    elif checks["strict_down_dog_gate"] and checks["dominant_leg_strict"] and checks["hips_peak"]:
        if checks["support_visible"] and checks["both_side_bonus_gate"] and checks["dominant_visible"]:
            score = 100
            status = "perfect"
            pose_label = "Correct Down Dog"
            feedback = "Beautiful Downward Dog. Hold steady."
            coach_text = "Keep the hips high, the spine long, and the breath steady."
            tips = ["Excellent posture.", "Keep breathing steadily.", "Maintain the inverted V shape."]
        else:
            score = 96
            status = "good"
            pose_label = "Down Dog"
            feedback = "Strong Downward Dog. Hold steady."
            coach_text = "Keep pressing the floor away and stay long through the spine."
            tips = ["Very good side alignment.", "Keep the shape steady."]
    else:
        score = 92
        status = "good"
        pose_label = "Down Dog"
        feedback = "Good Downward Dog. Refine the shape and hold steady."
        coach_text = "Keep the hips lifted and smooth out the main weak point."
        tips = ["Keep the hips high and the spine long."]

    return {
        "score": score,
        "status": status,
        "pose_label": pose_label,
        "feedback": feedback,
        "coach_text": coach_text,
        "tips": tips,
    }


def dd_pose_flags(checks, stable_score, pose_status):
    pose_ready = bool(
        checks.get("dominant_side_gate") or
        checks.get("soft_down_dog_gate") or
        (
            checks.get("hips_above_shoulders") and
            checks.get("hips_high") and
            checks.get("hip_fold_ok") and
            stable_score >= DD_POSE_READY_SCORE
        )
    )
    good_pose_ready = bool(
        pose_ready and
        stable_score >= 78 and
        (checks.get("soft_down_dog_gate") or checks.get("dominant_side_gate")) and
        str(pose_status).lower() in {"good", "perfect"}
    )
    hold_ready = bool(
        good_pose_ready and
        checks.get("dominant_side_gate") and
        checks.get("spine_long") and
        checks.get("hips_high") and
        stable_score >= DD_HOLD_READY_SCORE
    )

    return {
        "pose_ready": pose_ready,
        "good_pose_ready": good_pose_ready,
        "hold_ready": hold_ready,
    }


def dd_is_downdog_like(model_label, model_confidence, checks):
    label = str(model_label).lower()

    if checks.get("dominant_side_gate", False):
        return True

    if checks.get("soft_down_dog_gate", False):
        return True

    if ("down" in label or "dog" in label) and model_confidence >= 0.58:
        return True

    if (
        checks.get("hips_above_shoulders") and
        checks.get("hips_high") and
        checks.get("hip_fold_ok") and
        checks.get("core_shape_ready") and
        checks.get("detail_shape_ready")
    ):
        return True

    return False


# =========================================================
# STABILITY / HOLD / QUALITY
# =========================================================
def dd_update_stability_metrics(runtime, raw_pts):
    shoulder_center = (raw_pts[DD_LEFT_SHOULDER] + raw_pts[DD_RIGHT_SHOULDER]) / 2.0
    hip_center = (raw_pts[DD_LEFT_HIP] + raw_pts[DD_RIGHT_HIP]) / 2.0

    runtime.hip_center_history.append(float(hip_center[0]))
    runtime.shoulder_center_history.append(float(shoulder_center[0]))
    runtime.hip_height_history.append(float(hip_center[1]))
    runtime.spine_line_history.append(abs(float(shoulder_center[0] - hip_center[0])))


def dd_get_stability_feedback(runtime):
    hip_shift = dd_moving_std(runtime.hip_center_history)
    shoulder_shift = dd_moving_std(runtime.shoulder_center_history)
    hip_height_wobble = dd_moving_std(runtime.hip_height_history)
    spine_wobble = dd_moving_std(runtime.spine_line_history)

    feedback = []
    penalty = 0

    if hip_shift > 0.017:
        feedback.append("Keep your hips steadier")
        penalty += 4
    if shoulder_shift > 0.017:
        feedback.append("Stabilize your shoulders")
        penalty += 3
    if hip_height_wobble > 0.014:
        feedback.append("Try not to let the hips drop")
        penalty += 5
    if spine_wobble > 0.012:
        feedback.append("Keep the spine line more steady")
        penalty += 3

    return feedback, penalty


def dd_quality_from_score(score):
    if score >= 98:
        return "Perfect_DownDog"
    if score >= 88:
        return "Good_DownDog"
    if score >= 70:
        return "Needs_Correction"
    return "Not_Ready"


def dd_build_points_for_frontend(raw_pts, landmarks, analysis):
    dominant_side = analysis.get("dominant_side", "left")
    dominant_idxs = (
        [DD_LEFT_SHOULDER, DD_LEFT_ELBOW, DD_LEFT_WRIST, DD_LEFT_HIP, DD_LEFT_KNEE, DD_LEFT_ANKLE]
        if dominant_side == "left" else
        [DD_RIGHT_SHOULDER, DD_RIGHT_ELBOW, DD_RIGHT_WRIST, DD_RIGHT_HIP, DD_RIGHT_KNEE, DD_RIGHT_ANKLE]
    )

    points = []
    for idx in DD_SELECTED_POINTS + DD_EXTRA_POINTS:
        visibility = float(landmarks[idx].visibility)
        if visibility < DD_POINT_VISIBILITY_MIN:
            continue

        is_dominant = idx in dominant_idxs

        points.append({
            "name": DD_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": DD_YELLOW,
            "radius": 7 if is_dominant else 6,
            "visible": True,
            "visibility": round(visibility, 3),
            "role": "dominant" if is_dominant else "support",
        })
    return points


def dd_build_angle_texts(raw_pts, landmarks, analysis):
    dominant_side = analysis.get("dominant_side", "left")
    support_visible = analysis.get("support_vis", 0.0) >= DD_SUPPORT_VISIBILITY_MIN
    items = []

    if dominant_side == "left":
        primary = [(DD_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), DD_YELLOW),
                   (DD_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), DD_YELLOW)]
        secondary = [(DD_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), DD_CYAN),
                     (DD_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), DD_CYAN)]
    else:
        primary = [(DD_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), DD_YELLOW),
                   (DD_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), DD_YELLOW)]
        secondary = [(DD_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), DD_CYAN),
                     (DD_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), DD_CYAN)]

    for idx, value, color in primary + (secondary if support_visible else []):
        if float(landmarks[idx].visibility) < 0.30:
            continue
        items.append({
            "text": f"{int(round(float(value)))}°",
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
        })
    return items


def dd_build_angle_texts_v2(raw_pts, landmarks, analysis):
    dominant_side = analysis.get("dominant_side", "left")
    support_visible = analysis.get("support_vis", 0.0) >= DD_SUPPORT_VISIBILITY_MIN
    items = []

    if dominant_side == "left":
        primary = [
            (DD_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), DD_YELLOW, "left_elbow"),
            (DD_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), DD_YELLOW, "left_knee"),
        ]
        secondary = [
            (DD_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), DD_CYAN, "right_elbow"),
            (DD_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), DD_CYAN, "right_knee"),
        ]
    else:
        primary = [
            (DD_RIGHT_ELBOW, analysis["angles"].get("right_elbow_angle", 0), DD_YELLOW, "right_elbow"),
            (DD_RIGHT_KNEE, analysis["angles"].get("right_knee_angle", 0), DD_YELLOW, "right_knee"),
        ]
        secondary = [
            (DD_LEFT_ELBOW, analysis["angles"].get("left_elbow_angle", 0), DD_CYAN, "left_elbow"),
            (DD_LEFT_KNEE, analysis["angles"].get("left_knee_angle", 0), DD_CYAN, "left_knee"),
        ]

    for idx, value, color, joint_key in primary + (secondary if support_visible else []):
        if float(landmarks[idx].visibility) < 0.30:
            continue
        items.append({
            "text": f"{int(round(float(value)))}{DD_DEGREE_SIGN}",
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
            "joint_key": joint_key,
        })
    return items


dd_build_angle_texts = dd_build_angle_texts_v2


def dd_pose_success(**kwargs):
    payload = {
        "pose": "Unknown",
        "model_pose": "Unknown",
        "quality": "N/A",
        "feedback": "Waiting for pose.",
        "coach_text": "Press the floor away and lift the hips up.",
        "status": "warning",
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
    return dd_api_success(**payload)


# =========================================================
# MAIN PROCESS
# =========================================================
def process_down_dog_request(request):
    runtime = None
    try:
        runtime = dd_get_runtime(request)

        if request.POST.get("reset") == "true":
            dd_reset_runtime_state(runtime)

        uploaded_file = request.FILES["image"]
        frame = dd_read_uploaded_image(uploaded_file)

        if frame is None:
            return dd_api_error("Invalid image file", status=400)

        frame = dd_enhance_frame(frame)

        low_light, brightness = dd_check_lighting(frame)
        if low_light:
            dd_reset_runtime_state(runtime)
            return dd_pose_success(
                pose="Low Light",
                feedback="Room lighting is too low for accurate pose detection.",
                coach_text="Improve the lighting and try again.",
                status="warning",
                details=["Increase room lighting", "Face the light source", "Avoid a dark background"],
            )

        landmarks = dd_detect_landmarks(frame)
        has_landmarks = landmarks is not None
        stable_has_landmarks = dd_smooth_boolean(runtime.detection_history, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            dd_reset_runtime_state(runtime)
            return dd_pose_success(
                pose="Unknown",
                feedback="No human pose detected.",
                coach_text="Show your full body clearly in the camera.",
                status="unknown",
                details=["Show your full body", "Stand where the camera can see you clearly", "Move slightly back"],
            )

        if landmarks is None:
            dd_clear_point_history(runtime)
            return dd_pose_success(
                pose="Tracking...",
                feedback="Hold still while detection stabilizes.",
                coach_text="Keep the pose steady.",
                details=["Hold still", "Keep your full body in frame"],
            )

        features_df, lm_dict, _ = dd_build_feature_dataframe_from_landmarks(landmarks)

        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = raw_pts.copy()

        for idx in DD_SELECTED_POINTS + DD_EXTRA_POINTS:
            sx, sy, sz = dd_smooth_point(runtime, f"dd_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])
            smoothed_pts[idx] = [sx, sy, sz]

        full_body_visible, visible_count, avg_visibility = dd_check_body_visibility(lm_dict)
        stable_full_body_visible = dd_smooth_boolean(runtime.visibility_history, full_body_visible)
        framing_feedback = dd_check_frame_position(smoothed_pts)

        if not full_body_visible and not stable_full_body_visible:
            dd_reset_runtime_state(runtime)
            details = [
                "Show more of both hands and feet in frame",
                "Keep shoulders, hips, knees, and ankles visible",
                "Move only a little if needed",
            ]
            details.extend(framing_feedback)

            return dd_pose_success(
                pose="Body Not Visible",
                feedback="Adjust your position so the full Down Dog shape is visible.",
                coach_text="Show your whole Down Dog shape in the frame.",
                details=dd_dedupe_list(details, max_items=3),
            )

        raw_model_label, confidence = dd_predict_model_label(features_df)
        stable_model_label = dd_smooth_label(runtime.pose_history, raw_model_label)

        analysis = analyze_down_dog_pose(smoothed_pts, landmarks)
        joint_states, smoothed_checks = dd_build_joint_states(runtime, analysis)
        points = dd_build_points_for_frontend(smoothed_pts, landmarks, analysis)
        angle_texts = dd_build_angle_texts(smoothed_pts, landmarks, analysis)

        dd_update_stability_metrics(runtime, smoothed_pts)
        stability_tips, stability_penalty = dd_get_stability_feedback(runtime)

        pose_eval = dd_score_down_dog_pose(analysis, smoothed_checks)
        combined_score = max(0, pose_eval["score"] - stability_penalty)
        stable_score = dd_smooth_score(runtime, combined_score)
        is_downdog = dd_is_downdog_like(stable_model_label, confidence, smoothed_checks)
        pose_flags = dd_pose_flags(smoothed_checks, stable_score, pose_eval["status"])

        if not pose_flags["pose_ready"] and not is_downdog and stable_score < 75:
            tips = []
            tips.extend(pose_eval["tips"])
            tips.extend([
                "Lift the hips high",
                "Straighten the arms and legs",
                "Make a clean inverted V shape",
            ])
            tips.extend(framing_feedback)

            return dd_pose_success(
                pose="Not Down Dog",
                model_pose=stable_model_label,
                quality="Not_Ready",
                feedback=pose_eval["feedback"],
                coach_text=pose_eval["coach_text"],
                status="warning",
                confidence=round(float(confidence), 3),
                score=max(0, min(65, stable_score)),
                angles=analysis["angles"],
                details=dd_dedupe_list(tips, max_items=3),
                points=points,
                angle_texts=angle_texts,
                joint_states=joint_states,
                pose_ready=False,
                hold_ready=False,
            )

        if pose_flags["hold_ready"] and pose_eval["status"] == "perfect" and stable_score >= 95:
            pose_name = "Correct Down Dog"
            status = "perfect"
            feedback_text = pose_eval["feedback"]
            coach_text = pose_eval["coach_text"]
        elif pose_flags["good_pose_ready"]:
            pose_name = "Down Dog"
            status = "good"
            feedback_text = pose_eval["feedback"]
            coach_text = pose_eval["coach_text"]
        else:
            pose_name = "Down Dog Needs Correction"
            status = "warning"
            feedback_text = pose_eval["feedback"]
            coach_text = "Fix the main weak point and keep the hips lifting."

        stable_feedback = dd_smooth_feedback(runtime, feedback_text)

        tips = []
        tips.extend(pose_eval["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        return dd_pose_success(
            pose=pose_name,
            model_pose=stable_model_label,
            quality=dd_quality_from_score(stable_score),
            feedback=stable_feedback,
            coach_text=coach_text,
            status=status,
            confidence=round(float(confidence), 3),
            score=stable_score,
            angles=analysis["angles"],
            details=dd_dedupe_list(tips, max_items=3, exclude=[stable_feedback, coach_text]),
            points=points,
            angle_texts=angle_texts,
            joint_states=joint_states,
            pose_ready=pose_flags["pose_ready"] or is_downdog,
            hold_ready=pose_flags["hold_ready"],
        )

    except Exception as e:
        import traceback
        print("process_down_dog_request error:", str(e))
        traceback.print_exc()
        return JsonResponse({ 
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }, status=500)
    finally:
        if runtime is not None:
            dd_store_runtime(request, runtime)
