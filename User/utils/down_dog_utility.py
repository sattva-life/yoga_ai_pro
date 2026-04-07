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
DD_POSE_HISTORY = deque(maxlen=10)
DD_SCORE_HISTORY = deque(maxlen=8)
DD_FEEDBACK_HISTORY = deque(maxlen=8)

DD_HIP_CENTER_HISTORY = deque(maxlen=20)
DD_SHOULDER_CENTER_HISTORY = deque(maxlen=20)
DD_HIP_HEIGHT_HISTORY = deque(maxlen=20)
DD_SPINE_LINE_HISTORY = deque(maxlen=20)

DD_VISIBILITY_HISTORY = deque(maxlen=8)
DD_DETECTION_HISTORY = deque(maxlen=8)

DD_HOLD_START = None
DD_BEST_HOLD_TIME = 0.0
DD_PERFECT_HOLD_COUNT = 0

DD_POINT_HISTORY = {}
DD_POINT_HISTORY_SIZE = 6


# =========================================================
# MEDIAPIPE
# =========================================================
mp_pose = mp.solutions.pose
down_dog_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.62,
    min_tracking_confidence=0.62,
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


def dd_smooth_score(new_score):
    DD_SCORE_HISTORY.append(float(new_score))
    return int(round(sum(DD_SCORE_HISTORY) / len(DD_SCORE_HISTORY)))


def dd_smooth_feedback(new_feedback):
    DD_FEEDBACK_HISTORY.append(str(new_feedback))
    return Counter(DD_FEEDBACK_HISTORY).most_common(1)[0][0]


def dd_smooth_boolean(history, value):
    history.append(bool(value))
    true_count = sum(history)
    return true_count >= max(1, len(history) // 2 + 1)


def dd_smooth_point(key, x, y, z):
    if key not in DD_POINT_HISTORY:
        DD_POINT_HISTORY[key] = deque(maxlen=DD_POINT_HISTORY_SIZE)

    DD_POINT_HISTORY[key].append((float(x), float(y), float(z)))
    xs = [p[0] for p in DD_POINT_HISTORY[key]]
    ys = [p[1] for p in DD_POINT_HISTORY[key]]
    zs = [p[2] for p in DD_POINT_HISTORY[key]]

    return (
        float(sum(xs) / len(xs)),
        float(sum(ys) / len(ys)),
        float(sum(zs) / len(zs)),
    )


def dd_clear_point_history():
    DD_POINT_HISTORY.clear()


def dd_reset_runtime_state():
    global DD_HOLD_START, DD_PERFECT_HOLD_COUNT

    DD_POSE_HISTORY.clear()
    DD_SCORE_HISTORY.clear()
    DD_FEEDBACK_HISTORY.clear()
    DD_HIP_CENTER_HISTORY.clear()
    DD_SHOULDER_CENTER_HISTORY.clear()
    DD_HIP_HEIGHT_HISTORY.clear()
    DD_SPINE_LINE_HISTORY.clear()
    DD_VISIBILITY_HISTORY.clear()
    DD_DETECTION_HISTORY.clear()
    dd_clear_point_history()

    DD_HOLD_START = None
    DD_PERFECT_HOLD_COUNT = 0


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
    visible_count = sum(v > 0.28 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    shoulders_ok = (
        lm_dict["left_shoulder"]["visibility"] > 0.40 or
        lm_dict["right_shoulder"]["visibility"] > 0.40
    )
    hips_ok = (
        lm_dict["left_hip"]["visibility"] > 0.40 or
        lm_dict["right_hip"]["visibility"] > 0.40
    )
    wrists_ok = (
        lm_dict["left_wrist"]["visibility"] > 0.30 or
        lm_dict["right_wrist"]["visibility"] > 0.30
    )
    ankles_ok = (
        lm_dict["left_ankle"]["visibility"] > 0.30 or
        lm_dict["right_ankle"]["visibility"] > 0.30
    )

    full_body_visible = visible_count >= 8 and shoulders_ok and hips_ok and wrists_ok and ankles_ok
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

    hips_above_shoulders = hip_center[1] < shoulder_center[1] - 0.008
    hips_high_enough = (wrist_center[1] - hip_center[1]) / torso_size > 0.46
    hips_peak_good = (ankle_center[1] - hip_center[1]) / torso_size > 0.28

    spine_long = base_length > 1.34
    side_profile_clear = arm_line_length > 1.00 and leg_line_length > 1.05
    arm_length_ok = shoulder_to_wrist > 0.95
    leg_length_ok = hip_to_ankle > 1.00

    dominant_arm_soft = dom_elbow_angle > 150
    # CHANGED: 160 is more realistic for human flexibility without failing everyone
    dominant_arm_strict = dom_elbow_angle > 160 
    dominant_leg_soft = dom_knee_angle > 150
    dominant_leg_strict = dom_knee_angle > 160

    support_arm_soft = sup_elbow_angle > 145
    support_leg_soft = sup_knee_angle > 145

    # CHANGED: 55 degrees is practically bent in half. In Down Dog, the shoulders 
    # form a straight line with the torso, so it should be > 135
    shoulder_open_ok = dom_shoulder_open > 135 
    hip_fold_ok = dom_hip_fold < 122

    hands_width_ratio = dd_distance(lw[:2], rw[:2]) / (dd_distance(ls[:2], rs[:2]) + 1e-6)
    feet_width_ratio = dd_distance(la[:2], ra[:2]) / (dd_distance(lh[:2], rh[:2]) + 1e-6)

    hands_width_ok = 0.72 <= hands_width_ratio <= 1.75
    feet_width_ok = 0.62 <= feet_width_ratio <= 1.75

    # CHANGED: Comparing nose X against wrist X breaks completely in a side profile. 
    # To check if the head is "between the arms", we just make sure the head drops 
    # properly below the shoulders (Y axis).
    head_between_arms = nose[1] > shoulder_center[1] + 0.04

    shoulder_symmetry = abs(float(ls[1] - rs[1])) / torso_size < 0.18
    hip_symmetry = abs(float(lh[1] - rh[1])) / torso_size < 0.18
    balanced_shape = shoulder_symmetry and hip_symmetry

    heel_lift = abs(float(heel[1] - a[1])) / torso_size
    heel_lift_other = abs(float(other_heel[1] - oa[1])) / torso_size
    heels_reasonable = heel_lift < 0.24
    both_heels_reasonable = heel_lift < 0.24 and heel_lift_other < 0.28

    dominant_side_gate = (
        hips_above_shoulders and
        hips_high_enough and
        dominant_arm_soft and
        dominant_leg_soft and
        shoulder_open_ok and
        hip_fold_ok and
        head_between_arms and
        spine_long and
        side_profile_clear and
        arm_length_ok and
        leg_length_ok
    )

    both_side_bonus_gate = (
        support_vis >= 0.52 and
        support_arm_soft and
        support_leg_soft and
        balanced_shape and
        hands_width_ok and
        feet_width_ok and
        both_heels_reasonable
    )

    strict_down_dog_gate = dominant_side_gate and (support_vis < 0.52 or both_side_bonus_gate)

    score = 0
    status = "warning"
    pose_label = "Not Down Dog"
    main_feedback = "Move into Downward Dog position."
    tips = ["Place both hands and feet firmly on the floor."]

    if not hips_above_shoulders:
        score = 35
        main_feedback = "Lift your hips higher."
        tips = ["Push the floor away and send the hips up."]

    elif not hips_high_enough:
        score = 48
        main_feedback = "Lift your hips more to build the inverted V."
        tips = ["Press strongly through your hands and feet."]

    elif not dominant_arm_soft:
        score = 60
        main_feedback = "Straighten your supporting arm line more."
        tips = ["Keep the elbow long and firm.", "Push the floor away."]

    elif not dominant_leg_soft:
        score = 72
        main_feedback = "Lengthen your main visible leg more."
        tips = ["Straighten the knee as much as comfortable."]

    elif not shoulder_open_ok:
        score = 78
        main_feedback = "Push back through your hands more."
        tips = ["Reach your chest back toward your thighs.", "Open the shoulders."]

    elif not hip_fold_ok:
        score = 82
        main_feedback = "Fold more from the hips."
        tips = ["Lift the hips and draw the ribs back."]

    elif not spine_long:
        score = 86
        main_feedback = "Lengthen your spine more."
        tips = ["Reach the hips up and the chest back."]

    elif not head_between_arms:
        score = 88
        main_feedback = "Relax your head more between the arms."
        tips = ["Let the neck stay soft and natural.", "Look toward your toes."]

    elif not hands_width_ok:
        score = 90
        main_feedback = "Adjust your hands slightly."
        tips = ["Keep the hands around shoulder width."]

    elif not feet_width_ok:
        score = 90
        main_feedback = "Adjust your feet slightly."
        tips = ["Keep the feet around hip width."]

    elif strict_down_dog_gate and dominant_arm_strict and dominant_leg_strict and hips_peak_good:
        if support_vis >= 0.52 and both_side_bonus_gate:
            score = 100
            status = "perfect"
            pose_label = "Correct Down Dog"
            main_feedback = "Perfect Down Dog. Hold steady."
            tips = ["Excellent posture.", "Keep breathing steadily.", "Maintain the inverted V shape."]
        else:
            score = 96
            status = "perfect"  # Changed from 'good' so the timer reliably triggers
            pose_label = "Down Dog"
            main_feedback = "Strong Down Dog. Hold steady."
            tips = ["Very good side alignment.", "Keep the shape steady."]

    else:
        score = 92
        status = "good"
        pose_label = "Down Dog"
        main_feedback = "Good Down Dog. Refine and hold steady."
        tips = ["Keep the hips high and the spine long."]

    checks = {
        "dominant_side": dominant_side,
        "dominant_side_name": side_name,
        "dominant_vis": round(float(dominant_vis), 3),
        "support_vis": round(float(support_vis), 3),
        "dominant_arm_soft": dominant_arm_soft,
        "dominant_arm_strict": dominant_arm_strict,
        "dominant_leg_soft": dominant_leg_soft,
        "dominant_leg_strict": dominant_leg_strict,
        "support_arm_soft": support_arm_soft,
        "support_leg_soft": support_leg_soft,
        "hips_above_shoulders": hips_above_shoulders,
        "hips_high_enough": hips_high_enough,
        "hips_peak_good": hips_peak_good,
        "head_between_arms": head_between_arms,
        "spine_long": spine_long,
        "shoulder_open_ok": shoulder_open_ok,
        "hip_fold_ok": hip_fold_ok,
        "side_profile_clear": side_profile_clear,
        "arm_length_ok": arm_length_ok,
        "leg_length_ok": leg_length_ok,
        "balanced_shape": balanced_shape,
        "hands_width_ok": hands_width_ok,
        "feet_width_ok": feet_width_ok,
        "heels_reasonable": heels_reasonable,
        "both_heels_reasonable": both_heels_reasonable,
        "dominant_side_gate": dominant_side_gate,
        "both_side_bonus_gate": both_side_bonus_gate,
        "strict_down_dog_gate": strict_down_dog_gate,
    }

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": tips,
        "angles": {
            "dominant_elbow_angle": round(float(dom_elbow_angle), 1),
            "dominant_knee_angle": round(float(dom_knee_angle), 1),
            "support_elbow_angle": round(float(sup_elbow_angle), 1),
            "support_knee_angle": round(float(sup_knee_angle), 1),
            "left_elbow_angle": round(dd_calculate_angle(ls[:2], raw_pts[DD_LEFT_ELBOW][:2], lw[:2]), 1),
            "right_elbow_angle": round(dd_calculate_angle(rs[:2], raw_pts[DD_RIGHT_ELBOW][:2], rw[:2]), 1),
            "left_knee_angle": round(dd_calculate_angle(lh[:2], raw_pts[DD_LEFT_KNEE][:2], la[:2]), 1),
            "right_knee_angle": round(dd_calculate_angle(rh[:2], raw_pts[DD_RIGHT_KNEE][:2], ra[:2]), 1),
        },
        "checks": checks,
    }


def dd_is_downdog_like(model_label, model_confidence, analysis):
    label = str(model_label).lower()
    checks = analysis["checks"]

    if not checks.get("dominant_side_gate", False):
        return False

    if ("down" in label or "dog" in label) and model_confidence >= 0.58:
        return True

    if (
        checks.get("hips_above_shoulders") and
        checks.get("hips_high_enough") and
        checks.get("dominant_arm_soft") and
        checks.get("dominant_leg_soft") and
        checks.get("shoulder_open_ok") and
        checks.get("hip_fold_ok") and
        checks.get("head_between_arms") and
        checks.get("spine_long")
    ):
        return True

    return False


# =========================================================
# STABILITY / HOLD / QUALITY
# =========================================================
def dd_update_stability_metrics(raw_pts):
    shoulder_center = (raw_pts[DD_LEFT_SHOULDER] + raw_pts[DD_RIGHT_SHOULDER]) / 2.0
    hip_center = (raw_pts[DD_LEFT_HIP] + raw_pts[DD_RIGHT_HIP]) / 2.0

    DD_HIP_CENTER_HISTORY.append(float(hip_center[0]))
    DD_SHOULDER_CENTER_HISTORY.append(float(shoulder_center[0]))
    DD_HIP_HEIGHT_HISTORY.append(float(hip_center[1]))
    DD_SPINE_LINE_HISTORY.append(abs(float(shoulder_center[0] - hip_center[0])))


def dd_get_stability_feedback():
    hip_shift = dd_moving_std(DD_HIP_CENTER_HISTORY)
    shoulder_shift = dd_moving_std(DD_SHOULDER_CENTER_HISTORY)
    hip_height_wobble = dd_moving_std(DD_HIP_HEIGHT_HISTORY)
    spine_wobble = dd_moving_std(DD_SPINE_LINE_HISTORY)

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


def dd_update_hold_state(is_downdog, full_body_visible, low_light):
    global DD_HOLD_START, DD_BEST_HOLD_TIME

    valid_hold = is_downdog and full_body_visible and not low_light

    if valid_hold:
        if DD_HOLD_START is None:
            DD_HOLD_START = time.time()
        hold_time = time.time() - DD_HOLD_START
        DD_BEST_HOLD_TIME = max(DD_BEST_HOLD_TIME, hold_time)
    else:
        hold_time = 0.0
        DD_HOLD_START = None

    return hold_time, DD_BEST_HOLD_TIME


def dd_hold_bonus(hold_time):
    if hold_time >= 10:
        return 8
    if hold_time >= 7:
        return 6
    if hold_time >= 5:
        return 4
    if hold_time >= 3:
        return 2
    return 0


def dd_quality_from_score(score):
    if score >= 98:
        return "Perfect_DownDog"
    if score >= 88:
        return "Good_DownDog"
    if score >= 70:
        return "Needs_Correction"
    return "Not_Ready"


def dd_build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    dominant_side = checks.get("dominant_side", "left")
    dominant_idxs = (
        [DD_LEFT_SHOULDER, DD_LEFT_ELBOW, DD_LEFT_WRIST, DD_LEFT_HIP, DD_LEFT_KNEE, DD_LEFT_ANKLE]
        if dominant_side == "left" else
        [DD_RIGHT_SHOULDER, DD_RIGHT_ELBOW, DD_RIGHT_WRIST, DD_RIGHT_HIP, DD_RIGHT_KNEE, DD_RIGHT_ANKLE]
    )

    points = []
    for idx in DD_SELECTED_POINTS + DD_EXTRA_POINTS:
        visibility = float(landmarks[idx].visibility)
        if visibility < 0.24:
            continue

        is_dominant = idx in dominant_idxs
        
        # Keep radius 5 for hidden limbs so the frontend knows to ignore them
        radius = 7 if is_dominant else 5
        color = DD_GREEN if analysis["score"] >= 90 and is_dominant else DD_YELLOW

        if idx in [DD_LEFT_ELBOW, DD_RIGHT_ELBOW] and (
            (idx in dominant_idxs and not checks.get("dominant_arm_soft")) or
            (idx not in dominant_idxs and checks.get("support_vis", 0) >= 0.52 and not checks.get("support_arm_soft"))
        ):
            color = DD_RED
            radius = 8
        if idx in [DD_LEFT_KNEE, DD_RIGHT_KNEE] and (
            (idx in dominant_idxs and not checks.get("dominant_leg_soft")) or
            (idx not in dominant_idxs and checks.get("support_vis", 0) >= 0.52 and not checks.get("support_leg_soft"))
        ):
            color = DD_RED
            radius = 8
        if idx in [DD_LEFT_HIP, DD_RIGHT_HIP] and not checks.get("hips_high_enough"):
            color = DD_RED
            radius = 8
        if idx in [DD_LEFT_SHOULDER, DD_RIGHT_SHOULDER] and not checks.get("shoulder_open_ok") and idx in dominant_idxs:
            color = DD_RED
            radius = 8
        if idx == DD_NOSE and not checks.get("head_between_arms"):
            color = DD_RED
            radius = 8

        points.append({
            "name": DD_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": float(np.clip(raw_pts[idx][0], 0.0, 1.0)),
            "y": float(np.clip(raw_pts[idx][1], 0.0, 1.0)),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
        })
    return points


def dd_build_angle_texts(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    dominant_side = checks.get("dominant_side", "left")
    support_visible = checks.get("support_vis", 0.0) >= 0.52
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


# =========================================================
# MAIN PROCESS
# =========================================================
def process_down_dog_request(request):
    global DD_PERFECT_HOLD_COUNT, DD_HOLD_START

    try:
        uploaded_file = request.FILES["image"]
        frame = dd_read_uploaded_image(uploaded_file)

        if frame is None:
            return dd_api_error("Invalid image file", status=400)

        frame = dd_enhance_frame(frame)

        low_light, brightness = dd_check_lighting(frame)
        if low_light:
            dd_reset_runtime_state()
            return dd_api_success(
                pose="Low Light",
                model_pose="Unknown",
                quality="N/A",
                feedback="Room lighting is too low for accurate pose detection.",
                coach_text="Improve the lighting and try again.",
                status="warning",
                confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles={},
                details=["Increase room lighting", "Face the light source", "Avoid a dark background"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        landmarks = dd_detect_landmarks(frame)
        has_landmarks = landmarks is not None
        stable_has_landmarks = dd_smooth_boolean(DD_DETECTION_HISTORY, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            dd_reset_runtime_state()
            return dd_api_success(
                pose="Unknown",
                model_pose="Unknown",
                quality="N/A",
                feedback="No human pose detected.",
                coach_text="Show your full body clearly in the camera.",
                status="unknown",
                confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles={},
                details=["Show your full body", "Stand where the camera can see you clearly", "Move slightly back"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        if landmarks is None:
            dd_clear_point_history()
            return dd_api_success(
                pose="Tracking...",
                model_pose="Unknown",
                quality="N/A",
                feedback="Hold still while detection stabilizes.",
                coach_text="Keep the pose steady.",
                status="warning",
                confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles={},
                details=["Hold still", "Keep your full body in frame"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        features_df, lm_dict, _ = dd_build_feature_dataframe_from_landmarks(landmarks)

        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        smoothed_pts = raw_pts.copy()

        for idx in DD_SELECTED_POINTS + DD_EXTRA_POINTS:
            sx, sy, sz = dd_smooth_point(f"dd_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])
            smoothed_pts[idx] = [sx, sy, sz]

        full_body_visible, visible_count, avg_visibility = dd_check_body_visibility(lm_dict)
        stable_full_body_visible = dd_smooth_boolean(DD_VISIBILITY_HISTORY, full_body_visible)
        framing_feedback = dd_check_frame_position(smoothed_pts)

        if not full_body_visible and not stable_full_body_visible:
            dd_reset_runtime_state()
            details = [
                "Show more of both hands and feet in frame",
                "Keep shoulders, hips, knees, and ankles visible",
                "Move only a little if needed",
            ]
            details.extend(framing_feedback)

            return dd_api_success(
                pose="Body Not Visible",
                model_pose="Unknown",
                quality="N/A",
                feedback="Adjust your position so the full Down Dog shape is visible.",
                coach_text="Show your whole Down Dog shape in the frame.",
                status="warning",
                confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles={},
                details=dd_dedupe_list(details, max_items=3),
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        raw_model_label, confidence = dd_predict_model_label(features_df)
        stable_model_label = dd_smooth_label(DD_POSE_HISTORY, raw_model_label)

        analysis = analyze_down_dog_pose(smoothed_pts, landmarks)
        points = dd_build_points_for_frontend(smoothed_pts, landmarks, analysis)
        angle_texts = dd_build_angle_texts(smoothed_pts, landmarks, analysis)

        is_downdog = dd_is_downdog_like(stable_model_label, confidence, analysis)

        if not is_downdog:
            DD_PERFECT_HOLD_COUNT = 0
            DD_HOLD_START = None

            tips = []
            tips.extend(analysis.get("tips", []))
            tips.extend([
                "Lift the hips high",
                "Straighten the arms and legs",
                "Make a clean inverted V shape",
            ])
            tips.extend(framing_feedback)

            return dd_api_success(
                pose="Not Down Dog",
                model_pose=stable_model_label,
                quality="Not_Ready",
                feedback=analysis.get("main_feedback", "Move into Downward Dog position."),
                coach_text=analysis.get("main_feedback", "Move into Downward Dog position."),
                status="warning",
                confidence=round(float(confidence), 3),
                score=max(0, min(65, analysis.get("score", 0))),
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles=analysis.get("angles", {}),
                details=dd_dedupe_list(tips, max_items=3),
                perfect_hold=False,
                points=points,
                angle_texts=angle_texts,
            )

        dd_update_stability_metrics(smoothed_pts)
        stability_tips, stability_penalty = dd_get_stability_feedback()

        base_score = analysis["score"]
        combined_score = max(0, base_score - stability_penalty)

        hold_time, best_hold = dd_update_hold_state(
            is_downdog=is_downdog,
            full_body_visible=stable_full_body_visible,
            low_light=low_light,
        )

        combined_score = min(100, combined_score + dd_hold_bonus(hold_time))
        stable_score = dd_smooth_score(combined_score)

        strict_visibility_ok = analysis["checks"].get("dominant_vis", 0) >= 0.62
        both_side_ready = (
            analysis["checks"].get("support_vis", 0) >= 0.52 and
            analysis["checks"].get("both_side_bonus_gate", False)
        )

        if (
            analysis["checks"].get("strict_down_dog_gate") and
            strict_visibility_ok and
            both_side_ready and
            stable_score >= 98 and
            hold_time >= 3.0
        ):
            pose_name = "Correct Down Dog"
            status = "perfect"
            feedback_text = "Correct Down Dog"
            coach_text = "Excellent. Hold steady."
        elif stable_score >= 90:
            pose_name = "Down Dog"
            status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = "Good shape. Refine and hold steady."
        else:
            pose_name = "Down Dog Needs Correction"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "You are close. Fix the main weak point."

        stable_feedback = dd_smooth_feedback(feedback_text)

        if (
            pose_name == "Correct Down Dog" and
            hold_time >= 3.2 and
            strict_visibility_ok and
            both_side_ready
        ):
            DD_PERFECT_HOLD_COUNT += 1
        else:
            DD_PERFECT_HOLD_COUNT = 0

        tips = []
        if hold_time >= 5:
            tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        return dd_api_success(
            pose=pose_name,
            model_pose=stable_model_label,
            quality=dd_quality_from_score(stable_score),
            feedback=stable_feedback,
            coach_text=coach_text,
            status=status,
            confidence=round(float(confidence), 3),
            score=stable_score,
            hold_time=round(float(hold_time), 1),
            best_hold_time=round(float(best_hold), 1),
            angles=analysis["angles"],
            details=dd_dedupe_list(tips, max_items=3, exclude=[stable_feedback, coach_text]),
            perfect_hold=DD_PERFECT_HOLD_COUNT >= 3,
            points=points,
            angle_texts=angle_texts,
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