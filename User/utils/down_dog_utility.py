from pathlib import Path
from collections import deque, Counter
import time
import pickle

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
DD_POSE_HISTORY = deque(maxlen=12)
DD_SCORE_HISTORY = deque(maxlen=8)
DD_FEEDBACK_HISTORY = deque(maxlen=8)

DD_HIP_CENTER_HISTORY = deque(maxlen=20)
DD_SHOULDER_CENTER_HISTORY = deque(maxlen=20)
DD_HIP_HEIGHT_HISTORY = deque(maxlen=20)
DD_SPINE_LINE_HISTORY = deque(maxlen=20)

DD_HOLD_START = None
DD_BEST_HOLD_TIME = 0.0
DD_PERFECT_HOLD_COUNT = 0


# =========================================================
# MEDIAPIPE SETUP
# =========================================================
mp_pose = mp.solutions.pose
down_dog_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
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
}


# =========================================================
# FULL MODEL FEATURE SET
# =========================================================
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

DD_ALL_LANDMARK_NAME_TO_INDEX = {
    name: idx for idx, name in enumerate(DD_ALL_LANDMARK_NAMES)
}

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
    "hip_angle",
    "shoulder_angle",
]

DD_MODEL_COLUMNS = DD_FEATURE_COLUMNS + DD_ANGLE_COLUMNS


# =========================================================
# COLORS
# =========================================================
DD_GREEN = "#00ff66"
DD_RED = "#ff3b30"
DD_YELLOW = "#ffd60a"
DD_GRAY = "#cfcfcf"


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
    raise FileNotFoundError(
        f"Could not find model file: {filename}\nChecked:\n{checked}"
    )


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
# SMOOTHING HELPERS
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


# =========================================================
# BASIC HELPERS
# =========================================================
def dd_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def dd_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.04, beta=5)


def dd_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = down_dog_pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    return results.pose_landmarks.landmark


def dd_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 58, brightness


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
# FEATURE CREATION
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

    points = {name: [vals["x"], vals["y"]] for name, vals in lm_dict.items()}

    angles = {
        "left_knee_angle": dd_calculate_angle(
            points["left_hip"], points["left_knee"], points["left_ankle"]
        ),
        "right_knee_angle": dd_calculate_angle(
            points["right_hip"], points["right_knee"], points["right_ankle"]
        ),
        "left_elbow_angle": dd_calculate_angle(
            points["left_shoulder"], points["left_elbow"], points["left_wrist"]
        ),
        "right_elbow_angle": dd_calculate_angle(
            points["right_shoulder"], points["right_elbow"], points["right_wrist"]
        ),
        "hip_angle": dd_calculate_angle(
            points["left_shoulder"], points["left_hip"], points["left_knee"]
        ),
        "shoulder_angle": dd_calculate_angle(
            points["left_elbow"], points["left_shoulder"], points["left_hip"]
        ),
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
    features_df = features_df.astype(np.float32)
    scaled_features = downdog_scaler.transform(features_df)

    prediction = downdog_model.predict(scaled_features)[0]

    confidence = 0.50
    if hasattr(downdog_model, "predict_proba"):
        probs = downdog_model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probs))

    return str(prediction), confidence

# =========================================================
# VISIBILITY / FRAMING
# =========================================================
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
    visible_count = sum(v > 0.22 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    shoulders_ok = (
        lm_dict["left_shoulder"]["visibility"] > 0.35 and
        lm_dict["right_shoulder"]["visibility"] > 0.35
    )
    hips_ok = (
        lm_dict["left_hip"]["visibility"] > 0.35 and
        lm_dict["right_hip"]["visibility"] > 0.35
    )

    wrists_ok = (
        lm_dict["left_wrist"]["visibility"] > 0.18 or
        lm_dict["right_wrist"]["visibility"] > 0.18
    )
    ankles_ok = (
        lm_dict["left_ankle"]["visibility"] > 0.18 or
        lm_dict["right_ankle"]["visibility"] > 0.18
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

    if width > 0.96:
        feedback.append("Move slightly away from the camera")

    if height > 0.98:
        feedback.append("Show a little more of your full body")

    if width < 0.34:
        feedback.append("Move a little closer so your body is larger in frame")

    center_x = (min_x + max_x) / 2
    if center_x < 0.16:
        feedback.append("Move slightly to the right")
    elif center_x > 0.84:
        feedback.append("Move slightly to the left")

    return feedback


# =========================================================
# ANALYSIS
# =========================================================
def analyze_down_dog_pose(raw_pts):
    ls, rs = raw_pts[DD_LEFT_SHOULDER], raw_pts[DD_RIGHT_SHOULDER]
    le, re = raw_pts[DD_LEFT_ELBOW], raw_pts[DD_RIGHT_ELBOW]
    lw, rw = raw_pts[DD_LEFT_WRIST], raw_pts[DD_RIGHT_WRIST]
    lh, rh = raw_pts[DD_LEFT_HIP], raw_pts[DD_RIGHT_HIP]
    lk, rk = raw_pts[DD_LEFT_KNEE], raw_pts[DD_RIGHT_KNEE]
    la, ra = raw_pts[DD_LEFT_ANKLE], raw_pts[DD_RIGHT_ANKLE]
    nose = raw_pts[DD_NOSE]

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    wrist_center = (lw + rw) / 2.0
    ankle_center = (la + ra) / 2.0

    torso_size = dd_distance(shoulder_center[:2], hip_center[:2]) + 1e-6
    base_length = dd_distance(wrist_center[:2], ankle_center[:2]) / torso_size

    left_elbow_angle = dd_calculate_angle(ls[:2], le[:2], lw[:2])
    right_elbow_angle = dd_calculate_angle(rs[:2], re[:2], rw[:2])
    left_knee_angle = dd_calculate_angle(lh[:2], lk[:2], la[:2])
    right_knee_angle = dd_calculate_angle(rh[:2], rk[:2], ra[:2])

    left_shoulder_open = dd_calculate_angle(le[:2], ls[:2], lh[:2])
    right_shoulder_open = dd_calculate_angle(re[:2], rs[:2], rh[:2])

    left_hip_fold = dd_calculate_angle(ls[:2], lh[:2], lk[:2])
    right_hip_fold = dd_calculate_angle(rs[:2], rh[:2], rk[:2])

    hands_shoulder_width_ratio = dd_distance(lw[:2], rw[:2]) / (dd_distance(ls[:2], rs[:2]) + 1e-6)
    feet_hip_width_ratio = dd_distance(la[:2], ra[:2]) / (dd_distance(lh[:2], rh[:2]) + 1e-6)

    hips_above_shoulders = hip_center[1] < shoulder_center[1] - 0.012
    hips_high_enough = (wrist_center[1] - hip_center[1]) / torso_size > 0.48
    hips_peak_good = (ankle_center[1] - hip_center[1]) / torso_size > 0.26

    arms_straight_strict = left_elbow_angle > 158 and right_elbow_angle > 158
    arms_straight_soft = left_elbow_angle > 146 and right_elbow_angle > 146

    legs_long_strict = left_knee_angle > 164 and right_knee_angle > 164
    legs_long_soft = left_knee_angle > 146 and right_knee_angle > 146

    shoulder_open_ok = left_shoulder_open > 52 and right_shoulder_open > 52
    hip_fold_ok = left_hip_fold < 122 and right_hip_fold < 122

    spine_long = base_length > 1.38

    head_between_arms = (
        nose[0] > min(lw[0], rw[0]) - 0.10 and
        nose[0] < max(lw[0], rw[0]) + 0.10
    )

    shoulder_symmetry = abs(float(ls[1] - rs[1])) / torso_size < 0.15
    hip_symmetry = abs(float(lh[1] - rh[1])) / torso_size < 0.15
    balanced_shape = shoulder_symmetry and hip_symmetry

    hands_width_ok = 0.70 <= hands_shoulder_width_ratio <= 1.95
    feet_width_ok = 0.50 <= feet_hip_width_ratio <= 1.90

    strict_down_dog_gate = (
        hips_above_shoulders and
        hips_high_enough and
        arms_straight_soft and
        legs_long_soft and
        shoulder_open_ok and
        hip_fold_ok and
        head_between_arms and
        spine_long
    )

    score = 0
    status = "warning"
    pose_label = "Not Down Dog"
    main_feedback = "Move into Downward Dog position."
    tips = ["Place both hands and feet firmly on the floor."]

    if not hips_above_shoulders:
        score = 30
        main_feedback = "Lift your hips higher."
        tips = ["Push the floor away and send the hips up."]

    elif not hips_high_enough:
        score = 42
        main_feedback = "Lift your hips more to form the inverted V shape."
        tips = ["Press strongly through your hands and feet."]

    elif not arms_straight_soft:
        score = 54
        main_feedback = "Straighten both arms more."
        tips = ["Keep elbows long and firm.", "Press evenly through both palms."]

    elif not shoulder_open_ok:
        score = 64
        main_feedback = "Open your shoulders more."
        tips = ["Send your chest gently toward your thighs."]

    elif not legs_long_soft:
        score = 72
        main_feedback = "Lengthen your legs more."
        tips = ["Straighten the knees as much as comfortable."]

    elif not hip_fold_ok:
        score = 80
        main_feedback = "Fold more from the hips."
        tips = ["Reach the chest back and the hips up."]

    elif not head_between_arms:
        score = 86
        main_feedback = "Keep your head relaxed between your arms."
        tips = ["Let the neck stay soft and natural."]

    elif not spine_long:
        score = 90
        main_feedback = "Lengthen your spine more."
        tips = ["Reach your hips up and your chest back."]

    elif not balanced_shape:
        score = 92
        main_feedback = "Make your shape more even on both sides."
        tips = ["Balance your shoulders and hips."]

    elif not hands_width_ok:
        score = 91
        main_feedback = "Adjust your hands slightly."
        tips = ["Keep your hands around shoulder width."]

    elif not feet_width_ok:
        score = 91
        main_feedback = "Adjust your feet slightly."
        tips = ["Keep your feet around hip width."]

    elif strict_down_dog_gate and arms_straight_strict and legs_long_strict and hips_peak_good:
        score = 100
        status = "perfect"
        pose_label = "Correct Down Dog"
        main_feedback = "Perfect Down Dog. Hold steady."
        tips = [
            "Excellent posture.",
            "Keep breathing steadily.",
            "Maintain the inverted V shape.",
        ]

    else:
        score = 95
        status = "good"
        pose_label = "Down Dog"
        main_feedback = "Very good Down Dog. Refine and hold steady."
        tips = ["Keep the hips high and the spine long."]

    checks = {
        "arms_straight_strict": arms_straight_strict,
        "arms_straight_soft": arms_straight_soft,
        "legs_long_strict": legs_long_strict,
        "legs_long_soft": legs_long_soft,
        "hips_above_shoulders": hips_above_shoulders,
        "hips_high_enough": hips_high_enough,
        "hips_peak_good": hips_peak_good,
        "head_between_arms": head_between_arms,
        "spine_long": spine_long,
        "shoulder_open_ok": shoulder_open_ok,
        "hip_fold_ok": hip_fold_ok,
        "balanced_shape": balanced_shape,
        "hands_width_ok": hands_width_ok,
        "feet_width_ok": feet_width_ok,
        "strict_down_dog_gate": strict_down_dog_gate,
    }

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": tips,
        "angles": {
            "left_elbow_angle": round(float(left_elbow_angle), 1),
            "right_elbow_angle": round(float(right_elbow_angle), 1),
            "left_knee_angle": round(float(left_knee_angle), 1),
            "right_knee_angle": round(float(right_knee_angle), 1),
            "left_shoulder_open": round(float(left_shoulder_open), 1),
            "right_shoulder_open": round(float(right_shoulder_open), 1),
            "left_hip_fold": round(float(left_hip_fold), 1),
            "right_hip_fold": round(float(right_hip_fold), 1),
        },
        "checks": checks,
    }


def dd_is_downdog_like(model_label, model_confidence, analysis):
    label = str(model_label).lower()
    checks = analysis["checks"]

    if not checks.get("strict_down_dog_gate", False):
        return False

    if ("down" in label or "dog" in label) and model_confidence >= 0.60:
        return True

    if (
        checks.get("hips_above_shoulders") and
        checks.get("hips_high_enough") and
        checks.get("arms_straight_soft") and
        checks.get("legs_long_soft") and
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

    hip_center_x = float(hip_center[0])
    shoulder_center_x = float(shoulder_center[0])
    hip_height = float(hip_center[1])
    spine_shift = abs(float(shoulder_center[0] - hip_center[0]))

    DD_HIP_CENTER_HISTORY.append(hip_center_x)
    DD_SHOULDER_CENTER_HISTORY.append(shoulder_center_x)
    DD_HIP_HEIGHT_HISTORY.append(hip_height)
    DD_SPINE_LINE_HISTORY.append(spine_shift)


def dd_get_stability_feedback():
    hip_shift = dd_moving_std(DD_HIP_CENTER_HISTORY)
    shoulder_shift = dd_moving_std(DD_SHOULDER_CENTER_HISTORY)
    hip_height_wobble = dd_moving_std(DD_HIP_HEIGHT_HISTORY)
    spine_wobble = dd_moving_std(DD_SPINE_LINE_HISTORY)

    feedback = []
    penalty = 0

    if hip_shift > 0.018:
        feedback.append("Keep your hips steadier")
        penalty += 5

    if shoulder_shift > 0.018:
        feedback.append("Stabilize your shoulders")
        penalty += 4

    if hip_height_wobble > 0.015:
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
        return 10
    if hold_time >= 7:
        return 7
    if hold_time >= 5:
        return 5
    if hold_time >= 3:
        return 2
    return 0


def dd_quality_from_score(score):
    if score >= 96:
        return "Perfect_DownDog"
    if score >= 84:
        return "Good_DownDog"
    if score >= 65:
        return "Needs_Correction"
    return "Not_Ready"


# =========================================================
# OVERLAY HELPERS
# =========================================================
def dd_build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    points = []

    for idx in DD_SELECTED_POINTS:
        lm = landmarks[idx]
        visibility = float(lm.visibility)

        if visibility < 0.18:
            continue

        color = DD_GREEN if analysis["score"] >= 88 else DD_YELLOW
        radius = 6

        if idx in [DD_LEFT_ELBOW, DD_RIGHT_ELBOW] and not checks.get("arms_straight_soft"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_KNEE, DD_RIGHT_KNEE] and not checks.get("legs_long_soft"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_HIP, DD_RIGHT_HIP] and not checks.get("hips_high_enough"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_SHOULDER, DD_RIGHT_SHOULDER] and not checks.get("shoulder_open_ok"):
            color = DD_RED
            radius = 7

        if idx == DD_NOSE and not checks.get("head_between_arms"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_WRIST, DD_RIGHT_WRIST] and not checks.get("hands_width_ok"):
            color = DD_YELLOW
            radius = 7

        if idx in [DD_LEFT_ANKLE, DD_RIGHT_ANKLE] and not checks.get("feet_width_ok"):
            color = DD_YELLOW
            radius = 7

        points.append({
            "name": DD_POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": dd_clip01(raw_pts[idx][0]),
            "y": dd_clip01(raw_pts[idx][1]),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
        })

    return points


def dd_build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    mapping = [
        (DD_LEFT_ELBOW, analysis.get("angles", {}).get("left_elbow_angle", 0)),
        (DD_RIGHT_ELBOW, analysis.get("angles", {}).get("right_elbow_angle", 0)),
        (DD_LEFT_KNEE, analysis.get("angles", {}).get("left_knee_angle", 0)),
        (DD_RIGHT_KNEE, analysis.get("angles", {}).get("right_knee_angle", 0)),
    ]

    for idx, value in mapping:
        lm = landmarks[idx]
        if float(lm.visibility) < 0.18:
            continue

        items.append({
            "text": f"{int(round(float(value)))}°",
            "x": dd_clip01(raw_pts[idx][0]),
            "y": dd_clip01(raw_pts[idx][1]),
            "color": DD_YELLOW,
        })

    return items


# =========================================================
# MAIN PROCESS FUNCTION
# =========================================================
def process_down_dog_request(request):
    global DD_PERFECT_HOLD_COUNT

    try:
        uploaded_file = request.FILES["image"]
        frame = dd_read_uploaded_image(uploaded_file)

        if frame is None:
            return dd_api_error("Invalid image file", status=400)

        frame = dd_enhance_frame(frame)

        low_light, brightness = dd_check_lighting(frame)
        if low_light:
            DD_POSE_HISTORY.clear()
            DD_SCORE_HISTORY.clear()
            DD_FEEDBACK_HISTORY.clear()
            DD_PERFECT_HOLD_COUNT = 0

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
                details=[
                    "Increase room lighting",
                    "Face the light source",
                    "Avoid dark background",
                ],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        landmarks = dd_detect_landmarks(frame)
        if not landmarks:
            DD_POSE_HISTORY.clear()
            DD_SCORE_HISTORY.clear()
            DD_FEEDBACK_HISTORY.clear()
            DD_PERFECT_HOLD_COUNT = 0

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
                details=[
                    "Show full body",
                    "Stand where camera can see you clearly",
                    "Move slightly back",
                ],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        features_df, lm_dict, _ = dd_build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        full_body_visible, visible_count, avg_visibility = dd_check_body_visibility(lm_dict)
        framing_feedback = dd_check_frame_position(raw_pts)

        if not full_body_visible:
            DD_POSE_HISTORY.clear()
            DD_SCORE_HISTORY.clear()
            DD_FEEDBACK_HISTORY.clear()
            DD_PERFECT_HOLD_COUNT = 0

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

        analysis = analyze_down_dog_pose(raw_pts)
        points = dd_build_points_for_frontend(raw_pts, landmarks, analysis)
        angle_texts = dd_build_angle_texts(raw_pts, landmarks, analysis)

        is_downdog = dd_is_downdog_like(stable_model_label, confidence, analysis)

        if not is_downdog:
            DD_PERFECT_HOLD_COUNT = 0

            tips = []
            tips.extend(analysis.get("tips", []))
            tips.extend([
                "Lift the hips high",
                "Straighten the arms and legs",
                "Make an inverted V shape",
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
                score=max(0, min(55, analysis.get("score", 0))),
                hold_time=0.0,
                best_hold_time=round(float(DD_BEST_HOLD_TIME), 1),
                angles=analysis.get("angles", {}),
                details=dd_dedupe_list(tips, max_items=3),
                perfect_hold=False,
                points=points,
                angle_texts=angle_texts,
            )

        dd_update_stability_metrics(raw_pts)
        stability_tips, stability_penalty = dd_get_stability_feedback()

        base_score = analysis["score"]
        combined_score = max(0, base_score - stability_penalty)

        hold_time, best_hold = dd_update_hold_state(
            is_downdog=is_downdog,
            full_body_visible=full_body_visible,
            low_light=low_light,
        )

        combined_score = min(100, combined_score + dd_hold_bonus(hold_time))
        stable_score = dd_smooth_score(combined_score)

        if stable_score >= 96 and analysis["checks"].get("strict_down_dog_gate"):
            pose_name = "Correct Down Dog"
            status = "perfect"
            feedback_text = "Correct Down Dog"
            coach_text = "Excellent. Hold steady."
        elif stable_score >= 84:
            pose_name = "Down Dog"
            status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = "Good shape. Refine and hold steady."
        elif stable_score >= 65:
            pose_name = "Down Dog Needs Correction"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "You are close. Fix one or two points."
        else:
            pose_name = "Not Ready Yet"
            status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "Rebuild the inverted V shape."

        stable_feedback = dd_smooth_feedback(feedback_text)

        if (
            stable_score >= 97 and
            hold_time >= 2.5 and
            analysis["checks"].get("strict_down_dog_gate")
        ):
            DD_PERFECT_HOLD_COUNT += 1
        else:
            DD_PERFECT_HOLD_COUNT = 0

        if DD_PERFECT_HOLD_COUNT >= 3:
            pose_name = "Correct Down Dog"
            status = "perfect"
            stable_feedback = "Correct Down Dog"
            coach_text = "Excellent. Hold steady."

        tips = []
        if hold_time >= 5:
            tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        cleaned_tips = dd_dedupe_list(
            tips,
            max_items=3,
            exclude=[stable_feedback, coach_text]
        )

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
            details=cleaned_tips,
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