from pathlib import Path
from collections import deque, Counter
import time
import math

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


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
POINT_HISTORY_SIZE = 6


# =========================================================
# SAFE PATH / MODEL LOAD
# =========================================================
BASE_DIR = Path(settings.BASE_DIR)


def resolve_model_path(filename: str) -> Path:
    candidates = [
        BASE_DIR / "Ml_Models" / filename,
        BASE_DIR / "Ml_models" / filename,
        BASE_DIR / "ml_models" / filename,
        BASE_DIR / "models" / filename,
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

pose_model = None
pose_label_encoder = None
defect_model = None
defect_label_encoder = None
pose = None

mp_pose = mp.solutions.pose


def load_models():
    global pose_model, pose_label_encoder, defect_model, defect_label_encoder, pose

    if pose_model is None:
        pose_model = joblib.load(POSE_MODEL_PATH)

    if pose_label_encoder is None:
        pose_label_encoder = joblib.load(POSE_LABEL_ENCODER_PATH)

    if defect_model is None:
        defect_model = joblib.load(DEFECT_MODEL_PATH)

    if defect_label_encoder is None:
        defect_label_encoder = joblib.load(DEFECT_LABEL_ENCODER_PATH)

    if pose is None:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.62,
            min_tracking_confidence=0.62,
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
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

SELECTED_POINTS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
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
    LEFT_HEEL: "left_heel",
    RIGHT_HEEL: "right_heel",
    LEFT_FOOT_INDEX: "left_foot_index",
    RIGHT_FOOT_INDEX: "right_foot_index",
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


# =========================================================
# BASIC PAGES
# =========================================================
def HomePage(request):
    return render(request, "User/home_page.html")


def camera_page(request):
    return render(request, "User/camera.html")


# =========================================================
# RESPONSE HELPERS
# =========================================================
def api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})


def api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


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


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def clear_point_history():
    POINT_HISTORY.clear()


def adaptive_smooth_point(key, x, y):
    """
    Better than plain average.
    Keeps points stable when user is still,
    but reacts faster when user moves.
    """
    x = clip01(x)
    y = clip01(y)

    if key not in POINT_HISTORY:
        POINT_HISTORY[key] = {
            "x": x,
            "y": y,
            "trail": deque(maxlen=POINT_HISTORY_SIZE),
        }
        POINT_HISTORY[key]["trail"].append((x, y))
        return x, y

    prev_x = POINT_HISTORY[key]["x"]
    prev_y = POINT_HISTORY[key]["y"]

    dist = math.hypot(x - prev_x, y - prev_y)

    if dist > 0.08:
        alpha = 0.82
    elif dist > 0.04:
        alpha = 0.58
    else:
        alpha = 0.32

    smoothed_x = (alpha * x) + ((1 - alpha) * prev_x)
    smoothed_y = (alpha * y) + ((1 - alpha) * prev_y)

    POINT_HISTORY[key]["x"] = smoothed_x
    POINT_HISTORY[key]["y"] = smoothed_y
    POINT_HISTORY[key]["trail"].append((smoothed_x, smoothed_y))

    xs = [p[0] for p in POINT_HISTORY[key]["trail"]]
    ys = [p[1] for p in POINT_HISTORY[key]["trail"]]

    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


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


def safe_dist_2d(a, b):
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))


# =========================================================
# IMAGE HELPERS
# =========================================================
def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def enhance_frame(frame):
    """
    Keep geometry unchanged.
    Only mild denoise/contrast improvement.
    Do NOT resize or warp, otherwise overlay points drift.
    """
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.convertScaleAbs(frame, alpha=1.02, beta=2)
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
    return brightness < 60, brightness


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
    visible_count = sum(v > 0.52 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    ankles_visible = (
        lm_dict["left_ankle"]["visibility"] > 0.42 and
        lm_dict["right_ankle"]["visibility"] > 0.42
    )

    shoulders_visible = (
        lm_dict["left_shoulder"]["visibility"] > 0.55 and
        lm_dict["right_shoulder"]["visibility"] > 0.55
    )

    hips_visible = (
        lm_dict["left_hip"]["visibility"] > 0.55 and
        lm_dict["right_hip"]["visibility"] > 0.55
    )

    full_body_visible = (
        visible_count >= 10 and
        ankles_visible and
        shoulders_visible and
        hips_visible
    )
    return full_body_visible, visible_count, avg_visibility


def check_frame_position(raw_pts):
    xs = [float(raw_pts[idx][0]) for idx in SELECTED_POINTS if idx < len(raw_pts)]
    ys = [float(raw_pts[idx][1]) for idx in SELECTED_POINTS if idx < len(raw_pts)]

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
# TREE ANALYSIS
# =========================================================
def analyze_tree_pose(raw_pts):
    ls, rs = raw_pts[LEFT_SHOULDER], raw_pts[RIGHT_SHOULDER]
    le, re = raw_pts[LEFT_ELBOW], raw_pts[RIGHT_ELBOW]
    lw, rw = raw_pts[LEFT_WRIST], raw_pts[RIGHT_WRIST]
    lh, rh = raw_pts[LEFT_HIP], raw_pts[RIGHT_HIP]
    lk, rk = raw_pts[LEFT_KNEE], raw_pts[RIGHT_KNEE]
    la, ra = raw_pts[LEFT_ANKLE], raw_pts[RIGHT_ANKLE]
    lfi, rfi = raw_pts[LEFT_FOOT_INDEX], raw_pts[RIGHT_FOOT_INDEX]

    left_knee = calculate_angle(lh[:2], lk[:2], la[:2])
    right_knee = calculate_angle(rh[:2], rk[:2], ra[:2])
    left_elbow = calculate_angle(ls[:2], le[:2], lw[:2])
    right_elbow = calculate_angle(rs[:2], re[:2], rw[:2])
    left_hip = calculate_angle(ls[:2], lh[:2], lk[:2])
    right_hip = calculate_angle(rs[:2], rh[:2], rk[:2])

    shoulder_diff = abs(float(ls[1] - rs[1]))
    hip_diff = abs(float(lh[1] - rh[1]))

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    vertical = hip_center + np.array([0, -0.25, 0], dtype=np.float32)
    torso_tilt = calculate_angle(shoulder_center[:2], hip_center[:2], vertical[:2])

    if left_knee > right_knee:
        standing_side = "left"
        stand_knee_idx = LEFT_KNEE
        bent_knee_idx = RIGHT_KNEE
        stand_ankle_idx = LEFT_ANKLE
        bent_ankle_idx = RIGHT_ANKLE
        stand_foot_idx = LEFT_FOOT_INDEX
        stand_angle = left_knee
        bent_angle = right_knee
        raised_foot = ra
        raised_foot_index = rfi
        stand_hip = lh
        stand_knee = lk
        raised_knee = rk
        grounded_ankle = la
    else:
        standing_side = "right"
        stand_knee_idx = RIGHT_KNEE
        bent_knee_idx = LEFT_KNEE
        stand_ankle_idx = RIGHT_ANKLE
        bent_ankle_idx = LEFT_ANKLE
        stand_foot_idx = RIGHT_FOOT_INDEX
        stand_angle = right_knee
        bent_angle = left_knee
        raised_foot = la
        raised_foot_index = lfi
        stand_hip = rh
        stand_knee = rk
        raised_knee = lk
        grounded_ankle = ra

    # -------------------------------
    # HANDS
    # -------------------------------
    left_wrist_above_shoulder = lw[1] < ls[1] + 0.03
    right_wrist_above_shoulder = rw[1] < rs[1] + 0.03
    hands_overhead = left_wrist_above_shoulder and right_wrist_above_shoulder

    wrists_close = abs(float(lw[0] - rw[0])) < 0.11
    wrists_same_height = abs(float(lw[1] - rw[1])) < 0.09
    prayer_hands = (
        wrists_close and
        wrists_same_height and
        lw[1] < shoulder_center[1] + 0.12 and
        rw[1] < shoulder_center[1] + 0.12
    )

    hands_ok = hands_overhead or prayer_hands

    # -------------------------------
    # ARM / BODY ALIGNMENT
    # -------------------------------
    arms_straight = left_elbow >= 128 and right_elbow >= 128
    shoulders_level = shoulder_diff < 0.075
    hips_level = hip_diff < 0.085
    torso_ok = torso_tilt < 18
    hand_align_ok = np.linalg.norm(lw[:2] - rw[:2]) < 0.44

    # -------------------------------
    # FOOT / LEG STRICT RULES
    # -------------------------------
    dist_to_stand_knee = float(np.linalg.norm(raised_foot[:2] - stand_knee[:2]))
    dist_to_stand_hip = float(np.linalg.norm(raised_foot[:2] - stand_hip[:2]))

    raised_foot_lift = abs(float(raised_foot[1] - grounded_ankle[1]))
    one_foot_lifted = raised_foot_lift > 0.055

    raised_foot_above_ground_ankle = float(raised_foot[1]) < float(grounded_ankle[1]) - 0.045
    raised_foot_not_on_floor = float(raised_foot_index[1]) < float(raw_pts[stand_foot_idx][1]) - 0.02

    # Foot must be on opposite inner calf or thigh, not hanging low, not both feet on floor
    foot_on_inner_leg = (dist_to_stand_knee < 0.16) or (dist_to_stand_hip < 0.19)
    foot_not_too_low = float(raised_foot[1]) < float(grounded_ankle[1]) - 0.035
    foot_place_ok = (
        foot_on_inner_leg and
        foot_not_too_low and
        raised_foot_above_ground_ankle and
        raised_foot_not_on_floor
    )

    center_x = float(hip_center[0])
    knee_offset = abs(float(raised_knee[0] - center_x))
    knee_open_ok = knee_offset > 0.075 and 45 <= bent_angle <= 125

    support_line = abs(float(stand_hip[0] - raw_pts[stand_ankle_idx][0]))
    balance_ok = support_line < 0.10

    feet_together_on_floor = abs(float(la[1] - ra[1])) < 0.035
    both_legs_straight = left_knee > 156 and right_knee > 156

    # -------------------------------
    # STRICT TREE GATE
    # VERY IMPORTANT:
    # now hands_ok is REQUIRED
    # -------------------------------
    strict_tree_gate = (
        stand_angle >= 162 and
        45 <= bent_angle <= 128 and
        one_foot_lifted and
        foot_place_ok and
        knee_open_ok and
        hands_ok and
        torso_ok and
        not both_legs_straight and
        not feet_together_on_floor
    )

    checks = {
        "standing_leg": stand_angle >= 164,
        "bent_leg": 45 <= bent_angle <= 128,
        "foot_place": foot_place_ok,
        "knee_open": knee_open_ok,
        "hands_up": hands_ok,
        "arms_straight": arms_straight,
        "hand_align": hand_align_ok,
        "torso": torso_ok,
        "shoulders": shoulders_level,
        "hips": hips_level,
        "balance": balance_ok,
        "one_foot_lifted": one_foot_lifted,
        "strict_tree_gate": strict_tree_gate,
    }

    weights = {
        "standing_leg": 22,
        "bent_leg": 15,
        "foot_place": 22,
        "knee_open": 12,
        "hands_up": 12,
        "arms_straight": 4,
        "hand_align": 2,
        "torso": 5,
        "shoulders": 2,
        "hips": 2,
        "balance": 2,
    }

    score = 0
    for key, ok in checks.items():
        if key in weights and ok:
            score += weights[key]
    score = min(100, score)

    priority_feedback = []
    if not checks["one_foot_lifted"]:
        priority_feedback.append("Lift one foot fully off the floor")
    if not checks["foot_place"]:
        priority_feedback.append("Place the raised foot firmly on the opposite inner calf or inner thigh")
    if not checks["knee_open"]:
        priority_feedback.append("Open the bent knee outward more")
    if not checks["standing_leg"]:
        priority_feedback.append("Straighten the standing leg more")
    if not checks["hands_up"]:
        priority_feedback.append("Raise your hands overhead or bring them to prayer")
    if not checks["arms_straight"] and hands_overhead:
        priority_feedback.append("Straighten both arms")
    if not checks["torso"]:
        priority_feedback.append("Keep your torso upright")
    if not checks["shoulders"]:
        priority_feedback.append("Level your shoulders")
    if not checks["hips"]:
        priority_feedback.append("Level your hips")
    if not checks["balance"]:
        priority_feedback.append("Steady your balance over the standing foot")

    # Correct Tree ONLY when all strict conditions are satisfied
    if checks["strict_tree_gate"] and score >= 90:
        main_feedback = "Correct Tree pose"
        status = "perfect"
        pose_label = "Correct Tree"
    elif checks["strict_tree_gate"] and score >= 75:
        main_feedback = priority_feedback[0] if priority_feedback else "Good Tree pose"
        status = "good"
        pose_label = "Tree Pose"
    elif score >= 50:
        main_feedback = priority_feedback[0] if priority_feedback else "You are close. Make small corrections."
        status = "warning"
        pose_label = "Tree Needs Correction"
    else:
        main_feedback = priority_feedback[0] if priority_feedback else "Stand in Tree pose"
        status = "warning"
        pose_label = "Not Ready Yet"

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": priority_feedback[:4],
        "standing_side": standing_side,
        "stand_knee_idx": stand_knee_idx,
        "bent_knee_idx": bent_knee_idx,
        "stand_ankle_idx": stand_ankle_idx,
        "bent_ankle_idx": bent_ankle_idx,
        "angles": {
            "left_knee_angle": round(left_knee, 2),
            "right_knee_angle": round(right_knee, 2),
            "left_elbow_angle": round(left_elbow, 2),
            "right_elbow_angle": round(right_elbow, 2),
            "left_hip_angle": round(left_hip, 2),
            "right_hip_angle": round(right_hip, 2),
            "torso_tilt": round(torso_tilt, 2),
        },
        "stand_angle": float(stand_angle),
        "bent_angle": float(bent_angle),
        "hands_up": bool(hands_ok),
        "checks": checks,
    }


def is_tree_like(model_label, model_confidence, analysis):
    label = str(model_label).lower()
    checks = analysis["checks"]

    if not checks["strict_tree_gate"]:
        return False

    if "tree" in label and model_confidence >= 0.52:
        return True

    if (
        checks["standing_leg"] and
        checks["bent_leg"] and
        checks["one_foot_lifted"] and
        checks["foot_place"] and
        checks["knee_open"]
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
    global TREE_HOLD_START, BEST_HOLD_TIME

    valid_tree = (
        is_tree and
        defect_label in ["Perfect_Tree", "Bent_Support_Leg", "Low_Hands", "Torso_Lean", "Poor_Balance", "N/A"] and
        full_body_visible and
        not low_light
    )

    if valid_tree:
        if TREE_HOLD_START is None:
            TREE_HOLD_START = time.time()
        hold_time = time.time() - TREE_HOLD_START
        BEST_HOLD_TIME = max(BEST_HOLD_TIME, hold_time)
    else:
        hold_time = 0.0
        TREE_HOLD_START = None

    return hold_time, BEST_HOLD_TIME


def hold_bonus(hold_time):
    if hold_time >= 10:
        return 10
    if hold_time >= 7:
        return 7
    if hold_time >= 5:
        return 5
    if hold_time >= 3:
        return 2
    return 0


def choose_quality_label(analysis, defect_label, defect_confidence):
    checks = analysis["checks"]

    if analysis["score"] >= 92 and checks["balance"] and checks["torso"] and checks["strict_tree_gate"]:
        return "Perfect_Tree"

    if defect_confidence >= 0.66 and defect_label != "N/A":
        if defect_label == "Poor_Balance" and checks["balance"] and analysis["score"] >= 82:
            return "Perfect_Tree"
        return defect_label

    if not checks["standing_leg"]:
        return "Bent_Support_Leg"
    if not checks["hands_up"]:
        return "Low_Hands"
    if not checks["torso"]:
        return "Torso_Lean"
    if not checks["balance"]:
        return "Poor_Balance"

    return "Perfect_Tree"


# =========================================================
# FRONTEND OVERLAY HELPERS
# =========================================================
def build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    stand_knee_idx = analysis["stand_knee_idx"]
    bent_knee_idx = analysis["bent_knee_idx"]
    stand_ankle_idx = analysis["stand_ankle_idx"]
    bent_ankle_idx = analysis["bent_ankle_idx"]

    joint_colors = {idx: GRAY for idx in range(len(raw_pts))}
    joint_radius = {idx: 5 for idx in range(len(raw_pts))}

    for idx in SELECTED_POINTS:
        if idx < len(raw_pts):
            joint_colors[idx] = GREEN

    joint_colors[stand_knee_idx] = GREEN if checks["standing_leg"] else RED
    joint_colors[bent_knee_idx] = GREEN if (checks["bent_leg"] and checks["knee_open"]) else RED
    joint_colors[stand_ankle_idx] = GREEN if checks["standing_leg"] else RED
    joint_colors[bent_ankle_idx] = GREEN if checks["foot_place"] else RED

    joint_colors[LEFT_ELBOW] = GREEN if checks["arms_straight"] else RED
    joint_colors[RIGHT_ELBOW] = GREEN if checks["arms_straight"] else RED
    joint_colors[LEFT_WRIST] = GREEN if checks["hands_up"] else RED
    joint_colors[RIGHT_WRIST] = GREEN if checks["hands_up"] else RED
    joint_colors[LEFT_SHOULDER] = GREEN if checks["shoulders"] else RED
    joint_colors[RIGHT_SHOULDER] = GREEN if checks["shoulders"] else RED
    joint_colors[LEFT_HIP] = GREEN if checks["hips"] else RED
    joint_colors[RIGHT_HIP] = GREEN if checks["hips"] else RED

    points = []
    visible_ids = set()

    for idx in SELECTED_POINTS:
        lm = landmarks[idx]
        visibility = float(lm.visibility)

        cutoff = 0.46
        if idx in [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]:
            cutoff = 0.34

        if visibility < cutoff:
            continue

        # IMPORTANT:
        # Send raw normalized coordinates directly.
        # Frontend already mirrors preview, so backend must not mirror x.
        x = float(raw_pts[idx][0])
        y = float(raw_pts[idx][1])

        sx, sy = adaptive_smooth_point(f"joint_{idx}", x, y)
        visible_ids.add(f"joint_{idx}")

        points.append({
            "name": POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": clip01(sx),
            "y": clip01(sy),
            "color": joint_colors.get(idx, GRAY),
            "radius": joint_radius.get(idx, 5),
            "visible": True,
        })

    stale_keys = [k for k in list(POINT_HISTORY.keys()) if k.startswith("joint_") and k not in visible_ids]
    for k in stale_keys:
        POINT_HISTORY.pop(k, None)

    return points


def build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    mapping = [
        (LEFT_KNEE, analysis["angles"]["left_knee_angle"]),
        (RIGHT_KNEE, analysis["angles"]["right_knee_angle"]),
        (LEFT_ELBOW, analysis["angles"]["left_elbow_angle"]),
        (RIGHT_ELBOW, analysis["angles"]["right_elbow_angle"]),
    ]

    for idx, value in mapping:
        lm = landmarks[idx]
        if float(lm.visibility) < 0.45:
            continue

        sx, sy = adaptive_smooth_point(f"angle_{idx}", raw_pts[idx][0], raw_pts[idx][1])
        items.append({
            "text": str(int(value)),
            "x": clip01(sx),
            "y": clip01(sy),
            "color": YELLOW
        })

    return items


# =========================================================
# MAIN API
# =========================================================
@csrf_exempt
def predict_yoga_pose(request):
    load_models()
    global PERFECT_HOLD_COUNT

    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    try:
        uploaded_file = request.FILES["image"]
        frame = read_uploaded_image(uploaded_file)

        if frame is None:
            return api_error("Invalid image file", status=400)

        frame = enhance_frame(frame)

        low_light, brightness = check_lighting(frame)
        if low_light:
            PERFECT_HOLD_COUNT = 0
            DEFECT_HISTORY.clear()
            POSE_HISTORY.clear()
            SCORE_HISTORY.clear()
            FEEDBACK_HISTORY.clear()
            clear_point_history()

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
                best_hold_time=round(BEST_HOLD_TIME, 1),
                angles={},
                details=[
                    "Increase room lighting",
                    "Face the light source",
                    "Avoid dark background"
                ],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        landmarks = detect_landmarks(frame)
        if not landmarks:
            PERFECT_HOLD_COUNT = 0
            DEFECT_HISTORY.clear()
            POSE_HISTORY.clear()
            SCORE_HISTORY.clear()
            FEEDBACK_HISTORY.clear()
            clear_point_history()

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
                best_hold_time=round(BEST_HOLD_TIME, 1),
                angles={},
                details=[
                    "Show full body",
                    "Improve room lighting",
                    "Stay centered in frame"
                ],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        features_df, _, _ = build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        full_body_visible, visible_count, avg_visibility = check_body_visibility(
            extract_raw_landmark_dict(landmarks)
        )
        framing_feedback = check_frame_position(raw_pts)

        if not full_body_visible:
            PERFECT_HOLD_COUNT = 0
            DEFECT_HISTORY.clear()
            POSE_HISTORY.clear()
            SCORE_HISTORY.clear()
            FEEDBACK_HISTORY.clear()
            clear_point_history()

            details = [
                "Show full body clearly",
                "Keep both feet visible",
                "Stand in the center of the camera"
            ]
            details.extend(framing_feedback)

            return api_success(
                pose="Body Not Visible",
                model_pose="Unknown",
                quality="N/A",
                feedback="Move a little back. Full body should be visible.",
                coach_text="Show your complete body in the frame.",
                status="warning",
                confidence=0.0,
                defect_confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(BEST_HOLD_TIME, 1),
                angles={},
                details=dedupe_text_list(details, max_items=3),
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        predicted_label, confidence = predict_pose_label(features_df)
        stable_predicted_label = smooth_label(POSE_HISTORY, predicted_label)

        analysis = analyze_tree_pose(raw_pts)
        tree_like = is_tree_like(stable_predicted_label, confidence, analysis)

        if tree_like:
            raw_defect_label, defect_confidence = predict_defect_label(features_df)
            defect_label = smooth_label(DEFECT_HISTORY, raw_defect_label)
            defect_label = choose_quality_label(analysis, defect_label, defect_confidence)
        else:
            defect_label = "N/A"
            defect_confidence = 0.0
            DEFECT_HISTORY.clear()

        update_stability_metrics(raw_pts)
        stability_tips, stability_penalty = get_stability_feedback()

        base_score = analysis["score"]
        if defect_label != "N/A":
            model_defect_score = calculate_defect_score(defect_label)
            combined_score = int(round((base_score * 0.72) + (model_defect_score * 0.28)))
        else:
            combined_score = base_score

        combined_score -= stability_penalty
        combined_score += hold_bonus(0.0)
        combined_score = max(0, min(100, combined_score))

        hold_time, best_hold = update_hold_state(tree_like, defect_label, full_body_visible, low_light)
        combined_score = max(0, min(100, combined_score + hold_bonus(hold_time)))

        stable_score = smooth_score(combined_score)

        if not tree_like:
            PERFECT_HOLD_COUNT = 0
            SCORE_HISTORY.clear()
            FEEDBACK_HISTORY.clear()

            details = [
                "Lift one foot onto the opposite inner leg",
                "Open the bent knee outward",
                "Bring hands to prayer or overhead"
            ]
            details.extend(framing_feedback)

            return api_success(
                pose="Not Tree Pose",
                model_pose=stable_predicted_label,
                quality="N/A",
                feedback="Stand in a proper Tree pose first.",
                coach_text="Lift one foot onto the opposite inner leg to enter Tree pose.",
                status="warning",
                confidence=round(float(confidence), 3),
                defect_confidence=0.0,
                score=0,
                hold_time=0.0,
                best_hold_time=round(float(best_hold), 1),
                angles=analysis["angles"],
                details=dedupe_text_list(details, max_items=3),
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        if defect_label == "Perfect_Tree" and analysis["checks"]["strict_tree_gate"] and stable_score >= 90:
            pose_name = "Correct Tree"
            stable_status = "perfect"
            feedback_text = "Correct Tree pose"
            coach_text = "Excellent. Hold steady."
        elif analysis["checks"]["strict_tree_gate"] and stable_score >= 75:
            pose_name = "Tree Pose"
            stable_status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = analysis["tips"][0] if analysis["tips"] else "Good form. Hold the pose."
        else:
            pose_name = "Tree Needs Correction"
            stable_status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = analysis["tips"][0] if analysis["tips"] else "Adjust posture more."

        # Never allow 'Correct Tree pose' if hands are wrong
        if not analysis["checks"]["hands_up"]:
            pose_name = "Tree Needs Correction"
            stable_status = "warning"
            feedback_text = "Raise your hands overhead or bring them to prayer"
            coach_text = "Raise your hands overhead or bring them to prayer"

        # Never allow 'Correct Tree pose' if foot placement is wrong
        if not analysis["checks"]["foot_place"]:
            pose_name = "Tree Needs Correction"
            stable_status = "warning"
            feedback_text = "Place the raised foot on the opposite inner leg"
            coach_text = "Place the raised foot on the opposite inner calf or inner thigh"

        stable_feedback = smooth_feedback(feedback_text)
        if stability_tips:
            coach_text = stability_tips[0]

        stable_feedback = smooth_feedback(feedback_text)

        if stable_score >= 95 and hold_time >= 2.0 and defect_label == "Perfect_Tree":
            PERFECT_HOLD_COUNT += 1
        else:
            PERFECT_HOLD_COUNT = 0

        if PERFECT_HOLD_COUNT >= 2:
            pose_name = "Correct Tree"
            stable_status = "perfect"
            stable_feedback = "Correct Tree pose"
            coach_text = "Excellent. Hold steady."

        tips = []
        if hold_time >= 5:
            tips.append(f"Great hold! {hold_time:.1f}s")
        tips.extend(get_defect_tips(defect_label))
        tips.extend(analysis["tips"])
        tips.extend(stability_tips)
        tips.extend(framing_feedback)

        cleaned_tips = dedupe_text_list(
            tips,
            max_items=3,
            exclude=[stable_feedback, coach_text]
        )

        points = build_points_for_frontend(raw_pts, landmarks, analysis)
        angle_texts = build_angle_texts(raw_pts, landmarks, analysis)

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
            hold_time=round(float(hold_time), 1),
            best_hold_time=round(float(best_hold), 1),
            angles=analysis["angles"],
            details=cleaned_tips,
            perfect_hold=PERFECT_HOLD_COUNT >= 2,
            points=points,
            angle_texts=angle_texts,
        )

    except Exception as e:
        print("predict_yoga_pose error:", str(e))
        return api_error(str(e), status=500)