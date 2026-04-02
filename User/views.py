from pathlib import Path
from collections import deque, Counter
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle


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
POINT_HISTORY_SIZE = 7


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


def smooth_point(key, x, y):
    if key not in POINT_HISTORY:
        POINT_HISTORY[key] = deque(maxlen=POINT_HISTORY_SIZE)

    POINT_HISTORY[key].append((float(x), float(y)))

    xs = [p[0] for p in POINT_HISTORY[key]]
    ys = [p[1] for p in POINT_HISTORY[key]]

    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


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
    # keep original size for perfect overlay mapping
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
        "pose_label": "Not Ready", "score": 0, "status": "warning",
        "main_feedback": msg, "tips": ["Keep your standing leg completely straight."], "standing_side": "none",
        "stand_knee_idx": LEFT_KNEE, "bent_knee_idx": RIGHT_KNEE, 
        "stand_ankle_idx": LEFT_ANKLE, "bent_ankle_idx": RIGHT_ANKLE,
        "angles": {"left_knee_angle": 0, "right_knee_angle": 0, "torso_tilt": 0},
        "checks": {"strict_tree_gate": False}
    }


# =========================================================
# TREE ANALYSIS (PROPORTIONAL COACHING FIX)
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

    # -----------------------------------------------------
    # MAIN JOINT ANGLES
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # FOOT / LEG CHECKS
    # -----------------------------------------------------
    foot_lift_height = abs(float(bent_ankle[1] - stand_ankle[1])) / torso_size
    one_foot_lifted = foot_lift_height > 0.18

    foot_vs_knee_y = abs(float(bent_ankle[1] - stand_knee[1])) / torso_size
    not_on_knee_joint = foot_vs_knee_y > 0.14

    foot_to_knee_x = abs(float(bent_ankle[0] - stand_knee[0])) / torso_size
    foot_to_hip_x = abs(float(bent_ankle[0] - stand_hip[0])) / torso_size
    foot_close_to_leg = min(foot_to_knee_x, foot_to_hip_x) < 0.45

    ankle_to_stand_ankle_y = float(stand_ankle[1] - bent_ankle[1]) / torso_size
    ankle_to_stand_hip_y = float(stand_hip[1] - bent_ankle[1]) / torso_size

    foot_zone = "unknown"
    if ankle_to_stand_ankle_y < 0.14:
        foot_zone = "low"
    elif 0.14 <= ankle_to_stand_ankle_y <= 0.82:
        foot_zone = "calf"
    elif 0.02 <= ankle_to_stand_hip_y <= 0.72:
        foot_zone = "thigh"

    foot_height_ok = foot_zone in ["calf", "thigh"]

    # -----------------------------------------------------
    # KNEE / STANDING LEG
    # -----------------------------------------------------
    knee_open_distance = abs(float(bent_knee[0] - stand_hip[0])) / torso_size
    knee_open_ok = knee_open_distance > 0.20 and bent_angle < 148

    standing_leg_ok = stand_angle >= 158

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

    # -----------------------------------------------------
    # TORSO
    # -----------------------------------------------------
    vertical_ref = hip_center + np.array([0.0, -0.30, 0.0], dtype=np.float32)
    torso_tilt = calculate_angle(shoulder_center[:2], hip_center[:2], vertical_ref[:2])
    torso_ok = torso_tilt <= 14.0

    # -----------------------------------------------------
    # HANDS / ARMS
    # -----------------------------------------------------
    wrist_distance = float(np.linalg.norm(lw[:2] - rw[:2])) / torso_size
    wrist_height_diff = abs(float(lw[1] - rw[1])) / torso_size

    left_wrist_above_left_shoulder = lw[1] < ls[1] - 0.10
    right_wrist_above_right_shoulder = rw[1] < rs[1] - 0.10

    elbows_straight = left_elbow_angle > 150 and right_elbow_angle > 150
    elbows_soft_ok = left_elbow_angle > 138 and right_elbow_angle > 138

    hands_symmetric = wrist_height_diff < 0.10
    hands_not_too_wide = wrist_distance < 0.35
    hands_close_for_prayer = wrist_distance < 0.12

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
        elbows_straight
    )

    hands_ready = prayer_hands or hands_up

    # -----------------------------------------------------
    # FINAL STRICT TREE GATE
    # -----------------------------------------------------
    strict_tree_gate = (
        not is_neutral_standing and
        one_foot_lifted and
        foot_place_ok and
        knee_open_ok and
        standing_leg_ok and
        torso_ok and
        hands_ready
    )

    # -----------------------------------------------------
    # COACHING
    # -----------------------------------------------------
    pose_label = "Not Tree Pose"
    status = "warning"
    score = 0
    main_f = "Lift one foot onto the opposite inner leg."
    tips = ["Stand tall.", "Shift weight to one leg."]

    if is_neutral_standing:
        pose_label = "Standing Still"
        score = 10
        main_f = "Step 1: Shift weight onto one leg and lift the other foot."
        tips = ["Pick one standing leg.", "Lift the other foot off the floor."]

    elif not one_foot_lifted:
        score = 25
        main_f = "Step 1: Lift your foot higher off the floor."
        tips = ["Bring the foot toward the opposite inner leg."]

    elif not not_on_knee_joint:
        score = 42
        main_f = "Step 2: Move your foot away from the knee joint."
        tips = ["Place the foot on inner calf or thigh, never on the knee."]

    elif not foot_place_ok:
        score = 55
        if foot_zone == "low":
            main_f = "Step 2: Place the foot a little higher on calf or thigh."
            tips = ["Your foot is lifted correctly. Now set it on calf or thigh."]
        elif not foot_close_to_leg:
            main_f = "Step 2: Bring the lifted foot closer to the standing leg."
            tips = ["Press the foot gently into the inner leg."]
        else:
            main_f = "Step 2: Keep the foot stable on the inner leg."
            tips = ["Hold the lifted foot steady and avoid sliding."]

    elif not knee_open_ok:
        score = 66
        main_f = "Step 3: Open the bent knee more to the side."
        tips = ["Rotate the hip outward.", "Let the knee point sideways."]

    elif not standing_leg_ok:
        score = 76
        main_f = "Step 4: Straighten the standing leg."
        tips = ["Press down through the grounded foot.", "Keep the support leg long and firm."]

    elif not torso_ok:
        score = 84
        main_f = "Step 5: Stand taller through your spine."
        tips = ["Keep shoulders over hips.", "Look straight ahead."]

    elif not hands_ready:
        score = 75
        status = "warning"
        pose_label = "Tree Pose"
        main_f = "Step 6: Bring your hands to prayer or raise them overhead correctly."
        tips = ["Your leg position is good.", "Now correct the hand and arm position."]

    elif hands_up and not elbows_straight:
        score = 78
        status = "warning"
        pose_label = "Tree Pose"
        main_f = "Straighten both elbows more when raising your hands overhead."
        tips = ["Reach upward through both arms.", "Keep both elbows long and balanced."]

    else:
        score = 100
        status = "perfect"
        pose_label = "Correct Tree"
        main_f = "Perfect Pose! Hold steady and breathe."
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
        checks.get("elbows_straight") and
        checks.get("one_foot_lifted")
    ):
        return True

    return False

# =========================================================
# DEFECT / COACHING / STABILITY
# =========================================================
def calculate_defect_score(defect_label):
    score = 100
    penties = {
        "Perfect_Tree": 0,
        "Bent_Support_Leg": 18,
        "Low_Hands": 12,
        "Torso_Lean": 18,
        "Poor_Balance": 12,
    }
    score -= penties.get(defect_label, 18)
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
    if hold_time >= 10: return 10
    if hold_time >= 7: return 7
    if hold_time >= 5: return 5
    if hold_time >= 3: return 2
    return 0


def choose_quality_label(analysis, defect_label, defect_confidence):
    checks = analysis["checks"]

    if (
        analysis["score"] >= 90 and
        checks.get("torso") and
        checks.get("strict_tree_gate") and
        checks.get("hands_ready") and
        checks.get("elbows_straight")
    ):
        return "Perfect_Tree"

    if defect_confidence >= 0.65 and defect_label != "N/A":
        return defect_label

    if not checks.get("standing_leg"):
        return "Bent_Support_Leg"

    if not checks.get("hands_ready") or not checks.get("elbows_straight"):
        return "Low_Hands"

    if not checks.get("torso"):
        return "Torso_Lean"

    return "Perfect_Tree"

# =========================================================
# FRONTEND OVERLAY HELPERS (VISIBILITY FIX)
# =========================================================
def build_points_for_frontend(raw_pts, landmarks, analysis):
    checks = analysis["checks"]
    stand_knee_idx = analysis.get("stand_knee_idx", LEFT_KNEE)
    bent_knee_idx = analysis.get("bent_knee_idx", RIGHT_KNEE)
    stand_ankle_idx = analysis.get("stand_ankle_idx", LEFT_ANKLE)
    bent_ankle_idx = analysis.get("bent_ankle_idx", RIGHT_ANKLE)

    points = []
    for idx in SELECTED_POINTS:
        lm = landmarks[idx]
        visibility = float(lm.visibility)

        if visibility < 0.18:
            continue

        color = GREEN if analysis["score"] >= 86 else YELLOW
        radius = 6

        if idx == stand_knee_idx and not checks.get("standing_leg"):
            color = RED
            radius = 7
        if idx == bent_knee_idx and not checks.get("knee_open"):
            color = RED
            radius = 7
        if idx == bent_ankle_idx and not checks.get("foot_place"):
            color = RED
            radius = 7
        if idx == stand_ankle_idx and not checks.get("balance"):
            color = YELLOW
            radius = 7
        if idx in [LEFT_WRIST, RIGHT_WRIST] and not checks.get("hands_up") and analysis["score"] >= 70:
            color = RED
            radius = 7

        points.append({
            "name": POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": clip01(raw_pts[idx][0]),
            "y": clip01(raw_pts[idx][1]),
            "color": color,
            "radius": radius,
            "visible": True,
            "visibility": round(visibility, 3),
        })

    return points

def build_angle_texts(raw_pts, landmarks, analysis):
    items = []
    mapping = [
        (LEFT_KNEE, analysis.get("angles", {}).get("left_knee_angle", 0)),
        (RIGHT_KNEE, analysis.get("angles", {}).get("right_knee_angle", 0)),
        (LEFT_ELBOW, analysis.get("angles", {}).get("left_elbow_angle", 0)),
        (RIGHT_ELBOW, analysis.get("angles", {}).get("right_elbow_angle", 0)),
    ]

    for idx, value in mapping:
        lm = landmarks[idx]
        if float(lm.visibility) < 0.18:
            continue

        items.append({
            "text": f"{int(round(float(value)))}°",
            "x": clip01(raw_pts[idx][0]),
            "y": clip01(raw_pts[idx][1]),
            "color": YELLOW,
        })

    return items
# =========================================================
# MAIN API
# =========================================================
@csrf_exempt
def predict_yoga_pose(request):
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
                details=["Increase room lighting", "Face the light source", "Avoid dark background"],
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
                details=["Show full body", "Improve room lighting", "Stay centered in frame"],
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        features_df, _, _ = build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        lm_dict = extract_raw_landmark_dict(landmarks)
        full_body_visible, visible_count, avg_visibility = check_body_visibility(lm_dict)
        framing_feedback = check_frame_position(raw_pts)

        if not full_body_visible:
            PERFECT_HOLD_COUNT = 0
            DEFECT_HISTORY.clear()
            POSE_HISTORY.clear()
            SCORE_HISTORY.clear()
            FEEDBACK_HISTORY.clear()
            clear_point_history()

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
                best_hold_time=round(float(BEST_HOLD_TIME), 1),
                angles={},
                details=dedupe_text_list(details, max_items=3),
                perfect_hold=False,
                points=[],
                angle_texts=[],
            )

        predicted_label, confidence = predict_pose_label(features_df)
        stable_predicted_label = smooth_label(POSE_HISTORY, predicted_label)

        analysis = analyze_tree_pose(raw_pts)
        points = build_points_for_frontend(raw_pts, landmarks, analysis)
        angle_texts = build_angle_texts(raw_pts, landmarks, analysis)

        tree_like = is_tree_like(stable_predicted_label, confidence, analysis)

        if not tree_like:
            DEFECT_HISTORY.clear()
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
                coach_text=analysis.get("main_feedback", "Lift one foot onto the opposite inner leg to enter Tree pose."),
                status="warning",
                confidence=round(float(confidence), 3),
                defect_confidence=0.0,
                score=max(0, min(45, analysis.get("score", 0))),
                hold_time=0.0,
                best_hold_time=round(float(BEST_HOLD_TIME), 1),
                angles=analysis.get("angles", {}),
                details=dedupe_text_list(tips, max_items=3),
                perfect_hold=False,
                points=points,
                angle_texts=angle_texts,
            )

        raw_defect_label, raw_defect_confidence = predict_defect_label(features_df)

        if raw_defect_confidence >= 0.58:
            stable_defect = smooth_label(DEFECT_HISTORY, raw_defect_label)
            defect_label = choose_quality_label(analysis, stable_defect, raw_defect_confidence)
            defect_confidence = raw_defect_confidence
        else:
            defect_label = choose_quality_label(analysis, "N/A", 0.0)
            defect_confidence = 0.0

        update_stability_metrics(raw_pts)
        stability_tips, stability_penalty = get_stability_feedback()

        rule_score = analysis["score"]
        defect_score = calculate_defect_score(defect_label)
        combined_score = int(round((rule_score * 0.82) + (defect_score * 0.18)))
        combined_score = max(0, combined_score - stability_penalty)

        hold_time, best_hold = update_hold_state(
            is_tree=True,
            defect_label=defect_label,
            full_body_visible=full_body_visible,
            low_light=low_light
        )

        combined_score = min(100, combined_score + hold_bonus(hold_time))
        stable_score = smooth_score(combined_score)

        if stable_score >= 94 and analysis["checks"].get("strict_tree_gate") and defect_label in ["Perfect_Tree", "N/A"]:
            pose_name = "Correct Tree"
            stable_status = "perfect"
            feedback_text = "Correct Tree pose"
            coach_text = "Excellent. Hold steady."
        elif stable_score >= 80:
            pose_name = "Tree Pose"
            stable_status = "good"
            feedback_text = analysis["main_feedback"]
            coach_text = "Good pose. Refine one small correction."
        elif stable_score >= 58:
            pose_name = "Tree Needs Correction"
            stable_status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "You are close. Make one or two corrections."
        else:
            pose_name = "Not Ready Yet"
            stable_status = "warning"
            feedback_text = analysis["main_feedback"]
            coach_text = "Enter Tree pose and hold steady."

        stable_feedback = smooth_feedback(feedback_text)

        if (
            stable_score >= 96 and
            hold_time >= 2.5 and
            defect_label == "Perfect_Tree" and
            analysis["checks"].get("strict_tree_gate")
        ):
            PERFECT_HOLD_COUNT += 1
        else:
            PERFECT_HOLD_COUNT = 0

        if PERFECT_HOLD_COUNT >= 3 and analysis["checks"].get("strict_tree_gate"):
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
            angles=analysis.get("angles", {}),
            details=cleaned_tips,
            perfect_hold=PERFECT_HOLD_COUNT >= 3,
            points=points,
            angle_texts=angle_texts,
        )

    except Exception as e:
        print("predict_yoga_pose error:", str(e))
        return api_error(str(e), status=500)
    












# =========================================================
# DOWN DOG - MEDIAPIPE SETUP
# =========================================================
mp_pose = mp.solutions.pose
down_dog_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
)


# =========================================================
# DOWN DOG - MAIN LANDMARK INDEXES USED FOR ANALYSIS/OVERLAY
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
# DOWN DOG - FULL 33 LANDMARKS FOR MODEL INPUT (138 FEATURES)
# 33 landmarks * 4 values (x,y,z,visibility) = 132
# + 6 angle features = 138
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

DD_GREEN = "#00ff66"
DD_RED = "#ff3b30"
DD_YELLOW = "#ffd60a"


# =========================================================
# DOWN DOG - MODEL LOAD
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
# DOWN DOG - PAGE
# =========================================================
def down_dog_live_page(request):
    return render(request, "User/downdog_camera.html")


# =========================================================
# DOWN DOG - RESPONSE HELPERS
# =========================================================
def down_dog_api_success(**kwargs):
    return JsonResponse({"success": True, **kwargs})


def down_dog_api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


# =========================================================
# DOWN DOG - BASIC HELPERS
# =========================================================
def dd_clean_text(text):
    return " ".join(str(text).strip().split())


def dd_dedupe_list(items, max_items=None):
    output = []
    seen = set()

    for item in items:
        if not item:
            continue

        text = dd_clean_text(item)
        key = text.lower()

        if not key or key in seen:
            continue

        seen.add(key)
        output.append(text)

        if max_items and len(output) >= max_items:
            break

    return output


def dd_read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def dd_enhance_frame(frame):
    return cv2.convertScaleAbs(frame, alpha=1.03, beta=4)


def dd_detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = down_dog_pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    return results.pose_landmarks.landmark


def dd_check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 50, brightness


def dd_clip01(value):
    return float(np.clip(value, 0.0, 1.0))


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


# =========================================================
# DOWN DOG - FEATURE CREATION
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
    features_array = features_df.to_numpy(dtype=np.float32)
    scaled_features = downdog_scaler.transform(features_array)

    prediction = downdog_model.predict(scaled_features)[0]

    confidence = 0.50
    if hasattr(downdog_model, "predict_proba"):
        probs = downdog_model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probs))

    return str(prediction), confidence


# =========================================================
# DOWN DOG - VISIBILITY / FRAMING
# =========================================================
def dd_check_body_visibility(lm_dict):
    important_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    visibilities = [lm_dict[name]["visibility"] for name in important_names]
    visible_count = sum(v > 0.30 for v in visibilities)
    avg_visibility = float(np.mean(visibilities))

    full_body_visible = visible_count >= 10
    return full_body_visible, visible_count, avg_visibility


def dd_check_frame_position(raw_pts):
    xs = [float(p[0]) for p in raw_pts[DD_SELECTED_POINTS]]
    ys = [float(p[1]) for p in raw_pts[DD_SELECTED_POINTS]]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    feedback = []

    if width > 0.97:
        feedback.append("Move slightly away from the camera")

    if height > 0.95:
        feedback.append("Show more of your full body in frame")

    center_x = (min_x + max_x) / 2
    if center_x < 0.18:
        feedback.append("Move slightly to the right")
    elif center_x > 0.82:
        feedback.append("Move slightly to the left")

    return feedback


# =========================================================
# DOWN DOG - ANALYSIS
# =========================================================
def analyze_down_dog_pose(raw_pts):
    ls, rs = raw_pts[DD_LEFT_SHOULDER], raw_pts[DD_RIGHT_SHOULDER]
    le, re = raw_pts[DD_LEFT_ELBOW], raw_pts[DD_RIGHT_ELBOW]
    lw, rw = raw_pts[DD_LEFT_WRIST], raw_pts[DD_RIGHT_WRIST]
    lh, rh = raw_pts[DD_LEFT_HIP], raw_pts[DD_RIGHT_HIP]
    lk, rk = raw_pts[DD_LEFT_KNEE], raw_pts[DD_RIGHT_KNEE]
    la, ra = raw_pts[DD_LEFT_ANKLE], raw_pts[DD_RIGHT_ANKLE]

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    wrist_center = (lw + rw) / 2.0
    ankle_center = (la + ra) / 2.0

    torso_size = float(np.linalg.norm(shoulder_center[:2] - hip_center[:2])) + 1e-6

    left_elbow_angle = dd_calculate_angle(ls[:2], le[:2], lw[:2])
    right_elbow_angle = dd_calculate_angle(rs[:2], re[:2], rw[:2])
    left_knee_angle = dd_calculate_angle(lh[:2], lk[:2], la[:2])
    right_knee_angle = dd_calculate_angle(rh[:2], rk[:2], ra[:2])

    left_shoulder_open = dd_calculate_angle(le[:2], ls[:2], lh[:2])
    right_shoulder_open = dd_calculate_angle(re[:2], rs[:2], rh[:2])

    arms_straight = left_elbow_angle > 150 and right_elbow_angle > 150
    legs_long = left_knee_angle > 145 and right_knee_angle > 145
    hips_above_shoulders = hip_center[1] < shoulder_center[1] - 0.02
    hips_high_enough = (wrist_center[1] - hip_center[1]) / torso_size > 0.55

    head_between_arms = (
        raw_pts[DD_NOSE][0] > min(lw[0], rw[0]) - 0.08 and
        raw_pts[DD_NOSE][0] < max(lw[0], rw[0]) + 0.08
    )

    hands_far_from_feet = np.linalg.norm(wrist_center[:2] - ankle_center[:2]) / torso_size > 1.45
    back_length_good = np.linalg.norm(shoulder_center[:2] - hip_center[:2]) / torso_size > 0.95
    shoulder_open_ok = left_shoulder_open > 40 and right_shoulder_open > 40

    score = 0
    status = "warning"
    pose_label = "Not Down Dog"
    main_feedback = "Move into Downward Dog position."
    tips = ["Place both hands and feet firmly on the floor."]

    if not hips_above_shoulders:
        score = 35
        main_feedback = "Lift your hips higher."
        tips = ["Push the floor away and raise the hips upward."]
    elif not arms_straight:
        score = 52
        main_feedback = "Straighten both arms more."
        tips = ["Press evenly through both palms."]
    elif not legs_long:
        score = 64
        main_feedback = "Lengthen your legs more."
        tips = ["Gently straighten the knees as much as comfortable."]
    elif not shoulder_open_ok:
        score = 74
        main_feedback = "Open the shoulders more."
        tips = ["Send your chest gently toward your thighs."]
    elif not head_between_arms:
        score = 80
        main_feedback = "Keep your head relaxed between your arms."
        tips = ["Let the neck stay soft and natural."]
    elif not hands_far_from_feet:
        score = 84
        main_feedback = "Step the feet slightly farther back."
        tips = ["Create more length through the whole body."]
    elif not hips_high_enough:
        score = 88
        main_feedback = "Lift the hips a little more."
        tips = ["Push strongly through hands and feet."]
    elif not back_length_good:
        score = 92
        main_feedback = "Lengthen your spine more."
        tips = ["Reach hips up and chest back."]
    else:
        score = 100
        status = "perfect"
        pose_label = "Correct Down Dog"
        main_feedback = "Perfect Down Dog. Hold steady."
        tips = [
            "Excellent posture.",
            "Keep breathing steadily.",
            "Maintain the inverted V shape.",
        ]

    checks = {
        "arms_straight": arms_straight,
        "legs_long": legs_long,
        "hips_above_shoulders": hips_above_shoulders,
        "hips_high_enough": hips_high_enough,
        "head_between_arms": head_between_arms,
        "hands_far_from_feet": hands_far_from_feet,
        "back_length_good": back_length_good,
        "shoulder_open_ok": shoulder_open_ok,
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
        },
        "checks": checks,
    }


# =========================================================
# DOWN DOG - OVERLAY HELPERS
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

        if idx in [DD_LEFT_ELBOW, DD_RIGHT_ELBOW] and not checks.get("arms_straight"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_KNEE, DD_RIGHT_KNEE] and not checks.get("legs_long"):
            color = DD_RED
            radius = 7

        if idx in [DD_LEFT_HIP, DD_RIGHT_HIP] and not checks.get("hips_above_shoulders"):
            color = DD_RED
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
# DOWN DOG - MAIN API
# =========================================================
@csrf_exempt
def down_dog_live_api(request):
    if request.method != "POST":
        return down_dog_api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return down_dog_api_error("No image uploaded", status=400)

    try:
        uploaded_file = request.FILES["image"]
        frame = dd_read_uploaded_image(uploaded_file)

        if frame is None:
            return down_dog_api_error("Invalid image file", status=400)

        frame = dd_enhance_frame(frame)

        low_light, brightness = dd_check_lighting(frame)
        if low_light:
            return down_dog_api_success(
                pose="Low Light",
                model_pose="Unknown",
                quality="N/A",
                feedback="Room lighting is too low for accurate pose detection.",
                coach_text="Improve the lighting and try again.",
                status="warning",
                confidence=0.0,
                score=0,
                angles={},
                details=[
                    "Increase room lighting",
                    "Face the light source",
                    "Avoid dark background",
                ],
                points=[],
                angle_texts=[],
            )

        landmarks = dd_detect_landmarks(frame)
        if not landmarks:
            return down_dog_api_success(
                pose="Unknown",
                model_pose="Unknown",
                quality="N/A",
                feedback="No human pose detected.",
                coach_text="Show your full body clearly in the camera.",
                status="unknown",
                confidence=0.0,
                score=0,
                angles={},
                details=[
                    "Show full body",
                    "Stand where camera can see you clearly",
                    "Move slightly back",
                ],
                points=[],
                angle_texts=[],
            )

        features_df, lm_dict, angles = dd_build_feature_dataframe_from_landmarks(landmarks)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # optional debug
        print("DownDog feature count:", features_df.shape[1])

        full_body_visible, visible_count, avg_visibility = dd_check_body_visibility(lm_dict)
        framing_feedback = dd_check_frame_position(raw_pts)

        if not full_body_visible:
            details = [
                "Show full body clearly",
                "Keep hands, hips, knees, and ankles visible",
                "Move a little farther from the camera",
            ]
            details.extend(framing_feedback)

            return down_dog_api_success(
                pose="Body Not Visible",
                model_pose="Unknown",
                quality="N/A",
                feedback="Move back slightly so full body is visible.",
                coach_text="Show your whole Down Dog shape in the frame.",
                status="warning",
                confidence=0.0,
                score=0,
                angles={},
                details=dd_dedupe_list(details, max_items=3),
                points=[],
                angle_texts=[],
            )

        model_label, confidence = dd_predict_model_label(features_df)
        analysis = analyze_down_dog_pose(raw_pts)

        points = dd_build_points_for_frontend(raw_pts, landmarks, analysis)
        angle_texts = dd_build_angle_texts(raw_pts, landmarks, analysis)

        pose_name = analysis["pose_label"]
        status = analysis["status"]
        score = analysis["score"]

        if score >= 95:
            quality = "Perfect_DownDog"
        elif score >= 80:
            quality = "Good_DownDog"
        elif score >= 60:
            quality = "Needs_Correction"
        else:
            quality = "Not_Ready"

        details = []
        details.extend(analysis["tips"])
        details.extend(framing_feedback)

        return down_dog_api_success(
            pose=pose_name,
            model_pose=model_label,
            quality=quality,
            feedback=analysis["main_feedback"],
            coach_text=analysis["main_feedback"],
            status=status,
            confidence=round(float(confidence), 3),
            score=score,
            angles=analysis["angles"],
            details=dd_dedupe_list(details, max_items=3),
            points=points,
            angle_texts=angle_texts,
        )

    except Exception as e:
        import traceback
        print("down_dog_live_api error:", str(e))
        traceback.print_exc()

        return JsonResponse({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }, status=500)