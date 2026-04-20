from pathlib import Path
from collections import deque, Counter
from dataclasses import dataclass, field
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
GODDESS_POSE_HISTORY_SIZE = 5
GODDESS_DEFECT_HISTORY_SIZE = 5
GODDESS_SCORE_HISTORY_SIZE = 6
GODDESS_FEEDBACK_HISTORY_SIZE = 6
GODDESS_STABILITY_HISTORY_SIZE = 15
GODDESS_DETECTION_HISTORY_SIZE = 5
GODDESS_VISIBILITY_HISTORY_SIZE = 5
GODDESS_POINT_HISTORY_SIZE = 5
GODDESS_BOOLEAN_HISTORY_SIZE = 4

GODDESS_KNEE_BEND_MIN = 85.0
GODDESS_KNEE_BEND_MAX = 145.0
GODDESS_IDEAL_KNEE_MIN = 95.0
GODDESS_IDEAL_KNEE_MAX = 130.0
GODDESS_HIP_DEPTH_MAX = 155.0
GODDESS_STANCE_WIDTH_MIN_RATIO = 1.35
GODDESS_IDEAL_STANCE_WIDTH_RATIO = 1.55
GODDESS_SHOULDER_LEVEL_MAX_DIFF = 0.09
GODDESS_TORSO_CENTER_MAX_OFFSET = 0.35
GODDESS_ELBOW_MIN_ANGLE = 70.0
GODDESS_ELBOW_MAX_ANGLE = 125.0
GODDESS_SHOULDER_MIN_ANGLE = 55.0
GODDESS_SHOULDER_MAX_ANGLE = 145.0
GODDESS_POINT_VISIBILITY_MIN = 0.22
GODDESS_PRAYER_HANDS_MAX_WIDTH_RATIO = 0.42
GODDESS_PRAYER_MIDLINE_MAX_OFFSET = 0.28
GODDESS_PRAYER_WRIST_HEIGHT_MIN_RATIO = -0.08
GODDESS_PRAYER_WRIST_HEIGHT_MAX_RATIO = 1.25
GODDESS_PRAYER_CHEST_DISTANCE_MAX_RATIO = 0.88
GODDESS_PRAYER_BALANCE_MAX_RATIO = 0.24
GODDESS_PRAYER_ELBOW_MAX_ANGLE = 165.0

GODDESS_SESSION_RUNTIME_KEY = "goddess_runtime_v1"


@dataclass
class GoddessRuntime:
    pose_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_POSE_HISTORY_SIZE))
    defect_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_DEFECT_HISTORY_SIZE))
    score_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_SCORE_HISTORY_SIZE))
    feedback_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_FEEDBACK_HISTORY_SIZE))
    torso_center_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_STABILITY_HISTORY_SIZE))
    shoulder_tilt_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_STABILITY_HISTORY_SIZE))
    detection_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_DETECTION_HISTORY_SIZE))
    visibility_history: deque = field(default_factory=lambda: deque(maxlen=GODDESS_VISIBILITY_HISTORY_SIZE))
    point_history: dict = field(default_factory=dict)
    boolean_histories: dict = field(default_factory=dict)


def goddess_runtime_to_session_data(runtime):
    return {
        "pose_history": list(runtime.pose_history),
        "defect_history": list(runtime.defect_history),
        "score_history": list(runtime.score_history),
        "feedback_history": list(runtime.feedback_history),
        "torso_center_history": list(runtime.torso_center_history),
        "shoulder_tilt_history": list(runtime.shoulder_tilt_history),
        "detection_history": list(runtime.detection_history),
        "visibility_history": list(runtime.visibility_history),
        "point_history": {
            str(key): [[float(x), float(y), float(z)] for x, y, z in points]
            for key, points in runtime.point_history.items()
        },
        "boolean_histories": {
            str(key): [bool(value) for value in values]
            for key, values in runtime.boolean_histories.items()
        },
    }


def goddess_runtime_from_session_data(data):
    runtime = GoddessRuntime()
    if not isinstance(data, dict):
        return runtime

    runtime.pose_history = deque(data.get("pose_history", []), maxlen=GODDESS_POSE_HISTORY_SIZE)
    runtime.defect_history = deque(data.get("defect_history", []), maxlen=GODDESS_DEFECT_HISTORY_SIZE)
    runtime.score_history = deque(data.get("score_history", []), maxlen=GODDESS_SCORE_HISTORY_SIZE)
    runtime.feedback_history = deque(data.get("feedback_history", []), maxlen=GODDESS_FEEDBACK_HISTORY_SIZE)
    runtime.torso_center_history = deque(data.get("torso_center_history", []), maxlen=GODDESS_STABILITY_HISTORY_SIZE)
    runtime.shoulder_tilt_history = deque(data.get("shoulder_tilt_history", []), maxlen=GODDESS_STABILITY_HISTORY_SIZE)
    runtime.detection_history = deque(data.get("detection_history", []), maxlen=GODDESS_DETECTION_HISTORY_SIZE)
    runtime.visibility_history = deque(data.get("visibility_history", []), maxlen=GODDESS_VISIBILITY_HISTORY_SIZE)
    runtime.point_history = {
        str(key): deque(
            [(float(x), float(y), float(z)) for x, y, z in values],
            maxlen=GODDESS_POINT_HISTORY_SIZE,
        )
        for key, values in data.get("point_history", {}).items()
    }
    runtime.boolean_histories = {
        str(key): deque([bool(value) for value in values], maxlen=GODDESS_BOOLEAN_HISTORY_SIZE)
        for key, values in data.get("boolean_histories", {}).items()
    }
    return runtime

# =========================================================
# SAFE PATH / MODEL LOAD
# =========================================================
BASE_DIR = Path(settings.BASE_DIR)

def resolve_model_path(filename: str) -> Path:
    candidates = [
        BASE_DIR / "Ml_Models" / filename,
        BASE_DIR / "Ml_models" / filename,
        BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"CRITICAL: Could not find model file: {filename}")

pose_model = joblib.load(resolve_model_path("goddess_pose_model.pkl"))
pose_label_encoder = joblib.load(resolve_model_path("goddess_label_encoder.pkl"))
defect_model = joblib.load(resolve_model_path("goddess_defect_model.pkl"))
defect_label_encoder = joblib.load(resolve_model_path("goddess_defect_label_encoder.pkl"))

# =========================================================
# MEDIAPIPE SETUP & LANDMARKS
# =========================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
)

FEATURE_COLS = []
for i in range(33):
    FEATURE_COLS.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
FEATURE_COLS.extend(["left_knee_angle", "right_knee_angle", "hip_angle"])

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
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
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
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

GREEN = "#00ff66"
RED = "#ff3b30"
YELLOW = "#ffd60a"
GRAY = "#cfcfcf"
DEGREE_SIGN = chr(176)

# =========================================================
# RESPONSE / RUNTIME HELPERS
# =========================================================
def api_success(**kwargs): 
    return JsonResponse({"success": True, **kwargs})

def api_error(message, status=400): 
    return JsonResponse({"success": False, "error": str(message)}, status=status)

def goddess_pose_success(**kwargs):
    payload = {
        "pose": "Unknown",
        "model_pose": "Unknown",
        "quality": "N/A",
        "status": "warning",
        "score": 0,
        "confidence": 0.0,
        "defect_confidence": 0.0,
        "feedback": "Waiting for pose.",
        "coach_text": "Step into the center of the frame.",
        "details": [],
        "angles": {},
        "points": [],
        "angle_texts": [],
        "joint_states": {},
        "pose_ready": False,
        "hold_ready": False,
        "hold_time": 0.0,
        "best_hold_time": 0.0,
        "perfect_hold": False,
    }
    payload.update(kwargs)
    return api_success(**payload)


def goddess_alignment_count(checks):
    return sum(
        1 for key in ("knees_tracking", "shoulders_level", "torso_centered")
        if bool(checks.get(key))
    )

def goddess_get_runtime(request):
    if request.session.session_key is None:
        request.session.save()

    return goddess_runtime_from_session_data(request.session.get(GODDESS_SESSION_RUNTIME_KEY))


def goddess_store_runtime(request, runtime):
    if request.session.session_key is None:
        request.session.save()
    request.session[GODDESS_SESSION_RUNTIME_KEY] = goddess_runtime_to_session_data(runtime)
    request.session.modified = True

def reset_runtime_state(runtime=None):
    if runtime is None:
        return

    runtime.pose_history.clear()
    runtime.defect_history.clear()
    runtime.score_history.clear()
    runtime.feedback_history.clear()
    runtime.torso_center_history.clear()
    runtime.shoulder_tilt_history.clear()
    runtime.detection_history.clear()
    runtime.visibility_history.clear()
    runtime.point_history.clear()
    runtime.boolean_histories.clear()

# =========================================================
# TEXT / SMOOTHING HELPERS
# =========================================================

def clean_text(text): 
    return " ".join(str(text).strip().split())

def dedupe_text_list(items, max_items=4, exclude=None):
    exclude_keys = {clean_text(x).lower() for x in (exclude or []) if x}
    output = []
    seen = set()
    for item in items:
        if not item: 
            continue
        text = clean_text(item)
        key = text.lower()
        if not key or key in seen or key in exclude_keys: 
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= max_items: 
            break
    return output

def smooth_label(history, new_label):
    history.append(str(new_label))
    return Counter(history).most_common(1)[0][0]

def smooth_score(runtime, new_score):
    runtime.score_history.append(float(new_score))
    return int(round(sum(runtime.score_history) / len(runtime.score_history)))

def smooth_feedback(runtime, new_feedback):
    runtime.feedback_history.append(str(new_feedback))
    return Counter(runtime.feedback_history).most_common(1)[0][0]

def smooth_boolean(history, value):
    history.append(bool(value))
    true_count = sum(history)
    if len(history) < 3:
        return true_count >= 1
    return true_count >= max(2, len(history) // 2 + 1)

def smooth_runtime_boolean(runtime, key, value, maxlen=GODDESS_BOOLEAN_HISTORY_SIZE):
    if key not in runtime.boolean_histories:
        runtime.boolean_histories[key] = deque(maxlen=maxlen)
    return smooth_boolean(runtime.boolean_histories[key], value)

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

def clip01(value): 
    return float(np.clip(value, 0.0, 1.0))

def moving_std(values):
    if len(values) < 3:
        return 0.0
    return float(np.std(list(values)))

def distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def goddess_angle_between(angle, min_angle, max_angle):
    return float(min_angle) <= float(angle) <= float(max_angle)

def goddess_angle_at_most(angle, threshold):
    return float(angle) <= float(threshold)

def goddess_knee_bend_ok(angle):
    return goddess_angle_between(angle, GODDESS_KNEE_BEND_MIN, GODDESS_KNEE_BEND_MAX)

def goddess_elbow_ok(angle):
    return goddess_angle_between(angle, GODDESS_ELBOW_MIN_ANGLE, GODDESS_ELBOW_MAX_ANGLE)

# =========================================================
# POINT SMOOTHING
# =========================================================
def smooth_point(runtime, key, x, y, z):
    if key not in runtime.point_history:
        runtime.point_history[key] = deque(maxlen=GODDESS_POINT_HISTORY_SIZE)

    runtime.point_history[key].append((float(x), float(y), float(z)))
    xs = [point[0] for point in runtime.point_history[key]]
    ys = [point[1] for point in runtime.point_history[key]]
    zs = [point[2] for point in runtime.point_history[key]]

    return (
        float(sum(xs) / len(xs)),
        float(sum(ys) / len(ys)),
        float(sum(zs) / len(zs)),
    )

# =========================================================
# VISION / FRAMING CHECKS
# =========================================================
def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 45, brightness

def is_human_confident(landmarks):
    """
    THE GHOST FILTER: To prevent the "Move Back" error in an empty room,
    we look specifically at the Face. Ghost detections never have clear faces.
    """
    face_points = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER]
    visibilities = sorted([landmarks[i].visibility for i in face_points], reverse=True)
    
    # Take the top 3 most confident points around the head/shoulders
    top_3_avg = sum(visibilities[:3]) / 3.0
    return top_3_avg > 0.46

def check_body_visibility(landmarks):
    indexes = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    ]
    visible_count = sum(1 for idx in indexes if float(landmarks[idx].visibility) >= 0.34)
    avg_visibility = sum(float(landmarks[idx].visibility) for idx in indexes) / len(indexes)

    shoulders_visible = landmarks[LEFT_SHOULDER].visibility > 0.38 and landmarks[RIGHT_SHOULDER].visibility > 0.38
    hips_visible = landmarks[LEFT_HIP].visibility > 0.42 and landmarks[RIGHT_HIP].visibility > 0.42
    knees_visible = landmarks[LEFT_KNEE].visibility > 0.34 and landmarks[RIGHT_KNEE].visibility > 0.34
    ankles_visible = landmarks[LEFT_ANKLE].visibility > 0.30 and landmarks[RIGHT_ANKLE].visibility > 0.30
    return shoulders_visible and hips_visible and knees_visible and ankles_visible, visible_count, avg_visibility

def check_frame_position(raw_pts, landmarks):
    visible_pts = [raw_pts[i] for i in SELECTED_POINTS if landmarks[i].visibility > 0.5]
    if len(visible_pts) < 4:
        return []

    xs = [float(p[0]) for p in visible_pts]
    ys = [float(p[1]) for p in visible_pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    center_x = sum(xs) / len(xs)

    feedback = []
    if width > 0.90 or height > 1.05:
        feedback.append("Move a little away from the camera")

    if center_x < 0.25:
        feedback.append("Move slightly to the right")
    elif center_x > 0.75:
        feedback.append("Move slightly to the left")

    return feedback

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def build_goddess_dataframe(landmarks):
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])

    l_knee_angle = calculate_angle([row[23*4], row[23*4+1]], [row[25*4], row[25*4+1]], [row[27*4], row[27*4+1]])
    r_knee_angle = calculate_angle([row[24*4], row[24*4+1]], [row[26*4], row[26*4+1]], [row[28*4], row[28*4+1]])
    hip_angle = calculate_angle([row[11*4], row[11*4+1]], [row[23*4], row[23*4+1]], [row[25*4], row[25*4+1]])

    ref_x = row[23*4]
    ref_y = row[23*4+1]
    
    for i in range(33):
        row[i*4] -= ref_x
        row[i*4+1] -= ref_y

    row.extend([l_knee_angle, r_knee_angle, hip_angle])
    df = pd.DataFrame([row], columns=FEATURE_COLS)
    
    return df, l_knee_angle, r_knee_angle, hip_angle

def analyze_goddess_pose(raw_pts, landmarks):
    ls = raw_pts[LEFT_SHOULDER]
    rs = raw_pts[RIGHT_SHOULDER]
    le = raw_pts[LEFT_ELBOW]
    re = raw_pts[RIGHT_ELBOW]
    lw = raw_pts[LEFT_WRIST]
    rw = raw_pts[RIGHT_WRIST]
    lh = raw_pts[LEFT_HIP]
    rh = raw_pts[RIGHT_HIP]
    lk = raw_pts[LEFT_KNEE]
    rk = raw_pts[RIGHT_KNEE]
    la = raw_pts[LEFT_ANKLE]
    ra = raw_pts[RIGHT_ANKLE]

    left_knee_angle = calculate_angle(lh[:2], lk[:2], la[:2])
    right_knee_angle = calculate_angle(rh[:2], rk[:2], ra[:2])
    left_hip_angle = calculate_angle(ls[:2], lh[:2], lk[:2])
    right_hip_angle = calculate_angle(rs[:2], rh[:2], rk[:2])
    hip_angle = (left_hip_angle + right_hip_angle) / 2.0
    left_elbow_angle = calculate_angle(ls[:2], le[:2], lw[:2])
    right_elbow_angle = calculate_angle(rs[:2], re[:2], rw[:2])
    left_shoulder_angle = calculate_angle(le[:2], ls[:2], lh[:2])
    right_shoulder_angle = calculate_angle(re[:2], rs[:2], rh[:2])

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    chest_center = (shoulder_center + hip_center) / 2.0
    knee_center = (lk + rk) / 2.0
    shoulder_width = max(distance(ls[:2], rs[:2]), 1e-4)
    ankle_width = abs(float(la[0] - ra[0]))
    stance_width_ratio = ankle_width / shoulder_width
    torso_size = max(distance(shoulder_center[:2], hip_center[:2]), shoulder_width, 1e-4)

    center_x = float(hip_center[0])
    left_hip_out = abs(float(lh[0] - center_x))
    right_hip_out = abs(float(rh[0] - center_x))
    left_knee_out = abs(float(lk[0] - center_x))
    right_knee_out = abs(float(rk[0] - center_x))
    left_ankle_out = abs(float(la[0] - center_x))
    right_ankle_out = abs(float(ra[0] - center_x))

    left_knee_tracking = left_knee_out >= left_hip_out + 0.005 and left_knee_out <= left_ankle_out + 0.12
    right_knee_tracking = right_knee_out >= right_hip_out + 0.005 and right_knee_out <= right_ankle_out + 0.12

    left_elbow_lifted = float(le[1]) <= float(ls[1]) + 0.16
    right_elbow_lifted = float(re[1]) <= float(rs[1]) + 0.16
    left_wrist_lifted = float(lw[1]) <= float(le[1]) + 0.18
    right_wrist_lifted = float(rw[1]) <= float(re[1]) + 0.18
    wrist_distance_ratio = distance(lw[:2], rw[:2]) / shoulder_width
    wrist_center_x = float(lw[0] + rw[0]) / 2.0
    wrist_center_y = float(lw[1] + rw[1]) / 2.0
    wrist_midline_offset = abs(wrist_center_x - float(shoulder_center[0])) / shoulder_width
    wrist_height_from_shoulders = (wrist_center_y - float(shoulder_center[1])) / torso_size
    wrists_balanced = abs(float(lw[1] - rw[1])) / torso_size <= GODDESS_PRAYER_BALANCE_MAX_RATIO
    left_wrist_to_chest = distance(lw[:2], chest_center[:2]) / torso_size
    right_wrist_to_chest = distance(rw[:2], chest_center[:2]) / torso_size
    wrists_near_chest = (
        left_wrist_to_chest <= GODDESS_PRAYER_CHEST_DISTANCE_MAX_RATIO and
        right_wrist_to_chest <= GODDESS_PRAYER_CHEST_DISTANCE_MAX_RATIO
    )
    prayer_elbows_soft = (
        float(left_elbow_angle) <= GODDESS_PRAYER_ELBOW_MAX_ANGLE and
        float(right_elbow_angle) <= GODDESS_PRAYER_ELBOW_MAX_ANGLE
    )
    prayer_hands = (
        prayer_elbows_soft and
        wrist_midline_offset <= GODDESS_PRAYER_MIDLINE_MAX_OFFSET and
        GODDESS_PRAYER_WRIST_HEIGHT_MIN_RATIO <= wrist_height_from_shoulders <= GODDESS_PRAYER_WRIST_HEIGHT_MAX_RATIO and
        wrists_balanced and
        (
            wrist_distance_ratio <= GODDESS_PRAYER_HANDS_MAX_WIDTH_RATIO or
            wrists_near_chest
        )
    )

    checks = {
        "left_knee_bent": goddess_knee_bend_ok(left_knee_angle),
        "right_knee_bent": goddess_knee_bend_ok(right_knee_angle),
        "left_knee_ideal": goddess_angle_between(left_knee_angle, GODDESS_IDEAL_KNEE_MIN, GODDESS_IDEAL_KNEE_MAX),
        "right_knee_ideal": goddess_angle_between(right_knee_angle, GODDESS_IDEAL_KNEE_MIN, GODDESS_IDEAL_KNEE_MAX),
        "left_knee_tracking": left_knee_tracking,
        "right_knee_tracking": right_knee_tracking,
        "knees_tracking": left_knee_tracking and right_knee_tracking,
        "stance_width_ok": stance_width_ratio >= GODDESS_STANCE_WIDTH_MIN_RATIO,
        "stance_width_ideal": stance_width_ratio >= GODDESS_IDEAL_STANCE_WIDTH_RATIO,
        "hips_depth_ok": hip_angle <= GODDESS_HIP_DEPTH_MAX and knee_center[1] >= hip_center[1],
        "shoulders_level": (abs(float(ls[1] - rs[1])) / torso_size) <= GODDESS_SHOULDER_LEVEL_MAX_DIFF,
        "torso_centered": (abs(float(shoulder_center[0] - hip_center[0])) / shoulder_width) <= GODDESS_TORSO_CENTER_MAX_OFFSET,
        "left_elbow_ok": goddess_elbow_ok(left_elbow_angle) and left_elbow_lifted and left_wrist_lifted,
        "right_elbow_ok": goddess_elbow_ok(right_elbow_angle) and right_elbow_lifted and right_wrist_lifted,
        "left_shoulder_ok": goddess_angle_between(left_shoulder_angle, GODDESS_SHOULDER_MIN_ANGLE, GODDESS_SHOULDER_MAX_ANGLE),
        "right_shoulder_ok": goddess_angle_between(right_shoulder_angle, GODDESS_SHOULDER_MIN_ANGLE, GODDESS_SHOULDER_MAX_ANGLE),
        "prayer_hands": prayer_hands,
    }
    checks["arms_raised"] = (
        checks["left_elbow_ok"] and checks["right_elbow_ok"] and
        checks["left_shoulder_ok"] and checks["right_shoulder_ok"]
    )
    checks["hands_ready"] = checks["arms_raised"] or checks["prayer_hands"]
    checks["core_goddess_gate"] = (
        checks["stance_width_ok"] and
        checks["left_knee_bent"] and
        checks["right_knee_bent"] and
        checks["hips_depth_ok"]
    )
    checks["strict_goddess_gate"] = (
        checks["core_goddess_gate"] and
        checks["knees_tracking"] and
        checks["shoulders_level"] and
        checks["torso_centered"] and
        checks["hands_ready"]
    )

    return {
        "angles": {
            "left_knee_angle": round(float(left_knee_angle), 1),
            "right_knee_angle": round(float(right_knee_angle), 1),
            "hip_angle": round(float(hip_angle), 1),
            "left_hip_angle": round(float(left_hip_angle), 1),
            "right_hip_angle": round(float(right_hip_angle), 1),
            "left_elbow_angle": round(float(left_elbow_angle), 1),
            "right_elbow_angle": round(float(right_elbow_angle), 1),
            "left_shoulder_angle": round(float(left_shoulder_angle), 1),
            "right_shoulder_angle": round(float(right_shoulder_angle), 1),
        },
        "measures": {
            "stance_width_ratio": round(float(stance_width_ratio), 3),
            "shoulder_level_diff": round(float(abs(ls[1] - rs[1]) / torso_size), 3),
            "torso_center_offset": round(float(abs(shoulder_center[0] - hip_center[0]) / shoulder_width), 3),
            "wrist_distance_ratio": round(float(wrist_distance_ratio), 3),
            "left_wrist_to_chest": round(float(left_wrist_to_chest), 3),
            "right_wrist_to_chest": round(float(right_wrist_to_chest), 3),
        },
        "checks": checks,
    }

def build_frontend_angles(analysis):
    angles = analysis["angles"]
    measures = analysis["measures"]
    return {
        **angles,
        "left_knee": angles["left_knee_angle"],
        "right_knee": angles["right_knee_angle"],
        "hip": angles["hip_angle"],
        "left_elbow": angles["left_elbow_angle"],
        "right_elbow": angles["right_elbow_angle"],
        "left_shoulder": angles["left_shoulder_angle"],
        "right_shoulder": angles["right_shoulder_angle"],
        "stance_width_ratio": measures["stance_width_ratio"],
        "shoulder_level_diff": measures["shoulder_level_diff"],
        "torso_center_offset": measures["torso_center_offset"],
    }

# =========================================================
# DEFECT SCORING & STABILITY
# =========================================================
def get_goddess_defect_info(defect_label):
    mapping = {
        "Perfect_Goddess": {
            "score": 100, 
            "main": "Beautiful Goddess pose. Hold steady.", 
            "coach": "Keep breathing and stay tall through the spine.", 
            "tips": ["Keep breathing steadily.", "Keep engaging your glutes."]
        },
        "Hips_Too_High": {
            "score": 70, 
            "main": "Bend your knees deeper.", 
            "coach": "Sink your hips until thighs are parallel.", 
            "tips": ["Drop your hips lower.", "Ensure knees track over toes."]
        },
        "Knees_Caving_In": {
            "score": 60, 
            "main": "Push your knees outward.", 
            "coach": "Track your knees over your toes.", 
            "tips": ["Engage your outer glutes.", "Keep knees wide."]
        },
        "Stance_Too_Narrow": {
            "score": 65, 
            "main": "Step your feet wider apart.", 
            "coach": "Widen your stance past your shoulders.", 
            "tips": ["Take a wider step.", "Point toes out 45 degrees."]
        },
        "Raise_Your_Arms": {
            "score": 80, 
            "main": "Bring your hands to prayer at the chest or lift them into a cactus shape.", 
            "coach": "Choose one clear arm position and hold it steadily.", 
            "tips": ["Bring palms together at the chest or lift elbows to shoulder height.", "Keep the arm position balanced."]
        },
        "Level_Your_Shoulders": {
            "score": 85, 
            "main": "Level your shoulders.", 
            "coach": "Don't lean to the side.", 
            "tips": ["Keep torso straight and centered.", "Engage core."]
        },
        "Uneven_Squat": {
            "score": 75, 
            "main": "Center your weight evenly.", 
            "coach": "Balance weight between both legs.", 
            "tips": ["Shift weight to center.", "Press through both heels."]
        }
    }
    return mapping.get(defect_label, {
        "score": 50, 
        "main": "Adjust your posture.", 
        "coach": "Follow the guidance.", 
        "tips": ["Check alignment."]
    })

def goddess_should_apply_defect(defect_label, checks):
    if defect_label == "Hips_Too_High":
        return not checks.get("hips_depth_ok")
    if defect_label == "Knees_Caving_In":
        return not checks.get("knees_tracking")
    if defect_label == "Stance_Too_Narrow":
        return not checks.get("stance_width_ok")
    if defect_label == "Raise_Your_Arms":
        return not checks.get("hands_ready")
    if defect_label == "Level_Your_Shoulders":
        return not checks.get("shoulders_level")
    if defect_label == "Uneven_Squat":
        return not checks.get("torso_centered")
    return True

def goddess_build_joint_states(runtime, analysis):
    checks = analysis["checks"]
    angles = analysis["angles"]
    measures = analysis["measures"]

    keys = [
        "left_knee_bent",
        "right_knee_bent",
        "left_knee_tracking",
        "right_knee_tracking",
        "knees_tracking",
        "stance_width_ok",
        "stance_width_ideal",
        "hips_depth_ok",
        "shoulders_level",
        "torso_centered",
        "left_elbow_ok",
        "right_elbow_ok",
        "left_shoulder_ok",
        "right_shoulder_ok",
        "prayer_hands",
        "arms_raised",
        "hands_ready",
        "core_goddess_gate",
        "strict_goddess_gate",
    ]
    smoothed = {key: smooth_runtime_boolean(runtime, key, checks[key]) for key in keys}



    joint_states = {
        "left_knee": {
            "ok": smoothed["left_knee_bent"] and smoothed["left_knee_tracking"],
            "angle": angles["left_knee_angle"],
            "min": GODDESS_KNEE_BEND_MIN,
            "max": GODDESS_KNEE_BEND_MAX,
            "tracking_ok": smoothed["left_knee_tracking"],
        },
        "right_knee": {
            "ok": smoothed["right_knee_bent"] and smoothed["right_knee_tracking"],
            "angle": angles["right_knee_angle"],
            "min": GODDESS_KNEE_BEND_MIN,
            "max": GODDESS_KNEE_BEND_MAX,
            "tracking_ok": smoothed["right_knee_tracking"],
        },
        "knees_tracking": {
            "ok": smoothed["knees_tracking"],
            "message": "Press both knees outward over the toes.",
        },
        "stance_width": {
            "ok": smoothed["stance_width_ok"],
            "value": measures["stance_width_ratio"],
            "threshold": GODDESS_STANCE_WIDTH_MIN_RATIO,
        },
        "hips_depth": {
            "ok": smoothed["hips_depth_ok"],
            "angle": angles["hip_angle"],
            "threshold": GODDESS_HIP_DEPTH_MAX,
        },
        "shoulders_level": {
            "ok": smoothed["shoulders_level"],
            "value": measures["shoulder_level_diff"],
            "threshold": GODDESS_SHOULDER_LEVEL_MAX_DIFF,
        },
        "torso_centered": {
            "ok": smoothed["torso_centered"],
            "value": measures["torso_center_offset"],
            "threshold": GODDESS_TORSO_CENTER_MAX_OFFSET,
        },
        "left_elbow": {
            "ok": smoothed["left_elbow_ok"] or smoothed["prayer_hands"],
            "angle": angles["left_elbow_angle"],
            "min": GODDESS_ELBOW_MIN_ANGLE,
            "max": GODDESS_ELBOW_MAX_ANGLE,
        },
        "right_elbow": {
            "ok": smoothed["right_elbow_ok"] or smoothed["prayer_hands"],
            "angle": angles["right_elbow_angle"],
            "min": GODDESS_ELBOW_MIN_ANGLE,
            "max": GODDESS_ELBOW_MAX_ANGLE,
        },
        "left_shoulder": {
            "ok": smoothed["left_shoulder_ok"] or smoothed["prayer_hands"],
            "angle": angles["left_shoulder_angle"],
            "min": GODDESS_SHOULDER_MIN_ANGLE,
            "max": GODDESS_SHOULDER_MAX_ANGLE,
        },
        "right_shoulder": {
            "ok": smoothed["right_shoulder_ok"] or smoothed["prayer_hands"],
            "angle": angles["right_shoulder_angle"],
            "min": GODDESS_SHOULDER_MIN_ANGLE,
            "max": GODDESS_SHOULDER_MAX_ANGLE,
        },
        "prayer_hands": {
            "ok": smoothed["prayer_hands"],
            "message": "Prayer hands at the heart center are valid.",
        },
        "arms_raised": {
            "ok": smoothed["arms_raised"],
            "message": "Lift both elbows into a cactus shape.",
        },
        "hands_ready": {
            "ok": smoothed["hands_ready"],
            "mode": "prayer" if smoothed["prayer_hands"] else ("cactus" if smoothed["arms_raised"] else "transition"),
            "message": "Choose cactus arms or prayer hands at the chest.",
        },
    }

    return joint_states, smoothed

def goddess_pose_flags(checks, stable_score, pose_status):
    pose_ready = bool(checks.get("core_goddess_gate"))
    alignment_count = goddess_alignment_count(checks)
    hands_ready = bool(checks.get("hands_ready"))
    balanced_goddess_gate = bool(
        checks.get("core_goddess_gate") and
        hands_ready and
        bool(checks.get("knees_tracking")) and
        alignment_count >= 2
    )
    correction_ready = pose_ready and not balanced_goddess_gate
    good_pose_ready = bool(balanced_goddess_gate and stable_score >= 84)
    hold_ready = bool(
        balanced_goddess_gate and
        stable_score >= 88 and
        str(pose_status).lower() in {"perfect", "good"}
    )

    return {
        "pose_ready": pose_ready,
        "correction_ready": correction_ready,
        "good_pose_ready": good_pose_ready,
        "hold_ready": hold_ready,
    }

def score_goddess_pose(analysis, checks, defect_label=None, defect_confidence=0.0):
    tips = []
    score = 92
    status = "good"
    pose_label = "Goddess Pose"
    feedback = "Strong Goddess pose. Keep refining the alignment."
    coach_text = "Keep the chest lifted and the knees tracking outward."

    if not checks["stance_width_ok"]:
        score = 64
        status = "warning"
        pose_label = "Goddess Pose Needs Correction"
        feedback = "Step your feet wider apart."
        coach_text = "Widen your stance past your shoulders."
        tips = ["Step wider", "Keep toes turned out comfortably"]
    elif not (checks["left_knee_bent"] and checks["right_knee_bent"]):
        score = 72
        status = "warning"
        pose_label = "Goddess Pose Needs Correction"
        feedback = "Bend both knees into the squat."
        coach_text = "Sink your hips and bend both knees."
        tips = ["Bend both knees", "Keep knees comfortable and wide"]
    elif not checks["hips_depth_ok"]:
        score = 76
        status = "warning"
        pose_label = "Goddess Pose Needs Correction"
        feedback = "Lower your hips slightly."
        coach_text = "Sink your hips while keeping your chest lifted."
        tips = ["Lower hips gently", "Keep the spine tall"]
    elif not checks["knees_tracking"]:
        score = 78
        status = "warning"
        pose_label = "Goddess Pose Needs Correction"
        feedback = "Press your knees outward."
        coach_text = "Track your knees over your toes."
        tips = ["Press knees outward", "Engage outer hips"]
    elif not checks["shoulders_level"]:
        score = 88 if checks["core_goddess_gate"] and checks["hands_ready"] else 84
        status = "good" if checks["core_goddess_gate"] and checks["hands_ready"] else "warning"
        pose_label = "Correct Goddess" if checks["core_goddess_gate"] and checks["hands_ready"] else "Goddess Pose Needs Correction"
        feedback = "Your Goddess shape is there. Relax and level the shoulders."
        coach_text = "Keep the chest open and soften the shoulders evenly."
        tips = ["Relax the neck", "Broaden the chest", "Keep both shoulders level"]
    elif not checks["torso_centered"]:
        score = 86 if checks["core_goddess_gate"] and checks["hands_ready"] else 84
        status = "good" if checks["core_goddess_gate"] and checks["hands_ready"] else "warning"
        pose_label = "Correct Goddess" if checks["core_goddess_gate"] and checks["hands_ready"] else "Goddess Pose Needs Correction"
        feedback = "Your Goddess shape is good. Center the torso a little more."
        coach_text = "Balance the ribs over the hips and settle evenly through both legs."
        tips = ["Stand tall", "Keep the ribs over the hips", "Balance your weight evenly"]
    elif not checks["hands_ready"]:
        score = 74
        status = "warning"
        pose_label = "Goddess Pose Needs Correction"
        feedback = "Bring your hands to prayer at the chest or lift them into a cactus shape."
        coach_text = "Choose one clear arm position and hold it steadily."
        tips = ["Bring palms together at the chest or lift elbows to shoulder height.", "Keep the arm position balanced."]
    elif checks["strict_goddess_gate"] and checks["stance_width_ideal"]:
        score = 100
        status = "perfect"
        pose_label = "Correct Goddess"
        feedback = "Beautiful Goddess pose. Hold steady."
        coach_text = "Keep breathing and stay tall through the spine."
        tips = ["Keep breathing steadily", "Keep knees wide", "Stay tall through the spine"]
    else:
        score = 94
        pose_label = "Correct Goddess"
        feedback = "Strong Goddess pose. Stay steady."
        coach_text = "Keep the ribs stacked over the hips and stay balanced."
        tips = ["Refine evenly on both sides", "Keep breathing"]

    if (
        defect_label and
        defect_label != "Perfect_Goddess" and
        float(defect_confidence or 0.0) >= 0.70 and
        goddess_should_apply_defect(defect_label, checks)
    ):
        defect_info = get_goddess_defect_info(defect_label)
        if defect_info["score"] < score:
            score = max(defect_info["score"], score - 8)
            feedback = defect_info["main"]
            coach_text = defect_info["coach"]
            tips = defect_info["tips"] + tips
            status = "good" if score >= 82 else "warning"
            pose_label = "Goddess Pose Needs Correction"

    return {
        "score": int(max(0, min(100, score))),
        "status": status,
        "pose_label": pose_label,
        "feedback": feedback,
        "coach_text": coach_text,
        "tips": tips,
    }

def goddess_is_pose_like(model_label, model_confidence, checks):
    label = str(model_label or "").lower()
    if checks.get("strict_goddess_gate"):
        return True
    if checks.get("core_goddess_gate") and checks.get("hands_ready"):
        return True
    if "goddess" in label and "not" not in label and float(model_confidence or 0.0) >= 0.68 and checks.get("core_goddess_gate"):
        return True
    return False

def quality_from_score(score):
    if score >= 95:
        return "Perfect_Goddess"
    if score >= 85:
        return "Good_Goddess"
    if score >= 70:
        return "Needs_Correction"
    return "Not_Ready"

def predict_pose_label(features_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = pose_model.predict(features_df)[0]
        label = pose_label_encoder.inverse_transform([pred])[0]

        confidence = 0.90
        if hasattr(pose_model, "predict_proba"):
            probs = pose_model.predict_proba(features_df)[0]
            confidence = float(np.max(probs))
        return label, confidence

def predict_defect_label(features_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(defect_model, "predict_proba"):
            prob_array = defect_model.predict_proba(features_df)[0]
            defect_idx = int(np.argmax(prob_array))
            label = defect_label_encoder.inverse_transform([defect_idx])[0]
            return label, float(prob_array[defect_idx])

        pred = defect_model.predict(features_df)[0]
        label = defect_label_encoder.inverse_transform([pred])[0]
        return label, 0.90

def update_stability_metrics(runtime, raw_pts):
    hip_center_x = float(raw_pts[LEFT_HIP][0] + raw_pts[RIGHT_HIP][0]) / 2.0
    shoulder_tilt = abs(float(raw_pts[LEFT_SHOULDER][1] - raw_pts[RIGHT_SHOULDER][1]))
    runtime.torso_center_history.append(hip_center_x)
    runtime.shoulder_tilt_history.append(shoulder_tilt)

def get_stability_feedback(runtime):
    feedback = []
    penalty = 0

    if moving_std(runtime.torso_center_history) > 0.020:
        feedback.append("Steady your balance.")
        penalty += 4
    if moving_std(runtime.shoulder_tilt_history) > 0.015:
        feedback.append("Keep your shoulders more stable.")
        penalty += 3

    return feedback, penalty

def build_points_for_frontend(raw_pts, landmarks, runtime=None):
    runtime = runtime or GoddessRuntime()
    points = []
    for idx in SELECTED_POINTS:
        lm = landmarks[idx]
        visibility = float(lm.visibility)
        
        if visibility < GODDESS_POINT_VISIBILITY_MIN:
            continue

        sx, sy, sz = smooth_point(runtime, f"goddess_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])

        points.append({
            "name": POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": clip01(sx),
            "y": clip01(sy),
            "color": YELLOW,
            "radius": 6,
            "visible": True,
            "visibility": round(visibility, 3),
        })
        
    return points

def build_angle_texts(raw_pts, landmarks, analysis):
    angle_specs = [
        (LEFT_KNEE, analysis["angles"]["left_knee_angle"], YELLOW, "left_knee"),
        (RIGHT_KNEE, analysis["angles"]["right_knee_angle"], YELLOW, "right_knee"),
        (LEFT_HIP, analysis["angles"]["hip_angle"], YELLOW, "hips_depth"),
        (LEFT_ELBOW, analysis["angles"]["left_elbow_angle"], YELLOW, "left_elbow"),
        (RIGHT_ELBOW, analysis["angles"]["right_elbow_angle"], YELLOW, "right_elbow"),
    ]

    items = []
    for idx, value, color, joint_key in angle_specs:
        if float(landmarks[idx].visibility) < 0.30:
            continue
        items.append({
            "text": f"{int(round(float(value)))}{DEGREE_SIGN}",
            "x": clip01(raw_pts[idx][0]),
            "y": clip01(raw_pts[idx][1]),
            "color": color,
            "joint_key": joint_key,
        })
    return items

# =========================================================
# MAIN API PROCESS
# =========================================================
def process_goddess_pose_request(request):
    runtime = None
    try:
        runtime = goddess_get_runtime(request)

        if request.POST.get("reset") == "true":
            reset_runtime_state(runtime)

        uploaded_file = request.FILES["image"]
        frame = read_uploaded_image(uploaded_file)

        if frame is None:
            return api_error("Invalid image file", status=400)

        low_light, _brightness = check_lighting(frame)
        if low_light:
            reset_runtime_state(runtime)
            return goddess_pose_success(
                pose="Low Light",
                status="warning",
                feedback="Room lighting is too low.",
                coach_text="Improve room lighting.",
                details=["Increase lighting", "Avoid dark backgrounds"],
            )

        landmarks = detect_landmarks(frame)
        has_landmarks = landmarks is not None and is_human_confident(landmarks)
        stable_has_landmarks = smooth_boolean(runtime.detection_history, has_landmarks)

        if not has_landmarks and not stable_has_landmarks:
            reset_runtime_state(runtime)
            return goddess_pose_success(
                pose="Unknown",
                status="unknown",
                feedback="No human detected.",
                coach_text="Step into the center of the frame.",
                details=["Ensure you are fully visible", "Face the camera", "Move slightly back"],
            )

        if landmarks is None:
            runtime.point_history.clear()
            return goddess_pose_success(
                pose="Tracking...",
                status="warning",
                feedback="Hold still while tracking stabilizes.",
                coach_text="Keep your full body visible.",
                details=["Hold still", "Keep your full body in frame"],
            )

        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        full_body_visible, _visible_count, _avg_visibility = check_body_visibility(landmarks)
        stable_full_body_visible = smooth_boolean(runtime.visibility_history, full_body_visible)

        if not full_body_visible and not stable_full_body_visible:
            reset_runtime_state(runtime)
            framing_errors = check_frame_position(raw_pts, landmarks)
            details = ["Show full body clearly", "Keep both feet visible"] + framing_errors
            return goddess_pose_success(
                pose="Body Not Visible",
                status="warning",
                feedback="Move a little back. Full body should be visible.",
                coach_text="Move back until your full body is visible.",
                details=dedupe_text_list(details),
            )

        smoothed_pts = raw_pts.copy()
        for idx in SELECTED_POINTS:
            sx, sy, sz = smooth_point(runtime, f"goddess_{idx}", raw_pts[idx][0], raw_pts[idx][1], raw_pts[idx][2])
            smoothed_pts[idx] = [sx, sy, sz]

        df, _l_knee_angle, _r_knee_angle, _hip_angle = build_goddess_dataframe(landmarks)
        raw_pose_label, pose_confidence = predict_pose_label(df)
        stable_pose_label = smooth_label(runtime.pose_history, raw_pose_label)
        raw_defect_label, defect_confidence = predict_defect_label(df)
        stable_defect_label = smooth_label(runtime.defect_history, raw_defect_label)

        analysis = analyze_goddess_pose(smoothed_pts, landmarks)
        joint_states, smoothed_checks = goddess_build_joint_states(runtime, analysis)
        points = build_points_for_frontend(smoothed_pts, landmarks, runtime)
        angle_texts = build_angle_texts(smoothed_pts, landmarks, analysis)

        update_stability_metrics(runtime, smoothed_pts)
        stability_tips, stability_penalty = get_stability_feedback(runtime)

        pose_eval = score_goddess_pose(analysis, smoothed_checks, stable_defect_label, defect_confidence)
        stable_score = smooth_score(runtime, max(0, pose_eval["score"] - stability_penalty))
        is_goddess = goddess_is_pose_like(stable_pose_label, pose_confidence, smoothed_checks)
        pose_flags = goddess_pose_flags(smoothed_checks, stable_score, pose_eval["status"])
        framing_errors = check_frame_position(smoothed_pts, landmarks)

        if not pose_flags["pose_ready"] and not is_goddess and stable_score < 75:
            tips = [
                "Step into a wide stance",
                "Bend both knees outward",
                "Bring the hands to prayer at the chest or lift the elbows into cactus arms",
            ] + framing_errors
            return goddess_pose_success(
                pose="Not Goddess Pose",
                model_pose=stable_pose_label,
                quality="Not_Ready",
                status="warning",
                score=max(0, min(65, stable_score)),
                confidence=round(float(pose_confidence), 3),
                defect_confidence=round(float(defect_confidence), 3),
                feedback="Take a wide stance and bend your knees.",
                coach_text="Step wide and sink into a squat.",
                details=dedupe_text_list(tips, max_items=4),
                angles=build_frontend_angles(analysis),
                points=points,
                angle_texts=angle_texts,
                joint_states=joint_states,
                pose_ready=False,
                hold_ready=False,
            )

        if pose_flags["hold_ready"]:
            status = "perfect"
            pose_name = "Correct Goddess"
        elif pose_flags["good_pose_ready"]:
            status = "good"
            pose_name = "Correct Goddess"
        elif pose_flags["pose_ready"] or is_goddess:
            status = "warning"
            pose_name = "Goddess Pose Needs Correction"
        else:
            status = "warning"
            pose_name = "Not Goddess Pose"

        stable_feedback = smooth_feedback(runtime, pose_eval["feedback"])
        tips = pose_eval["tips"] + stability_tips + framing_errors
        quality = (
            "Perfect_Goddess" if pose_flags["hold_ready"] else
            "Good_Goddess" if pose_flags["good_pose_ready"] else
            "Needs_Correction" if (pose_flags["pose_ready"] or is_goddess) else
            "Not_Ready"
        )

        return goddess_pose_success(
            pose=pose_name,
            model_pose=stable_pose_label,
            quality=quality,
            status=status,
            score=stable_score,
            confidence=round(float(pose_confidence), 3),
            defect_confidence=round(float(defect_confidence), 3),
            feedback=stable_feedback,
            coach_text=pose_eval["coach_text"],
            details=dedupe_text_list(tips, max_items=4, exclude=[stable_feedback, pose_eval["coach_text"]]),
            angles=build_frontend_angles(analysis),
            points=points,
            angle_texts=angle_texts,
            joint_states=joint_states,
            pose_ready=pose_flags["pose_ready"] or is_goddess,
            hold_ready=pose_flags["hold_ready"],
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return api_error(str(e), status=500)
    finally:
        if runtime is not None:
            goddess_store_runtime(request, runtime)
