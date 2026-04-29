import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from django.contrib.sessions.middleware import SessionMiddleware
from django.core import mail
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings

from User.models import Users
from User.session_auth import APP_USER_SESSION_KEY
from User.utils import down_dog_utility as dd
from User.utils import goddess_utility as gd
from User.utils import tree_utility as tr
from User.utils import warrior_utility as wr


class DownDogUtilityTests(TestCase):
    def build_request(self, reset=False):
        request = SimpleNamespace(
            POST={"reset": "true"} if reset else {},
            FILES={
                "image": SimpleUploadedFile(
                    "frame.jpg",
                    b"fake-image-bytes",
                    content_type="image/jpeg",
                )
            },
            COOKIES={},
            session={},
        )

        middleware = SessionMiddleware(lambda req: None)
        middleware.process_request(request)
        request.session.save()
        return request

    def parse_json(self, response):
        return json.loads(response.content.decode("utf-8"))

    def make_fake_landmarks(self, visibility=0.95):
        return [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility)
            for _ in range(33)
        ]

    def make_down_dog_landmarks_for_points(self, dominant_side="left", support_visible=True):
        landmarks = self.make_fake_landmarks()
        for index in range(33):
            landmarks[index].x = 0.40 + (index % 3) * 0.05
            landmarks[index].y = 0.20 + (index % 5) * 0.08

        support_indexes = (
            [dd.DD_RIGHT_SHOULDER, dd.DD_RIGHT_ELBOW, dd.DD_RIGHT_WRIST, dd.DD_RIGHT_HIP, dd.DD_RIGHT_KNEE, dd.DD_RIGHT_ANKLE]
            if dominant_side == "left" else
            [dd.DD_LEFT_SHOULDER, dd.DD_LEFT_ELBOW, dd.DD_LEFT_WRIST, dd.DD_LEFT_HIP, dd.DD_LEFT_KNEE, dd.DD_LEFT_ANKLE]
        )

        for idx in support_indexes:
            landmarks[idx].visibility = 0.92 if support_visible else 0.20

        return landmarks

    def build_analysis(self, left_elbow_ok=True, dominant_leg_straight=True, spine_long=True, head_between_arms=True):
        core_shape_count = sum([
            True,
            True,
            left_elbow_ok,
            dominant_leg_straight,
            True,
            True,
            spine_long,
        ])
        detail_shape_count = sum([
            head_between_arms,
            True,
            True,
            True,
            True,
            True,
        ])
        dominant_side_gate = (
            left_elbow_ok and
            core_shape_count >= 5 and
            detail_shape_count >= 3
        )
        soft_down_dog_gate = core_shape_count >= dd.DD_CORE_SHAPE_READY_MIN and detail_shape_count >= dd.DD_DETAIL_SHAPE_READY_MIN
        return {
            "dominant_side": "left",
            "dominant_side_name": "Left",
            "dominant_vis": 0.92,
            "support_vis": 0.75,
            "angles": {
                "dominant_elbow_angle": 150.0 if left_elbow_ok else 149.0,
                "dominant_knee_angle": 145.0 if dominant_leg_straight else 136.0,
                "dominant_shoulder_open_angle": 132.0,
                "dominant_hip_fold_angle": 126.0,
                "support_elbow_angle": 145.0,
                "support_knee_angle": 140.0,
                "left_elbow_angle": 150.0 if left_elbow_ok else 149.0,
                "right_elbow_angle": 145.0,
                "left_knee_angle": 145.0 if dominant_leg_straight else 136.0,
                "right_knee_angle": 140.0,
            },
            "measures": {
                "hips_height_ratio": 0.46,
                "hips_peak_ratio": 0.34,
                "hands_width_ratio": 1.0,
                "feet_width_ratio": 1.0,
                "core_shape_count": core_shape_count,
                "detail_shape_count": detail_shape_count,
            },
            "checks": {
                "dominant_visible": True,
                "support_visible": True,
                "dominant_arm_straight": left_elbow_ok,
                "support_arm_straight": True,
                "dominant_leg_straight": dominant_leg_straight,
                "dominant_leg_strict": dominant_leg_straight,
                "support_leg_straight": True,
                "hips_above_shoulders": True,
                "hips_high": True,
                "hips_peak": True,
                "head_between_arms": head_between_arms,
                "spine_long": spine_long,
                "shoulder_open": True,
                "hip_fold_ok": True,
                "side_profile_clear": True,
                "arm_length_ok": True,
                "leg_length_ok": True,
                "core_shape_count": core_shape_count,
                "detail_shape_count": detail_shape_count,
                "core_shape_ready": core_shape_count >= dd.DD_CORE_SHAPE_READY_MIN,
                "detail_shape_ready": detail_shape_count >= dd.DD_DETAIL_SHAPE_READY_MIN,
                "soft_down_dog_gate": soft_down_dog_gate,
                "balanced_shape": True,
                "hands_width_ok": True,
                "feet_width_ok": True,
                "heels_reasonable": True,
                "both_heels_reasonable": True,
                "dominant_side_gate": dominant_side_gate,
                "both_side_bonus_gate": True,
                "strict_down_dog_gate": dominant_side_gate and dominant_leg_straight and spine_long,
                "left_elbow_ok": left_elbow_ok,
                "right_elbow_ok": True,
                "left_knee_ok": dominant_leg_straight,
                "right_knee_ok": True,
            },
        }

    def test_reset_flag_clears_active_runtime_histories(self):
        request = self.build_request(reset=True)
        runtime = dd.dd_get_runtime(request)
        runtime.pose_history.append("Down Dog")
        runtime.score_history.append(99.0)
        runtime.feedback_history.append("Hold steady")
        runtime.point_history["dd_13"] = [(0.1, 0.1, 0.0)]
        runtime.boolean_histories["left_elbow_ok"] = [True, False]
        dd.dd_store_runtime(request, runtime)

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(dd, "dd_read_uploaded_image", return_value=frame), \
             patch.object(dd, "dd_enhance_frame", return_value=frame), \
             patch.object(dd, "dd_check_lighting", return_value=(True, 12.0)):
            response = dd.process_down_dog_request(request)

        data = self.parse_json(response)
        runtime = dd.dd_get_runtime(request)
        self.assertEqual(data["pose"], "Low Light")
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(len(runtime.pose_history), 0)
        self.assertEqual(len(runtime.score_history), 0)
        self.assertEqual(len(runtime.feedback_history), 0)
        self.assertEqual(runtime.point_history, {})
        self.assertEqual(runtime.boolean_histories, {})

    def test_relaxed_angle_threshold_helpers_accept_edges_and_reject_just_below(self):
        self.assertTrue(dd.dd_angle_at_least(dd.DD_DOMINANT_ELBOW_STRAIGHT_ANGLE, dd.DD_DOMINANT_ELBOW_STRAIGHT_ANGLE))
        self.assertFalse(dd.dd_angle_at_least(dd.DD_DOMINANT_ELBOW_STRAIGHT_ANGLE - 0.1, dd.DD_DOMINANT_ELBOW_STRAIGHT_ANGLE))

        self.assertTrue(dd.dd_angle_at_least(dd.DD_SUPPORT_ELBOW_STRAIGHT_ANGLE, dd.DD_SUPPORT_ELBOW_STRAIGHT_ANGLE))
        self.assertFalse(dd.dd_angle_at_least(dd.DD_SUPPORT_ELBOW_STRAIGHT_ANGLE - 0.1, dd.DD_SUPPORT_ELBOW_STRAIGHT_ANGLE))

        self.assertTrue(dd.dd_angle_at_least(dd.DD_DOMINANT_KNEE_STRAIGHT_ANGLE, dd.DD_DOMINANT_KNEE_STRAIGHT_ANGLE))
        self.assertFalse(dd.dd_angle_at_least(dd.DD_DOMINANT_KNEE_STRAIGHT_ANGLE - 0.1, dd.DD_DOMINANT_KNEE_STRAIGHT_ANGLE))

        self.assertTrue(dd.dd_angle_at_least(dd.DD_SUPPORT_KNEE_STRAIGHT_ANGLE, dd.DD_SUPPORT_KNEE_STRAIGHT_ANGLE))
        self.assertFalse(dd.dd_angle_at_least(dd.DD_SUPPORT_KNEE_STRAIGHT_ANGLE - 0.1, dd.DD_SUPPORT_KNEE_STRAIGHT_ANGLE))

        self.assertTrue(dd.dd_angle_at_least(dd.DD_SHOULDER_OPEN_MIN_ANGLE, dd.DD_SHOULDER_OPEN_MIN_ANGLE))
        self.assertFalse(dd.dd_angle_at_least(dd.DD_SHOULDER_OPEN_MIN_ANGLE - 0.1, dd.DD_SHOULDER_OPEN_MIN_ANGLE))

        self.assertTrue(dd.dd_angle_at_most(dd.DD_HIP_FOLD_MAX_ANGLE, dd.DD_HIP_FOLD_MAX_ANGLE))
        self.assertFalse(dd.dd_angle_at_most(dd.DD_HIP_FOLD_MAX_ANGLE + 0.1, dd.DD_HIP_FOLD_MAX_ANGLE))

    def test_joint_states_include_kinematic_keys_and_use_hysteresis(self):
        runtime = dd.DownDogRuntime()

        joint_states, _ = dd.dd_build_joint_states(runtime, self.build_analysis(left_elbow_ok=True))
        self.assertIn("left_elbow", joint_states)
        self.assertIn("right_elbow", joint_states)
        self.assertIn("left_knee", joint_states)
        self.assertIn("right_knee", joint_states)
        self.assertIn("hips_high", joint_states)
        self.assertIn("spine_long", joint_states)
        self.assertIn("shoulders_open", joint_states)
        self.assertIn("head_between_arms", joint_states)
        self.assertIn("hands_width", joint_states)
        self.assertIn("feet_width", joint_states)
        self.assertTrue(joint_states["left_elbow"]["ok"])
        self.assertEqual(joint_states["left_elbow"]["threshold"], dd.DD_DOMINANT_ELBOW_STRAIGHT_ANGLE)

        joint_states, _ = dd.dd_build_joint_states(runtime, self.build_analysis(left_elbow_ok=False))
        self.assertTrue(joint_states["left_elbow"]["ok"], "A single noisy frame should not flip the elbow state red.")

        joint_states, _ = dd.dd_build_joint_states(runtime, self.build_analysis(left_elbow_ok=False))
        self.assertFalse(joint_states["left_elbow"]["ok"], "Two consecutive bad frames should eventually flip the elbow state.")

    def test_down_dog_angle_texts_include_joint_keys_and_clean_symbols(self):
        raw_pts = np.zeros((33, 3), dtype=np.float32)
        landmarks = self.make_fake_landmarks()
        analysis = self.build_analysis(left_elbow_ok=True)

        items = dd.dd_build_angle_texts(raw_pts, landmarks, analysis)

        self.assertTrue(items)
        self.assertTrue(all("joint_key" in item for item in items))
        self.assertTrue(all("Â" not in item["text"] for item in items))

    def test_down_dog_points_skip_extra_foot_clutter_and_hidden_support_side(self):
        raw_pts = np.zeros((33, 3), dtype=np.float32)
        for idx in range(33):
            raw_pts[idx] = [0.35 + (idx % 4) * 0.08, 0.20 + (idx % 6) * 0.09, 0.0]

        landmarks = self.make_down_dog_landmarks_for_points(dominant_side="left", support_visible=False)
        analysis = self.build_analysis()
        analysis["support_vis"] = 0.18

        points = dd.dd_build_points_for_frontend(raw_pts, landmarks, analysis)
        names = {point["name"] for point in points}

        self.assertNotIn("left_heel", names)
        self.assertNotIn("right_heel", names)
        self.assertNotIn("left_foot_index", names)
        self.assertNotIn("right_foot_index", names)
        self.assertNotIn("right_elbow", names)
        self.assertNotIn("right_knee", names)

    def test_down_dog_pose_flags_hold_only_when_ready(self):
        analysis = self.build_analysis()
        pose_eval = dd.dd_score_down_dog_pose(analysis, analysis["checks"])
        flags = dd.dd_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertTrue(flags["pose_ready"])
        self.assertTrue(flags["good_pose_ready"])
        self.assertTrue(flags["hold_ready"])

    def test_down_dog_soft_knee_is_recognized_but_not_hold_ready(self):
        analysis = self.build_analysis(dominant_leg_straight=False)
        pose_eval = dd.dd_score_down_dog_pose(analysis, analysis["checks"])
        flags = dd.dd_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertTrue(flags["pose_ready"])
        self.assertFalse(flags["hold_ready"])
        self.assertLess(pose_eval["score"], dd.DD_HOLD_READY_SCORE)

    def test_down_dog_soft_elbow_still_counts_as_pose_ready(self):
        analysis = self.build_analysis(left_elbow_ok=False)
        pose_eval = dd.dd_score_down_dog_pose(analysis, analysis["checks"])
        flags = dd.dd_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])
        is_like = dd.dd_is_downdog_like("down dog", 0.40, analysis["checks"])

        self.assertTrue(flags["pose_ready"])
        self.assertTrue(is_like)
        self.assertFalse(flags["hold_ready"])

    def test_no_landmarks_response_is_safe(self):
        request = self.build_request()
        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(dd, "dd_read_uploaded_image", return_value=frame), \
             patch.object(dd, "dd_enhance_frame", return_value=frame), \
             patch.object(dd, "dd_check_lighting", return_value=(False, 100.0)), \
             patch.object(dd, "dd_detect_landmarks", return_value=None):
            response = dd.process_down_dog_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Unknown")
        self.assertEqual(data["score"], 0)
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(data["points"], [])
        self.assertEqual(data["angle_texts"], [])
        self.assertEqual(data["joint_states"], {})

    def test_partial_visibility_response_is_safe(self):
        request = self.build_request()
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        landmarks = self.make_fake_landmarks()

        with patch.object(dd, "dd_read_uploaded_image", return_value=frame), \
             patch.object(dd, "dd_enhance_frame", return_value=frame), \
             patch.object(dd, "dd_check_lighting", return_value=(False, 100.0)), \
             patch.object(dd, "dd_detect_landmarks", return_value=landmarks), \
             patch.object(dd, "dd_build_feature_dataframe_from_landmarks", return_value=(MagicMock(), {}, {})), \
             patch.object(dd, "dd_check_body_visibility", return_value=(False, 3, 0.21)):
            response = dd.process_down_dog_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Body Not Visible")
        self.assertEqual(data["score"], 0)
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(data["points"], [])
        self.assertEqual(data["angle_texts"], [])
        self.assertEqual(data["joint_states"], {})



class GoddessUtilityTests(TestCase):
    def build_request(self, reset=False):
        request = SimpleNamespace(
            POST={"reset": "true"} if reset else {},
            FILES={
                "image": SimpleUploadedFile(
                    "frame.jpg",
                    b"fake-image-bytes",
                    content_type="image/jpeg",
                )
            },
            COOKIES={},
            session={},
        )

        middleware = SessionMiddleware(lambda req: None)
        middleware.process_request(request)
        request.session.save()
        return request

    def parse_json(self, response):
        return json.loads(response.content.decode("utf-8"))

    def make_fake_landmarks(self, visibility=0.95):
        return [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility)
            for _ in range(33)
        ]

    def build_analysis(
        self,
        left_knee_ok=True,
        arms_raised=True,
        prayer_hands=False,
        knees_tracking=True,
        shoulders_level=True,
        torso_centered=True,
    ):
        return {
            "angles": {
                "left_knee_angle": 145.0 if left_knee_ok else 145.1,
                "right_knee_angle": 112.0,
                "hip_angle": 140.0,
                "left_hip_angle": 140.0,
                "right_hip_angle": 140.0,
                "left_elbow_angle": 90.0,
                "right_elbow_angle": 92.0,
                "left_shoulder_angle": 90.0,
                "right_shoulder_angle": 91.0,
            },
            "measures": {
                "stance_width_ratio": 1.55,
                "shoulder_level_diff": 0.02 if shoulders_level else 0.12,
                "torso_center_offset": 0.05 if torso_centered else 0.42,
                "wrist_distance_ratio": 0.20,
                "left_wrist_to_chest": 0.44,
                "right_wrist_to_chest": 0.45,
            },
            "checks": {
                "left_knee_bent": left_knee_ok,
                "right_knee_bent": True,
                "left_knee_ideal": False,
                "right_knee_ideal": True,
                "left_knee_tracking": knees_tracking,
                "right_knee_tracking": knees_tracking,
                "knees_tracking": knees_tracking,
                "stance_width_ok": True,
                "stance_width_ideal": True,
                "hips_depth_ok": True,
                "shoulders_level": shoulders_level,
                "torso_centered": torso_centered,
                "left_elbow_ok": arms_raised,
                "right_elbow_ok": arms_raised,
                "left_shoulder_ok": arms_raised,
                "right_shoulder_ok": arms_raised,
                "prayer_hands": prayer_hands,
                "arms_raised": arms_raised,
                "hands_ready": arms_raised or prayer_hands,
                "core_goddess_gate": left_knee_ok,
                "strict_goddess_gate": left_knee_ok and knees_tracking and shoulders_level and torso_centered and (arms_raised or prayer_hands),
            },
        }

    def test_reset_flag_clears_active_goddess_runtime(self):
        request = self.build_request(reset=True)
        runtime = gd.goddess_get_runtime(request)
        runtime.pose_history.append("Goddess")
        runtime.defect_history.append("Perfect_Goddess")
        runtime.score_history.append(99.0)
        runtime.feedback_history.append("Hold steady")
        runtime.point_history["goddess_25"] = [(0.1, 0.1, 0.0)]
        runtime.boolean_histories["left_knee_bent"] = [True, False]
        gd.goddess_store_runtime(request, runtime)

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(gd, "read_uploaded_image", return_value=frame), \
             patch.object(gd, "check_lighting", return_value=(True, 12.0)):
            response = gd.process_goddess_pose_request(request)

        data = self.parse_json(response)
        runtime = gd.goddess_get_runtime(request)
        self.assertEqual(data["pose"], "Low Light")
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(len(runtime.pose_history), 0)
        self.assertEqual(len(runtime.defect_history), 0)
        self.assertEqual(len(runtime.score_history), 0)
        self.assertEqual(len(runtime.feedback_history), 0)
        self.assertEqual(runtime.point_history, {})
        self.assertEqual(runtime.boolean_histories, {})

    def test_goddess_relaxed_thresholds_accept_edges_and_reject_just_outside(self):
        self.assertTrue(gd.goddess_knee_bend_ok(gd.GODDESS_KNEE_BEND_MIN))
        self.assertFalse(gd.goddess_knee_bend_ok(gd.GODDESS_KNEE_BEND_MIN - 0.1))
        self.assertTrue(gd.goddess_knee_bend_ok(gd.GODDESS_KNEE_BEND_MAX))
        self.assertFalse(gd.goddess_knee_bend_ok(gd.GODDESS_KNEE_BEND_MAX + 0.1))

        self.assertTrue(gd.goddess_elbow_ok(gd.GODDESS_ELBOW_MIN_ANGLE))
        self.assertFalse(gd.goddess_elbow_ok(gd.GODDESS_ELBOW_MIN_ANGLE - 0.1))
        self.assertTrue(gd.goddess_elbow_ok(gd.GODDESS_ELBOW_MAX_ANGLE))
        self.assertFalse(gd.goddess_elbow_ok(gd.GODDESS_ELBOW_MAX_ANGLE + 0.1))

        self.assertTrue(gd.goddess_angle_at_most(155.0, gd.GODDESS_HIP_DEPTH_MAX))
        self.assertFalse(gd.goddess_angle_at_most(155.1, gd.GODDESS_HIP_DEPTH_MAX))

    def test_goddess_joint_states_include_expected_keys_and_hysteresis(self):
        runtime = gd.GoddessRuntime()

        joint_states, _ = gd.goddess_build_joint_states(runtime, self.build_analysis(left_knee_ok=True))
        self.assertIn("left_knee", joint_states)
        self.assertIn("right_knee", joint_states)
        self.assertIn("stance_width", joint_states)
        self.assertIn("knees_tracking", joint_states)
        self.assertIn("hips_depth", joint_states)
        self.assertIn("shoulders_level", joint_states)
        self.assertIn("left_elbow", joint_states)
        self.assertIn("right_elbow", joint_states)
        self.assertTrue(joint_states["left_knee"]["ok"])
        self.assertEqual(joint_states["left_knee"]["max"], gd.GODDESS_KNEE_BEND_MAX)

        joint_states, _ = gd.goddess_build_joint_states(runtime, self.build_analysis(left_knee_ok=False))
        self.assertTrue(joint_states["left_knee"]["ok"], "A single noisy frame should not flip the knee state red.")

        joint_states, _ = gd.goddess_build_joint_states(runtime, self.build_analysis(left_knee_ok=False))
        self.assertFalse(joint_states["left_knee"]["ok"], "Two consecutive bad frames should eventually flip the knee state.")

    def test_goddess_angle_texts_include_joint_keys(self):
        raw_pts = np.zeros((33, 3), dtype=np.float32)
        landmarks = self.make_fake_landmarks()
        analysis = self.build_analysis(left_knee_ok=True)

        items = gd.build_angle_texts(raw_pts, landmarks, analysis)

        self.assertTrue(items)
        self.assertTrue(all("joint_key" in item for item in items))

    def test_goddess_missing_arms_is_warning_and_not_hold_ready(self):
        analysis = self.build_analysis(left_knee_ok=True, arms_raised=False, prayer_hands=False)
        pose_eval = gd.score_goddess_pose(analysis, analysis["checks"])
        flags = gd.goddess_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertEqual(pose_eval["status"], "warning")
        self.assertEqual(pose_eval["pose_label"], "Goddess Pose Needs Correction")
        self.assertLess(pose_eval["score"], 85)
        self.assertTrue(flags["pose_ready"])
        self.assertFalse(flags["hold_ready"])

    def test_goddess_prayer_hands_counts_as_valid_pose_variation(self):
        runtime = gd.GoddessRuntime()
        analysis = self.build_analysis(left_knee_ok=True, arms_raised=False, prayer_hands=True)
        joint_states, _ = gd.goddess_build_joint_states(runtime, analysis)
        pose_eval = gd.score_goddess_pose(analysis, analysis["checks"])
        flags = gd.goddess_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertTrue(joint_states["hands_ready"]["ok"])
        self.assertTrue(joint_states["prayer_hands"]["ok"])
        self.assertTrue(joint_states["left_elbow"]["ok"])
        self.assertEqual(joint_states["hands_ready"]["mode"], "prayer")
        self.assertEqual(pose_eval["status"], "perfect")
        self.assertEqual(pose_eval["pose_label"], "Correct Goddess")
        self.assertTrue(flags["hold_ready"])

    def test_goddess_display_model_label_follows_final_pose_when_model_lags(self):
        self.assertEqual(
            gd.goddess_display_model_label("Not_Goddess", "Correct Goddess", pose_ready=True),
            "Goddess",
        )
        self.assertEqual(
            gd.goddess_display_model_label("Not_Goddess", "Not Goddess Pose", pose_ready=False),
            "Not_Goddess",
        )

    def test_goddess_prayer_hands_with_one_noisy_upper_body_check_still_counts(self):
        analysis = self.build_analysis(
            left_knee_ok=True,
            arms_raised=False,
            prayer_hands=True,
            knees_tracking=True,
            shoulders_level=False,
            torso_centered=True,
        )
        pose_eval = gd.score_goddess_pose(analysis, analysis["checks"])
        flags = gd.goddess_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertEqual(pose_eval["pose_label"], "Correct Goddess")
        self.assertEqual(pose_eval["status"], "good")
        self.assertTrue(flags["good_pose_ready"])
        self.assertTrue(flags["hold_ready"])

    def test_goddess_needs_two_alignment_signals_for_hold_ready(self):
        analysis = self.build_analysis(
            left_knee_ok=True,
            arms_raised=False,
            prayer_hands=True,
            knees_tracking=False,
            shoulders_level=False,
            torso_centered=True,
        )
        pose_eval = gd.score_goddess_pose(analysis, analysis["checks"])
        flags = gd.goddess_pose_flags(analysis["checks"], pose_eval["score"], pose_eval["status"])

        self.assertFalse(flags["good_pose_ready"])
        self.assertFalse(flags["hold_ready"])

    def test_goddess_no_landmarks_response_is_safe(self):
        request = self.build_request()
        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(gd, "read_uploaded_image", return_value=frame), \
             patch.object(gd, "check_lighting", return_value=(False, 100.0)), \
             patch.object(gd, "detect_landmarks", return_value=None):
            response = gd.process_goddess_pose_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Unknown")
        self.assertEqual(data["score"], 0)
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(data["points"], [])
        self.assertEqual(data["angle_texts"], [])
        self.assertEqual(data["joint_states"], {})

    def test_goddess_partial_visibility_response_is_safe(self):
        request = self.build_request()
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        landmarks = self.make_fake_landmarks()

        with patch.object(gd, "read_uploaded_image", return_value=frame), \
             patch.object(gd, "check_lighting", return_value=(False, 100.0)), \
             patch.object(gd, "detect_landmarks", return_value=landmarks), \
             patch.object(gd, "is_human_confident", return_value=True), \
             patch.object(gd, "check_body_visibility", return_value=(False, 3, 0.21)):
            response = gd.process_goddess_pose_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Body Not Visible")
        self.assertEqual(data["score"], 0)
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(data["points"], [])
        self.assertEqual(data["angle_texts"], [])
        self.assertEqual(data["joint_states"], {})


class TreeUtilityTests(TestCase):
    def build_request(self, reset=False):
        request = SimpleNamespace(
            POST={"reset": "true"} if reset else {},
            FILES={
                "image": SimpleUploadedFile(
                    "frame.jpg",
                    b"fake-image-bytes",
                    content_type="image/jpeg",
                )
            },
            COOKIES={},
            session={},
        )

        middleware = SessionMiddleware(lambda req: None)
        middleware.process_request(request)
        request.session.save()
        return request

    def parse_json(self, response):
        return json.loads(response.content.decode("utf-8"))

    def make_fake_landmarks(self, visibility=0.95):
        return [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility)
            for _ in range(33)
        ]

    def build_analysis(self, standing_side="left", standing_leg=True, lifted_leg=True, knee_open=True, torso=True, hands_ready=True, left_elbow_ok=True):
        right_elbow_ok = True
        return {
            "standing_side": standing_side,
            "angles": {
                "left_knee_angle": 160.0 if standing_side == "left" else 124.0,
                "right_knee_angle": 160.0 if standing_side == "right" else 124.0,
                "left_elbow_angle": 150.0 if left_elbow_ok else 118.0,
                "right_elbow_angle": 150.0 if right_elbow_ok else 118.0,
                "torso_tilt": 8.0 if torso else 20.0,
            },
            "checks": {
                "standing_leg": standing_leg,
                "foot_place": lifted_leg,
                "no_knee_pressure": lifted_leg,
                "knee_open": knee_open,
                "hands_ready": hands_ready,
                "prayer_hands": False,
                "hands_up": hands_ready,
                "elbows_straight": left_elbow_ok and right_elbow_ok,
                "elbows_soft_ok": left_elbow_ok and right_elbow_ok,
                "left_elbow_ok": left_elbow_ok,
                "right_elbow_ok": right_elbow_ok,
                "left_arm_ready": left_elbow_ok and hands_ready,
                "right_arm_ready": right_elbow_ok and hands_ready,
                "hands_symmetric": True,
                "hands_not_too_wide": True,
                "torso": torso,
                "one_foot_lifted": lifted_leg,
                "strict_tree_gate": standing_leg and lifted_leg and knee_open and torso and hands_ready,
            },
        }

    def build_neutral_tree_points(self):
        raw_pts = np.zeros((33, 3), dtype=np.float32)
        raw_pts[tr.LEFT_SHOULDER] = [0.45, 0.20, 0.0]
        raw_pts[tr.RIGHT_SHOULDER] = [0.55, 0.20, 0.0]
        raw_pts[tr.LEFT_ELBOW] = [0.45, 0.34, 0.0]
        raw_pts[tr.RIGHT_ELBOW] = [0.55, 0.34, 0.0]
        raw_pts[tr.LEFT_WRIST] = [0.45, 0.48, 0.0]
        raw_pts[tr.RIGHT_WRIST] = [0.55, 0.48, 0.0]
        raw_pts[tr.LEFT_HIP] = [0.46, 0.45, 0.0]
        raw_pts[tr.RIGHT_HIP] = [0.54, 0.45, 0.0]
        raw_pts[tr.LEFT_KNEE] = [0.46, 0.65, 0.0]
        raw_pts[tr.RIGHT_KNEE] = [0.54, 0.65, 0.0]
        raw_pts[tr.LEFT_ANKLE] = [0.46, 0.85, 0.0]
        raw_pts[tr.RIGHT_ANKLE] = [0.54, 0.85, 0.0]
        return raw_pts

    def test_tree_reset_flag_clears_runtime_and_hold_state(self):
        request = self.build_request(reset=True)
        tr.POSE_HISTORY.append("Tree")
        tr.DEFECT_HISTORY.append("Perfect_Tree")
        tr.SCORE_HISTORY.append(98.0)
        tr.FEEDBACK_HISTORY.append("Hold steady")
        tr.TORSO_CENTER_HISTORY.append(0.5)
        tr.SHOULDER_TILT_HISTORY.append(0.1)
        tr.TORSO_LEAN_HISTORY.append(0.1)
        tr.POINT_HISTORY["tree_11"] = [(0.1, 0.2)]
        tr.TREE_HOLD_START = 123.0
        tr.BEST_HOLD_TIME = 8.5
        tr.PERFECT_HOLD_COUNT = 2

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(tr, "read_uploaded_image", return_value=frame), \
             patch.object(tr, "enhance_frame", return_value=frame), \
             patch.object(tr, "check_lighting", return_value=(True, 12.0)):
            response = tr.process_yoga_pose_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Low Light")
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(len(tr.POSE_HISTORY), 0)
        self.assertEqual(len(tr.DEFECT_HISTORY), 0)
        self.assertEqual(len(tr.SCORE_HISTORY), 0)
        self.assertEqual(len(tr.FEEDBACK_HISTORY), 0)
        self.assertEqual(len(tr.TORSO_CENTER_HISTORY), 0)
        self.assertEqual(len(tr.SHOULDER_TILT_HISTORY), 0)
        self.assertEqual(len(tr.TORSO_LEAN_HISTORY), 0)
        self.assertEqual(tr.POINT_HISTORY, {})
        self.assertEqual(tr.BOOLEAN_HISTORY, {})
        self.assertEqual(tr.BEST_HOLD_TIME, 0.0)
        self.assertEqual(tr.PERFECT_HOLD_COUNT, 0)

    def test_tree_no_landmarks_response_is_safe(self):
        request = self.build_request()
        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(tr, "read_uploaded_image", return_value=frame), \
             patch.object(tr, "enhance_frame", return_value=frame), \
             patch.object(tr, "check_lighting", return_value=(False, 100.0)), \
             patch.object(tr, "detect_landmarks", return_value=None):
            response = tr.process_yoga_pose_request(request)

        data = self.parse_json(response)
        self.assertEqual(data["pose"], "Unknown")
        self.assertEqual(data["score"], 0)
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(data["points"], [])
        self.assertEqual(data["angle_texts"], [])
        self.assertEqual(data["joint_states"], {})

    def test_tree_joint_states_include_expected_keys_and_use_hysteresis(self):
        tr.reset_tree_runtime_state()

        joint_states, _ = tr.build_tree_joint_states(self.build_analysis(left_elbow_ok=True))
        self.assertIn("left_knee", joint_states)
        self.assertIn("right_knee", joint_states)
        self.assertIn("standing_leg", joint_states)
        self.assertIn("lifted_leg", joint_states)
        self.assertIn("torso", joint_states)
        self.assertIn("hands", joint_states)
        self.assertIn("left_elbow", joint_states)
        self.assertIn("right_elbow", joint_states)
        self.assertTrue(joint_states["left_elbow"]["ok"])
        self.assertEqual(joint_states["left_elbow"]["threshold"], tr.TREE_ELBOW_SOFT_MIN)

        joint_states, _ = tr.build_tree_joint_states(self.build_analysis(left_elbow_ok=False))
        self.assertTrue(joint_states["left_elbow"]["ok"], "A single noisy frame should not immediately flip the elbow state red.")

        joint_states, _ = tr.build_tree_joint_states(self.build_analysis(left_elbow_ok=False))
        self.assertFalse(joint_states["left_elbow"]["ok"], "Two consecutive bad frames should eventually flip the elbow state.")

    def test_tree_feedback_drops_step_labels(self):
        analysis = tr.analyze_tree_pose(self.build_neutral_tree_points())

        self.assertNotIn("Step", analysis["main_feedback"])
        self.assertTrue(analysis["coach_text"])

    def test_tree_points_are_smoothed_before_frontend_response(self):
        tr.reset_tree_runtime_state()
        landmarks = self.make_fake_landmarks()
        first_frame = self.build_neutral_tree_points()
        second_frame = first_frame.copy()
        second_frame[tr.LEFT_KNEE] = [0.92, 0.18, 0.0]
        analysis = self.build_analysis()
        analysis["score"] = 92

        tr.smooth_tree_points(first_frame, landmarks)
        smoothed_second = tr.smooth_tree_points(second_frame, landmarks)
        points = tr.build_points_for_frontend(smoothed_second, landmarks, analysis)
        left_knee = next(point for point in points if point["name"] == "left_knee")

        self.assertLess(smoothed_second[tr.LEFT_KNEE][0], second_frame[tr.LEFT_KNEE][0])
        self.assertAlmostEqual(left_knee["x"], float(smoothed_second[tr.LEFT_KNEE][0]), places=3)

    def test_tree_angle_texts_use_clean_degree_symbol(self):
        tr.reset_tree_runtime_state()
        landmarks = self.make_fake_landmarks()
        raw_pts = self.build_neutral_tree_points()
        tr.smooth_tree_points(raw_pts, landmarks)
        analysis = self.build_analysis()

        items = tr.build_angle_texts(raw_pts, landmarks, analysis)

        self.assertTrue(items)
        self.assertTrue(all(tr.TREE_DEGREE_SIGN in item["text"] for item in items))
        self.assertTrue(all("Ã‚" not in item["text"] for item in items))


class WarriorUtilityTests(TestCase):
    def build_request(self):
        request = SimpleNamespace(
            POST={},
            FILES={
                "image": SimpleUploadedFile(
                    "frame.jpg",
                    b"fake-image-bytes",
                    content_type="image/jpeg",
                )
            },
            COOKIES={},
            session={},
        )

        middleware = SessionMiddleware(lambda req: None)
        middleware.process_request(request)
        request.session.save()
        return request

    def parse_json(self, response):
        return json.loads(response.content.decode("utf-8"))

    def make_fake_landmarks(self, visibility=0.95):
        return [
            SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility)
            for _ in range(33)
        ]

    def build_analysis(self, front_side="left", arms_level=True, front_knee_bent=True, back_leg_soft=True):
        return {
            "score": 96 if arms_level and front_knee_bent and back_leg_soft else 78,
            "status": "perfect" if arms_level and front_knee_bent and back_leg_soft else "good",
            "main_feedback": "Perfect Warrior! Hold steady." if arms_level and front_knee_bent and back_leg_soft else "Refine your Warrior II.",
            "tips": ["Keep your gaze soft over the front fingertips"],
            "angles": {
                "left_knee_angle": 104.0,
                "right_knee_angle": 168.0,
                "left_elbow_angle": 174.0,
                "right_elbow_angle": 173.0,
            },
            "checks": {
                "front_side": front_side,
                "stance_ratio": 1.58,
                "stance_wide_enough": True,
                "stance_ideal": True,
                "front_knee_bent": front_knee_bent,
                "front_knee_ideal": True,
                "back_leg_soft": back_leg_soft,
                "back_leg_straight": True,
                "arms_reaching": True,
                "arms_level": arms_level,
                "torso_centered": True,
                "front_knee_over_ankle": True,
                "core_warrior_gate": front_knee_bent and back_leg_soft,
                "balanced_warrior_gate": arms_level and front_knee_bent and back_leg_soft,
            },
        }

    def test_warrior_reset_clears_runtime_histories_on_low_light(self):
        request = self.build_request()
        runtime = wr.wr_get_runtime(request)
        runtime.pose_history.append("Warrior Pose")
        runtime.score_history.append(92.0)
        runtime.feedback_history.append("Hold steady")
        runtime.point_history["wr_25"] = [(0.1, 0.2, 0.0)]
        runtime.hold_start = 123.0
        runtime.best_hold_time = 7.2
        runtime.perfect_hold_count = 2
        wr.wr_store_runtime(request, runtime)

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(wr, "wr_read_uploaded_image", return_value=frame), \
             patch.object(wr, "wr_enhance_frame", return_value=frame), \
             patch.object(wr, "wr_check_lighting", return_value=(True, 12.0)):
            response = wr.process_warrior_pose_request(request)

        data = self.parse_json(response)
        runtime = wr.wr_get_runtime(request)
        self.assertEqual(data["pose"], "Low Light")
        self.assertEqual(data["hold_time"], 0.0)
        self.assertEqual(data["best_hold_time"], 0.0)
        self.assertEqual(len(runtime.pose_history), 0)
        self.assertEqual(len(runtime.score_history), 0)
        self.assertEqual(len(runtime.feedback_history), 0)
        self.assertEqual(runtime.point_history, {})
        self.assertEqual(runtime.best_hold_time, 0.0)
        self.assertEqual(runtime.perfect_hold_count, 0)

    def test_warrior_angle_texts_use_clean_degree_symbol(self):
        raw_pts = np.zeros((33, 3), dtype=np.float32)
        landmarks = self.make_fake_landmarks()
        analysis = self.build_analysis()

        items = wr.wr_build_angle_texts(raw_pts, landmarks, analysis)

        self.assertTrue(items)
        self.assertTrue(all("Â" not in item["text"] for item in items))
        self.assertTrue(all(wr.WR_DEGREE_SIGN in item["text"] for item in items))
        self.assertTrue(all(item.get("joint_key") for item in items))

    def test_warrior_points_keep_recently_visible_joint_markers(self):
        runtime = wr.WarriorRuntime()
        runtime.point_history["wr_25"] = [(0.2, 0.6, 0.0)]

        raw_pts = np.zeros((33, 3), dtype=np.float32)
        for idx in range(33):
            raw_pts[idx] = [0.35 + (idx % 4) * 0.08, 0.20 + (idx % 5) * 0.09, 0.0]

        landmarks = self.make_fake_landmarks()
        landmarks[wr.WR_LEFT_KNEE].visibility = 0.05

        points = wr.wr_build_points_for_frontend(runtime, raw_pts, landmarks, self.build_analysis(front_side="left"))
        names = {point["name"] for point in points}
        left_knee = next(point for point in points if point["name"] == "left_knee")

        self.assertIn("left_knee", names)
        self.assertAlmostEqual(left_knee["x"], 0.2, places=3)
        self.assertAlmostEqual(left_knee["y"], 0.6, places=3)

    def test_warrior_visible_point_jump_snaps_to_current_body(self):
        runtime = wr.WarriorRuntime()
        runtime.point_history["wr_25"] = [(0.1, 0.1, 0.0)]

        x, y, z = wr.wr_smooth_point(runtime, "wr_25", 0.55, 0.56, 0.0, visibility=0.95)

        self.assertAlmostEqual(x, 0.55, places=3)
        self.assertAlmostEqual(y, 0.56, places=3)
        self.assertAlmostEqual(z, 0.0, places=3)

    def test_warrior_display_model_label_follows_final_pose_when_model_lags(self):
        self.assertEqual(
            wr.wr_display_model_label("Not_Warrior", "Correct Warrior", pose_ready=True),
            "Warrior",
        )
        self.assertEqual(
            wr.wr_display_model_label("Not_Warrior", "Not Warrior Pose", pose_ready=False),
            "Not_Warrior",
        )

    def test_warrior_pose_flags_allow_hold_ready_for_balanced_pose(self):
        analysis = self.build_analysis()
        flags = wr.wr_pose_flags(analysis["checks"], stable_score=96, pose_status=analysis["status"])

        self.assertTrue(flags["pose_ready"])
        self.assertTrue(flags["good_pose_ready"])
        self.assertTrue(flags["hold_ready"])


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    DEFAULT_FROM_EMAIL="noreply@sattvalife.test",
)
class ReportEmailViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = Users(
            name="Asha",
            email="asha@example.com",
            address="Kochi",
            status=Users.STATUS_ACCEPTED,
            photo="user_photos/profile.jpg",
        )
        self.user.set_password("Password@123")
        self.user.save()

        session = self.client.session
        session[APP_USER_SESSION_KEY] = self.user.pk
        session.save()

    def test_email_pose_report_sends_pdf_attachment_to_current_user(self):
        report = SimpleUploadedFile(
            "tree-report.pdf",
            b"%PDF-1.4 report",
            content_type="application/pdf",
        )

        response = self.client.post(
            "/user/email-report/",
            {"pose": "Tree Pose", "report": report},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, [self.user.email])
        self.assertIn("Tree Pose", mail.outbox[0].subject)
        self.assertEqual(mail.outbox[0].attachments[0][0], "tree-report.pdf")
