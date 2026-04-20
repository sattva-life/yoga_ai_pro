import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from django.contrib.sessions.middleware import SessionMiddleware
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from User.utils import down_dog_utility as dd
from User.utils import goddess_utility as gd
from User.utils import tree_utility as tr


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

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(dd, "dd_read_uploaded_image", return_value=frame), \
             patch.object(dd, "dd_enhance_frame", return_value=frame), \
             patch.object(dd, "dd_check_lighting", return_value=(True, 12.0)):
            response = dd.process_down_dog_request(request)

        data = self.parse_json(response)
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

        frame = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(gd, "read_uploaded_image", return_value=frame), \
             patch.object(gd, "check_lighting", return_value=(True, 12.0)):
            response = gd.process_goddess_pose_request(request)

        data = self.parse_json(response)
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
