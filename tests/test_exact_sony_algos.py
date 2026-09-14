import unittest
import struct
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pdaf import (
        PDAFPoint,
        PDAFGrid,
        BoundingBox,
        DefocusEvaluation,
        MarioAfcStateMachine,
        MarioAfcStateEnum
    )
    from contrast_af import (
        ContrastAFEngine,
        ContrastAFState
    )
    from eye_face_arbiter import (
        EyeCandidate,
        FaceCandidate,
        EyeSelector,
        PriorityFaceArbiter,
        SelectedEye,
        TrackingTargetTier
    )
    from staple_tracker import (
        SingleTargetStapleTracker,
        DualTargetStapleManager,
        TrackedTargetResult
    )
except ImportError:
    from sony_af_tracker.pdaf import (
        PDAFPoint,
        PDAFGrid,
        BoundingBox,
        DefocusEvaluation,
        MarioAfcStateMachine,
        MarioAfcStateEnum
    )
    from sony_af_tracker.contrast_af import (
        ContrastAFEngine,
        ContrastAFState
    )
    from sony_af_tracker.eye_face_arbiter import (
        EyeCandidate,
        FaceCandidate,
        EyeSelector,
        PriorityFaceArbiter,
        SelectedEye,
        TrackingTargetTier
    )
    from sony_af_tracker.staple_tracker import (
        SingleTargetStapleTracker,
        DualTargetStapleManager,
        TrackedTargetResult
    )


class TestExactSonyPDAFAndDebounce(unittest.TestCase):
    """Verifies exact sub_271A90, sub_6FC676, and sub_8E5BF8 algorithms."""

    def test_hardware_20byte_record_unpacking(self):
        """Verifies 20-byte record unpacking and binning shift (sub_2718A4)."""
        # x=100, y=200, flags=0x30 (reliable), df=-120, peak=800, norm=1000
        rec = struct.pack('<hhIiII', 100, 200, 0x30, -120, 800, 1000)
        pt = PDAFPoint.from_hw_bytes(rec, index=42, grid_x=10, grid_y=15)

        self.assertEqual(pt.index, 42)
        self.assertEqual(pt.defocus, -120.0)
        self.assertEqual(pt.error_flags, 0x30)
        self.assertAlmostEqual(pt.confidence, 0.80, places=3)

    def test_sub_6fc676_metric_accumulation(self):
        """Verifies exact error mask 0x90CF and metric accumulation from sub_6FC676."""
        grid = PDAFGrid()
        points = [
            # Valid, reliable, lock-on (df=0.2 within dof=0.5)
            PDAFPoint(0, 0, 0, 0.1, 0.1, defocus=0.2, confidence=0.8, error_flags=0x30),
            # Valid, reliable, out of lock-on (df=2.5)
            PDAFPoint(1, 0, 1, 0.1, 0.2, defocus=-2.5, confidence=0.9, error_flags=0x30),
            # Hardware error (flag & 0x90CF != 0) -> must be discarded
            PDAFPoint(2, 0, 2, 0.1, 0.3, defocus=0.1, confidence=0.9, error_flags=0x0001),
            # Hardware error (0x8000 bit set) -> discarded
            PDAFPoint(3, 0, 3, 0.1, 0.4, defocus=0.0, confidence=0.9, error_flags=0x8000),
        ]

        res = grid.evaluate_points(points, dof_tolerance=0.50)

        # 4 total points, 2 without error (points 0 and 1)
        self.assertEqual(res.num_no_err, 2)
        self.assertEqual(res.num_reliable, 2)
        self.assertEqual(res.num_lockon, 1)
        # sum_df_abs = abs(0.2) + abs(-2.5) = 2.7
        self.assertAlmostEqual(res.sum_df_abs, 2.7, places=3)
        # min_df_abs among lock-on points = 0.2
        self.assertAlmostEqual(res.min_df_abs, 0.2, places=3)

    def test_sub_8e5bf8_debounce_matrix(self):
        """Verifies exact 5x5 debounce matrix timing from sub_8E5BF8 (dword_C747C4)."""
        fsm = MarioAfcStateMachine()
        fsm.state = MarioAfcStateEnum.LOCKED  # Slot 1 in firmware (sub_8E5BF8)
        fsm.state_entry_ticks = [None] * 5
        fsm.state_entry_ticks[1] = 1000000  # Entered at tick 1,000,000

        # Attempt transition to TRACKING (Slot 2) after only 200,000 ticks
        # Required ticks: dword_C747C4[1 * 5 + 2] = 400,000 ticks
        current_tick = 1200000  # Elapsed = 200,000 < 400,000
        success = fsm.try_transition(MarioAfcStateEnum.TRACKING, current_tick, force=False)
        self.assertFalse(success, "Transition must be rejected when elapsed ticks < debounce threshold")
        self.assertEqual(fsm.state, MarioAfcStateEnum.LOCKED)

        # Attempt transition after 500,000 ticks from initial request (1,200,000 + 500,000 = 1,700,000 >= 400,000)
        current_tick = 1700000
        success = fsm.try_transition(MarioAfcStateEnum.TRACKING, current_tick, force=False)
        self.assertTrue(success, "Transition must succeed when elapsed ticks >= debounce threshold")
        self.assertEqual(fsm.state, MarioAfcStateEnum.TRACKING)


class TestExactSonyEyeAndFaceAlgorithms(unittest.TestCase):
    """Verifies sub_278778, sub_278858, sub_709FD6, and sub_272420."""

    def setUp(self):
        self.eye_selector = EyeSelector(
            face_size_min=16.0,
            face_size_max=320.0,
            upper_ratio=8.0,
            lower_ratio=3.0
        )
        self.arbiter = PriorityFaceArbiter()

    def test_sub_278858_piecewise_linear_radius(self):
        """Verifies exact formula: v7 = (v5*v4) + (s - v4)*(v6*v3 - v5*v4)/(v3 - v4) >> 4."""
        # s <= v4 (16.0): val = v5 * s = 8.0 * 10.0 = 80.0 -> >> 4 = 5.0
        r_small = self.eye_selector.compute_spatial_gating_radius(10.0)
        self.assertAlmostEqual(r_small, 5.0, delta=0.5)

        # s >= v3 (320.0): val = v6 * s = 3.0 * 350.0 = 1050.0 -> >> 4 = 65
        r_large = self.eye_selector.compute_spatial_gating_radius(350.0)
        self.assertAlmostEqual(r_large, 65.0, delta=0.5)

    def test_sub_709fd6_spatial_gating_outlier_rejection(self):
        """Verifies that an eye detection outside spatial acceptance radius is rejected."""
        face = FaceCandidate(
            face_id=1, left=100.0, top=100.0, right=200.0, bottom=200.0, confidence=0.9,
            # Eye right at normal position
            eye_right=EyeCandidate(x=130.0, y=130.0, confidence=0.85),
            # Eye left way off in the corner (outlier/false positive detection)
            eye_left=EyeCandidate(x=800.0, y=800.0, confidence=0.95)
        )

        chosen_eye, eye_cand = self.eye_selector.select_eye(face, tracking_center=(150.0, 150.0))
        self.assertEqual(chosen_eye, SelectedEye.RIGHT_EYE)
        self.assertEqual(eye_cand.x, 130.0)

    def test_sub_272420_min_reliable_defocus_selection(self):
        """Verifies that the face with minimum defocus (|Δd|) among reliable candidates is selected."""
        face1 = FaceCandidate(
            face_id=10, left=0.1, top=0.1, right=0.3, bottom=0.3,
            confidence=0.95, defocus_error=4.5, reliable_pdaf_count=10
        )
        face2 = FaceCandidate(
            face_id=20, left=0.4, top=0.4, right=0.6, bottom=0.6,
            confidence=0.75, defocus_error=0.15, reliable_pdaf_count=8  # Min defocus!
        )
        face3 = FaceCandidate(
            face_id=30, left=0.7, top=0.7, right=0.9, bottom=0.9,
            confidence=0.99, defocus_error=0.05, reliable_pdaf_count=0  # Zero reliable PDAF points
        )

        # Candidate 3 has zero reliable PDAF points, so candidate 2 must be chosen (Min reliable DF FaceID)
        chosen, tier, eye = self.arbiter.update_candidates([face1, face2, face3])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.face_id, 20)


class TestExactSonyContrastAndTracking(unittest.TestCase):
    """Verifies sub_8A5122, sub_8A520C, and sub_705BC6."""

    def test_sub_8a5122_and_sub_8a520c_transitions(self):
        """Verifies state dispatch functions from Camera::LC::MovieContrastAF."""
        engine = ContrastAFEngine()

        # sub_8A5122: when gradient detected -> transition to YAMA
        st = engine.transition_from_wob(has_gradient=True, is_idle=False, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st, ContrastAFState.YAMA)

        # sub_8A5122: when speed mode and high speed requested -> HIGH_SPEED
        st2 = engine.transition_from_wob(has_gradient=False, is_idle=False, is_speed_mode=True, is_high_speed_req=True)
        self.assertEqual(st2, ContrastAFState.HIGH_SPEED)

        # sub_8A520C: when stay_in_yama is True -> returns YAMA
        st3 = engine.transition_from_yama(stay_in_yama=True, is_idle=False, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st3, ContrastAFState.YAMA)
        # sub_8A520C: when speed mode and high speed requested -> HIGH_SPEED
        st4 = engine.transition_from_yama(stay_in_yama=False, is_idle=False, is_speed_mode=True, is_high_speed_req=True)
        self.assertEqual(st4, ContrastAFState.HIGH_SPEED)

    def test_sub_705bc6_displacement_integration(self):
        """Verifies position integration and 20px border margin check from sub_705BC6."""
        tracker = SingleTargetStapleTracker(target_id=0)
        # Create frame of size 200x200
        frame = np.ones((200, 200, 3), dtype=np.float32) * 50.0
        # Put target inside safe region (e.g. at 50, 50)
        frame[50:80, 50:80, :] = 200.0
        tracker.init_target(frame, (50.0, 50.0, 30.0, 30.0))

        # Track on same frame
        res = tracker.track(frame)
        self.assertTrue(res.is_reliable)
        self.assertEqual(res.error_code, 0)

        # Test border caution: place target near boundary (x=10, y=10 within 20px margin)
        frame_border = np.ones((200, 200, 3), dtype=np.float32) * 50.0
        frame_border[10:40, 10:40, :] = 200.0
        tracker.init_target(frame_border, (10.0, 10.0, 30.0, 30.0))
        res_border = tracker.track(frame_border)
        # Border check should flag error_code 3 (near boundary caution)
        self.assertEqual(res_border.error_code, 3)


if __name__ == '__main__':
    unittest.main()
