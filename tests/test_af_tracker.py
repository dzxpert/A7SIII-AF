"""
Unit Test Suite for Sony Alpha AF & Tracking Subsystems
Verifies clean-room implementations against reverse engineered specifications.
"""

import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pdaf import (
        PDAFGrid,
        BoundingBox,
        AFAreaType,
        MarioAfcStateEnum,
        MarioAfcStateMachine,
        PDAFPoint
    )
    from contrast_af import (
        ContrastAFEngine,
        ContrastAFState,
        ImagerNoiseEstimator
    )
    from staple_tracker import (
        HannWindow2D,
        ColorHistogramLearner,
        SingleTargetStapleTracker,
        DualTargetStapleManager
    )
    from eye_face_arbiter import (
        EyeSelector,
        PriorityFaceArbiter,
        SelectedEye,
        TrackingTargetTier,
        EyeCandidate,
        FaceCandidate
    )
except ImportError:
    from sony_af_tracker.pdaf import (
        PDAFGrid,
        BoundingBox,
        AFAreaType,
        MarioAfcStateEnum,
        MarioAfcStateMachine,
        PDAFPoint
    )
    from sony_af_tracker.contrast_af import (
        ContrastAFEngine,
        ContrastAFState,
        ImagerNoiseEstimator
    )
    from sony_af_tracker.staple_tracker import (
        HannWindow2D,
        ColorHistogramLearner,
        SingleTargetStapleTracker,
        DualTargetStapleManager
    )
    from sony_af_tracker.eye_face_arbiter import (
        EyeSelector,
        PriorityFaceArbiter,
        SelectedEye,
        TrackingTargetTier,
        EyeCandidate,
        FaceCandidate
    )


class TestPDAFSubsystem(unittest.TestCase):
    def setUp(self):
        self.grid = PDAFGrid()

    def test_grid_geometry(self):
        """Verify 841-point (29x29) layout matching Sony zpd_max."""
        self.assertEqual(len(self.grid.points), 841)
        self.assertEqual(self.grid.points[0].index, 0)
        self.assertEqual(self.grid.points[-1].index, 840)

        # First point at top-left
        pt0 = self.grid.points[0]
        self.assertEqual((pt0.grid_x, pt0.grid_y), (0, 0))
        self.assertAlmostEqual(pt0.norm_x, 0.5 / 29.0)
        self.assertAlmostEqual(pt0.norm_y, 0.5 / 29.0)

    def test_area_selection(self):
        """Verify point counts for various Sony focus areas."""
        wide_pts = self.grid.get_active_points_for_area(AFAreaType.WIDE)
        self.assertEqual(len(wide_pts), 841)

        center_pts = self.grid.get_active_points_for_area(AFAreaType.CENTER)
        self.assertEqual(len(center_pts), 25) # 5x5

        spot_s = self.grid.get_active_points_for_area(AFAreaType.FLEXIBLE_SPOT_S, (0.5, 0.5))
        self.assertEqual(len(spot_s), 1)

        spot_m = self.grid.get_active_points_for_area(AFAreaType.FLEXIBLE_SPOT_M, (0.5, 0.5))
        self.assertEqual(len(spot_m), 9) # 3x3

        spot_l = self.grid.get_active_points_for_area(AFAreaType.FLEXIBLE_SPOT_L, (0.5, 0.5))
        self.assertEqual(len(spot_l), 25) # 5x5

        zone = self.grid.get_active_points_for_area(AFAreaType.ZONE, (0.5, 0.5))
        self.assertEqual(len(zone), 81) # 9x9

    def test_defocus_metric_evaluation(self):
        """Verify sub_271A90 defocus evaluation logic."""
        test_pts = [
            PDAFPoint(0, 0, 0, 0.1, 0.1, defocus=0.2, confidence=0.85, error_flags=0x30),
            PDAFPoint(1, 0, 1, 0.1, 0.2, defocus=-0.3, confidence=0.90, error_flags=0x30),
            PDAFPoint(2, 0, 2, 0.1, 0.3, defocus=5.0, confidence=0.10, error_flags=0), # Low conf / no 0x30
            PDAFPoint(3, 0, 3, 0.1, 0.4, defocus=0.1, confidence=0.95, error_flags=0x9000), # Sensor error
        ]

        res = self.grid.evaluate_points(test_pts, confidence_thresh=0.4, dof_tolerance=0.5)
        self.assertEqual(res.total_points, 4)
        self.assertEqual(res.num_no_err, 3) # Excludes pt 3
        self.assertEqual(res.num_reliable, 2) # Pts 0 and 1
        self.assertEqual(res.num_lockon, 2) # Both within 0.5 DoF
        self.assertAlmostEqual(res.sum_df_abs, 5.5) # sub_6FC676 accumulates all num_no_err
        self.assertTrue(res.is_locked)

    def test_mario_afc_state_machine(self):
        """Verify MarioAfcState transitions from STOP to LOCKED."""
        sm = MarioAfcStateMachine()
        self.assertEqual(sm.state, MarioAfcStateEnum.STOP)

        # S1 half-press
        sm.handle_event("S1_PRESS")
        self.assertEqual(sm.state, MarioAfcStateEnum.WAIT)

        # Target acquisition
        sm.handle_event("TARGET_UPDATE")
        self.assertEqual(sm.state, MarioAfcStateEnum.SEARCH)

        # Far out of focus (|Δd| = 4.5 > approach_thresh 3.0)
        eval_far = self.grid.evaluate_points([
            PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=4.5, confidence=0.8)
        ])
        sm.handle_event("EVAL_STEP", eval_far)
        self.assertEqual(sm.state, MarioAfcStateEnum.APPROACH)

        # Closing in (|Δd| = 2.0 <= 3.0)
        eval_mid = self.grid.evaluate_points([
            PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=2.0, confidence=0.8)
        ])
        sm.handle_event("EVAL_STEP", eval_mid)
        self.assertEqual(sm.state, MarioAfcStateEnum.TRACKING)

        # Near peak (|Δd| = 0.5 <= 0.8)
        eval_near = self.grid.evaluate_points([
            PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=0.5, confidence=0.8)
        ])
        sm.handle_event("EVAL_STEP", eval_near)
        self.assertEqual(sm.state, MarioAfcStateEnum.MOVE_TO_PEAK)

        # Locked in focus (|Δd| = 0.1 <= 0.5, locked)
        eval_locked = self.grid.evaluate_points([
            PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=0.1, confidence=0.9)
        ])
        sm.handle_event("EVAL_STEP", eval_locked)
        self.assertEqual(sm.state, MarioAfcStateEnum.LOCKED)


class TestContrastAFSubsystem(unittest.TestCase):
    def test_contrast_calculation(self):
        """Verify spatial gradient contrast energy."""
        flat_img = np.ones((50, 50), dtype=np.float32) * 128.0
        self.assertAlmostEqual(ContrastAFEngine.calculate_contrast(flat_img), 0.0)

        edge_img = np.zeros((50, 50), dtype=np.float32)
        edge_img[:, 25:] = 255.0
        c_val = ContrastAFEngine.calculate_contrast(edge_img)
        self.assertGreater(c_val, 10.0)

    def test_wobble_and_yama_simulation(self):
        """Simulate hill-climbing sweep with peak at pos=65.0."""
        engine = ContrastAFEngine(wobble_amplitude=0.2, wobble_cycles=2, lens_min_pos=0.0, lens_max_pos=100.0)
        engine.start_af(initial_lens_pos=50.0)
        self.assertEqual(engine.state, ContrastAFState.WOBBLE)

        def synthetic_contrast(pos):
            # Gaussian contrast peak at pos 65.0
            return float(100.0 * np.exp(-0.5 * ((pos - 65.0) / 8.0) ** 2))

        # Run multiple cycles
        for _ in range(40):
            pos = engine.current_pos
            c = synthetic_contrast(pos)
            new_pos, state = engine.step(c)
            if state == ContrastAFState.IN_FOCUS:
                break

        # Should find peak close to 65.0
        self.assertEqual(engine.state, ContrastAFState.IN_FOCUS)
        self.assertAlmostEqual(engine.current_pos, 65.0, delta=2.0)


class TestStapleTrackerSubsystem(unittest.TestCase):
    def test_hann_window(self):
        """Verify 2D Hann window energy and boundary zeros."""
        w, h = 33, 33  # Odd dimension has exact center at index 16
        win = HannWindow2D.generate(w, h)
        self.assertEqual(win.shape, (h, w))
        self.assertAlmostEqual(win[0, 0], 0.0, places=4)
        self.assertAlmostEqual(win[h - 1, w - 1], 0.0, places=4)
        self.assertAlmostEqual(win[h // 2, w // 2], 1.0, places=2)

    def test_tracker_target_tracking(self):
        """Verify correlation filter tracking on synthetic translating object."""
        tracker = SingleTargetStapleTracker(target_id=0)

        # Create canvas with a bright square at (40, 40)
        frame1 = np.ones((120, 120, 3), dtype=np.float32) * 40.0
        frame1[40:60, 40:60, :] = 220.0
        init_box = (40.0, 40.0, 20.0, 20.0)

        tracker.init_target(frame1, init_box)
        self.assertTrue(tracker.is_tracking)

        # Move object by dx=4, dy=3 in frame 2
        frame2 = np.ones((120, 120, 3), dtype=np.float32) * 40.0
        frame2[43:63, 44:64, :] = 220.0

        res = tracker.track(frame2)
        self.assertTrue(res.is_reliable)
        self.assertAlmostEqual(res.x, 44.0, delta=2.0)
        self.assertAlmostEqual(res.y, 43.0, delta=2.0)

    def test_dual_target_manager(self):
        """Verify dual target tracking manager matching sub_280C3C."""
        mgr = DualTargetStapleManager()
        frame = np.ones((100, 100, 3), dtype=np.float32) * 50.0
        frame[20:35, 20:35] = 200.0
        frame[60:75, 60:75] = 200.0

        mgr.set_target(0, frame, (20.0, 20.0, 15.0, 15.0))
        mgr.set_target(1, frame, (60.0, 60.0, 15.0, 15.0))

        results = mgr.process_frame(frame)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].target_id, 0)
        self.assertEqual(results[1].target_id, 1)


class TestEyeAndFaceArbiter(unittest.TestCase):
    def setUp(self):
        self.eye_selector = EyeSelector()
        self.arbiter = PriorityFaceArbiter()

    def test_adaptive_threshold(self):
        """Verify scale-adaptive threshold formula from sub_278858."""
        thresh_small = self.eye_selector.compute_adaptive_threshold(face_size=20.0)
        thresh_large = self.eye_selector.compute_adaptive_threshold(face_size=250.0)
        # Small faces have tighter search tolerance than large portraits
        self.assertLess(thresh_small, thresh_large)

    def test_eye_selection_proximity(self):
        """Verify closer eye selection to tracking center from sub_278778."""
        face = FaceCandidate(
            face_id=1, left=0.3, top=0.3, right=0.5, bottom=0.5, confidence=0.9,
            eye_left=EyeCandidate(x=0.45, y=0.35, confidence=0.95),
            eye_right=EyeCandidate(x=0.35, y=0.35, confidence=0.95)
        )

        # Center point close to right eye (0.35, 0.35)
        chosen_eye, eye_cand = self.eye_selector.select_eye(face, tracking_center=(0.34, 0.35))
        self.assertEqual(chosen_eye, SelectedEye.RIGHT_EYE)
        self.assertEqual(eye_cand.x, 0.35)

        # Center point close to left eye (0.45, 0.35)
        chosen_eye_l, eye_cand_l = self.eye_selector.select_eye(face, tracking_center=(0.46, 0.35))
        self.assertEqual(chosen_eye_l, SelectedEye.LEFT_EYE)
        self.assertEqual(eye_cand_l.x, 0.45)

    def test_face_priority_arbitration(self):
        """Verify defocus-weighted face selection from sub_271C00 & sub_271FA8."""
        face1 = FaceCandidate(
            face_id=1, left=0.1, top=0.1, right=0.3, bottom=0.3,
            confidence=0.8, defocus_error=3.5, reliable_pdaf_count=5 # Far out of focus
        )
        face2 = FaceCandidate(
            face_id=2, left=0.5, top=0.5, right=0.7, bottom=0.7,
            confidence=0.8, defocus_error=0.2, reliable_pdaf_count=5 # Nearly in focus
        )

        chosen, tier, eye = self.arbiter.update_candidates([face1, face2])
        # Face 2 should be chosen due to lower defocus error
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.face_id, 2)


if __name__ == '__main__':
    unittest.main()
