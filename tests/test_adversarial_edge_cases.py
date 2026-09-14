"""
Sony Alpha 7S III (BIONZ XR CXD90057) - Adversarial Stress & Edge-Case Test Suite
R2 Automated Adversarial Test Execution for all 12 Reversed Functions:
1. sub_272420: Defocus Face Selection (8-slot, 60-byte records, min_df, reliability gating)
2. sub_271C00: Defocus Face Priority Arbiter
3. sub_271FA8: Defocus Face Queue Reordering & Hysteresis
4. sub_709FD6: Eye Spatial Gating Outlier Rejection
5. sub_278778: Eye Selection Proximity
6. sub_278858: Piecewise Linear Radius Interpolation
7. sub_2718A4: Hardware 20-byte Record Unpacking
8. sub_271A90: Defocus Info Translation & Bitmask Filter (0x90CF)
9. sub_6FC676: PDAF Metric Accumulation
10. sub_8E5BF8: Mario AFC 5x5 Debounce Matrix
11. sub_8A5122: Contrast AF Wobble State Transition (CStateWob)
12. sub_8A520C: Contrast AF Yama State Transition (CStateYama)
Plus sub_705BC6 & sub_71E28A: Staple Tracking displacement integration & 20px border margin checks.
"""

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
        MarioAfcStateEnum,
        AFAreaType
    )
    from contrast_af import (
        ContrastAFEngine,
        ContrastAFState,
        ImagerNoiseEstimator
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
        TrackedTargetResult,
        CorrelationFilterTracker,
        ColorHistogramLearner,
        HannWindow2D
    )
except ImportError:
    from sony_af_tracker.pdaf import (
        PDAFPoint,
        PDAFGrid,
        BoundingBox,
        DefocusEvaluation,
        MarioAfcStateMachine,
        MarioAfcStateEnum,
        AFAreaType
    )
    from sony_af_tracker.contrast_af import (
        ContrastAFEngine,
        ContrastAFState,
        ImagerNoiseEstimator
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
        TrackedTargetResult,
        CorrelationFilterTracker,
        ColorHistogramLearner,
        HannWindow2D
    )


class TestPDAFHardwareAndAccumulationAdversarial(unittest.TestCase):
    """
    Adversarial stress-testing of PDAF Hardware Records, Error Bitmasks,
    and Accumulation Metrics (sub_2718A4, sub_271A90, sub_6FC676).
    """

    def setUp(self):
        self.grid = PDAFGrid()

    def test_zero_reliable_points_all_error_flags_set(self):
        """
        Stress Test: All 841 points have error bitmask 0x90CF set.
        Firmware sub_271A90 condition: (*v10 & 0x90CF) == 0.
        All points must be rejected; no accumulation allowed.
        """
        bad_points = []
        for i in range(841):
            bad_points.append(
                PDAFPoint(
                    index=i, grid_x=i % 29, grid_y=i // 29,
                    norm_x=(i % 29 + 0.5) / 29.0, norm_y=(i // 29 + 0.5) / 29.0,
                    defocus=0.1, confidence=0.99,
                    error_flags=0x90CF  # Exact hardware invalid mask
                )
            )

        res = self.grid.evaluate_points(bad_points, dof_tolerance=0.5)
        self.assertEqual(res.total_points, 841)
        self.assertEqual(res.num_no_err, 0)
        self.assertEqual(res.num_reliable, 0)
        self.assertEqual(res.num_lockon, 0)
        self.assertEqual(res.sum_df_abs, 0.0)
        self.assertIsNone(res.min_df_abs)
        self.assertEqual(res.mean_defocus, 0.0)
        self.assertEqual(res.reliable_ratio, 0.0)
        self.assertEqual(res.lockon_ratio, 0.0)
        self.assertFalse(res.is_locked)

    def test_individual_error_flag_bits(self):
        """
        Stress Test: Verify that each bit in 0x90CF individually rejects the point:
        Bits: 0x8000, 0x1000, 0x0080, 0x0040, 0x0008, 0x0004, 0x0002, 0x0001.
        """
        mask_bits = [0x8000, 0x1000, 0x0080, 0x0040, 0x0008, 0x0004, 0x0002, 0x0001]
        for bit in mask_bits:
            pt = PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=0.1, confidence=0.99, error_flags=bit | 0x30)
            res = self.grid.evaluate_points([pt], dof_tolerance=0.5)
            self.assertEqual(res.num_no_err, 0, f"Point with error bit 0x{bit:04X} was not rejected!")

    def test_invalid_correlation_flags_parity(self):
        """
        Parity Test: When correlation reliability bit 0x30 is absent (error_flags=0x0000),
        firmware sub_6FC676 condition is: (v13 & 0x90CF) != 0 || (v13 & 0x30) == 0.
        Points with flag 0x30 set are reliable.
        """
        pt_reliable_hw = PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=0.2, confidence=0.2, error_flags=0x0030)
        pt_unreliable_hw = PDAFPoint(1, 0, 1, 0.5, 0.5, defocus=0.2, confidence=0.2, error_flags=0x0000)

        res1 = self.grid.evaluate_points([pt_reliable_hw], confidence_thresh=0.5)
        self.assertEqual(res1.num_reliable, 1)

        res2 = self.grid.evaluate_points([pt_unreliable_hw], confidence_thresh=0.5)
        self.assertEqual(res2.num_reliable, 0)

    def test_pdaf_correlation_reliability_fallback_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_6FC676):
        In firmware sub_6FC676, reliability is strictly gated by hardware flag:
        `if ( (v13 & 0x90CF) != 0 || (v13 & 0x30) == 0 )`
        If hardware correlation flag 0x30 is 0, the point MUST NOT be reliable.
        However, the current Python implementation has a generic fallback:
        `is_reliable = ((pt.error_flags & 0x30) != 0) or (pt.confidence >= confidence_thresh)`
        This allows points with invalid hardware correlation (error_flags=0x0000)
        to be marked reliable if synthetic confidence >= 0.35.
        """
        # Hardware correlation failed (error_flags = 0x0000), but synthetic confidence is high (0.95)
        pt_bad_hw_corr = PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=0.1, confidence=0.95, error_flags=0x0000)
        res = self.grid.evaluate_points([pt_bad_hw_corr], confidence_thresh=0.35)

        # Firmware expects num_reliable == 0. Asserting firmware parity exposes the discrepancy.
        self.assertEqual(res.num_reliable, 0,
                         "Discrepancy: Point with invalid hardware correlation (flag 0x30 unset) was marked reliable due to confidence fallback!")

    def test_empty_points_list(self):
        """Stress Test: PDAF evaluation on empty point set."""
        res = self.grid.evaluate_points([])
        self.assertEqual(res.total_points, 0)
        self.assertEqual(res.num_no_err, 0)
        self.assertEqual(res.num_reliable, 0)
        self.assertEqual(res.num_lockon, 0)
        self.assertIsNone(res.min_df_abs)
        self.assertFalse(res.is_locked)

    def test_optical_bound_exact_boundaries(self):
        """
        Stress Test: Firmware sub_6FC676 condition: *v10 < df && df <= *v9.
        Lower bound is strict (<), upper bound is inclusive (<=).
        """
        # Lower bound = -0.5, Upper bound = +0.5
        pt_exact_min = PDAFPoint(0, 0, 0, 0.5, 0.5, defocus=-0.50, confidence=0.8, error_flags=0x30)
        pt_just_above_min = PDAFPoint(1, 0, 1, 0.5, 0.5, defocus=-0.4999, confidence=0.8, error_flags=0x30)
        pt_exact_max = PDAFPoint(2, 0, 2, 0.5, 0.5, defocus=0.50, confidence=0.8, error_flags=0x30)
        pt_just_above_max = PDAFPoint(3, 0, 3, 0.5, 0.5, defocus=0.5001, confidence=0.8, error_flags=0x30)

        res_min = self.grid.evaluate_points([pt_exact_min], dof_tolerance=0.50)
        self.assertEqual(res_min.num_lockon, 0, "Exact min bound (-0.5) must not be included (strict <)")

        res_above_min = self.grid.evaluate_points([pt_just_above_min], dof_tolerance=0.50)
        self.assertEqual(res_above_min.num_lockon, 1, "Defocus just above min bound must be lock-on")

        res_max = self.grid.evaluate_points([pt_exact_max], dof_tolerance=0.50)
        self.assertEqual(res_max.num_lockon, 1, "Exact max bound (+0.5) must be included (inclusive <=)")

        res_above_max = self.grid.evaluate_points([pt_just_above_max], dof_tolerance=0.50)
        self.assertEqual(res_above_max.num_lockon, 0, "Defocus just above max bound must not be lock-on")

    def test_extreme_defocus_values_no_crash(self):
        """Stress Test: Defocus values with extreme positive and negative magnitudes."""
        extreme_pts = [
            PDAFPoint(0, 0, 0, 0.1, 0.1, defocus=1e8, confidence=0.9, error_flags=0x30),
            PDAFPoint(1, 0, 1, 0.2, 0.2, defocus=-1e8, confidence=0.9, error_flags=0x30),
            PDAFPoint(2, 0, 2, 0.3, 0.3, defocus=0.0, confidence=0.9, error_flags=0x30),
        ]
        res = self.grid.evaluate_points(extreme_pts, dof_tolerance=0.5)
        self.assertEqual(res.num_no_err, 3)
        self.assertEqual(res.num_reliable, 3)
        self.assertEqual(res.num_lockon, 1)  # Only df=0.0 is within 0.5
        self.assertAlmostEqual(res.sum_df_abs, 2e8, places=1)
        self.assertAlmostEqual(res.mean_defocus, 0.0, places=3)
        self.assertAlmostEqual(res.min_df_abs, 0.0, places=3)

    def test_hardware_20byte_record_unpacking_adversarial_bytes(self):
        """
        Stress Test: sub_2718A4 unpacking 20-byte record with zero normalizer,
        negative coordinates, and maximum signed 32-bit values.
        """
        # Normalizer is 0 -> should not raise ZeroDivisionError
        rec_zero_norm = struct.pack('<hhIiII', -100, -200, 0x30, -500, 800, 0)
        pt = PDAFPoint.from_hw_bytes(rec_zero_norm, index=10, grid_x=5, grid_y=5)
        self.assertEqual(pt.defocus, -500.0)
        self.assertEqual(pt.confidence, 800.0)  # max(1, 0) prevents zero division

        # Maximum signed 32-bit defocus
        rec_max_df = struct.pack('<hhIiII', 0, 0, 0x30, 2147483647, 1000, 1000)
        pt_max = PDAFPoint.from_hw_bytes(rec_max_df, index=11, grid_x=5, grid_y=5)
        self.assertEqual(pt_max.defocus, 2147483647.0)


class TestMarioDebounceMatrixAdversarial(unittest.TestCase):
    """
    Adversarial stress-testing of Mario AFC 5x5 Debounce Matrix (sub_8E5BF8 & dword_C747C4).
    """

    def setUp(self):
        self.fsm = MarioAfcStateMachine()

    def test_rapid_state_chattering_suppression(self):
        """
        Stress Test: Rapid state chattering across Mario 5x5 debounce matrix.
        Oscillating between LOCKED (slot 1) and TRACKING (slot 2) every 20,000 ticks.
        Required duration: dword_C747C4[1 * 5 + 2] = 400,000 ticks.
        Because each transition request resets the other slots,
        rapid toggling must NEVER allow transition to TRACKING.
        """
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 0

        # Simulate rapid chattering over 2,000,000 ticks
        for tick in range(0, 2000000, 20000):
            # Attempt to chatter to TRACKING (slot 2)
            res1 = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, tick, force=False)
            self.assertFalse(res1, f"Transition allowed prematurely at tick {tick}!")
            # Immediate chatter back to LOCKED (slot 1)
            self.fsm.try_transition(MarioAfcStateEnum.LOCKED, tick + 5000, force=False)

        self.assertEqual(self.fsm.state, MarioAfcStateEnum.LOCKED)

    def test_exact_debounce_boundary_slot_1_to_2(self):
        """
        Stress Test: Exact debounce boundary for Slot 1 -> Slot 2 (400,000 ticks).
        Entry tick: 1,000,000.
        At 1,399,999 (elapsed 399,999 < 400,000) -> MUST FAIL.
        At 1,400,000 (elapsed 400,000 >= 400,000) -> MUST SUCCEED.
        """
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 1000000

        # Initial request at 1,000,000
        ok_init = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 1000000, force=False)
        self.assertFalse(ok_init)

        # Attempt at 399,999 ticks elapsed
        ok_boundary_fail = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 1399999, force=False)
        self.assertFalse(ok_boundary_fail, "Transition must fail at elapsed = 399,999 ticks")
        self.assertEqual(self.fsm.state, MarioAfcStateEnum.LOCKED)

        # Attempt at 400,000 ticks elapsed
        ok_boundary_pass = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 1400000, force=False)
        self.assertTrue(ok_boundary_pass, "Transition must succeed at elapsed = 400,000 ticks")
        self.assertEqual(self.fsm.state, MarioAfcStateEnum.TRACKING)

    def test_exact_debounce_boundary_slot_1_to_3(self):
        """
        Stress Test: Exact debounce boundary for Slot 1 -> Slot 3 (10,000,000 ticks).
        dword_C747C4[1 * 5 + 3] = 10,000,000 ticks.
        Base tick: 1,000,000.
        """
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 1000000

        # Initial request at tick 1,000,000
        self.fsm.try_transition(MarioAfcStateEnum.SEARCH, 1000000, force=False)

        # 9,999,999 ticks elapsed (tick 10,999,999) -> Fail
        ok_fail = self.fsm.try_transition(MarioAfcStateEnum.SEARCH, 10999999, force=False)
        self.assertFalse(ok_fail)

        # 10,000,000 ticks elapsed (tick 11,000,000) -> Pass
        ok_pass = self.fsm.try_transition(MarioAfcStateEnum.SEARCH, 11000000, force=False)
        self.assertTrue(ok_pass)
        self.assertEqual(self.fsm.state, MarioAfcStateEnum.SEARCH)

    def test_tick_zero_sentinel_collision_vulnerability(self):
        """
        Adversarial Vulnerability Test (sub_8E5BF8):
        In sub_8E5BF8:
        `if ( v6 == 0 ) *v5 = a3;`
        `a1[2..6]` stores entry ticks initialized to 0. If current_tick (a3) is 0,
        *v5 is written with 0. On subsequent calls, `v6 == 0` is STILL True!
        This causes the timer to overwrite the slot entry tick on the NEXT call,
        effectively losing the initial tick measurement and delaying debounce!
        Using None as uninitialized sentinel prevents tick 0 collision.
        """
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 0

        # Initial request at tick 0 to Slot 3 (SEARCH, 10,000,000 ticks debounce from Slot 1)
        self.fsm.try_transition(MarioAfcStateEnum.SEARCH, 0, force=False)

        # At tick 10,000,000, elapsed = 10,000,000 - 0 = 10,000,000 >= 10,000,000 -> Pass
        ok = self.fsm.try_transition(MarioAfcStateEnum.SEARCH, 10000000, force=False)
        self.assertTrue(ok,
                        "Vulnerability: Initializing transition at tick=0 failed debounce at exact elapsed threshold due to zero-sentinel collision!")

    def test_force_bypass_debounce(self):
        """Stress Test: Force flag (a4 == 1 in sub_8E5BF8) bypasses all debounce timing."""
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 1000000

        # Force transition after only 1 tick
        ok = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 1000001, force=True)
        self.assertTrue(ok, "Forced transition must bypass debounce threshold")
        self.assertEqual(self.fsm.state, MarioAfcStateEnum.TRACKING)

    def test_timer_jitter_and_backwards_ticks(self):
        """Stress Test: Hardware timer wraps or jags backwards."""
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 500000

        # Initial request recorded at 500000 to TRACKING (slot 2)
        self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 500000, force=False)

        # Clock goes backwards to 450000 (jitter / wrap)
        ok_backward = self.fsm.try_transition(MarioAfcStateEnum.TRACKING, 450000, force=False)
        self.assertFalse(ok_backward, "Transition must be rejected when elapsed ticks are negative")

    def test_sudden_sensor_dropout_locked_to_tracking_to_wait(self):
        """Stress Test: Sudden loss of PDAF points while LOCKED."""
        self.fsm.state = MarioAfcStateEnum.LOCKED
        self.fsm.state_entry_ticks = [None] * 5
        self.fsm.state_entry_ticks[1] = 0

        # Dropout: 0 reliable points. First step initiates debounce timer (400,000 ticks to TRACKING).
        eval_dropout = DefocusEvaluation(total_points=20, num_reliable=0, is_locked=False)
        st0 = self.fsm.handle_event("EVAL_STEP", eval_dropout, tick=1000)
        self.assertEqual(st0, MarioAfcStateEnum.LOCKED, "LOCKED state should be held during 400,000 tick debounce window")

        # After 400,000 ticks of sustained dropout, transition to TRACKING succeeds
        st1 = self.fsm.handle_event("EVAL_STEP", eval_dropout, tick=401000)
        self.assertEqual(st1, MarioAfcStateEnum.TRACKING)

        # Continued dropout in TRACKING -> transitions immediately to WAIT (matrix[2][0] == 0)
        st2 = self.fsm.handle_event("EVAL_STEP", eval_dropout, tick=402000)
        self.assertEqual(st2, MarioAfcStateEnum.WAIT)


class TestFaceAndEyeArbiterAdversarial(unittest.TestCase):
    """
    Adversarial stress-testing of Face and Eye Selection & Spatial Gating:
    - sub_272420: Defocus Face Selection (8-slot, 60-byte, min_df)
    - sub_271C00 / sub_271FA8: Queue capacity and hysteresis
    - sub_278858: Piecewise linear radius interpolation (R >> 4)
    - sub_709FD6: Euclidean distance spatial gating
    - sub_278778: Inter-ocular geometry
    """

    def setUp(self):
        self.eye_selector = EyeSelector()
        self.arbiter = PriorityFaceArbiter()

    def test_defocus_face_selection_empty_candidate_list(self):
        """Stress Test: sub_272420 with 0 candidates."""
        chosen, tier, eye = self.arbiter.update_candidates([])
        self.assertIsNone(chosen)
        self.assertEqual(tier, TrackingTargetTier.OBJECT)
        self.assertIsNone(eye)
        self.assertEqual(len(self.arbiter.priority_queue), 0)

    def test_sub_272420_all_unreliable_slots_rejection_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_272420):
        In firmware sub_272420:
        `if ( v13 != 0 && v12 == 1 ) { ... v9 = 1; }`
        `if ( v9 << 31 != 0 ) { ... *(_BYTE *)(a1 + 12) = 1; }`
        `return v9;`
        If ALL 8 slots fail the reliability / point count gate, v9 remains 0,
        and NO face is selected (sub_272420 returns 0, a1+12 is not set).
        In the current Python implementation (eye_face_arbiter.py:256):
        `pool = reliable_candidates if reliable_candidates else candidates`
        It falls back to selecting an unreliable candidate face!
        """
        f1 = FaceCandidate(face_id=1, left=0.1, top=0.1, right=0.3, bottom=0.3,
                           confidence=0.8, defocus_error=1.5, reliable_pdaf_count=0)
        f2 = FaceCandidate(face_id=2, left=0.4, top=0.4, right=0.6, bottom=0.6,
                           confidence=0.8, defocus_error=2.0, reliable_pdaf_count=0)

        chosen, tier, eye = self.arbiter.update_candidates([f1, f2])
        # Firmware parity requires chosen to be None when all candidates are unreliable.
        self.assertIsNone(chosen,
                          "Discrepancy: Defocus Face Arbiter selected an unreliable face when all slots had 0 reliable PDAF points!")

    def test_defocus_face_selection_tie_breaking_min_df_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_272420 tie-breaking):
        Two faces with exact same min_df (defocus_error=0.20).
        Candidate 1 (slot 0): conf 0.50.
        Candidate 2 (slot 1): conf 0.95.
        Firmware sub_272420 keeps earlier candidate:
        `if ( v9 << 31 != 0 && *(_DWORD *)(v10 + 24) >= v7 ) v9 = 1; else { v2 = ...; v7 = ...; }`
        Because v10+24 >= v7, Candidate 2 does NOT replace Candidate 1!
        In Python: `sorted(pool, key=lambda f: (f.defocus_error, -f.confidence))`
        Python breaks ties by confidence, selecting Candidate 2 instead of Candidate 1.
        """
        f1 = FaceCandidate(face_id=101, left=0.1, top=0.1, right=0.3, bottom=0.3,
                           confidence=0.50, defocus_error=0.20, reliable_pdaf_count=5)
        f2 = FaceCandidate(face_id=102, left=0.4, top=0.4, right=0.6, bottom=0.6,
                           confidence=0.95, defocus_error=0.20, reliable_pdaf_count=5)

        self.arbiter.current_top_id = None
        chosen, tier, eye = self.arbiter.update_candidates([f1, f2])
        self.assertEqual(chosen.face_id, 101,
                         "Discrepancy: sub_272420 tie-breaker replaced earlier slot with higher-confidence later slot!")

    def test_priority_queue_capacity_limit_8(self):
        """
        Stress Test: Priority queue linked list capacity matching sub_271FA8 (8 slots).
        Feed 15 candidates; queue must be strictly capped at 8.
        """
        candidates = [
            FaceCandidate(face_id=i, left=0.1, top=0.1, right=0.2, bottom=0.2,
                          confidence=0.9, defocus_error=float(i) * 0.1, reliable_pdaf_count=5)
            for i in range(15)
        ]
        self.arbiter.update_candidates(candidates)
        self.assertEqual(len(self.arbiter.priority_queue), 8)

    def test_hysteresis_lockon_retention(self):
        """
        Stress Test: PreFaceIsInLockOnRange hysteresis retention from sub_271C00.
        When curr_top defocus is within lockon range (<= 0.8), it is retained
        even if another face has slightly lower defocus (within switching delta).
        """
        f1 = FaceCandidate(face_id=1, left=0.1, top=0.1, right=0.3, bottom=0.3,
                           confidence=0.9, defocus_error=0.50, reliable_pdaf_count=5)
        f2 = FaceCandidate(face_id=2, left=0.4, top=0.4, right=0.6, bottom=0.6,
                           confidence=0.9, defocus_error=0.40, reliable_pdaf_count=5)

        self.arbiter.current_top_id = 1  # Assume f1 was previously locked
        chosen, tier, eye = self.arbiter.update_candidates([f1, f2])
        # f1 should be retained because 0.50 <= 0.80 and diff (0.50 - 0.40 = 0.10) < 1.2
        self.assertEqual(chosen.face_id, 1)

    def test_piecewise_linear_radius_boundary_conditions(self):
        """
        Stress Test: Piecewise linear radius interpolation sub_278858 with extreme scales:
        - scale <= face_size_min (16.0): val = v5 * s (upper_ratio 8.0)
        - scale >= face_size_max (320.0): val = v6 * s (lower_ratio 3.0)
        - scale >> face_size_max (100,000.0) -> R >> 4
        - scale <= 0.0 (negative or zero scale)
        """
        # Negative scale (scale < 8.0 returns 0.0 per sub_278858)
        r_neg = self.eye_selector.compute_spatial_gating_radius(-50.0)
        self.assertEqual(r_neg, 0.0)

        # Zero scale (scale < 8.0 returns 0.0 per sub_278858)
        r_zero = self.eye_selector.compute_spatial_gating_radius(0.0)
        self.assertEqual(r_zero, 0.0)

        # Exact face_size_min (16.0): 8 * 16 = 128 -> >> 4 = 8.0
        r_min = self.eye_selector.compute_spatial_gating_radius(16.0)
        self.assertAlmostEqual(r_min, 8.0, places=2)

        # Exact face_size_max (320.0): 3 * 320 = 960 -> >> 4 = 60.0
        r_max = self.eye_selector.compute_spatial_gating_radius(320.0)
        self.assertAlmostEqual(r_max, 60.0, places=2)

        # R >> 4 extreme (scale = 100,000.0): 3 * 100,000 = 300,000 -> >> 4 = 18,750.0
        r_huge = self.eye_selector.compute_spatial_gating_radius(100000.0)
        self.assertAlmostEqual(r_huge, 18750.0, places=2)

    def test_eye_spatial_gating_bounding_box_bypass_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_709FD6):
        In firmware sub_709FD6, spatial acceptance check is strictly:
        `v7 >= (dx*dx) + (dy*dy)` where v7 = R^2 from sub_278858.
        If distance exceeds R^2, it MUST return false.
        In the current Python implementation (eye_face_arbiter.py:164):
        `if dist_sq_r <= radius_sq or (eye_r.x >= face.left and eye_r.x <= face.right ...):`
        An unverified fallback accepts eye candidates that are far outside R^2
        simply because they are inside the face bounding box.
        """
        # Face of size 200x200 (left=0, right=200, top=0, bottom=200). Center at (100, 100).
        # At scale 200: R ~ 39.4 px, R^2 ~ 1558.
        # Place eye at (10, 10): dist^2 = (90)^2 + (90)^2 = 16200 >> 1558 (more than 10x radius!).
        face = FaceCandidate(
            face_id=1, left=0.0, top=0.0, right=200.0, bottom=200.0, confidence=0.9,
            eye_right=EyeCandidate(x=10.0, y=10.0, confidence=0.90)
        )
        chosen_eye, cand = self.eye_selector.select_eye(face, tracking_center=(100.0, 100.0))

        # Firmware sub_709FD6 strictly rejects this outlier!
        self.assertEqual(chosen_eye, SelectedEye.NONE,
                         "Discrepancy: sub_709FD6 spatial gating accepted an outlier eye (dist^2 >> R^2) due to bounding-box bypass!")

    def test_extreme_head_yaw_handover(self):
        """
        Stress Test: Head yaw angle > 65 degrees without eye detections
        must trigger hierarchical handover from FACE tier to PERSON_BODY tier.
        """
        face_frontal = FaceCandidate(
            face_id=1, left=0.2, top=0.2, right=0.4, bottom=0.4, confidence=0.9,
            yaw_angle=15.0, reliable_pdaf_count=5
        )
        _, tier_frontal, _ = self.arbiter.update_candidates([face_frontal])
        self.assertEqual(tier_frontal, TrackingTargetTier.FACE)

        face_profile = FaceCandidate(
            face_id=2, left=0.2, top=0.2, right=0.4, bottom=0.4, confidence=0.9,
            yaw_angle=75.0, reliable_pdaf_count=5  # Head turned > 65 deg
        )
        _, tier_profile, _ = self.arbiter.update_candidates([face_profile])
        self.assertEqual(tier_profile, TrackingTargetTier.PERSON_BODY)


class TestContrastAFTransitionsAdversarial(unittest.TestCase):
    """
    Adversarial stress-testing of Movie Contrast AF State Machine:
    - sub_8A5122: CStateWob transition logic
    - sub_8A520C: CStateYama transition logic
    - Zero contrast gradient, flat peaks, sudden slope reversal
    - CImagerNoiseEstimator extreme ISO and exposure times
    """

    def setUp(self):
        self.engine = ContrastAFEngine()

    def test_sub_8a5122_transition_truth_table(self):
        """
        Stress Test: Exhaustive truth-table verification of sub_8A5122 (CStateWob).
        """
        # When has_gradient is True -> always YAMA
        st1 = self.engine.transition_from_wob(has_gradient=True, is_idle=False, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st1, ContrastAFState.YAMA)

        # When has_gradient is False, is_idle is False, is_speed_mode is True, is_high_speed_req is True -> HIGH_SPEED
        st2 = self.engine.transition_from_wob(has_gradient=False, is_idle=False, is_speed_mode=True, is_high_speed_req=True)
        self.assertEqual(st2, ContrastAFState.HIGH_SPEED)

        # When has_gradient is False, is_idle is False, is_speed_mode is True, is_high_speed_req is False -> WOBBLE
        st3 = self.engine.transition_from_wob(has_gradient=False, is_idle=False, is_speed_mode=True, is_high_speed_req=False)
        self.assertEqual(st3, ContrastAFState.WOBBLE)

        # When has_gradient is False, is_idle is True -> IRREGULAR_WOBBLE
        st4 = self.engine.transition_from_wob(has_gradient=False, is_idle=True, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st4, ContrastAFState.IRREGULAR_WOBBLE)

    def test_sub_8a520c_transition_truth_table(self):
        """
        Stress Test: Exhaustive truth-table verification of sub_8A520C (CStateYama).
        Verifies exact firmware branches:
        - a3 != 0 (stay_in_yama) -> off_F320CC (CStateYama)
        - a4 == 0, a5 == 1, a6 != 0 (speed_mode, high_speed) -> off_F320D8 (CStateHiSpeed)
        - a4 == 0, a5 == 1, a6 == 0 (speed_mode, wobble) -> off_F320D4 (CStateWob)
        - a4 == 0, a5 == 0, a6 == 0, a7 == 0, trans_mode == 1 -> &off_F320C4 (CStateTrans)
        - a4 == 0, a5 == 0, a6 == 0, a7 == 0, trans_mode == 0 -> off_F320CC (CStateYama)
        - a4 != 0 (is_idle) -> off_F320D0 (CStateIrregularWob)
        """
        # stay_in_yama -> YAMA
        st1 = self.engine.transition_from_yama(stay_in_yama=True, is_idle=False, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st1, ContrastAFState.YAMA)

        # Speed mode + high speed -> HIGH_SPEED
        st2 = self.engine.transition_from_yama(stay_in_yama=False, is_idle=False, is_speed_mode=True, is_high_speed_req=True)
        self.assertEqual(st2, ContrastAFState.HIGH_SPEED)

        # Speed mode + wobble -> WOBBLE
        st3 = self.engine.transition_from_yama(stay_in_yama=False, is_idle=False, is_speed_mode=True, is_high_speed_req=False)
        self.assertEqual(st3, ContrastAFState.WOBBLE)

        # Normal mode + trans_mode enabled -> TRANSITION (Camera::LC::MovieContrastAF::CStateTrans)
        st4 = self.engine.transition_from_yama(stay_in_yama=False, is_idle=False, is_speed_mode=False, is_high_speed_req=False, trans_mode=True)
        self.assertEqual(st4, ContrastAFState.TRANSITION)

        # Idle / aborted -> IRREGULAR_WOBBLE
        st5 = self.engine.transition_from_yama(stay_in_yama=False, is_idle=True, is_speed_mode=False, is_high_speed_req=False)
        self.assertEqual(st5, ContrastAFState.IRREGULAR_WOBBLE)

    def test_sub_8a520c_sudden_reversal_premature_in_focus_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_8A520C):
        In contrast AF hill-climbing (YAMA), when moving in the search direction,
        if the very first step experiences a sharp contrast drop (e.g. wrong direction or scene motion),
        firmware sub_8A520C initiates coarse search or direction reversal.
        However, the current implementation:
        `if current_contrast < (self.peak_contrast * self.peak_drop_ratio): self.state = ContrastAFState.IN_FOCUS`
        blindly treats ANY drop below peak_drop_ratio as a confirmed peak,
        even if no peak was ever approached, locking onto the initial starting position!
        """
        self.engine.start_af(50.0)
        # Feed wobble cycles to transition to YAMA with direction +1
        for i in range(4):
            self.engine.step(10.0 + i)
            self.engine.step(5.0)

        self.assertEqual(self.engine.state, ContrastAFState.YAMA)

        # Immediate sharp drop on step 1 of YAMA (from peak 11.5 down to 2.0)
        new_pos, state = self.engine.step(2.0)

        # Firmware expects coarse search or reversal, NOT premature IN_FOCUS lock at pos 50.0!
        self.assertNotEqual(state, ContrastAFState.IN_FOCUS,
                            "Discrepancy: Contrast AF falsely declared IN_FOCUS on immediate drop without finding a true peak!")

    def test_zero_contrast_gradient_endless_sweep(self):
        """
        Stress Test: Completely flat scene with zero contrast gradient (e.g. lens cap on).
        Wobble detects no gradient (|delta_c| <= thresh) and enters HIGH_SPEED.
        Verifies behavior across 60 steps without crashing or floating-point errors.
        """
        self.engine.start_af(50.0)
        for _ in range(60):
            pos, state = self.engine.step(0.0)
            self.assertGreaterEqual(pos, self.engine.lens_min_pos)
            self.assertLessEqual(pos, self.engine.lens_max_pos)
        self.assertIn(state, [ContrastAFState.WOBBLE, ContrastAFState.HIGH_SPEED])

    def test_imager_noise_estimator_extremes(self):
        """
        Stress Test: Sensor noise estimator under extreme ISO (100 to 409,600)
        and extreme shutter speeds (1/8000s to 30s).
        """
        estimator = ImagerNoiseEstimator(base_noise=0.05)

        noise_base = estimator.estimate_noise(iso=100.0, exposure_time=1.0 / 8000.0)
        self.assertAlmostEqual(noise_base, 0.0505, delta=0.01)

        noise_extreme = estimator.estimate_noise(iso=409600.0, exposure_time=30.0)
        self.assertGreater(noise_extreme, noise_base * 50)
        self.assertFalse(np.isnan(noise_extreme))
        self.assertFalse(np.isinf(noise_extreme))


class TestStapleTrackingBorderAndNoiseAdversarial(unittest.TestCase):
    """
    Adversarial stress-testing of Staple Visual Tracking Engine:
    - sub_705BC6: Displacement integration and 20px border margin checks
    - sub_71E28A: Staple processing loop
    - Extreme noise, sudden black frame dropouts, negative/zero scale factors
    - DualTargetStapleManager out-of-range slots
    """

    def test_sub_705bc6_all_four_border_violations(self):
        """
        Stress Test: 20-pixel border margin check from sub_705BC6:
        v13 = ((v11 - 20) | (v12 - 20)) < 0 || v12 + 20 > H || v11 + 20 > W
        Targets inside 20px of ANY of the 4 borders must be flagged:
        is_reliable = False, error_code = 3.
        """
        H, W = 200, 200
        frame = np.ones((H, W, 3), dtype=np.float32) * 50.0

        border_boxes = [
            ("Left Border", (10.0, 50.0, 30.0, 30.0)),              # x < 20
            ("Top Border", (50.0, 10.0, 30.0, 30.0)),               # y < 20
            ("Right Border", (W - 25.0, 50.0, 30.0, 30.0)),         # x + w + 20 > W (175 + 30 + 20 = 225 > 200)
            ("Bottom Border", (50.0, H - 25.0, 30.0, 30.0)),        # y + h + 20 > H
        ]

        for desc, box in border_boxes:
            tracker = SingleTargetStapleTracker(target_id=0)
            tracker.init_target(frame, box)
            res = tracker.track(frame)
            self.assertFalse(res.is_reliable, f"{desc} should be flagged unreliable")
            self.assertEqual(res.error_code, 3, f"{desc} must have error_code=3 (border caution)")

    def test_staple_black_frame_phantom_displacement_discrepancy(self):
        """
        Adversarial Discrepancy Test (sub_705BC6 / CorrelationFilterTracker):
        When a total sensor dropout occurs (completely black frame), the correlation
        response is all zeros. `np.argmax(response)` blindly indexes (0, 0),
        producing a phantom displacement:
        `dx = 0 - 32 = -32.0, dy = 0 - 32 = -32.0`
        This causes a violent position jump of half the patch width on a blank frame!
        When response has no peak (zero variance), displacement should be zero.
        """
        frame_good = np.ones((100, 100, 3), dtype=np.float32) * 50.0
        frame_good[40:60, 40:60] = 220.0

        tracker = SingleTargetStapleTracker(target_id=0)
        tracker.init_target(frame_good, (40.0, 40.0, 20.0, 20.0))

        black_frame = np.zeros((100, 100, 3), dtype=np.float32)
        res = tracker.track(black_frame)

        # Displacement vector should be near zero on total dropout, not jump by -32 px!
        self.assertAlmostEqual(res.dx, 0.0, delta=2.0,
                               msg=f"Discrepancy: Black frame caused a phantom displacement jump of dx={res.dx}, dy={res.dy}!")

    def test_high_sensor_noise_frame(self):
        """
        Stress Test: Extreme sensor noise (pure uniform noise frame).
        Tracking filter must reject spurious correlation peaks.
        """
        frame_good = np.ones((100, 100, 3), dtype=np.float32) * 50.0
        frame_good[40:60, 40:60] = 220.0

        tracker = SingleTargetStapleTracker(target_id=0)
        tracker.init_target(frame_good, (40.0, 40.0, 20.0, 20.0))

        np.random.seed(42)
        noise_frame = np.random.uniform(0.0, 255.0, (100, 100, 3)).astype(np.float32)
        res = tracker.track(noise_frame)
        self.assertFalse(res.is_reliable)

    def test_sub_705bc6_scale_factor_extremes(self):
        """
        Stress Test: sub_705BC6 fixed point displacement integration:
        norm_dx = dx / scale
        Test scale = 0.0 (zero division prevention), scale = -1.0, scale = 100.0.
        """
        tracker = SingleTargetStapleTracker(target_id=0)
        frame = np.ones((100, 100, 3), dtype=np.float32) * 50.0
        tracker.init_target(frame, (40.0, 40.0, 20.0, 20.0))

        # Force scale to 0.0
        tracker.scale = 0.0
        res0 = tracker.track(frame)
        self.assertFalse(np.isnan(res0.dx))
        self.assertFalse(np.isinf(res0.dx))

        # Force scale to extreme 1000.0
        tracker.scale = 1000.0
        res_huge = tracker.track(frame)
        self.assertFalse(np.isnan(res_huge.dx))

    def test_dual_target_manager_out_of_range_target_id(self):
        """
        Stress Test: sub_280C3C dual-slot manager handles invalid target slot IDs
        without raising IndexError.
        """
        mgr = DualTargetStapleManager()
        frame = np.ones((100, 100, 3), dtype=np.float32) * 50.0
        # Valid slots 0 and 1
        mgr.set_target(0, frame, (20.0, 20.0, 10.0, 10.0))
        mgr.set_target(1, frame, (50.0, 50.0, 10.0, 10.0))
        # Invalid slots
        mgr.set_target(-1, frame, (10.0, 10.0, 10.0, 10.0))
        mgr.set_target(5, frame, (10.0, 10.0, 10.0, 10.0))

        results = mgr.process_frame(frame)
        self.assertEqual(len(results), 2)
        self.assertEqual({r.target_id for r in results}, {0, 1})


if __name__ == '__main__':
    unittest.main()
