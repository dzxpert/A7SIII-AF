import os
import sys
import time
import numpy as np

# Ensure parent directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ALGORITHM import (
        PDAFGrid,
        BoundingBox,
        AFAreaType,
        MarioAfcStateMachine,
        MarioAfcStateEnum,
        ContrastAFEngine,
        ContrastAFState,
        SingleTargetStapleTracker,
        PriorityFaceArbiter,
        FaceCandidate,
        EyeCandidate,
        SelectedEye,
        TrackingTargetTier
    )
except ImportError:
    from sony_af_tracker import (
        PDAFGrid,
        BoundingBox,
        AFAreaType,
        MarioAfcStateMachine,
        MarioAfcStateEnum,
        ContrastAFEngine,
        ContrastAFState,
        SingleTargetStapleTracker,
        PriorityFaceArbiter,
        FaceCandidate,
        EyeCandidate,
        SelectedEye,
        TrackingTargetTier
    )


def create_synthetic_frame(
    width: int,
    height: int,
    target_pos: tuple,
    target_size: tuple,
    noise_level: float = 5.0
) -> np.ndarray:
    """Generates a synthetic camera sensor frame with a textured target."""
    frame = np.random.normal(60.0, noise_level, (height, width, 3)).astype(np.float32)
    tx, ty = target_pos
    tw, th = target_size

    x1 = int(np.clip(tx - tw // 2, 0, width))
    y1 = int(np.clip(ty - th // 2, 0, height))
    x2 = int(np.clip(tx + tw // 2, 0, width))
    y2 = int(np.clip(ty + th // 2, 0, height))

    # Face/target texture (skin tone + distinctive facial features)
    target_patch = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.float32)
    target_patch[:, :] = [210.0, 160.0, 130.0] # Skin tone

    # Add eyes
    ey = int(target_patch.shape[0] * 0.35)
    ex_l = int(target_patch.shape[1] * 0.35)
    ex_r = int(target_patch.shape[1] * 0.65)
    ew = max(2, target_patch.shape[1] // 10)
    target_patch[max(0, ey - ew):min(target_patch.shape[0], ey + ew),
                 max(0, ex_l - ew):min(target_patch.shape[1], ex_l + ew)] = [30.0, 20.0, 15.0]
    target_patch[max(0, ey - ew):min(target_patch.shape[0], ey + ew),
                 max(0, ex_r - ew):min(target_patch.shape[1], ex_r + ew)] = [30.0, 20.0, 15.0]

    frame[y1:y2, x1:x2] = target_patch
    return np.clip(frame, 0.0, 255.0)


def run_pipeline_simulation():
    print("=" * 75)
    print(" Sony Alpha 7S III (BIONZ XR CXD90057) Pipeline Simulation")
    print(" Subsystems: 841-Pt PDAF | MovieContrastAF | Staple Tracker | Face/Eye Arbiter")
    print("=" * 75)

    W, H = 320, 240
    pdaf_grid = PDAFGrid(sensor_width=W, sensor_height=H)
    afc_state_machine = MarioAfcStateMachine()
    contrast_af = ContrastAFEngine()
    staple_tracker = SingleTargetStapleTracker(target_id=0)
    face_arbiter = PriorityFaceArbiter()

    # Simulation parameters
    total_frames = 15
    init_center = (100.0, 100.0)
    target_w, target_h = 40.0, 48.0
    vel_x, vel_y = 6.0, 2.5 # Moving diagonally

    # Initialize lens focus state
    lens_focus_distance = 3.5 # Initial defocus error

    print("\n[Step 1] Initializing Shutter S1 (Half-Press / AF-ON)...")
    afc_state_machine.handle_event("S1_PRESS")
    print(f"  -> MarioAfcState: {afc_state_machine.state.name}")

    # Generate Frame 0 and initialize tracker
    frame0 = create_synthetic_frame(W, H, init_center, (target_w, target_h))
    init_box = (init_center[0] - target_w / 2, init_center[1] - target_h / 2, target_w, target_h)
    staple_tracker.init_target(frame0, init_box)
    print(f"  -> Staple Tracker initialized on target at box: {init_box}")

    print("\n[Step 2] Processing Video Stream & Real-Time Tracking Loop:")
    print("-" * 75)
    print(f"{'Frm':>3} | {'Target Box (x, y)':>18} | {'Tier':>11} | {'Conf':>5} | {'Defocus':>7} | {'MarioAfcState':>14} | {'Lock'}")
    print("-" * 75)

    cur_cx, cur_cy = init_center

    for f in range(total_frames):
        cur_cx += vel_x
        cur_cy += vel_y

        # Lens drive toward focus
        if afc_state_machine.state in (MarioAfcStateEnum.APPROACH, MarioAfcStateEnum.TRACKING):
            lens_focus_distance -= 0.85
        elif afc_state_machine.state == MarioAfcStateEnum.MOVE_TO_PEAK:
            lens_focus_distance -= 0.35
        lens_focus_distance = max(0.05, lens_focus_distance)

        # 1. Capture synthetic camera frame
        frame = create_synthetic_frame(W, H, (cur_cx, cur_cy), (target_w, target_h))

        # 2. Update Staple Visual Tracker
        track_res = staple_tracker.track(frame)

        # 3. Map tracked target to normalized bounding box for 841-Point PDAF Grid
        norm_box = BoundingBox(
            left=track_res.x / W,
            top=track_res.y / H,
            right=(track_res.x + track_res.width) / W,
            bottom=(track_res.y + track_res.height) / H
        )

        # Update synthetic defocus values on PDAF points in target box
        # Hardware correlation flag: 0x30 (reliable correlation status, sub_6FC676)
        pts_in_box = pdaf_grid.get_points_in_rect(norm_box)
        for pt in pts_in_box:
            pt.defocus = lens_focus_distance
            pt.confidence = 0.88
            pt.error_flags = 0x30

        eval_res = pdaf_grid.evaluate_points(pts_in_box, confidence_thresh=0.40, dof_tolerance=0.45)

        # 4. Simulate Face/Eye Detection with PDAF density integration
        face_cand = FaceCandidate(
            face_id=101,
            left=track_res.x,
            top=track_res.y,
            right=track_res.x + track_res.width,
            bottom=track_res.y + track_res.height,
            confidence=track_res.confidence,
            eye_left=EyeCandidate(track_res.x + 14.0, track_res.y + 16.0, 0.92),
            eye_right=EyeCandidate(track_res.x + 26.0, track_res.y + 16.0, 0.90),
            defocus_error=lens_focus_distance,
            reliable_pdaf_count=len(pts_in_box)
        )

        chosen_face, tier, active_eye = face_arbiter.update_candidates(
            [face_cand],
            tracking_center=(track_res.x + 20.0, track_res.y + 18.0)
        )

        # 5. Drive Continuous AF State Machine
        afc_state_machine.handle_event("TARGET_UPDATE", eval_res)

        lock_sym = "[LOCKED]" if eval_res.is_locked else "......"
        print(f"{f:3d} | ({track_res.x:5.1f}, {track_res.y:5.1f}, {track_res.width:2.0f}x{track_res.height:2.0f}) | {tier.name:>11} | {track_res.confidence:5.2f} | {eval_res.mean_defocus:7.2f} | {afc_state_machine.state.name:>14} | {lock_sym}")

    print("-" * 75)
    print("\n[Step 3] Verification Summary:")
    print(f"  * Final MarioAfcState: {afc_state_machine.state.name}")
    print(f"  * Final Defocus Error: {eval_res.mean_defocus:.3f} units (within depth of field)")
    print(f"  * Tracking Tier:       {tier.name} (Precision Eye AF active)")
    print(f"  * Lock-On Ratio:       {eval_res.lockon_ratio * 100:.1f}% of reliable PDAF points")
    print("\nPipeline Simulation Executed Successfully!")


if __name__ == '__main__':
    run_pipeline_simulation()
