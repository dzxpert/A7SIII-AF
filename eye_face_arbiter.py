"""
Sony Alpha 7S III - Eye Selection, Dynamic Thresholding & Face Defocus Arbitration
Clean-room implementation reversed from CXD90057 coprocessor firmware (`cpapp-b.bin`):
- `fc_alg_tracking_eye_selector.cpp`
  - Inter-ocular distance & Left/Right Eye selection (sub_278778)
  - Dynamic scale-adaptive confidence thresholding (sub_278858)
- `priority_defocus_face_selector.cpp`
  - Priority face arbitration & queue re-ordering (sub_271C00, sub_271FA8)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


class SelectedEye(Enum):
    NONE = 0
    RIGHT_EYE = 1       # Subject's right eye
    LEFT_EYE = 2        # Subject's left eye
    AUTO = 3


class TrackingTargetTier(Enum):
    EYE = 0             # Precision Eye AF (Human / Animal / Bird)
    FACE = 1            # Face AF (Frontal / Profile)
    PERSON_BODY = 2     # Upper body / torso (PDN fallback when face turns)
    OBJECT = 3          # General object (Staple tracker correlation / color)


@dataclass
class EyeCandidate:
    x: float
    y: float
    confidence: float


@dataclass
class FaceCandidate:
    face_id: int
    left: float
    top: float
    right: float
    bottom: float
    confidence: float
    eye_left: Optional[EyeCandidate] = None
    eye_right: Optional[EyeCandidate] = None
    yaw_angle: float = 0.0          # Estimated head pose (-90 left, +90 right)
    defocus_error: float = 0.0      # Evaluated PDAF defocus |Δd|
    reliable_pdaf_count: int = 0    # Count of reliable PDAF points within face rect (+40 in 60-byte record)
    lockon_pdaf_count: Optional[int] = None # Count of lockon points within face rect (+48 in 60-byte record)
    has_min_df_reliable: Optional[bool] = None # Small face reliability flag (+20 in 60-byte record)
    is_in_focus: bool = False

    @property
    def size(self) -> float:
        """Face diameter / diagonal size metric."""
        w = max(0.0, self.right - self.left)
        h = max(0.0, self.bottom - self.top)
        return float(np.sqrt(w * w + h * h))

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0


class EyeSelector:
    """
    Reversed from fc_alg_tracking_eye_selector.cpp:
    - sub_278778: Inter-ocular geometry & Left/Right eye selection
    - sub_278858 / sub_709FD6: Scale-adaptive spatial gating acceptance radius
    """

    def __init__(
        self,
        face_size_min: float = 16.0,
        face_size_max: float = 320.0,
        upper_ratio: float = 8.0,       # Strict threshold for tiny faces
        lower_ratio: float = 3.0,       # Relaxed threshold for large portraits
        min_confidence: float = 0.30
    ):
        self.face_size_min = face_size_min
        self.face_size_max = face_size_max
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.min_confidence = min_confidence

    def compute_spatial_gating_radius(self, scale: float) -> float:
        """
        Reversed from sub_278858 & sub_709FD6.
        Calculates spatial gating acceptance radius in pixels (dist^2 <= radius^2):
        v7 = (v5 * v4) + (s - v4) * (v6 * v3 - v5 * v4) / (v3 - v4)
        scaled by >> 4. Returns 0 if scale < scale_min (8.0).
        """
        s = float(scale)
        v3 = self.face_size_max
        v4 = self.face_size_min
        v5 = self.upper_ratio
        v6 = self.lower_ratio

        # sub_278858 validation and scale_min check:
        # if (*((unsigned __int16 *)a2 + 3) > a3) return 0;
        if s < 8.0 or v3 <= v4 or v6 >= v5 or (v6 * v3 <= v5 * v4):
            return 0.0

        if s <= v4:
            val = v5 * s
        elif s >= v3:
            val = v6 * s
        else:
            slope = float(v6 * v3 - v5 * v4) / float(v3 - v4)
            val = (v5 * v4) + (s - v4) * slope

        # sub_278858 fixed point scaling: return (unsigned __int16)(v7 >> 4)
        return float(int(val) >> 4)

    def compute_adaptive_threshold(self, face_size: float) -> float:
        """Compatibility alias for compute_spatial_gating_radius."""
        return self.compute_spatial_gating_radius(face_size)

    def select_eye(
        self,
        face: FaceCandidate,
        tracking_center: Tuple[float, float],
        user_preference: SelectedEye = SelectedEye.AUTO
    ) -> Tuple[SelectedEye, Optional[EyeCandidate]]:
        """
        Reversed from sub_278778 & sub_709FD6:
        1. Calculates inter-ocular distance: D_eye = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        2. Computes scale strictly as 2.0 * D_eye (sub_278778: v11 = 2 * v8)
        3. Computes spatial gating radius: radius = compute_spatial_gating_radius(scale)
        4. Tests spatial proximity gating: radius > 0 and (eye.x - tx)^2 + (eye.y - ty)^2 <= radius^2
        5. Selects closer eye to tracking center or user-preferred eye.
        """
        eye_r = face.eye_right
        eye_l = face.eye_left

        if eye_r is None and eye_l is None:
            return SelectedEye.NONE, None

        # Determine inter-ocular scale strictly matching sub_278778 (line 0x2787ba: v11 = 2 * v8)
        if eye_r is not None and eye_l is not None:
            d_eye = float(np.sqrt((eye_r.x - eye_l.x) ** 2 + (eye_r.y - eye_l.y) ** 2))
            inter_scale = 2.0 * d_eye
        else:
            inter_scale = face.size

        # Spatial gating acceptance radius (sub_278858 & sub_709FD6)
        # Convert normalized scale to pixel scale if face coordinates are normalized [0.0..1.0]
        scale_px = inter_scale * 320.0 if face.size <= 1.0 else inter_scale
        raw_radius = self.compute_spatial_gating_radius(scale_px)
        if face.size <= 1.0:
            radius = raw_radius / 320.0
        else:
            radius = raw_radius
        radius_sq = radius * radius

        # Reference anchor: tracking center or face center
        fx, fy = face.center
        tx, ty = tracking_center
        # If tracking center is normalized (e.g. 0.5, 0.5) while face is in pixel coords, use face center
        if (face.size > 1.0 and tx <= 1.0 and ty <= 1.0):
            ref_x, ref_y = fx, fy
        else:
            ref_x, ref_y = tx, ty

        valid_r = False
        dist_sq_r = 0.0
        if eye_r is not None and eye_r.confidence >= self.min_confidence:
            dist_sq_r = (eye_r.x - ref_x) ** 2 + (eye_r.y - ref_y) ** 2
            # Strict sub_709FD6 Euclidean spatial gating check: radius > 0 and dist^2 <= radius^2
            if radius > 0.0 and dist_sq_r <= radius_sq:
                valid_r = True

        valid_l = False
        dist_sq_l = 0.0
        if eye_l is not None and eye_l.confidence >= self.min_confidence:
            dist_sq_l = (eye_l.x - ref_x) ** 2 + (eye_l.y - ref_y) ** 2
            # Strict sub_709FD6 Euclidean spatial gating check: radius > 0 and dist^2 <= radius^2
            if radius > 0.0 and dist_sq_l <= radius_sq:
                valid_l = True

        if not valid_r and not valid_l:
            return SelectedEye.NONE, None

        if user_preference == SelectedEye.RIGHT_EYE and valid_r:
            return SelectedEye.RIGHT_EYE, eye_r
        elif user_preference == SelectedEye.LEFT_EYE and valid_l:
            return SelectedEye.LEFT_EYE, eye_l

        # Only one eye meets gating criteria
        if valid_r and not valid_l:
            return SelectedEye.RIGHT_EYE, eye_r
        if valid_l and not valid_r:
            return SelectedEye.LEFT_EYE, eye_l

        # Both eyes valid: compute proximity to active tracking point (sub_278778)
        # Prioritize nearer eye
        if dist_sq_r <= dist_sq_l:
            return SelectedEye.RIGHT_EYE, eye_r
        else:
            return SelectedEye.LEFT_EYE, eye_l


class PriorityFaceArbiter:
    """
    Reversed from priority_defocus_face_selector.cpp:
    - sub_271C00 / sub_272420: Priority Defocus Face Selection (Min Reliable DF FaceID)
    - sub_271FA8: Priority Queue Linked-List Re-ordering & Hysteresis
    """

    def __init__(
        self,
        max_priority_queue: int = 8,
        lockon_range_defocus: float = 0.8,
        min_reliable_pdaf_count: int = 4,
        density_multiplier: int = 128
    ):
        self.max_queue = max_priority_queue
        self.lockon_range_defocus = lockon_range_defocus
        self.min_reliable_pdaf_count = min_reliable_pdaf_count
        self.density_multiplier = density_multiplier

        self.current_top_id: Optional[int] = None
        self.priority_queue: List[FaceCandidate] = []
        self.eye_selector = EyeSelector()

    def update_candidates(
        self,
        candidates: List[FaceCandidate],
        tracking_center: Tuple[float, float] = (0.5, 0.5),
        is_movie_lockon: bool = False,
        is_movie_af_assist: bool = False
    ) -> Tuple[Optional[FaceCandidate], TrackingTargetTier, Optional[EyeCandidate]]:
        """
        Executes face arbitration matching sub_272420 & sub_271FA8:
        1. Filters candidates via sub_272420 60-byte face slot density & reliability gating:
           - Large faces (num_reliable >= 4): (num_lockon_reliable << 8) >= num_reliable * 128
           - Small faces (num_reliable < 4): has_min_df_reliable != 0
        2. Strictly returns None if no candidate satisfies gating (no fallback).
        3. Selects candidate with absolute minimum defocus |Δd| min_df, preserving slot tie-breaker.
        4. Retains locked face unconditionally if within lock-on tolerance (sub_271E34).
        5. Re-orders priority queue matching sub_271FA8 (capacity 8).
        Returns: (chosen_face, tracking_tier, active_eye)
        """
        if not candidates:
            self.current_top_id = None
            self.priority_queue.clear()
            return None, TrackingTargetTier.OBJECT, None

        # 1. Evaluate candidates via sub_272420 60-byte face slot density & reliability gating:
        valid_candidates: List[FaceCandidate] = []
        for f in candidates:
            # Slot validity (+0 is_valid)
            if f.confidence <= 0.0:
                continue

            num_reliable = f.reliable_pdaf_count  # +40
            lockon_count = f.lockon_pdaf_count if f.lockon_pdaf_count is not None else num_reliable  # +48

            if num_reliable >= self.min_reliable_pdaf_count:
                # Large face density gate: (num_lockon_reliable << 8) >= num_reliable * 128
                passed = (lockon_count << 8) >= (num_reliable * self.density_multiplier)
            else:
                # Small face reliability gate: has_min_df_reliable != 0
                has_min_df = f.has_min_df_reliable if f.has_min_df_reliable is not None else (num_reliable > 0)
                passed = bool(has_min_df)

            if passed:
                valid_candidates.append(f)

        if not valid_candidates:
            # sub_272420: v9 == 0 -> No face selected! Strictly return None, no fallback.
            self.current_top_id = None
            self.priority_queue.clear()
            return None, TrackingTargetTier.OBJECT, None

        # 2. Select candidate with absolute minimum defocus min_df (sub_272420)
        # Preserve earlier slot order tie-breaking without sorting by confidence:
        # if ( v9 << 31 != 0 && *(_DWORD *)(v10 + 24) >= v7 ) v9 = 1; else { v2 = ...; v7 = ...; }
        best_df_candidate = valid_candidates[0]
        min_v7 = abs(best_df_candidate.defocus_error)
        for cand in valid_candidates[1:]:
            cand_df = abs(cand.defocus_error)
            if cand_df < min_v7:
                min_v7 = cand_df
                best_df_candidate = cand

        # 3. Check current top candidate hysteresis (sub_271E34 / PreFaceIsInLockOnRange)
        curr_top_cand = None
        if self.current_top_id is not None:
            for f in candidates:
                if f.face_id == self.current_top_id:
                    curr_top_cand = f
                    break

        # If locked face remains within lockon tolerance, retain it unconditionally (sub_271E34)
        if curr_top_cand is not None and abs(curr_top_cand.defocus_error) <= self.lockon_range_defocus:
            selected_top = curr_top_cand
        else:
            selected_top = best_df_candidate

        # 4. Priority queue maintenance matching sub_271FA8 (capacity 8)
        # Move selected_top to front
        queue = [selected_top] + [f for f in candidates if f.face_id != selected_top.face_id]
        self.priority_queue = queue[:self.max_queue]
        self.current_top_id = selected_top.face_id

        # 5. Hierarchical handover check
        # Tier 1: Check Eye AF
        selected_eye_type, eye_target = self.eye_selector.select_eye(
            selected_top,
            tracking_center
        )

        if eye_target is not None:
            return selected_top, TrackingTargetTier.EYE, eye_target

        # Tier 2: Check Face AF (if head turned > 65 degrees, handover to Body)
        if abs(selected_top.yaw_angle) < 65.0:
            return selected_top, TrackingTargetTier.FACE, None

        # Tier 3: Person Body (PDN)
        return selected_top, TrackingTargetTier.PERSON_BODY, None
