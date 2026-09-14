"""
Sony Alpha 7S III - Phase Detection AutoFocus (PDAF) Subsystem
Clean-room implementation reversed from CXD90057 coprocessor firmware (`cpapp-b.bin`):
- `defocus_info_translator.cpp` (sub_2718A4, sub_2719D4, sub_271A90)
- `defocus_info_generator.cpp` (sub_2717E8)
- `Camera::LC::MarioAfcState` (vtable 0xc74978 - 0xc74b34)
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


class AFAreaType(Enum):
    WIDE = 0
    ZONE = 1
    CENTER = 2
    FLEXIBLE_SPOT_S = 3
    FLEXIBLE_SPOT_M = 4
    FLEXIBLE_SPOT_L = 5
    EXPANDED_FLEXIBLE_SPOT = 6
    TRACKING = 7


class MarioAfcStateEnum(Enum):
    STOP = 0
    WAIT = 1
    SEARCH = 2
    SCAN = 3
    APPROACH = 4
    TRACKING = 5
    MOVE_TO_PEAK = 6
    LOCKED = 7


@dataclass
class PDAFPoint:
    """
    Individual Phase Detection AF Site.
    Matches 20-byte hardware record layout in cpapp-b.bin:
    - Offset +0: X coordinate (sensor physical or binned >> 1, sub_2718A4)
    - Offset +2: Y coordinate
    - Offset +4: Status flags / error bitmask (0x90CF error mask, 0x30 reliability)
    - Offset +8: Signed 32-bit Defocus value (sub_6FC676, sub_2E7770)
    - Offset +12: Correlation peak height
    - Offset +16: Normalizer / Peak ratio
    """
    index: int                  # 0..840 (zpd_max = 841, 0 <= a3 <= 0x348)
    grid_x: int                 # 0..28
    grid_y: int                 # 0..28
    norm_x: float               # 0.0 .. 1.0 (normalized image width)
    norm_y: float               # 0.0 .. 1.0 (normalized image height)
    defocus: float = 0.0        # Signed defocus in focus units (0.0 = perfect focus)
    confidence: float = 0.0     # Cross-correlation peak confidence [0.0 .. 1.0]
    error_flags: int = 0x30     # Hardware status flags (0x30 = reliable, bitmask 0x90CF invalid)

    @classmethod
    def from_hw_bytes(cls, record_bytes: bytes, index: int, grid_x: int, grid_y: int) -> 'PDAFPoint':
        """Unpacks 20-byte hardware record from ZPD DMA buffer."""
        import struct
        x, y, flags, df, peak, norm = struct.unpack('<hhIiII', record_bytes[:20])
        # sub_2718A4 binning right-shift
        binned_x = x >> 1
        binned_y = y >> 1
        conf = float(peak) / float(max(1, norm))
        return cls(
            index=index,
            grid_x=grid_x,
            grid_y=grid_y,
            norm_x=float(grid_x + 0.5) / 29.0,
            norm_y=float(grid_y + 0.5) / 29.0,
            defocus=float(df),
            confidence=conf,
            error_flags=flags
        )


@dataclass
class BoundingBox:
    """Normalized bounding box [0.0, 1.0]"""
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class DefocusEvaluation:
    """
    Evaluation metrics for an area or bounding box.
    Matches log format and accumulation in sub_271A90 & sub_6FC676:
    num_no_err:%d num_reliable:%d num_lockon:%d sum_df_abs:%d min_df_abs:%d
    """
    total_points: int = 0
    num_no_err: int = 0
    num_reliable: int = 0
    num_lockon: int = 0
    sum_df_abs: float = 0.0
    min_df_abs: Optional[float] = None
    mean_defocus: float = 0.0
    reliable_ratio: float = 0.0
    lockon_ratio: float = 0.0
    is_locked: bool = False


class PDAFGrid:
    """
    841-Point (29x29) Phase Detection Grid Model
    Reversed from defocus_info_translator.cpp (sub_2718A4).
    """
    ZPD_MAX = 841
    GRID_ROWS = 29
    GRID_COLS = 29

    # Error flags mask matching decompiled sub_271A90: (*v10 & 0x90CF) == 0
    ERR_MASK_INVALID = 0x90CF

    def __init__(self, sensor_width: int = 4240, sensor_height: int = 2832):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.points: List[PDAFPoint] = []
        self._init_grid()

    def _init_grid(self):
        """Build 29x29 grid layout with sub-sampling normalization."""
        idx = 0
        for r in range(self.GRID_ROWS):
            norm_y = (r + 0.5) / self.GRID_ROWS
            for c in range(self.GRID_COLS):
                norm_x = (c + 0.5) / self.GRID_COLS
                pt = PDAFPoint(
                    index=idx,
                    grid_x=c,
                    grid_y=r,
                    norm_x=norm_x,
                    norm_y=norm_y
                )
                self.points.append(pt)
                idx += 1

    def get_points_in_rect(self, rect: BoundingBox) -> List[PDAFPoint]:
        """Find all PDAF points inside the specified normalized bounding box."""
        return [pt for pt in self.points if rect.contains(pt.norm_x, pt.norm_y)]

    def get_active_points_for_area(
        self,
        area_type: AFAreaType,
        spot_center: Tuple[float, float] = (0.5, 0.5),
        tracking_rect: Optional[BoundingBox] = None
    ) -> List[PDAFPoint]:
        """
        Reversed from sub_2717E8 (defocus_info_generator.cpp).
        Selects active points based on area mode and spot center.
        """
        if area_type == AFAreaType.WIDE:
            return self.points

        cx, cy = spot_center
        # Convert norm center to nearest grid coordinates
        gc_x = int(np.clip(cx * self.GRID_COLS, 0, self.GRID_COLS - 1))
        gc_y = int(np.clip(cy * self.GRID_ROWS, 0, self.GRID_ROWS - 1))

        if area_type == AFAreaType.CENTER:
            # 5x5 central cluster
            mid_x, mid_y = self.GRID_COLS // 2, self.GRID_ROWS // 2
            return [
                pt for pt in self.points
                if abs(pt.grid_x - mid_x) <= 2 and abs(pt.grid_y - mid_y) <= 2
            ]

        elif area_type == AFAreaType.ZONE:
            # 9x9 zone cluster around spot center
            return [
                pt for pt in self.points
                if abs(pt.grid_x - gc_x) <= 4 and abs(pt.grid_y - gc_y) <= 4
            ]

        elif area_type == AFAreaType.FLEXIBLE_SPOT_S:
            # Single closest point
            return [self.points[gc_y * self.GRID_COLS + gc_x]]

        elif area_type == AFAreaType.FLEXIBLE_SPOT_M:
            # 3x3 spot
            return [
                pt for pt in self.points
                if abs(pt.grid_x - gc_x) <= 1 and abs(pt.grid_y - gc_y) <= 1
            ]

        elif area_type == AFAreaType.FLEXIBLE_SPOT_L:
            # 5x5 spot
            return [
                pt for pt in self.points
                if abs(pt.grid_x - gc_x) <= 2 and abs(pt.grid_y - gc_y) <= 2
            ]

        elif area_type == AFAreaType.EXPANDED_FLEXIBLE_SPOT:
            # 3x3 plus 1-step cross expansion
            return [
                pt for pt in self.points
                if (abs(pt.grid_x - gc_x) <= 1 and abs(pt.grid_y - gc_y) <= 1)
                or (abs(pt.grid_x - gc_x) <= 2 and pt.grid_y == gc_y)
                or (abs(pt.grid_y - gc_y) <= 2 and pt.grid_x == gc_x)
            ]

        elif area_type == AFAreaType.TRACKING:
            if tracking_rect is not None:
                pts = self.get_points_in_rect(tracking_rect)
                if pts:
                    return pts
            # Fallback to center 3x3
            return [
                pt for pt in self.points
                if abs(pt.grid_x - gc_x) <= 1 and abs(pt.grid_y - gc_y) <= 1
            ]

        return self.points

    def evaluate_points(
        self,
        points: List[PDAFPoint],
        confidence_thresh: float = 0.35,
        dof_tolerance: float = 0.50,
        lockon_ratio_thresh: float = 0.60,
        min_optical_bound: Optional[float] = None,
        max_optical_bound: Optional[float] = None
    ) -> DefocusEvaluation:
        """
        Reversed from sub_271A90 & sub_6FC676:
        Accumulates metrics for active points inside target ROI polygon:
        1. Checks error bitmask (*v10 & 0x90CF == 0)
        2. Increments num_no_err, accumulates sum_df_abs += abs(defocus)
        3. Tests hardware correlation reliability ((v13 & 0x90CF) == 0 && (v13 & 0x30) != 0)
        4. Tests lock-on against optical tolerance bounds (*v10 < df && df <= *v9)
        5. Tracks min_df_abs
        """
        eval_res = DefocusEvaluation(total_points=len(points))
        if not points:
            return eval_res

        signed_defocus_sum = 0.0
        min_bound = min_optical_bound if min_optical_bound is not None else -dof_tolerance
        max_bound = max_optical_bound if max_optical_bound is not None else dof_tolerance

        for pt in points:
            # Error bitmask filter: sub_271A90 (*v10 & 0x90CF) == 0
            if (pt.error_flags & self.ERR_MASK_INVALID) != 0:
                continue
            eval_res.num_no_err += 1

            # Signed 32-bit defocus accumulation: sub_6FC676 *(v8 - 1) += v12
            abs_df = abs(pt.defocus)
            eval_res.sum_df_abs += abs_df

            # Hardware correlation reliability flag strictly checks (pt.error_flags & 0x30) != 0:
            # Matches firmware sub_6FC676: if ( (v13 & 0x90CF) != 0 || (v13 & 0x30) == 0 )
            # Synthetic confidence fallback removed to enforce strict hardware correlation status.
            is_reliable = ((pt.error_flags & 0x30) != 0)
            if is_reliable:
                eval_res.num_reliable += 1
                signed_defocus_sum += pt.defocus

                # Optical depth-of-field tolerance window: *v10 < df && df <= *v9
                if min_bound < pt.defocus <= max_bound:
                    eval_res.num_lockon += 1
                    if eval_res.min_df_abs is None or abs_df < eval_res.min_df_abs:
                        eval_res.min_df_abs = abs_df

        if eval_res.num_reliable > 0:
            eval_res.mean_defocus = signed_defocus_sum / eval_res.num_reliable
            eval_res.lockon_ratio = eval_res.num_lockon / float(eval_res.num_reliable)
            eval_res.reliable_ratio = eval_res.num_reliable / float(max(1, eval_res.num_no_err))
            eval_res.is_locked = (eval_res.lockon_ratio >= lockon_ratio_thresh)

        return eval_res


class MarioAfcStateMachine:
    """
    Continuous AutoFocus (AF-C) State Machine.
    Reversed from Camera::LC::MarioAfcState (cpapp-b.bin):
    - State transition debounce handler: sub_8E5BF8
    - Debounce timing matrix: dword_C747C4 (5x5 matrix in hardware clock ticks)
    - State values: dword_C74828 = (0, 1, 2, 0, 0)
    - 8 polymorphic states (vtables 0xc74978 - 0xc74b34)
    """

    # Exact 5x5 debounce matrix recovered from dword_C747C4 (clock ticks)
    DEBOUNCE_MATRIX_5X5 = [
        [0, 0, 0, 0, 0],
        [0, 0, 400000, 10000000, 0],   # Slot 1 -> 2: 400,000 ticks, 1 -> 3: 10,000,000 ticks
        [0, 0, 0, 10000000, 0],        # Slot 2 -> 3: 10,000,000 ticks
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    # Map the 8 MarioAfc states to the 5 debounce slots matching firmware sub_368100 & sub_8E5BF8:
    # Slot 0: STOP / WAIT
    # Slot 1: LOCKED (IN-FOCUS: |df| <= 3 * DOF, sub_368100, sub_8E5C7E)
    # Slot 2: TRACKING / MOVE_TO_PEAK (ACTIVE DRIVE: |df| > 3 * DOF)
    # Slot 3: SEARCH / APPROACH (FAR SEARCH / LOST TARGET)
    # Slot 4: SCAN (COARSE SWEEP)
    STATE_TO_SLOT = {
        MarioAfcStateEnum.STOP: 0,
        MarioAfcStateEnum.WAIT: 0,
        MarioAfcStateEnum.LOCKED: 1,
        MarioAfcStateEnum.TRACKING: 2,
        MarioAfcStateEnum.MOVE_TO_PEAK: 2,
        MarioAfcStateEnum.SEARCH: 3,
        MarioAfcStateEnum.APPROACH: 3,
        MarioAfcStateEnum.SCAN: 4,
    }

    # State attribute mapping from dword_C74828 (5 DWORDs: a1[1] = dword_C74828[a2]):
    STATE_ATTRIBUTES = [0, 1, 2, 0, 0]

    def __init__(
        self,
        approach_thresh: float = 3.0,
        move_to_peak_thresh: float = 0.8,
        lockon_thresh: float = 0.4
    ):
        self.state = MarioAfcStateEnum.STOP
        self.approach_thresh = approach_thresh
        self.move_to_peak_thresh = move_to_peak_thresh
        self.lockon_thresh = lockon_thresh
        self.history: List[MarioAfcStateEnum] = [self.state]

        # Timing tracking matching sub_8E5BF8 (in clock ticks)
        # Using None as uninitialized sentinel to avoid tick 0 collision (sub_8E5BF8: v6 == 0)
        self.state_entry_ticks: List[Optional[int]] = [None] * 5
        self.current_tick: int = 0
        self.state_entry_ticks[self.STATE_TO_SLOT[self.state]] = self.current_tick
        self.state_attribute: int = self.STATE_ATTRIBUTES[self.STATE_TO_SLOT[self.state]]

    def reset(self):
        self.state = MarioAfcStateEnum.STOP
        self.history = [self.state]
        self.state_entry_ticks = [None] * 5
        self.current_tick = 0
        self.state_entry_ticks[self.STATE_TO_SLOT[self.state]] = self.current_tick
        self.state_attribute = self.STATE_ATTRIBUTES[self.STATE_TO_SLOT[self.state]]

    def try_transition(
        self,
        new_state: MarioAfcStateEnum,
        current_tick: int,
        force: bool = False
    ) -> bool:
        """
        Reversed from sub_8E5BF8 (MarioAfcState::TransitionHandler):
        Checks debounce matrix dword_C747C4 to suppress rapid state toggling / chattering:
        v4 = &a1[a2]; v6 = v4[2]; v5 = v4 + 2;
        if ( v6 == 0 ) *v5 = a3;
        for ( i = 0; i != 5; ++i ) if ( a2 != i ) a1[i + 2] = 0;
        if ( a4 != 1 && a3 - *v5 < (unsigned int)dword_C747C4[5 * *a1 + a2] ) return 0;
        *a1 = a2; a1[1] = dword_C74828[a2]; return 1;
        """
        curr_slot = self.STATE_TO_SLOT[self.state]
        new_slot = self.STATE_TO_SLOT[new_state]

        # Initialize slot timestamp if not set (None sentinel avoids tick 0 collision)
        if self.state_entry_ticks[new_slot] is None:
            self.state_entry_ticks[new_slot] = current_tick

        # Reset other slot timestamps matching sub_8E5BF8 loop:
        for i in range(5):
            if new_slot != i:
                self.state_entry_ticks[i] = None

        # Check debounce duration from dword_C747C4
        required_ticks = self.DEBOUNCE_MATRIX_5X5[curr_slot][new_slot]
        elapsed = current_tick - self.state_entry_ticks[new_slot]
        if not force and (elapsed < 0 or elapsed < required_ticks):
            return False  # Debounce condition not met; transition suppressed

        self.state = new_state
        self.state_attribute = self.STATE_ATTRIBUTES[new_slot]
        # Matching sub_8E5BF8: entry timestamp (*v5) is NOT overwritten on transition success
        return True

    def handle_event(
        self,
        event: str,
        eval_result: Optional[DefocusEvaluation] = None,
        tick: Optional[int] = None,
        force: Optional[bool] = None
    ) -> MarioAfcStateEnum:
        """
        Processes camera input events (S1_PRESS, S1_RELEASE, TARGET_UPDATE)
        with sub_8E5BF8 debounce enforcement.
        """
        # Strict sub_8E5BF8 debounce enforcement without artificial bypass
        force_transition = bool(force)
        if tick is not None:
            self.current_tick = tick
        else:
            # Advance synthetic clock by 40,000 ticks (~40ms at 1MHz)
            self.current_tick += 40000

        prev_state = self.state
        target_state = self.state

        if event == "S1_RELEASE":
            target_state = MarioAfcStateEnum.STOP

        elif self.state == MarioAfcStateEnum.STOP:
            if event == "S1_PRESS":
                target_state = MarioAfcStateEnum.WAIT

        elif self.state == MarioAfcStateEnum.WAIT:
            if event in ("TARGET_UPDATE", "AF_TRIGGER"):
                target_state = MarioAfcStateEnum.SEARCH

        elif self.state == MarioAfcStateEnum.SEARCH:
            if eval_result and eval_result.num_reliable > 0:
                abs_df = abs(eval_result.mean_defocus)
                if abs_df > self.approach_thresh:
                    target_state = MarioAfcStateEnum.APPROACH
                elif abs_df > self.move_to_peak_thresh:
                    target_state = MarioAfcStateEnum.TRACKING
                else:
                    target_state = MarioAfcStateEnum.MOVE_TO_PEAK
            else:
                target_state = MarioAfcStateEnum.SCAN

        elif self.state == MarioAfcStateEnum.SCAN:
            if eval_result and eval_result.num_reliable >= 3:
                target_state = MarioAfcStateEnum.APPROACH

        elif self.state == MarioAfcStateEnum.APPROACH:
            if eval_result and eval_result.num_reliable > 0:
                abs_df = abs(eval_result.mean_defocus)
                if abs_df <= self.approach_thresh:
                    target_state = MarioAfcStateEnum.TRACKING
            else:
                target_state = MarioAfcStateEnum.SCAN

        elif self.state == MarioAfcStateEnum.TRACKING:
            if eval_result and eval_result.num_reliable > 0:
                abs_df = abs(eval_result.mean_defocus)
                if abs_df <= self.move_to_peak_thresh:
                    target_state = MarioAfcStateEnum.MOVE_TO_PEAK
                elif abs_df > self.approach_thresh:
                    target_state = MarioAfcStateEnum.APPROACH
            else:
                target_state = MarioAfcStateEnum.WAIT

        elif self.state == MarioAfcStateEnum.MOVE_TO_PEAK:
            if eval_result and eval_result.num_reliable > 0:
                if eval_result.is_locked:
                    target_state = MarioAfcStateEnum.LOCKED
                elif abs(eval_result.mean_defocus) > self.move_to_peak_thresh:
                    target_state = MarioAfcStateEnum.TRACKING
            else:
                target_state = MarioAfcStateEnum.WAIT

        elif self.state == MarioAfcStateEnum.LOCKED:
            if eval_result:
                if not eval_result.is_locked or abs(eval_result.mean_defocus) > self.move_to_peak_thresh:
                    target_state = MarioAfcStateEnum.TRACKING

        if target_state != self.state:
            if self.try_transition(target_state, self.current_tick, force=force_transition):
                self.history.append(self.state)

        return self.state
