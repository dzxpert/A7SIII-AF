"""
Sony Alpha 7S III - Contrast AutoFocus (CDAF) Subsystem
Clean-room implementation reversed from CXD90057 coprocessor firmware (`cpapp-b.bin`):
- `Camera::LC::MovieContrastAF`
  - CStateWob (sub_8A5122)
  - CStateYama (sub_8A520C)
  - CStateHiSpeed, CStateTrans, CStateIrregularWob, CStateNoUse
  - CImagerNoiseEstimator
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


class ContrastAFState(Enum):
    NO_USE = 0          # Idle / Inactive (CStateNoUse)
    WOBBLE = 1          # Micro-modulation for gradient sign (CStateWob)
    YAMA = 2            # Hill-climbing peak search (CStateYama)
    TRANSITION = 3      # State transition buffer (CStateTrans)
    HIGH_SPEED = 4      # Coarse rapid sweep (CStateHiSpeed)
    IRREGULAR_WOBBLE = 5 # Jittered modulation for flicker rejection (CStateIrregularWob)
    IN_FOCUS = 6        # Peak locked


@dataclass
class ContrastMetrics:
    """Focus evaluation metrics computed on image crop."""
    contrast_energy: float      # High-frequency Sobel/Laplacian gradient energy
    high_freq_ratio: float      # Ratio of high-frequency to mid-frequency energy
    noise_floor: float          # Estimated sensor noise floor
    signal_to_noise: float      # S/N ratio for peak verification


class ImagerNoiseEstimator:
    """
    Reversed from Camera::LC::MovieContrastAF::CImagerNoiseEstimator.
    Estimates noise floor as a function of ISO / sensor gain and scene illuminance.
    """
    def __init__(self, base_noise: float = 0.05):
        self.base_noise = base_noise

    def estimate_noise(self, iso: float, exposure_time: float) -> float:
        """
        Shot noise + read noise model:
        noise = base_noise * sqrt(ISO / 100) * (1 + 0.1 * sqrt(t))
        """
        gain_factor = np.sqrt(max(1.0, iso / 100.0))
        time_factor = 1.0 + 0.1 * np.sqrt(max(0.001, exposure_time))
        return float(self.base_noise * gain_factor * time_factor)


class ContrastAFEngine:
    """
    Movie Contrast AutoFocus Engine (MovieContrastAF).
    Implements the dual-loop Wobble + Yama algorithm.
    """

    def __init__(
        self,
        wobble_amplitude: float = 0.08,     # Lens micro-displacement delta
        wobble_cycles: int = 4,             # Number of +/- wobbles before direction decision
        gradient_threshold: float = 0.03,   # Minimum gradient to initiate Yama
        peak_drop_ratio: float = 0.90,      # Drop from peak to confirm overshoot (10% drop)
        yama_step_size: float = 1.0,        # Continuous drive step size
        lens_min_pos: float = 0.0,
        lens_max_pos: float = 100.0
    ):
        self.wobble_amplitude = wobble_amplitude
        self.wobble_cycles = wobble_cycles
        self.gradient_threshold = gradient_threshold
        self.peak_drop_ratio = peak_drop_ratio
        self.yama_step_size = yama_step_size
        self.lens_min_pos = lens_min_pos
        self.lens_max_pos = lens_max_pos

        self.noise_estimator = ImagerNoiseEstimator()
        self.state = ContrastAFState.NO_USE

        # Lens position state
        self.current_pos = 50.0             # Initial mid-focus position
        self.target_pos = 50.0
        self.search_direction = 1           # +1 = Far -> Near, -1 = Near -> Far

        # Wobble loop state
        self.wobble_step = 0
        self.wobble_base_pos = 50.0
        self.wobble_samples_plus: List[float] = []
        self.wobble_samples_minus: List[float] = []

        # Yama (Hill climbing) loop state
        self.peak_contrast = 0.0
        self.peak_lens_pos = 50.0
        self.has_climbed_in_yama: bool = False

        # History log
        self.contrast_history: List[Tuple[float, float]] = [] # (lens_pos, contrast)

    def start_af(self, initial_lens_pos: float = 50.0):
        """Activates Contrast AF."""
        self.current_pos = float(np.clip(initial_lens_pos, self.lens_min_pos, self.lens_max_pos))
        self.target_pos = self.current_pos
        self.wobble_base_pos = self.current_pos
        self.wobble_step = 0
        self.wobble_samples_plus.clear()
        self.wobble_samples_minus.clear()
        self.contrast_history.clear()
        self.peak_contrast = 0.0
        self.peak_lens_pos = self.current_pos
        self.has_climbed_in_yama = False
        self.state = ContrastAFState.WOBBLE

    def stop_af(self):
        self.state = ContrastAFState.NO_USE

    def transition_from_wob(
        self,
        has_gradient: bool,
        is_idle: bool,
        is_speed_mode: bool,
        is_high_speed_req: bool,
        trans_mode: bool = False
    ) -> ContrastAFState:
        """
        Reversed from sub_8A5122 (Camera::LC::MovieContrastAF::CStateWob::Transition):
        v6 = off_F320D0; // CStateIrregularWob
        if (a3 != 0) v6 = off_F320CC; // CStateYama
        else if (a4 == 0) {
            if (a5 == 1) v6 = (a6 != 0) ? off_F320D8 (&CStateHiSpeed) : result (this, &CStateWob);
            else {
                v6 = &off_F320C4; // &CStateTrans
                if (MEMORY[0x145662E] == 0) v6 = off_F320CC; // CStateYama
            }
        }
        *(_DWORD *)(a2 + 8) = v6;
        return result;
        """
        if has_gradient:
            return ContrastAFState.YAMA
        elif not is_idle:
            if is_speed_mode:
                return ContrastAFState.HIGH_SPEED if is_high_speed_req else ContrastAFState.WOBBLE
            else:
                return ContrastAFState.TRANSITION if trans_mode else ContrastAFState.YAMA
        return ContrastAFState.IRREGULAR_WOBBLE

    def transition_from_yama(
        self,
        stay_in_yama: bool,
        is_idle: bool,
        is_speed_mode: bool,
        is_high_speed_req: bool,
        flag_a7: bool = False,
        trans_mode: bool = False
    ) -> ContrastAFState:
        """
        Reversed from sub_8A520C (Camera::LC::MovieContrastAF::CStateYama::Transition):
        result = off_F320D0; // CStateIrregularWob
        if ( a3 != 0 ) {
            result = off_F320CC; // CStateYama
            goto LABEL_11;
        }
        if ( a4 == 0 ) {
            v8 = a6;
            if ( a5 == 1 ) {
                v9 = off_F320D8;   // CStateHiSpeed
                result = off_F320D4; // CStateWob
            } else {
                result = off_F320CC; // CStateYama
                if ( a6 != 0 || a7 != 0 ) goto LABEL_11;
                v9 = &off_F320C4;  // CStateTrans
                v8 = MEMORY[0x145662E];
            }
            if ( v8 != 0 ) result = v9;
        }
        LABEL_11:
        *(_DWORD *)(a2 + 8) = result;
        return result;
        """
        result = ContrastAFState.IRREGULAR_WOBBLE
        if stay_in_yama:
            return ContrastAFState.YAMA
        if not is_idle:
            if is_speed_mode:
                result = ContrastAFState.HIGH_SPEED if is_high_speed_req else ContrastAFState.WOBBLE
            else:
                result = ContrastAFState.YAMA
                if not is_high_speed_req and not flag_a7:
                    if trans_mode:
                        result = ContrastAFState.TRANSITION
        return result

    @staticmethod
    def calculate_contrast(image_gray: np.ndarray) -> float:
        """
        Computes high-frequency contrast energy via 2D spatial gradient (Sobel-like filter).
        """
        if image_gray.ndim != 2 or image_gray.size < 16:
            return 0.0
        # Fast finite-difference gradients
        gx = np.diff(image_gray, axis=1)
        gy = np.diff(image_gray, axis=0)
        energy_x = np.mean(gx ** 2)
        energy_y = np.mean(gy ** 2)
        return float(np.sqrt(energy_x + energy_y))

    def step(
        self,
        current_contrast: float,
        iso: float = 400.0,
        exposure_time: float = 1.0 / 60.0
    ) -> Tuple[float, ContrastAFState]:
        """
        Executes one AF cycle step.
        Returns (new_target_lens_pos, current_state).
        """
        self.contrast_history.append((self.current_pos, current_contrast))
        noise_floor = self.noise_estimator.estimate_noise(iso, exposure_time)

        if self.state == ContrastAFState.NO_USE:
            return self.current_pos, self.state

        elif self.state == ContrastAFState.WOBBLE:
            # Execute micro-wobbling: +delta, -delta alternately
            if self.wobble_step % 2 == 0:
                self.wobble_samples_plus.append(current_contrast)
                self.target_pos = float(np.clip(
                    self.wobble_base_pos - self.wobble_amplitude,
                    self.lens_min_pos,
                    self.lens_max_pos
                ))
            else:
                self.wobble_samples_minus.append(current_contrast)
                self.target_pos = float(np.clip(
                    self.wobble_base_pos + self.wobble_amplitude,
                    self.lens_min_pos,
                    self.lens_max_pos
                ))

            self.wobble_step += 1
            self.current_pos = self.target_pos

            # Once enough wobble cycles are collected, determine direction sign
            if len(self.wobble_samples_plus) >= self.wobble_cycles and len(self.wobble_samples_minus) >= self.wobble_cycles:
                mean_plus = float(np.mean(self.wobble_samples_plus))
                mean_minus = float(np.mean(self.wobble_samples_minus))
                delta_c = mean_plus - mean_minus

                effective_thresh = max(self.gradient_threshold, noise_floor * 1.5)

                if abs(delta_c) > effective_thresh:
                    # Direction determined! Drive transition via sub_8A5122
                    self.search_direction = 1 if delta_c > 0 else -1
                    self.peak_contrast = max(mean_plus, mean_minus)
                    self.peak_lens_pos = self.wobble_base_pos
                    self.has_climbed_in_yama = False
                    self.state = self.transition_from_wob(
                        has_gradient=True,
                        is_idle=False,
                        is_speed_mode=False,
                        is_high_speed_req=False
                    )
                else:
                    # Flat gradient -> high speed coarse scan via sub_8A5122
                    self.state = self.transition_from_wob(
                        has_gradient=False,
                        is_idle=False,
                        is_speed_mode=True,
                        is_high_speed_req=True
                    )

        elif self.state == ContrastAFState.HIGH_SPEED:
            # Step in search direction with large stride
            self.target_pos += self.search_direction * (self.yama_step_size * 2.5)

            # Check boundary bounce
            if self.target_pos >= self.lens_max_pos:
                self.target_pos = self.lens_max_pos
                self.search_direction = -1
            elif self.target_pos <= self.lens_min_pos:
                self.target_pos = self.lens_min_pos
                self.search_direction = 1

            self.current_pos = self.target_pos

            # If contrast begins rising above noise threshold, transition back to Wobble via sub_8A5122
            if len(self.contrast_history) >= 3:
                c_prev = self.contrast_history[-2][1]
                if (current_contrast - c_prev) > (noise_floor * 2.0):
                    self.wobble_base_pos = self.current_pos
                    self.wobble_step = 0
                    self.wobble_samples_plus.clear()
                    self.wobble_samples_minus.clear()
                    self.state = self.transition_from_wob(
                        has_gradient=False,
                        is_idle=False,
                        is_speed_mode=True,
                        is_high_speed_req=False
                    )

        elif self.state == ContrastAFState.YAMA:
            # Hill climbing along positive contrast slope
            if current_contrast > self.peak_contrast:
                self.peak_contrast = current_contrast
                self.peak_lens_pos = self.current_pos
                self.has_climbed_in_yama = True

            # Check if contrast dropped below peak (overshoot confirmed)
            if current_contrast < (self.peak_contrast * self.peak_drop_ratio):
                if self.has_climbed_in_yama:
                    # True peak found and confirmed! Return directly to peak position
                    self.target_pos = self.peak_lens_pos
                    self.current_pos = self.target_pos
                    self.state = ContrastAFState.IN_FOCUS
                else:
                    # Immediate drop without climbing (sub_8A520C): wrong direction or noise
                    # Reverse direction and transition via sub_8A520C to coarse search
                    self.search_direction *= -1
                    self.state = self.transition_from_yama(
                        stay_in_yama=False,
                        is_idle=False,
                        is_speed_mode=True,
                        is_high_speed_req=True
                    )
            else:
                # Continue driving in search direction matching sub_8A520C (stay_in_yama=True)
                self.state = self.transition_from_yama(
                    stay_in_yama=True,
                    is_idle=False,
                    is_speed_mode=False,
                    is_high_speed_req=False
                )
                self.target_pos += self.search_direction * self.yama_step_size
                if self.target_pos >= self.lens_max_pos or self.target_pos <= self.lens_min_pos:
                    # Hit boundary, reverse direction
                    self.search_direction *= -1
                self.current_pos = self.target_pos

        elif self.state == ContrastAFState.IN_FOCUS:
            # Monitor for scene changes
            if current_contrast < (self.peak_contrast * 0.70):
                # Lost focus, re-initiate wobble
                self.start_af(self.current_pos)

        return self.current_pos, self.state
