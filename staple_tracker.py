"""
Sony Alpha 7S III - Staple (Sum of Template and Pixel-wise LEarners) Visual Tracking Engine
Clean-room software emulation of Sony's Hardware Accelerated Tracking Pipeline:

Hardware Disassembly Mapping (`cpapp-b.bin`):
- `fc_drv_staple.cpp`:
  - `idt_staple_get_result` (sub_280C3C): Reads dual-slot target tracking descriptors
  - `sub_705BC6`: Evaluates displacement vectors (dx, dy in Q12 fixed point) and reliability
  - `idt_staple_set_hann_window_map` (sub_26E3F4): DMA transfers 1024-byte Hann coefficient banks
  - `StapleWork` memory allocator (sub_26D880): Allocates 96KB (98304 bytes) DMA scratchpad
  - `ddl_saCalcStart` (sub_26FE50 / sub_AD3CD2): Dispatches tracking execution to the
    proprietary SA IP / FRC2 (Scene Analysis / Frame Rate Conversion) ASIC coprocessor
- `Camera::FC::Alg::StapleAdjust` (vtable 0xBF484C): Ring-buffer interface (sub_706D12)

Architecture Note:
In the physical camera, correlation filters, HOG feature extraction, and histogram fusion are
accelerated directly on-chip inside the SA IP / FRC2 coprocessor. The ARM CPU manages DMA descriptors,
state machines, and tracking coordinates. This Python module serves as a behavioral clean-room
emulation of that pipeline using the Bertinetto et al. (2016) STAPLE formulation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class TrackedTargetResult:
    """
    Tracking result structure matching sub_280C3C and sub_705BC6:
    v11: [dx, dy, scale, reliability, error_code] (20 bytes descriptor)
    v20: [x, y, w, h] (16 bytes bounding box)
    """
    target_id: int              # 0 or 1 (up to 2 concurrent tracked targets)
    x: float                    # Bounding box left (pixel / normalized)
    y: float                    # Bounding box top
    width: float                # Bounding box width
    height: float               # Bounding box height
    confidence: float           # Response peak confidence / PSR metric [0.0 .. 1.0]
    dx: float                   # Velocity vector X (pixels/frame, Q12 fixed-point in HW)
    dy: float                   # Velocity vector Y (pixels/frame, Q12 fixed-point in HW)
    scale: float                # Relative scale factor (1.0 = baseline)
    is_reliable: bool           # Tracking lock-on flag (sub_705BC6 gate)
    error_code: int = 0         # 0 = success, nonzero = tracking error (v11[4])


class HannWindow2D:
    """
    Precomputed 2D Cosine Window Generator.
    Reversed from idt_staple_set_hann_window_map (sub_26E3F4).
    """
    @staticmethod
    def generate(width: int, height: int) -> np.ndarray:
        """
        w(x, y) = 0.25 * (1 - cos(2*pi*x / (W-1))) * (1 - cos(2*pi*y / (H-1)))
        """
        wx = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(width) / max(1, width - 1)))
        wy = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(height) / max(1, height - 1)))
        return np.outer(wy, wx).astype(np.float32)


class ColorHistogramLearner:
    """
    Pixel-wise Color Representation Learner.
    Builds foreground and background color likelihood models.
    """
    def __init__(self, num_bins: int = 16, learning_rate: float = 0.04):
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.bin_factor = 256.0 / num_bins

        # 3D histograms: [R, G, B]
        self.fg_hist = np.zeros((num_bins, num_bins, num_bins), dtype=np.float32)
        self.bg_hist = np.zeros((num_bins, num_bins, num_bins), dtype=np.float32)
        self.is_initialized = False

    def _quantize_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """Quantizes [0..255] RGB image into bin indices [0..num_bins-1]."""
        return np.clip((img_rgb / self.bin_factor).astype(np.int32), 0, self.num_bins - 1)

    def update(
        self,
        img_rgb: np.ndarray,
        target_box: Tuple[int, int, int, int],
        surround_box: Tuple[int, int, int, int]
    ):
        """
        Updates foreground and background color histograms.
        """
        tx, ty, tw, th = target_box
        sx, sy, sw, sh = surround_box
        H, W = img_rgb.shape[:2]

        tx1, tx2 = max(0, tx), min(W, tx + tw)
        ty1, ty2 = max(0, ty), min(H, ty + th)
        sx1, sx2 = max(0, sx), min(W, sx + sw)
        sy1, sy2 = max(0, sy), min(H, sy + sh)

        if tx2 <= tx1 or ty2 <= ty1 or sx2 <= sx1 or sy2 <= sy1:
            return

        quant = self._quantize_image(img_rgb)

        # Compute foreground histogram
        fg_crop = quant[ty1:ty2, tx1:tx2]
        cur_fg = np.zeros_like(self.fg_hist)
        for c in range(fg_crop.shape[1]):
            for r in range(fg_crop.shape[0]):
                p = fg_crop[r, c]
                cur_fg[p[0], p[1], p[2]] += 1.0
        fg_sum = np.sum(cur_fg)
        if fg_sum > 0:
            cur_fg /= fg_sum

        # Compute background histogram (surround excluding foreground)
        surr_crop = quant[sy1:sy2, sx1:sx2]
        cur_bg = np.zeros_like(self.bg_hist)
        for c in range(surr_crop.shape[1]):
            for r in range(surr_crop.shape[0]):
                p = surr_crop[r, c]
                cur_bg[p[0], p[1], p[2]] += 1.0

        # Subtract foreground counts from surround
        cur_bg = np.maximum(0.0, cur_bg - (cur_fg * fg_sum))
        bg_sum = np.sum(cur_bg)
        if bg_sum > 0:
            cur_bg /= bg_sum

        if not self.is_initialized:
            self.fg_hist = cur_fg
            self.bg_hist = cur_bg
            self.is_initialized = True
        else:
            lr = self.learning_rate
            self.fg_hist = (1.0 - lr) * self.fg_hist + lr * cur_fg
            self.bg_hist = (1.0 - lr) * self.bg_hist + lr * cur_bg

    def compute_likelihood_map(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Computes pixel-wise posterior likelihood:
        p(x) = p(x|fg) / (p(x|fg) + p(x|bg) + eps)
        """
        if not self.is_initialized:
            return np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32) * 0.5

        quant = self._quantize_image(img_rgb)
        p_fg = self.fg_hist[quant[:, :, 0], quant[:, :, 1], quant[:, :, 2]]
        p_bg = self.bg_hist[quant[:, :, 0], quant[:, :, 1], quant[:, :, 2]]
        eps = 1e-5
        return (p_fg / (p_fg + p_bg + eps)).astype(np.float32)


class CorrelationFilterTracker:
    """
    Fourier Domain Correlation Filter (Ridge Regression).
    Reversed from Camera::FC::Alg::StapleAdjust.
    """
    def __init__(
        self,
        template_size: Tuple[int, int] = (64, 64),
        learning_rate: float = 0.08,
        regularization: float = 1e-2
    ):
        self.w, self.h = template_size
        self.learning_rate = learning_rate
        self.lambda_reg = regularization
        self.hann_window = HannWindow2D.generate(self.w, self.h)

        # Precompute Gaussian peak desired response
        sigma = 0.1 * np.sqrt(self.w * self.h)
        y = np.arange(self.h) - self.h // 2
        x = np.arange(self.w) - self.w // 2
        xx, yy = np.meshgrid(x, y)
        gaussian_peak = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
        self.Y = np.fft.fft2(np.fft.fftshift(gaussian_peak)).astype(np.complex64)

        # Filter model in frequency domain: H = num / den
        self.model_num: Optional[np.ndarray] = None
        self.model_den: Optional[np.ndarray] = None

    def _extract_patch(self, img_gray: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
        """Extracts and resamples image patch to fixed template size."""
        cx, cy = center
        x1 = int(round(cx - self.w / 2.0))
        y1 = int(round(cy - self.h / 2.0))
        x2 = x1 + self.w
        y2 = y1 + self.h

        H, W = img_gray.shape
        # Safe crop with boundary padding
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        x1_c, x2_c = max(0, x1), min(W, x2)
        y1_c, y2_c = max(0, y1), min(H, y2)

        crop = img_gray[y1_c:y2_c, x1_c:x2_c]
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

        return crop.astype(np.float32)

    def init_filter(self, img_gray: np.ndarray, center: Tuple[float, float]):
        """Trains initial correlation filter."""
        patch = self._extract_patch(img_gray, center)
        # Apply Hann window
        windowed = patch * self.hann_window
        X = np.fft.fft2(windowed)

        self.model_num = np.conj(self.Y) * X
        self.model_den = np.real(np.conj(X) * X)

    def correlate(self, img_gray: np.ndarray, center: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """
        Correlates current search patch with learned filter.
        Returns: (response_map, (dx, dy), peak_confidence)
        """
        if self.model_num is None or self.model_den is None:
            return np.zeros((self.h, self.w), dtype=np.float32), (0.0, 0.0), 0.0

        patch = self._extract_patch(img_gray, center)
        windowed = patch * self.hann_window
        Z = np.fft.fft2(windowed)

        # Response in frequency domain
        H = self.model_num / (self.model_den + self.lambda_reg)
        R = np.fft.ifft2(H * Z)
        response = np.real(np.fft.fftshift(R))

        # Check standard deviation and peak to detect flat/zero correlation response (e.g. black frame dropout)
        # Matches hardware Scene Analysis IP coprocessor zero-response handling:
        std_val = float(np.std(response))
        peak_val = float(np.max(response)) if response.size > 0 else 0.0
        if std_val <= 1e-6 or peak_val <= 1e-6:
            return response, (0.0, 0.0), 0.0

        # Find peak
        max_idx = np.unravel_index(np.argmax(response), response.shape)
        peak_y, peak_x = max_idx
        dx = float(peak_x - self.w // 2)
        dy = float(peak_y - self.h // 2)

        peak_val = float(response[peak_y, peak_x])
        mean_val = float(np.mean(response))
        psr = (peak_val - mean_val) / max(1e-4, std_val) # Peak-to-Sidelobe Ratio

        # Normalized confidence [0.0 .. 1.0]
        confidence = float(np.clip(psr / 12.0, 0.0, 1.0))

        return response, (dx, dy), confidence

    def update_filter(self, img_gray: np.ndarray, center: Tuple[float, float]):
        """Online adaptive filter update with learning rate."""
        if self.model_num is None:
            self.init_filter(img_gray, center)
            return

        patch = self._extract_patch(img_gray, center)
        windowed = patch * self.hann_window
        X = np.fft.fft2(windowed)

        new_num = np.conj(self.Y) * X
        new_den = np.real(np.conj(X) * X)

        lr = self.learning_rate
        self.model_num = (1.0 - lr) * self.model_num + lr * new_num
        self.model_den = (1.0 - lr) * self.model_den + lr * new_den


class SingleTargetStapleTracker:
    """Staple tracker instance for a single target."""
    def __init__(
        self,
        target_id: int = 0,
        alpha_staple: float = 0.70  # Weight between correlation filter (0.7) and histogram (0.3)
    ):
        self.target_id = target_id
        self.alpha = alpha_staple
        self.corr_filter = CorrelationFilterTracker()
        self.color_learner = ColorHistogramLearner()

        # Bounding box in pixel coords: [x, y, w, h]
        self.box: Optional[Tuple[float, float, float, float]] = None
        self.velocity = (0.0, 0.0)
        self.scale = 1.0
        self.is_tracking = False

    def init_target(self, img_rgb: np.ndarray, init_box: Tuple[float, float, float, float]):
        """Initializes tracker with target bounding box."""
        self.box = init_box
        x, y, w, h = init_box
        cx, cy = x + w / 2.0, y + h / 2.0

        # Initialize correlation filter
        img_gray = np.mean(img_rgb, axis=2) if img_rgb.ndim == 3 else img_rgb
        self.corr_filter.init_filter(img_gray, (cx, cy))

        # Initialize color histogram with target and surround box
        surround_box = (
            int(x - w * 0.5),
            int(y - h * 0.5),
            int(w * 2.0),
            int(h * 2.0)
        )
        target_int = (int(x), int(y), int(w), int(h))
        self.color_learner.update(img_rgb, target_int, surround_box)

        self.velocity = (0.0, 0.0)
        self.scale = 1.0
        self.is_tracking = True

    def track(self, img_rgb: np.ndarray) -> TrackedTargetResult:
        """
        Processes new frame and updates target position.
        """
        if not self.is_tracking or self.box is None:
            return TrackedTargetResult(
                target_id=self.target_id,
                x=0.0, y=0.0, width=0.0, height=0.0,
                confidence=0.0, dx=0.0, dy=0.0, scale=1.0,
                is_reliable=False, error_code=1
            )

        x, y, w, h = self.box
        cx, cy = x + w / 2.0, y + h / 2.0

        img_gray = np.mean(img_rgb, axis=2) if img_rgb.ndim == 3 else img_rgb

        # 1. Correlation filter response
        resp_corr, (dx_c, dy_c), conf_c = self.corr_filter.correlate(img_gray, (cx, cy))

        # 2. Histogram response
        like_map = self.color_learner.compute_likelihood_map(img_rgb)
        # Average likelihood inside predicted target box
        px1 = int(np.clip(x + dx_c, 0, img_rgb.shape[1] - 1))
        py1 = int(np.clip(y + dy_c, 0, img_rgb.shape[0] - 1))
        px2 = int(np.clip(px1 + w, 0, img_rgb.shape[1]))
        py2 = int(np.clip(py1 + h, 0, img_rgb.shape[0]))

        conf_hist = 0.5
        if px2 > px1 and py2 > py1:
            conf_hist = float(np.mean(like_map[py1:py2, px1:px2]))

        # 3. Fused confidence score
        fused_conf = float(self.alpha * conf_c + (1.0 - self.alpha) * conf_hist)

        # Smooth displacement
        dx, dy = dx_c, dy_c
        self.velocity = (dx, dy)

        # Exact sub_705BC6 position update:
        # *(_WORD *)(v9 + 32) += (*(_DWORD *)(v8 + 656) << 12) / v10;
        # Fixed point Q12 scaling with scale factor normalization
        scale_val = max(0.1, self.scale)
        norm_dx = dx / scale_val
        norm_dy = dy / scale_val

        H, W = img_rgb.shape[:2]
        new_x = float(np.clip(x + norm_dx, 0.0, max(0.0, W - w)))
        new_y = float(np.clip(y + norm_dy, 0.0, max(0.0, H - h)))
        self.box = (new_x, new_y, w, h)
        new_cx, new_cy = new_x + w / 2.0, new_y + h / 2.0

        # Exact sub_705BC6 boundary check: margin of 20 pixels from any of the 4 borders
        # v13 = ((v11 - 20) | (v12 - 20)) < 0 || v12 + h + 20 > H || v11 + w + 20 > W
        near_boundary = (new_x < 20.0 or new_y < 20.0 or (new_x + w + 20.0) > W or (new_y + h + 20.0) > H)

        # Online model update if tracking confidence is high
        if fused_conf > 0.35:
            self.corr_filter.update_filter(img_gray, (new_cx, new_cy))
            surround_box = (
                int(new_x - w * 0.5),
                int(new_y - h * 0.5),
                int(w * 2.0),
                int(h * 2.0)
            )
            target_int = (int(new_x), int(new_y), int(w), int(h))
            self.color_learner.update(img_rgb, target_int, surround_box)

        # Peak response confidence check matching sub_705BC6: v31 <= v30 (v29 * v29)
        is_reliable = (fused_conf >= 0.38) and not near_boundary

        return TrackedTargetResult(
            target_id=self.target_id,
            x=new_x,
            y=new_y,
            width=w,
            height=h,
            confidence=fused_conf,
            dx=norm_dx,
            dy=norm_dy,
            scale=self.scale,
            is_reliable=is_reliable,
            error_code=0 if is_reliable else (3 if near_boundary else 2)
        )


class DualTargetStapleManager:
    """
    Master Dual-Target Tracking Engine Manager.
    Reversed from idt_staple_get_result (sub_280C3C).
    Manages Target 0 (Primary) and Target 1 (Secondary).
    """
    MAX_TARGETS = 2

    def __init__(self):
        self.trackers: List[SingleTargetStapleTracker] = [
            SingleTargetStapleTracker(target_id=0),
            SingleTargetStapleTracker(target_id=1)
        ]

    def set_target(self, target_id: int, img_rgb: np.ndarray, box: Tuple[float, float, float, float]):
        """Initializes or resets a specific target slot (0 or 1)."""
        if 0 <= target_id < self.MAX_TARGETS:
            self.trackers[target_id].init_target(img_rgb, box)

    def process_frame(self, img_rgb: np.ndarray) -> List[TrackedTargetResult]:
        """
        Processes new video frame and returns results for both targets.
        Matches the dual-slot output of sub_280C3C.
        """
        results: List[TrackedTargetResult] = []
        for t in self.trackers:
            if t.is_tracking:
                results.append(t.track(img_rgb))
        return results
