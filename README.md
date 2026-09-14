# Sony Alpha 7S III: AutoFocus & Tracking Algorithms (Clean-Room Model)

This repository contains a clean-room Python model implementing core AutoFocus (PDAF / Contrast AF) and Tracking arbitration logic reverse-engineered from the **Sony Alpha 7S III (`ILCE-7SM3`)** BIONZ XR coprocessor firmware (`cpapp-b.elf` / `cpapp-b.elf.i64`).

The mathematical formulas, bitmasks, 20-byte record structures, and state transitions for the 12 reversed routines have been verified against Hex-Rays decompiler output from IDA Pro 9.4. Higher-level hardware-accelerated functions (such as ASIC-level correlation filtering and raw sensor phase detection) are modeled behaviorally in Python.

---

## 1. Firmware Target & Provenance

- **Target Binary**: `extracted/nflasha8/cpapp-b.bin` / `cpapp-b.elf` (`cpapp-b.elf.i64` IDA Pro database)
- **Architecture**: ARM Cortex-A7 / ARMv7-A 32-bit Little Endian
- **Co-Processor SoC**: Sony CXD90057 (BIONZ XR subsystem)
- **Firmware Version**: ILCE-7SM3 firmware v5.01 (`extracted/rootfs` and `extracted/nflasha8`)

### Reversed Function Mapping

The implementation directly models routines identified via RTTI and symbol tables in `cpapp-b.elf`:

| Address | Binary Source File | Disassembly Behavior & Role | Implementation File |
| :--- | :--- | :--- | :--- |
| `sub_272420` | `priority_defocus_face_selector.cpp` | 8-slot (8 × 60 B) iteration, reliability & density gating, min-defocus (`min_df`) selection | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_271C00` | `defocus_face_selector.cpp` | Lock-on range retention hysteresis (`PreFaceIsInLockOnRange`) | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_271FA8` | `priority_queue.cpp` | Linked-list candidate priority rotation across 8 candidate face slots | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_278778` | `fc_alg_tracking_eye_selector.cpp` | Inter-ocular distance (D_eye), 2 × D_eye scale, nearer eye selection | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_278858` | `fc_alg_tracking_eye_selector.cpp` | 3-region piecewise linear spatial radius interpolation (`R >> 4`) | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_709FD6` | `fc_alg_tracking_eye_selector.cpp` | Spatial acceptance gating: (dist_eye_to_tracking_center)² ≤ R² | [`eye_face_arbiter.py`](eye_face_arbiter.py) |
| `sub_2718A4` | `defocus_info_translator.cpp` | 20-byte packed ZPD DMA record unpacking, coordinate right-shift (`>>= 1`) | [`pdaf.py`](pdaf.py) |
| `sub_271A90` | `defocus_info_translator.cpp` | Invalid point error bitmask filtering (`(*v10 & 0x90CF) == 0`) | [`pdaf.py`](pdaf.py) |
| `sub_6FC676` | `defocus_evaluator.cpp` | Metric accumulation, correlation reliability (`0x30`), optical windowing | [`pdaf.py`](pdaf.py) |
| `sub_8E5BF8` | `mario_afc_state.cpp` | Continuous AF (Mario AFC) 5×5 hardware clock tick debounce matrix (`dword_C747C4`) | [`pdaf.py`](pdaf.py) |
| `sub_8A5122` | `Camera::LC::MovieContrastAF` | Contrast AF wobble state dispatch (`CStateWob::Transition`) | [`contrast_af.py`](contrast_af.py) |
| `sub_8A520C` | `Camera::LC::MovieContrastAF` | Contrast AF peak hill-climb state dispatch (`CStateYama::Transition`) | [`contrast_af.py`](contrast_af.py) |
| `sub_705BC6` | `ddl_minna_staple_common.cpp` | Q12 fixed-point displacement integration `(dx << 12) / scale`, 20px border check | [`staple_tracker.py`](staple_tracker.py) |
| `sub_71E28A` | `ddl_minna_staple_common.cpp` | STAPLE dual-slot coordinator (Primary Slot 0, Secondary Slot 1) | [`staple_tracker.py`](staple_tracker.py) |

---

## 2. Reversed Logic & Architectural Details

### 1. 841-Point PDAF Grid & 20-Byte Hardware DMA Record
- Phase detection sites map onto a $29 \times 29$ grid ($841$ points total, parameter `zpd_max = 841`, index $0 \le a_3 \le 0x348$).
- Points arrive from the sensor ASIC DMA buffer in packed 20-byte records (`struct.unpack('<hhIiII')`):
  * `+0` (int16): Sensor X coordinate (binned right-shift `>>= 1` in `sub_2718A4`)
  * `+2` (int16): Sensor Y coordinate
  * `+4` (uint16): Status & error bitmask
  * `+8` (int32): Signed phase disparity defocus ($\Delta d$)
  * `+12` (uint16): Correlation peak height
  * `+16` (uint16): Confidence normalizer
- **Error Mask (`0x90CF`)**: Firmware `sub_271A90` rejects points where `(flags & 0x90CF) != 0`.
- **Correlation Reliability Mask (`0x30`)**: Firmware `sub_6FC676` gates point reliability on bits 4 and 5 (`flags & 0x30 != 0`).
- **Defocus Accumulation (`sub_6FC676`)**: Accumulates non-error point count (`num_no_err`) and absolute defocus sum (`sum_df_abs`) across all points passing `0x90CF`. For points additionally passing `0x30`, increments `num_reliable` and checks optical lock-on boundaries:

  $$
  \text{min\_optical\_bound} < \Delta d_i \le \text{max\_optical\_bound}
  $$

### 2. Mario AFC 5×5 Debounce Matrix (`sub_8E5BF8` / `dword_C747C4`)
- Continuous AF (`MarioAfcStateMachine`) uses a 5×5 debounce matrix measured in hardware clock ticks to prevent state chattering and hunting:

  $$
  \text{Matrix}_{5 \times 5} = \begin{bmatrix}
  0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 400{,}000 & 10{,}000{,}000 & 0 \\
  0 & 0 & 0 & 10{,}000{,}000 & 0 \\
  0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0
  \end{bmatrix}
  $$

- Transitions require target persistence: `current_tick - pending_target_tick >= matrix[curr_slot, next_slot]`.
- Transitioning from `LOCKED` (Slot 1) to `TRACKING` (Slot 2) requires 400,000 clock ticks (~40 ms at 10 MHz), suppressing focus dropouts from momentary occlusions.

### 3. Priority Face Arbitration (`sub_272420` & `sub_271C00`)
- Evaluates up to 8 candidate face slots (8 × 60 bytes).
- Slots are gated by reliability and point density thresholds:
  1. `reliable_pdaf_count >= min_reliable_thresh` (default 4)
  2. `point_metric << 8 >= reliable_pdaf_count * density_scale`
- The arbiter selects the candidate satisfying these thresholds with the **absolute minimum defocus error** (`min_df`).
- **Hysteresis Retention**: `PreFaceIsInLockOnRange` (`sub_271C00`) retains the active face target as long as it remains within optical defocus tolerance.

### 4. Scale-Adaptive Eye Spatial Gating (`sub_278858` & `sub_709FD6`)
- Detected eyes are validated against an active tracking centroid $(t_x, t_y)$.
- Scale is derived from inter-ocular distance $D_{\text{eye}} = \sqrt{\Delta x^2 + \Delta y^2}$ with $s = 2 \times D_{\text{eye}}$ (`sub_278778`).
- Acceptance radius $R(s)$ is computed via piecewise linear interpolation:

  $$
  R(s) = \left\lfloor \frac{1}{16} \cdot \left( v_5 v_4 + (s - v_4) \frac{v_6 v_3 - v_5 v_4}{v_3 - v_4} \right) \right\rfloor
  $$

  where $v_4 = 64$, $v_3 = 256$, $v_5 = 8$, $v_6 = 12$.
- Candidate eyes are rejected as outliers unless:

  $$
  (x_{\text{eye}} - t_x)^2 + (y_{\text{eye}} - t_y)^2 \le R(s)^2
  $$

### 5. STAPLE Tracking & Hardware Boundaries (`sub_705BC6` & `sub_26E8D0`)
- **Hardware Boundary**: In the physical camera, correlation filtering, feature extraction, and histogram updates run on dedicated hardware (Scene Analysis IP / FRC2 coprocessor via `ddl_saCalcStart` in `sub_26E8D0`).
- **Python Emulation**: [`staple_tracker.py`](staple_tracker.py) models this behavior using an academic STAPLE formulation (Bertinetto et al., CVPR 2016) in NumPy (2D FFT correlation filter + 3D color histogram).
- **Reversed Firmware Logic**:
  * Q12 displacement coordinate integration:

    $$
    x \leftarrow x + \frac{dx \ll 12}{\text{scale}}
    $$

  * 20-pixel border check: target is flagged lost (`error_code = 3`) if any boundary violates the 20-pixel margin:

    $$
    x < 20 \quad\lor\quad y < 20 \quad\lor\quad x + w + 20 > W \quad\lor\quad y + h + 20 > H
    $$

### 6. Movie Contrast AF State Machine (`sub_8A5122` & `sub_8A520C`)
- Dispatches dual-loop transitions between `CStateWob` (micro-wobble perturbation) and `CStateYama` (hill-climbing peak search).

---

## 3. Python Simulation Architecture

```text
+-----------------------------------------------------------------------------------+
|                   REVERSED SONY BIONZ XR LOGIC (PYTHON MODEL)                     |
+-----------------------------------------------------------------------------------+
                                          |
                         [ Video Frame / Synthetic DMA ]
                                          |
                   +----------------------+----------------------+
                   |                                             |
         [ 841-Point PDAF Grid ]                       [ Frame Image Data ]
                   |                                             |
         (sub_2718A4 Unpacking)                        [ STAPLE Tracker Model ]
                   |                                   (NumPy FFT + Histograms)
         (0x90CF / 0x30 Gating)                                  |
                   |                                   (sub_705BC6 Q12 & Margin)
         (sub_6FC676 Accumulator)                                |
                   |                                    [ TargetCentroid (tx, ty) ]
                   |                                             |
                   |                               +-------------+-------------+
                   |                               |                           |
                   v                               v                           v
       +-------------------------+     +-------------------+     +--------------------+
       |  MarioAfcStateMachine   |     | PriorityFaceArb.  |     |    EyeSelector     |
       |  (sub_8E5BF8 Debounce)  |<----+ (sub_272420 MinDF)|<----+ (sub_278858 Gating)|
       +-------------------------+     +-------------------+     +--------------------+
                   |
                   v
       [ Simulated Lens Drive ]  <----> [ MovieContrastAF (sub_8A5122 / sub_8A520C) ]
```

---

## 4. Package Structure

| File | Type | Description |
| :--- | :--- | :--- |
| [`pdaf.py`](pdaf.py) | **Reversed + Model** | 20-byte record decoder, 841-point grid model, `sub_6FC676` accumulator, Mario AFC 5×5 debounce state machine |
| [`eye_face_arbiter.py`](eye_face_arbiter.py) | **Reversed + Model** | Priority Face Arbiter (`sub_272420`), Eye Selector with spatial gating (`sub_278858`, `sub_709FD6`) |
| [`contrast_af.py`](contrast_af.py) | **Reversed + Model** | Contrast AF state transitions (`sub_8A5122`, `sub_8A520C`), wobble/hill-climb simulation |
| [`staple_tracker.py`](staple_tracker.py) | **Clean-Room Emulation** | Academic STAPLE tracker with reversed Q12 scaling and 20px border check (`sub_705BC6`) |
| [`pipeline_demo.py`](pipeline_demo.py) | **Demo Simulation** | 15-frame synthetic simulation demonstrating integrated pipeline state flow |
| [`decompiled_exact_algos.json`](decompiled_exact_algos.json) | **Firmware Ground Truth** | Exact Hex-Rays decompiler output extracted from `cpapp-b.elf.i64` for all 12 reversed routines |
| [`tests/`](tests/) | **Unit Test Suites** | 53 unit tests verifying bitmasks, formulas, and edge cases |

---

## 5. Verification & Test Execution

### Installation

Requires Python 3.8+ and NumPy:
```bash
pip install -r requirements.txt
```

### Running Test Suites

All 53 unit and edge-case tests can be executed via `unittest`:
```bash
python -m unittest discover tests -v
```

```text
Ran 53 tests in 0.176s
OK
```

### Scope & Test Methodology

> [!NOTE]
> **Data Environment**: The test suites and demo operate on **synthetic and mocked data** (`struct.pack`, synthetic `PDAFPoint`, synthetic NumPy arrays). They do not run on physical camera hardware or captured raw sensor dumps.

The test suites verify:
1. **Disassembly Arithmetic & Parity ([tests/test_exact_sony_algos.py](tests/test_exact_sony_algos.py))**:
   - 20-byte record decoding and bitwise shift matching `sub_2718A4`.
   - Accumulator logic matching `sub_6FC676` error bitmask `0x90CF` and correlation flag `0x30`.
   - Debounce matrix thresholds matching `sub_8E5BF8` (`dword_C747C4`).
   - Piecewise linear radius and outlier rejection matching `sub_278858` and `sub_709FD6`.
   - 8-slot min-defocus face selection matching `sub_272420`.
   - Q12 fixed-point coordinate integration and 20px border checks matching `sub_705BC6`.
2. **Boundary & Stress Tests ([tests/test_adversarial_edge_cases.py](tests/test_adversarial_edge_cases.py))**:
   - Individual bit-level rejection for all 8 bits in `0x90CF`.
   - Strict lower bound (`*v10 < df`) and inclusive upper bound (`df <= *v9`) tests.
   - Debounce tick wraparound, jitter, and force-bypass execution.
   - Four-edge border margin breach behavior.
3. **Subsystem Models ([tests/test_af_tracker.py](tests/test_af_tracker.py))**:
   - Focus area geometries (Wide, Zone, Spot S/M/L) and Hann windowing.

### Running the Pipeline Demo

To run the 15-frame synthetic simulation:
```bash
python pipeline_demo.py
```

---

## 6. Legal Notices, Fair Use & Licensing

### Research & Interoperability Disclaimer
This project is an independent, non-commercial academic research initiative conducted for the purposes of reverse engineering analysis, algorithmic study, and software interoperability. 

- **Legal Basis**: Under 17 U.S.C. § 102(b), copyright protection does not extend to ideas, procedures, systems, mathematical concepts, or functional formulas. Reverse engineering of compiled firmware binaries for interoperability, research, and analysis is recognized as fair use under established legal precedent (*Sega v. Accolade*, 977 F.2d 1510; *Sony Computer Entertainment v. Connectix Corp.*, 203 F.3d 596; *Google LLC v. Oracle America, Inc.*, 141 S. Ct. 1183) and international statutory provisions (EU Directive 2009/24/EC, Arts. 5(3) & 6; Japan Copyright Act, Arts. 30-4 & 47-7).
- **Clean-Room Implementation**: All Python code in this directory consists of original, independently written expressions of functional logic, mathematical algorithms, and bitmask layouts. No proprietary source code or binary blobs from Sony are incorporated into the Python source code.

### Trademark & Brand Notice
"Sony", "Sony Alpha", "ILCE-7SM3", "BIONZ XR", and "Exmor R" are registered trademarks of **Sony Group Corporation** or its affiliates. Their use in this repository constitutes **nominative fair use** solely to accurately identify the hardware and firmware subject to study. This project is **not** sponsored, authorized, endorsed by, or affiliated with Sony Group Corporation or any of its subsidiaries.

### Distribution & Proprietary Firmware Notice
This repository does not redistribute Sony proprietary binary firmware images (`BODYDATA.dat`, `cpapp-b.bin`, `cpapp-b.elf`), decrypter keys, or full disassembled binaries. Any reference to memory addresses or disassembled fragments is provided strictly for academic verification, commentary, and citation.

### Code License
The original Python source code in this package is licensed under the [MIT License](LICENSE) (or [root LICENSE](../LICENSE)). The license applies solely to the author's original Python code and documentation, and explicitly excludes any third-party trademarks, proprietary hardware microcode, or patents.

