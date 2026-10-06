# PartGuard: Offline Visual Inspection for Small Manufacturers

English | [繁體中文](README.zh-TW.md)

PartGuard is an **offline visual inspection prototype** designed for small manufacturers. It focuses on final inspection of small metal parts such as washers, nuts, and stamped rings before shipment.

The system runs entirely on a local device without uploading images to the cloud. For each detected part, PartGuard determines whether it passes inspection and, if not, reports the corresponding defect reason.

The current prototype is implemented with OpenCV and NumPy and can run directly on a standard laptop CPU. It was developed as a proof of concept for the ASUS UGen AI League (Battlefield Lightning).

> **Current validation scope**
>
> All quantitative results for metal-part inspection in this project are currently based on **programmatically generated synthetic images**, not photographs of real factory parts.  
> Real-world factors such as metallic glare, oil contamination, machining texture, and surface irregularities may affect inspection performance. Validation and calibration with real parts are therefore required before deployment.

---

## Supported Inspection Checks

| Check | Description | Reason Code |
|---|---|---|
| Outer diameter | Compared against the batch median or engineering drawing tolerances | `SIZE_SMALL`, `SIZE_LARGE` |
| Shape integrity | Detects chips, burr-like deformation, or obvious shape distortion | `DEFORMED` |
| Hole geometry | Detects missing holes, incorrect hole size, and off-center holes | `HOLE_MISSING`, `HOLE_SIZE`, `HOLE_OFFCENTER` |
| Surface rust | Detects orange-brown rust-like regions based on color | `RUST` |
| Surface scratches | Detects thin dark-line patterns on the part surface | `SCRATCH` |

Size inspection supports two modes:

- If `px_per_mm` and drawing-based size limits are configured, the system evaluates the measured physical size directly.
- If physical calibration has not yet been performed, the system uses the median outer diameter of the current batch as a reference and compares each part against it.

---

## System Architecture

```text
 Camera + diffuse lighting + gray reference card
                        |
                        v
+------------------- Edge Device -------------------+
|                                                   |
|  1. Image preprocessing                          |
|     - gray-card lighting calibration              |
|     - foreground / part segmentation              |
|     - hole segmentation                           |
|                                                   |
|  2. Feature extraction                            |
|     - size                                        |
|     - roundness / solidity                        |
|     - hole size / position                        |
|     - rust / scratch features                     |
|                                                   |
|  3. Rule-based inspection                         |
|     - drawing tolerance / batch statistics        |
|     - PASS / FAIL                                 |
|     - defect reason codes                         |
|                                                   |
|  Stage II (planned):                              |
|     learned defect classifier on ASUS UGen300     |
+---------------------------------------------------+
              |                         |
              v                         v
     Annotated image / UI        Local JSONL log
```

The current metal-part inspection pipeline runs entirely on the CPU, including image preprocessing, feature extraction, and rule-based inspection.

`partguard/backends.py` already includes a `HailoEmbedder` interface as an extension point for deploying a learned model on the ASUS UGen300 / Hailo-10H in a later stage. **UGen300 inference is not implemented yet.**

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
opencv-python
numpy
pyyaml
```

---

### 2. Generate Synthetic Test Data

```bash
python tools/make_parts_synthetic.py
```

This script generates synthetic washer images for testing, including both normal parts and several types of injected defects.

These images are generated programmatically and are **not photographs of real metal parts**.

---

### 3. Run the Smoke Tests

```bash
python tests/test_smoke.py
```

The tests verify that:

- normal parts are classified as PASS,
- the injected defect types are detected correctly,
- gray-card calibration compensates for reduced overall image brightness.

---

### 4. Run Metal-Part Inspection

Inspect a synthetic test image:

```bash
python run.py \
    --mode parts \
    --source samples/parts_batch_mixed.png \
    --lang en
```

Use Traditional Chinese output:

```bash
python run.py \
    --mode parts \
    --source samples/parts_batch_mixed.png \
    --lang zh
```

---

### 5. Run Live Camera Inspection

```bash
python run.py \
    --mode parts \
    --source 0 \
    --show
```

Press `q` to close the preview window.

---

## Outputs

Each inspection produces three types of output.

### Console Summary

Example:

```text
[parts_batch_mixed.png] FAIL

8 items: 1 pass, 7 need attention
- Item 2: deformed shape
- Item 3: hole missing
- Item 4: too large
- Item 5: hole size off
...
```

Use:

```bash
--lang en
```

or

```bash
--lang zh
```

to switch between English and Traditional Chinese output.

### Annotated Image

Inspection results are drawn on the input image and saved by default to:

```text
out/
```

Passing parts are labeled with `OK`, while failed parts are labeled with the corresponding reason code.

### JSONL Inspection Log

Each inspection is also recorded in:

```text
logs/inspections.jsonl
```

For example:

```json
{
  "ts": "2026-...",
  "mode": "parts",
  "source": "...",
  "passed": false,
  "latency_ms": 27.4,
  "details": [...]
}
```

This provides a local audit trail containing inspection results, defect details, and processing latency for later analysis.

---

## Synthetic Benchmark Results

The benchmark can be reproduced with:

```bash
python tools/benchmark_parts.py
```

The benchmark simulates:

- different overall lighting levels,
- different color casts,
- sensor noise,
- different synthetic defect severities.

By default, the benchmark uses a fixed random seed of `2026` and generates 100 batches per test condition, with 8 synthetic parts in each batch.

For the baseline benchmark, defect severity is set to `0.75`.

| Metric | Result |
|---|---|
| Defective parts detected | 100% (95% CI: 98.8–100%, 330 parts) |
| Correct defect reason reported | 100% |
| Good parts incorrectly rejected | 0% (95% CI: 0–0.8%, 470 parts) |
| Processing time per batch of 8 parts | About 27 ms median |
| Latency P95 | About 68 ms |

These latency measurements were collected on a standard CPU and **do not represent UGen300 performance**.

---

## Detection at Different Defect Severities

The benchmark also evaluates how detection performance changes as synthetic defects become less or more obvious.

A severity of `1.0` represents a very obvious injected defect. The severity value is only a relative parameter used by the synthetic image generator and has no physical unit.

| Defect Severity | Shape / Size | Hole | Surface |
|---|---:|---:|---:|
| 0.1 | 0% | 30%* | 0% |
| 0.25 | 0% | 32%* | 13% |
| 0.5 | 59% | 99% | 100% |
| 0.75 | 100% | 100% | 100% |
| 1.0 | 100% | 100% | 100% |

\* `HOLE_MISSING` can still be detected at low severity values, which is why the hole category retains roughly 30% detection at low severity. Hole-size and off-center defects require larger deviations before they can be detected reliably.

This also illustrates a limitation of the current rule-based approach: tighter thresholds can detect subtler defects, but they also increase the risk of rejecting normal manufacturing variation in real parts.

---

## Lighting and Gray-Card Calibration

Visual features of metal parts can change significantly with lighting conditions. PartGuard therefore supports simple color and brightness calibration using a **neutral gray reference card** fixed within the camera frame.

The gray-card region is configured in:

```yaml
configs/parts.yaml
```

For example:

```yaml
calibration:
  card_roi: [0.88, 0.86, 0.08, 0.08]
  target_gray: 128
```

The program measures the average value of the gray-card region in the B, G, and R channels and independently rescales each channel so that the reference card remains close to the configured gray level.

In the synthetic benchmark, without gray-card calibration, reducing overall brightness to approximately `0.7x` or lower can cause normal parts to be rejected.

With gray-card calibration enabled, the synthetic good parts produced no false alarms across approximately `0.6x ~ 1.4x` lighting gain and simulated color casts of approximately ±20%.

If the real setup **does not use a gray reference card**, remove the following section from the configuration:

```yaml
calibration:
```

Otherwise, the system may incorrectly treat another region of the image as the calibration reference.

Gray-card calibration also cannot recover information lost to overexposure. Once bright pixels are clipped to 255, the original image information is no longer available. Camera exposure should therefore be configured to avoid saturation.

---

## Sensor Noise

The benchmark also evaluates different levels of synthetic image noise.

With the current configuration, the system remains reliable at approximately `5 / 255` noise intensity. False alarms begin to appear around `10 / 255`, and inspection performance gradually breaks down at higher noise levels.

Scratch detection estimates the noise level on the part surface and dynamically increases the detection threshold as noise increases. This helps prevent sensor noise from being incorrectly classified as scratches.

The trade-off is that subtle real scratches may become harder to detect when the input image itself is highly noisy.

---

## Applying PartGuard to Real Parts

Before deploying the current prototype in a real manufacturing environment, the following calibration process is recommended:

1. **Fix the camera, fixture, and lighting**

   Keep the camera position and part placement as consistent as possible, and use uniform diffuse lighting to reduce metallic glare.

   For highly reflective materials, a polarizing filter may also help reduce specular reflections.

2. **Collect good-part samples**

   Capture at least 50 real parts that are known to be acceptable. This helps characterize the natural variation in size, shape, and surface appearance under the actual manufacturing process.

3. **Calibrate physical dimensions**

   Measure a part with known dimensions and determine:

   ```yaml
   px_per_mm
   ```

   Then configure the acceptable drawing limits in `configs/parts.yaml`:

   ```yaml
   size:
     min_mm: ...
     max_mm: ...
   ```

4. **Retune inspection thresholds**

   The current thresholds were tuned using synthetic data. For real deployment, parameters under:

   ```yaml
   shape:
   hole:
   surface:
   ```

   should be adjusted based on the distribution of real good parts.

5. **Validate with real defective parts**

   Finally, collect actual manufacturing defects and verify that the system detects the defects that need to be intercepted in production.

---

## ROI Break-Even Model

The project includes a simple ROI estimation tool:

```bash
python tools/roi_model.py
```

It estimates:

> How many defective lots must an automated inspection station prevent from reaching customers each year in order to offset the cost of the system?

For example, under the following **assumptions**:

```text
Annual system cost: 40,000
Cost per returned defective lot: 20,000
Defective-lot catch rate: 80%
```

the break-even point is approximately:

```text
2.5 lots / year
```

In other words, under these assumptions, preventing approximately 2.5 defective lots per year from reaching customers would offset the annual cost of the inspection station.

These values are provided only to demonstrate the calculation and **are not real factory cost data**. Actual ROI analysis should replace them with sourced equipment, labor, scrap, rework, and customer-return costs.

---

## Other Demonstration Modes

In addition to metal-part inspection, the repository contains two additional visual-inspection examples:

```text
partguard/bakery.py
partguard/bento.py
```

They correspond to:

```bash
--mode bakery
```

and:

```bash
--mode bento
```

These modes demonstrate how the same general inspection framework can be adapted to different application scenarios.

The Bento mode additionally supports:

```text
HSV histogram
ONNX backbone
Hailo backend (placeholder)
```

However, the current primary development and benchmarking focus of PartGuard is **metal-part inspection**.

---

## Current Limitations and Future Work

PartGuard is still a proof-of-concept prototype. The following limitations should be considered:

- **All current quantitative results for metal-part inspection are based on synthetic images.** Real metallic glare, oil contamination, machining texture, subtle burrs, and other production artifacts have not yet been validated.
- **The current inspection method is primarily rule-based.** Very subtle defects may not be detected reliably, and performance depends on threshold selection.
- **UGen300 integration is not yet implemented.** `HailoEmbedder` is currently only a placeholder interface and does not yet run actual Hailo-10H inference.
- **Current latency numbers were measured on a standard CPU.** They should not be interpreted as expected UGen300 performance.
- **Licenses for public industrial-defect datasets must be checked individually.** Some datasets are restricted to research or non-commercial use and may conflict with competition requirements or future commercial deployment. This repository does not include third-party dataset images.

Future work will focus on collecting and validating real factory-part data and evaluating a learned defect classifier deployed on the ASUS UGen300 to improve detection of subtle and complex surface defects.