# PartGuard: offline visual inspection for small factories

PartGuard checks small metal parts (washers, nuts, stamped rings) **on the device, without the cloud**, as the last
step before a lot ships. For every part it returns PASS or FAIL and the reason.

| Check | Reason codes |
|---|---|
| Outer diameter (vs. batch median or drawing tolerance) | `SIZE_SMALL`, `SIZE_LARGE` |
| Chips, burrs, deformation (solidity, roundness) | `DEFORMED` |
| Hole: missing, wrong size, off-centre | `HOLE_MISSING`, `HOLE_SIZE`, `HOLE_OFFCENTER` |
| Surface: rust (colour), scratches (thin dark lines) | `RUST`, `SCRATCH` |

Status: a **laptop prototype** (OpenCV + NumPy) built for the ASUS UGen AI League (Battlefield Lightning).
The same engine also ships food examples (bakery, bento tray) in `partguard/bakery.py` and `partguard/bento.py`.

> **Honest scope:** every number below comes from **synthetic** images (drawn washers with injected defects).
> Real metal adds glare, oil and machining texture that is not modelled. Real-part validation is the next step.

## Architecture

```
 Camera + diffuse light + gray reference card
                 |
                 v
 +---------------- edge node (mini-PC or SBC + ASUS UGen300) ----------------+
 |  1 Pre-process   light calibration (gray card), part and hole segmentation |
 |  2 Feature model learned defect classifier on UGen300 (Stage II, planned)  |
 |  3 Rules engine  drawing tolerances -> PASS/FAIL + reason codes            |
 +-----------------------------------------------------------------------------+
        |                                          |
        v                                          v
 station display / alert light               local JSONL log (audit)

 Inputs: part spec + tolerance file (configs/parts.yaml), golden-sample library
```

Today, stages 1 and 3 run on a CPU. The `HailoEmbedder` in `partguard/backends.py` is a placeholder for running a
learned model on the UGen300; the supported models and toolchain follow the contest platform documentation.

## Quick start

```bash
pip install -r requirements.txt
python tools/make_parts_synthetic.py      # create synthetic washer test images (drawings, not real metal)
python tests/test_smoke.py                # smoke tests
python run.py --mode parts --source samples/parts_batch_mixed.png --lang en
python run.py --mode parts --source 0 --show      # webcam, press q to quit
```

Outputs: a text verdict (`--lang en|zh`), an annotated image (`out/`) and a JSON log with latency (`logs/inspections.jsonl`).
Optional food examples: `python tools/make_synthetic.py`, then `--mode bakery` or `--mode bento`.

## Results (synthetic, reproducible)

```bash
python tools/benchmark_parts.py     # lighting / colour cast / noise sweeps and defect-strength sweep, 95% intervals
python tools/roi_model.py           # break-even model (all inputs are assumptions)
```

100 batches of 8 synthetic washers per condition (seed 2026). Defects at strength 0.75 unless noted.

| Metric | Result |
|---|---|
| Defective parts caught | 100% (95% CI 99-100%, 330 parts) |
| Correct defect named | 100% of those caught |
| Good parts wrongly flagged | 0% (95% CI 0-0.8%, 470 parts) |
| Latency per batch of 8 parts | about 27 ms median, 68 ms p95 (standard CPU, not UGen300) |

**Lighting.** Without a gray reference card, dim light (0.7x or darker) flags every good part. With the card, 0% of good
parts are flagged from 0.6x to 1.4x light and with colour casts of +/-20%.

**Detection limit.** Defect strength is an arbitrary scale (1.0 = obvious). Caught with the right reason:

| Strength | Shape / size | Hole | Surface |
|---|---|---|---|
| 0.1 | 0% | 30%* | 0% |
| 0.25 | 0% | 32%* | 13% |
| 0.5 | 59% | 99% | 100% |
| 0.75 and 1.0 | 100% | 100% | 100% |

\* a missing hole is detected at any strength; the other hole defects need strength 0.5.
Where detection stops depends on the thresholds. Tighter thresholds catch subtler defects but will raise false alarms
on the natural variation of real parts.

**Noise.** Reliable up to about 5 (of 255) sensor noise; false alarms begin near 10 and results break down by 20.
The scratch threshold adapts to the estimated noise instead of producing false scratches.

**Break-even (assumptions only).** With a station cost of 40,000 per year, 20,000 per returned lot and 80% of escaping
lots stopped, about 2.5 defective lots a year must be reaching customers for one station to pay for itself.
Replace every input with sourced numbers before relying on this.

## Gray-card calibration

Fix a neutral gray card in a corner of the camera frame (`calibration.card_roi` in `configs/parts.yaml`). Each frame is
rescaled so the card always reads the same gray. **If you do not use a card, delete the `calibration` block.**
Calibration cannot recover clipped highlights, so keep the camera exposure below saturation.

## Using real parts

1. Fix the camera and fixture. Use diffuse light; glare on metal is the biggest variable (a polarising filter can help).
2. Photograph 50+ good parts, measure `px_per_mm`, and enter the drawing tolerances in `configs/parts.yaml`.
3. Fit the thresholds to the distribution of your good parts (the current ones were fitted to synthetic good parts).
4. Collect real defective parts to verify they are caught.

## Limitations

- All numbers are synthetic; glare, oil and machining texture are not modelled.
- The rule-based method misses subtle defects (see the detection-limit table). A learned classifier on the UGen300 is
  the Stage II plan.
- The UGen300 integration is not implemented yet.
- Many public industrial-defect datasets are licensed for non-commercial use only, which may conflict with the contest
  rules and commercial use. Check each licence. This repository ships no third-party images.
- Latency was measured on a standard CPU and does not represent the UGen300.

A Traditional Chinese version of this document is in `README.zh-TW.md`.
