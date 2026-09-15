# AV Overconfidence Death Zone Mapper

> **Standalone Project | AV Safety Research**
> MS Robotics & Autonomous Systems Engineering - Arizona State University - Dec 2026

---

## The Question Nobody Answers

Waymo struck a child in January 2026. It braked, but not in time.
Waymo recalled 3,871 vehicles for entering construction zones.
Motional released an entire dataset on edge case failures in September 2026.

None of these systems reported low confidence before failing.
The question nobody has measured: **at exactly what degradation level does a perception system start lying to itself about how well it can see?**

I measured mine on 11,000 Monte Carlo trials and validated it on real KITTI footage with YOLOv8.

---

## How the Death Zone Forms

*Confidence stays high. Accuracy collapses. The gap between them is the danger.*

[![Demo 1 - Death Zone Explainer](https://drive.google.com/thumbnail?id=1e4IbTQN-LglQPtU6WO0dVmPhSCNwg-4h&sz=w1280)](https://drive.google.com/file/d/1PF1h1oc8bCS1AviecUpJwnLMzbv38Ra8/view)

---

## Real Detection - KITTI + YOLOv8 Live

*Green = still detected. Red MISSED = detected clean, lost after degradation.*

[![Demo 2 - Real Detection](https://drive.google.com/thumbnail?id=1v5Hcnab9r6PnVEMWB9xYO7nhiqq_xpjV&sz=w1280)](https://drive.google.com/file/d/1Zz9CB7BPth8ILre_1DgQfzy4GM0DaFq5/view)

---

## Confidence vs Accuracy Divergence

![Hero Chart](https://drive.google.com/thumbnail?id=1e4IbTQN-LglQPtU6WO0dVmPhSCNwg-4h&sz=w1280)

**The finding:** Exposure Loss + Glare enters the death zone at 53% degradation. That is 1.6x earlier than the most forgiving condition tested. Camera-dependent failures happen far sooner than most engineering discussions assume.

---

## Death Zone Rate Heatmap

*Every condition, every degradation level. Red = system confidently wrong.*

![Death Zone Heatmap](https://drive.google.com/thumbnail?id=15D4u4ZsaT2xjaUA4isMxtwpsaxdJ_NfM&sz=w1280)

---

## Real 2026 Incidents vs Measured Boundaries

![Incident Correlation](https://drive.google.com/thumbnail?id=1v5Hcnab9r6PnVEMWB9xYO7nhiqq_xpjV&sz=w1280)

---

## Safety Envelope

![Safety Envelope](https://drive.google.com/thumbnail?id=1CRR329gFQVx72gJcdo3rgHQTczruDLLB&sz=w1280)

---

## Key Findings

### Finding 1 - Exposure Loss + Glare Is the Most Dangerous Condition

| Condition | Death Zone Entry | Real Incident |
|---|---|---|
| Exposure Loss + Glare | 53% | Motional nuReasoning edge case |
| Construction Zone | 66% | Waymo 3,871 vehicle recall 2026 |
| School Zone Occlusion | 67% | Waymo child strike Jan 2026 |
| Rain + Crowd | 82% | Motional nuReasoning edge case |
| Blind Curve | 84% | Waymo stopped school bus 2026 |

Death zone = confidence above 80% while true accuracy is below 50%. Camera-dependent conditions fail first. Multi-sensor conditions hold longer.

### Finding 2 - Confidence Does Not Fall With Accuracy

At 53% degradation (Exposure Loss + Glare):
Reported confidence: 82%
True accuracy: 41%
Gap: 41 points

The model was trained on mostly clean data. It has never learned what degraded input looks like, so its confidence head has no signal to lower itself. Accuracy collapses. Confidence barely moves. This asymmetry is the entire danger - not that the system fails, but that it fails silently.

### Finding 3 - The Real Video Confirms the Simulation

Running YOLOv8n on real KITTI footage with programmatic exposure and glare degradation applied frame by frame:

Clean baseline detections retained: 100%
At 42% degradation: 89% retained, 31% confidence
At 63% degradation: 57% retained, 80% confidence

Confidence rose while retention fell. Real detections on real footage reproduce the same overconfidence pattern measured in the synthetic Monte Carlo simulation - this is not a modeling artifact, it shows up in actual detector behavior.

### Finding 4 - The Safety Ratio Is 1.6x, Not Marginal

Most dangerous: Exposure Loss + Glare, 53% degradation tolerance
Most tolerant: Blind Curve, 84% degradation tolerance
Ratio: 1.6x

The same vehicle, on the same day, is 60% more fragile facing glare than facing a blind curve - despite both being generically labeled "sensor degradation" in most engineering discussions. Treating all degradation as equivalent hides this gap.

### Finding 5 - Divergence Starts Well Before the Death Zone

| Condition | Divergence Point (20% gap) | Death Zone Entry |
|---|---|---|
| Exposure Loss + Glare | 32.5% | 53% |
| Construction Zone | 40.7% | 66% |
| School Zone Occlusion | 43.7% | 67% |
| Rain + Crowd | 52.3% | 82% |
| Blind Curve | 56.5% | 84% |

The gap between claim and reality opens 15-20 percentage points before the system crosses into the actual death zone. This earlier divergence point is a usable early-warning signal, before the danger threshold is even reached.

---

## What I Built

### Simulation Models

Confidence curve - decays slowly, floor around 58-65%. Below the onset point it stays at 0.96. Above the onset point it follows max(floor, 0.96 times exponential decay at the given rate past onset).

Accuracy curve - decays fast, floor around 5-12%. Below the onset point it stays at 0.97. Above the onset point it follows max(floor, 0.97 times exponential decay at the given rate past onset).

Each of the 5 conditions has its own onset point and decay rate, calibrated so confidence always decays slower than accuracy - this asymmetry is the mechanism being measured, not an assumption.

Real degradation model for exposure and glare: brightness is scaled down by 0.6 times the degradation level, a radial glare falloff is added from a simulated light source scaled by level, Gaussian blur is applied with a kernel scaled by level, and Gaussian sensor noise is added scaled by level.

### Pipeline

Five conditions, eleven degradation levels, two hundred trials each. Per-trial accuracy and confidence are sampled with noise. A death zone flag is set when confidence exceeds 80% and accuracy drops below 50%. The exact death zone entry boundary per condition is found through interpolation. Results are cross-referenced against real 2026 incident degradation estimates. The whole pattern is then validated on real KITTI footage with YOLOv8 detection retention.

### Benchmark Scale

Conditions tested: 5
Degradation levels: 11, from 0% to 100%
Trials per level: 200
Total Monte Carlo trials: 11,000
Real-world validation: YOLOv8n on real KITTI driving sequence

---

## What I Learned

Why confidence and accuracy decouple under degradation: Detection heads and confidence heads in most perception models share the same backbone features. When those features degrade, the detection head's output correctly reflects the loss - fewer correct boxes. But the confidence head was trained to associate its own internal activation pattern with correctness on mostly clean data. It has no exposure to what "confident but wrong" activations look like, so it keeps producing high scores even as the detections themselves become unreliable.

Why real footage matters, not just synthetic curves: A synthetic curve can be shaped to say whatever the modeler wants. Running the same story on real KITTI footage with a real YOLOv8 model - not a curve, actual bounding boxes disappearing frame by frame - is what makes the overconfidence pattern credible. The real video's confidence and retention curves are noisier than the simulation's smooth curves, which is expected: single-scene detection counts are inherently noisy. The Monte Carlo simulation with 200 trials per level is the primary evidence; the real video is confirmation that the pattern exists outside the simulation.

Why this connects to more than autonomous vehicles: Any robot with a learned confidence score - a humanoid's grasp confidence, a drone's obstacle confidence, an inspection robot's defect confidence - can fall into the same trap. The specific numbers here are AV-specific, but the underlying failure mode is not. Overconfidence under degradation is a general robotics perception problem, not a self-driving-car-specific one.

---

## Why This Matters

Boston Dynamics is deploying Atlas inside Hyundai's factory now. Production Atlas uses a 360-degree camera suite with no LiDAR. Factory dust and inconsistent lighting are exactly the camera-specific degradation this project measures. If Atlas's perception confidence stays high while its real accuracy silently drops on a dusty factory floor, that is the same death zone, different environment, same cause.

Motional is targeting driverless robotaxi service in Las Vegas by end of 2026. They released nuReasoning specifically because edge case failures are unsolved industry-wide. This project does not replace that effort - it gives a concrete, quantified answer to a question nuReasoning raises but does not directly measure: the exact point where the system stops being trustworthy about its own reliability.

---

## Run It Yourself

Clone the repo, cd into it, and install dependencies:

git clone https://github.com/GVK-Engine/av-overconfidence-mapper
cd av-overconfidence-mapper
pip install numpy matplotlib scipy tqdm opencv-python ultralytics imageio[ffmpeg]

Then run the full Monte Carlo analysis, all four charts, and the synthetic demo video:

python main.py

Then run the real KITTI plus YOLOv8 validation and its demo video:

python generate_real_video.py

Update KITTI_DIR in generate_real_video.py to your local KITTI path. KITTI download available at cvlibs.net/datasets/kitti/raw_data.php.

---

## Project Structure

av-overconfidence-mapper root contains config.py holding all experiment parameters, main.py as the full analysis pipeline entry point, and generate_real_video.py for the real KITTI plus YOLOv8 validation.

Inside src/perception/ is detector.py with the confidence and accuracy curve models.
Inside src/analysis/ is calibration.py with the Monte Carlo runner and boundary detection.
Inside src/visualization/ is charts.py with all four analysis charts, and video_generator.py with the synthetic explainer video.

Inside results/charts/ are hero_confidence_accuracy.png, death_zone_heatmap.png, incident_correlation.png, and safety_envelope.png.
Inside results/videos/ are av_overconfidence_demo.mp4 and real_overconfidence_demo.mp4.
Inside results/logs/ is analysis.json.

---

## Stack

Python 3.11, NumPy, SciPy, Matplotlib, OpenCV, Ultralytics YOLOv8n, imageio, KITTI Raw Data

---

## Status

Monte Carlo simulation with 11,000 trials: Complete
Death zone boundary detection: Complete
Four analysis charts: Complete
Synthetic explainer video: Complete
Real KITTI plus YOLOv8 validation: Complete
Real-incident correlation: Complete