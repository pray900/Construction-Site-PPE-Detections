# Construction Site Safety Detection

A Streamlit application that checks PPE compliance on construction sites. Three
object detectors were trained on the same 3-class dataset (`helmet`, `vest`,
`head`), and the app turns their raw detections into per-worker compliance
verdicts and a site-level safety score.

Images, video files and webcam snapshots are all supported.

## Results

All three models were trained and evaluated on the same splits
(17,248 train / 2,438 val / 2,455 test images). Figures below are the held-out
**test** set.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Weights |
|---|---|---|---|---|---|
| YOLOv5n | 0.895 | 0.548 | 0.889 | 0.845 | 5 MB |
| **YOLOv8s** | **0.898** | **0.559** | **0.894** | **0.857** | 22 MB |
| RT-DETR | 0.867 | 0.518 | 0.873 | 0.843 | 63 MB |

YOLOv8s gives the best accuracy overall. YOLOv5n is within half a point of it at
a quarter of the size, which makes it the sensible choice for edge deployment.
RT-DETR trails both here despite being by far the largest model.

Full training curves, per-class AP and confusion matrices are in `notebooks/`.

## Quick start

```bash
git clone <repo-url>
cd construction-site-safety-detection
git lfs pull                      # fetch the .pt weights
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9+ and [Git LFS](https://git-lfs.com) are required. On first run
Ultralytics may download additional support files.

## How it works

Detection alone does not say whether a *worker* is compliant — the model returns
helmets, vests and heads as independent boxes with no notion of which belongs to
whom. The app links them using a head-anchored geometric heuristic:

1. **Helmet zone** — a box 1.5× the head's width and height, sitting directly
   above the head. A helmet detection overlapping this zone counts as worn.
2. **Torso zone** — a box 2.5× wide and 4.5× tall, extending below the head. A
   vest detection overlapping this zone counts as worn.
3. **Second pass** — a helmet matched to no head is treated as its own worker,
   since a helmet viewed from the front often occludes the head entirely. That
   worker is known to have a helmet, so only the vest check runs.

Each worker scores 100 (COMPLIANT), 50 (PARTIAL) or 0 (NON-COMPLIANT). The site
score is the mean, mapped to a risk band:

| Score | Risk |
|---|---|
| ≥ 90 | LOW |
| 70–89 | MODERATE |
| 40–69 | HIGH |
| < 40 | CRITICAL |

## Project layout

```
app.py                    Streamlit application
requirements.txt          Python dependencies
notebooks/
  01_YOLOv5n.ipynb        Training + evaluation
  02_YOLOv8s.ipynb
  03_RTDETR.ipynb
weights/
  yolov5n_best.pt         Tracked with Git LFS
  yolov8s_best.pt
  rtdetr_best.pt
```

The dataset itself is not included in this repository.

## Limitations

- **Shared helmets.** Nothing prevents two nearby heads from matching the same
  helmet detection, so a bare-headed worker standing beside a helmeted one can
  be scored as compliant. Exclusive one-to-one assignment would fix this.
- **Fixed zone ratios.** The 1.5× / 2.5× / 4.5× multipliers assume roughly
  upright, front-facing workers at moderate distance. Crouching workers,
  overhead camera angles and heavy crowding all degrade the matching.
- **Video cost.** One inference runs per analysed frame. The sidebar stride
  control ("Analyse every Nth frame") trades temporal resolution for speed on
  long clips.
- **Codec support.** Annotated video is written as H.264 where the local OpenCV
  build supports it, and falls back to `mp4v` otherwise. The `mp4v` fallback
  downloads correctly but may not play in the browser preview.
- **Not production safety equipment.** Trained on a single public dataset and
  validated only against it. This is coursework, not a certified safety system.

## Author

Prayush Bahadur Shrestha — 25717671
University of Technology Sydney
Deep Learning and Convolutional Neural Networks — Assignment 3
