# Biometric Face Recognition — Access Control System
### Technical Project Report

> **Repository:** `Hagras99/AccessControl_Face-Recognition-system`
> **Date:** May 2026
> **Stack:** Python · OpenCV · scikit-learn · Flask · Vanilla JS/CSS

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset](#3-dataset)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Recognition Algorithms](#5-recognition-algorithms)
   - 5.1 [PCA — Eigenfaces (Holistic)](#51-pca--eigenfaces-holistic)
   - 5.2 [LBP — Local Binary Patterns (Texture)](#52-lbp--local-binary-patterns-texture)
6. [Matching & Similarity](#6-matching--similarity)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Experimental Setup](#8-experimental-setup)
9. [Web Interface (Flask Dashboard)](#9-web-interface-flask-dashboard)
10. [Results & Analysis](#10-results--analysis)
11. [Visualisation Dashboard](#11-visualisation-dashboard)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)
13. [References](#13-references)

---

## 1. Project Overview

This project is a **biometric face recognition system** designed for access-control evaluation. It implements and benchmarks two classical face recognition methods—**PCA (Eigenfaces)** and **LBP (Local Binary Patterns)**—against the well-known **ORL / AT&T Olivetti Faces** dataset.

The system is composed of:

| Component | Role |
|---|---|
| `main.py` | Core ML pipeline: loading, preprocessing, feature extraction, scoring, metrics, and plot generation |
| `app.py` | Lightweight Flask REST API serving the pipeline and results to the browser |
| `templates/index.html` | Single-page front-end for triggering evaluation and displaying results |
| `static/css/style.css` | Glassmorphism dark-themed responsive stylesheet |
| `static/js/script.js` | Async fetch logic + dynamic DOM rendering of metric cards |
| `dataset/` | ORL dataset — 40 subjects × 10 images each (PGM format) |
| `output/` | Generated dashboard PNG (`final_dashboard.png`) |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Browser (Client)                      │
│   ┌─────────────────────────────────────────────────┐    │
│   │  index.html  ←  style.css  ←  script.js         │    │
│   │  Click "Start Evaluation" → POST /api/run       │    │
│   └────────────────────────┬────────────────────────┘    │
└────────────────────────────┼─────────────────────────────┘
                             │ HTTP/JSON
┌────────────────────────────▼─────────────────────────────┐
│                    Flask Server (app.py)                   │
│  Route /          → render index.html                    │
│  Route /api/run   → calls run_pipeline() in main.py      │
│  Route /static/…  → serves CSS, JS, result PNG          │
└────────────────────────────┬─────────────────────────────┘
                             │ Python call
┌────────────────────────────▼─────────────────────────────┐
│               Recognition Pipeline (main.py)              │
│                                                           │
│  load_dataset()                                           │
│       │                                                   │
│  preprocess() ── Histogram Equalisation                   │
│       │                                                   │
│  split_dataset() ── 80% Enroll / 20% Probe               │
│       │                                                   │
│  ┌────┴────┐                                              │
│  │   PCA   │                  │   LBP   │                 │
│  │Eigenfaces│                 │Local BP  │                │
│  └────┬────┘                  └────┬────┘                 │
│       │                            │                      │
│  Feature Matrix              Feature Matrix               │
│       │                            │                      │
│  gen_imp_scores()            gen_imp_scores()             │
│  compute_metrics()           compute_metrics()            │
│  rank1_acc()                 rank1_acc()                  │
│  fpir_fnir()                 fpir_fnir()                  │
│       │                            │                      │
│  ─────────────── Dashboard ─────────────────             │
│        matplotlib 2×3 grid → output/final_dashboard.png  │
│        copy → static/images/results.png                  │
│                                                           │
│  Returns JSON dict  →  Flask  →  Browser                 │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Dataset

**ORL / AT&T Olivetti Faces Dataset**

| Property | Value |
|---|---|
| Subjects | 40 individuals |
| Images per subject | 10 |
| Total images | 400 |
| Format | PGM (Portable Graymap) |
| Image size (resized to) | 100 × 100 px |
| Variations captured | Lighting, facial expression (open/closed eyes, smiling/not), facial details (glasses/no-glasses), slight head pose variation |

The dataset is stored in `dataset/s1/` through `dataset/s40/`, with each sub-directory representing one subject and containing 10 `.pgm` files.

> **Why ORL?** It is a standard academic benchmark for face recognition, providing a clean, controlled environment suitable for classical (non-deep-learning) algorithms. Its 40-subject structure allows meaningful genuine/impostor score separation testing.

---

## 4. Preprocessing Pipeline

```
Raw PGM Image (112×92 or variable)
        │
  cv2.resize(img, (100, 100))      ← Standardise spatial resolution
        │
  cv2.equalizeHist(img)            ← Histogram Equalisation
        │
  Processed Image (100×100, uint8)
```

### Histogram Equalisation

Histogram Equalisation redistributes pixel intensity values across the full dynamic range [0, 255], dramatically reducing the effect of varying lighting conditions. This is crucial for PCA (which is sensitive to illumination), bringing it closer to the inherent illumination-robustness of LBP.

---

## 5. Recognition Algorithms

### 5.1 PCA — Eigenfaces (Holistic)

**Category:** Appearance-based / Holistic  
**Reference:** Turk & Pentland (1991)

#### How It Works

1. **Flatten** each 100×100 image into a 10,000-dimensional vector.
2. **Fit PCA** on the *enroll set only* (prevents data leakage into the probe set).
3. **Retain** the top-50 principal components (Eigenfaces) — directions of maximum variance.
4. **Project** all images (enroll + probe) into this 50-D subspace.
5. **Compare** projected vectors using cosine similarity.

```python
pca = PCA(n_components=min(50, len(enroll_idx) - 1))
pca.fit(X_enroll)          # fit ONLY on enroll set
FM_pca = pca.transform(X_all)  # project all
```

#### Key Properties
- **Strengths:** Compact representation; computationally efficient; well-understood mathematically.
- **Weaknesses:** Holistic — sensitive to illumination and pose variations. Mitigated here by histogram equalisation.
- **Data Leakage Guard:** PCA is fitted exclusively on the enroll partition. The probe set is never seen during training.

---

### 5.2 LBP — Local Binary Patterns (Texture)

**Category:** Local feature / Texture-based  
**Reference:** Ojala, Pietikäinen & Mäenpää (2002)

#### How It Works

For each pixel `c` in the image (excluding 1-pixel border), a binary code is computed by comparing the 8 surrounding neighbours clockwise from the top-left:

```
Neighbour > centre ? 1 : 0
```

The 8 bits form a byte (0–255). A global **histogram** of all these byte values (256 bins) is computed and L1-normalised to produce the feature vector.

```python
def lbp(img):
    c = img[1:-1, 1:-1]
    code = (
        ((img[0:-2, 0:-2] > c) << 7) |
        ((img[0:-2, 1:-1] > c) << 6) |
        ((img[0:-2, 2:]   > c) << 5) |
        ((img[1:-1, 2:]   > c) << 4) |
        ((img[2:,   2:]   > c) << 3) |
        ((img[2:,   1:-1] > c) << 2) |
        ((img[2:,   0:-2] > c) << 1) |
        ((img[1:-1, 0:-2] > c))
    )
    hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
    return hist / (np.sum(hist) + 1e-10)
```

> **Implementation note:** The entire operation uses NumPy array slicing — approximately **100× faster** than equivalent pixel-level Python loops.

#### Key Properties
- **Strengths:** Inherently illumination-robust (uses relative comparisons, not absolute intensities); fast; no training phase required.
- **Weaknesses:** Ignores spatial relationships unless spatial histograms (LBPH) are used; sensitive to noise at sharp edges.

---

## 6. Matching & Similarity

Both methods use **Cosine Similarity** for comparison:

```
sim(x, y) = (x · y) / (‖x‖ · ‖y‖)
```

- Range: [−1, 1], typically [0, 1] for non-negative feature vectors.
- Score → 1: highly similar (likely the same person).
- Score → 0: very dissimilar (likely different people).
- Numerically stabilised with `+ 1e-10` to prevent division by zero.

```python
def similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)
```

---

## 7. Evaluation Metrics

The system computes a full biometric evaluation suite:

| Metric | Description |
|---|---|
| **EER** (Equal Error Rate) | Threshold where FAR = FRR. Lower = better. Primary system-level metric. |
| **AUC** (Area Under ROC Curve) | Overall discriminative power. Closer to 1.0 = better. |
| **d' (d-prime)** | Statistical separability between genuine/impostor score distributions. Higher = better. |
| **TMR @ FMR=1%** | True Match Rate at a 1% False Match Rate operating point. |
| **TMR @ FMR=0.01%** | TMR at very strict 0.01% FMR — relevant for high-security access control. |
| **Rank-1 Accuracy** | Closed-set identification: does top-1 match belong to the correct subject? |
| **FPIR @ EER threshold** | False Positive Identification Rate at the EER operating threshold. |
| **FNIR @ EER threshold** | False Negative Identification Rate at the EER operating threshold. |

### Score Generation

- **Genuine scores:** Similarity between enroll and probe images of the *same* subject.
- **Impostor scores:** Similarity between enroll images of subject A and probe images of subject B (for all A ≠ B).

The ROC curve is built from these scores; EER is found at the crossing of the FAR and FRR curves.

---

## 8. Experimental Setup

| Parameter | Value |
|---|---|
| Dataset split | 80% enroll / 20% probe (per-subject, stratified) |
| PCA components | min(50, n_enroll − 1) |
| LBP bins | 256 (full byte) |
| Similarity metric | Cosine similarity |
| EER threshold | Auto-detected from ROC |
| Output resolution | 150 DPI PNG |

The 80/20 split is applied **per subject**, ensuring every class is represented in both enroll and probe partitions, even with only 10 images per subject (→ 8 enroll, 2 probe per person).

---

## 9. Web Interface (Flask Dashboard)

### Backend — `app.py`

```
GET  /           → Renders index.html (Jinja2 template)
POST /api/run    → Triggers run_pipeline(), returns JSON
                   {status, data: {pca: {...}, lbp: {...}, plot_url}}
GET  /static/…   → Serves CSS, JS, and the result PNG
```

### Frontend — Single Page Application

The UI is built with **Vanilla HTML/CSS/JavaScript** using the following design patterns:

#### Design System
- **Theme:** Dark glassmorphism (`#0f172a` background, `rgba(30,41,59,0.7)` card)
- **Typography:** Inter (Google Fonts) — weights 300/400/600/700
- **Animated background:** 3 blurred radial-gradient blobs with a continuous `float` keyframe animation
- **Cards:** `backdrop-filter: blur(16px)` glassmorphism panels with subtle border

#### Interaction Flow
1. User clicks **"Start Evaluation"**
2. Button disables, spinner appears, loading overlay shown
3. `fetch('/api/run', {method: 'POST'})` is sent
4. On success: metric cards for PCA and LBP are dynamically injected into the DOM
5. Dashboard PNG is loaded with a cache-busting `?t=<timestamp>` query parameter
6. Results container fades in with staggered CSS animations (`delay-1`, `delay-2`, `delay-3`)

---

## 10. Results & Analysis

> Results below are characteristic of the ORL dataset with the described setup. Actual values depend on the runtime run.

| Metric | PCA (Eigenfaces) | LBP |
|---|---|---|
| EER | ~5–12% | ~3–8% |
| AUC | ~0.95–0.99 | ~0.97–0.99 |
| d-prime | ~2.5–4.0 | ~3.0–5.0 |
| Rank-1 Accuracy | ~85–95% | ~90–97% |
| TMR @ 1% FMR | ~85–98% | ~90–99% |

### Interpretation

- **LBP** generally outperforms PCA on the ORL dataset because its local, relative comparisons are naturally illumination-tolerant—even after histogram equalisation, PCA still sees holistic pixel patterns that vary with lighting.
- **PCA** benefits significantly from histogram equalisation pre-processing, closing the gap vs. LBP.
- At strict operating points (TMR @ 0.01% FMR), both methods show significant drop, highlighting the challenge of ultra-low FAR biometric systems.
- The **d-prime** values above 2.5 indicate good distributional separation between genuine and impostor score populations.

---

## 11. Visualisation Dashboard

The system generates a 2×3 matplotlib grid saved to both `output/final_dashboard.png` and `static/images/results.png`:

| Position | Plot |
|---|---|
| [0,0] | PCA — Genuine vs. Impostor Score Distribution (histogram, density) |
| [0,1] | LBP — Genuine vs. Impostor Score Distribution |
| [0,2] | ROC Curve — PCA & LBP overlaid with EER annotations |
| [1,0] | EER Bar Chart — PCA vs. LBP (lower = better) |
| [1,1] | d-prime Bar Chart — PCA vs. LBP (higher = better) |
| [1,2] | Full Metrics Summary Table (styled with dark header, bold values) |

---

## 12. Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| PCA fitted only on enroll set | Strict no-leakage policy — probe data is never seen during feature space construction |
| Cosine similarity over Euclidean distance | Scale-invariant; better suited for high-dimensional feature vectors |
| NumPy-vectorised LBP | ~100× faster than Python loops; no external scikit-image dependency |
| Histogram equalisation for both methods | Reduces illumination sensitivity in PCA; negligible impact on LBP (still beneficial) |
| 50 PCA components | Empirically balances information retention vs. overfitting for 40-class, 8-enroll-per-class data |
| Single-page Flask app | Minimal operational overhead; no heavy framework needed for a research evaluation tool |
| Cache-busting on plot URL | Ensures the browser always fetches the freshly-generated PNG on re-run |
| Glassmorphism UI | Modern, accessible dark-mode design that clearly separates metric cards |

---

## 13. References

1. **Turk, M. & Pentland, A. (1991).** Eigenfaces for Recognition. *Journal of Cognitive Neuroscience*, 3(1), 71–86.
   - Foundation of the PCA/Eigenfaces method used in this project.

2. **Ojala, T., Pietikäinen, M. & Mäenpää, T. (2002).** Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI*, 24(7), 971–987.
   - Core paper behind the LBP feature descriptor.

3. **Samaria, F. & Harter, A. (1994).** Parameterisation of a Stochastic Model for Human Face Identification. *Proceedings of the 2nd IEEE Workshop on Applications of Computer Vision*, 138–142.
   - Original publication of the ORL / AT&T Olivetti Faces dataset.

4. **Pedregosa, F. et al. (2011).** Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
   - `sklearn.decomposition.PCA` and `sklearn.metrics.roc_curve`, `auc` used in this project.

5. **Bradski, G. (2000).** The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
   - `cv2.imread`, `cv2.resize`, `cv2.equalizeHist` used throughout the pipeline.

6. **ISO/IEC 19795-1:2021** — Biometric Performance Testing and Reporting — Part 1: Principles and Framework.
   - Standard definitions for EER, TMR, FMR, FNIR, FPIR metrics used in the evaluation.

7. **Phillips, P.J., et al. (2000).** The FERET Evaluation Methodology for Face-Recognition Algorithms. *IEEE TPAMI*, 22(10), 1090–1104.
   - Methodology reference for genuine/impostor score generation and biometric evaluation protocols.

---

*Report generated on 15 May 2026. All code analyses are based on the current repository state at `d:\AccessControl_Face-Recognition-system`.*
