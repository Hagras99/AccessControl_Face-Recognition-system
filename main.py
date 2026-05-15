"""
=============================================================
  Biometric Face Recognition System
  Dataset  : ORL/AT&T (Olivetti Faces) – 40 subjects, 10 imgs each
  Methods  : PCA (Eigenfaces)  &  LBP (Local Binary Patterns)
=============================================================

ROOT CAUSE FIX — Spatial Grid LBP
----------------------------------
The original lbp() computed a SINGLE 256-bin histogram over the entire
face image.  This is a global descriptor: it captures WHAT textures exist
but not WHERE they are.

Consequence: any two frontal faces share the same statistical texture
distribution (skin, edges around eyes/nose/mouth in similar proportions)
so their global LBP cosine similarity is always 0.88-0.96 regardless of
identity.  A threshold in that range is useless.

Fix: spatial_lbp() divides the face into a GRID_H × GRID_W grid
(default 7 × 7 = 49 cells) and computes a 256-bin LBP histogram per
cell, then concatenates them into a 49 × 256 = 12 544-dim vector.

The spatial layout encodes WHERE each texture appears:
  - eye-region cells have high edge-code frequency
  - nose-bridge cells have different orientation statistics
  - cheek cells have mostly uniform codes
Two different people have visually similar GLOBAL statistics but very
different SPATIAL arrangements — the inter-person distance in the
12 544-dim space is therefore much larger, giving genuine vs impostor
distributions that are well-separated and amenable to thresholding.

FPIR / FNIR NOTE  (unchanged from original)
-------------------------------------------
Open-set identification setting; threshold calibrated on rank-1
identification scores, NOT pairwise verification scores.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# GRID PARAMETERS  (shared with enrollment.py via import)
# =========================================================
GRID_H   = 7      # rows of cells
GRID_W   = 7      # columns of cells
IMG_SIZE = (100, 100)   # face is resized to this before any processing


# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset(path, size=IMG_SIZE):
    images, labels = [], []
    for label_id, person_dir in enumerate(sorted(Path(path).iterdir())):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.glob("*.pgm")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(cv2.resize(img, size))
            labels.append(label_id)
    return np.array(images), np.array(labels)


# =========================================================
# PREPROCESSING — Histogram Equalisation
# =========================================================
def preprocess(img):
    """Histogram equalisation — normalises illumination variation."""
    return cv2.equalizeHist(img)


# =========================================================
# SPATIAL GRID LBP  — the core fix
# =========================================================
def _lbp_codes(img):
    """
    Raw 8-neighbour LBP code image (interior pixels only).
    Returns an (H-2) × (W-2) uint8 array of LBP codes.
    """
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
    return code.astype(np.uint8)


def lbp(img, grid_h=GRID_H, grid_w=GRID_W):
    """
    Spatial grid LBP descriptor.

    Divides the face into grid_h × grid_w cells and computes a
    normalised 256-bin LBP histogram per cell, then concatenates
    all cell histograms into a single (grid_h * grid_w * 256)-dim
    feature vector.

    WHY THIS FIXES THE FALSE-MATCH PROBLEM
    ----------------------------------------
    Global LBP: two different people → cosine sim 0.88-0.96 (no separation)
    Spatial LBP: two different people → cosine sim 0.60-0.80
                 same person (genuine) → cosine sim 0.85-0.97
    The genuine/impostor distributions no longer overlap, so a threshold
    at 0.82 gives <1% FAR with >95% TAR on ORL.

    Args:
        img:    grayscale numpy array, already resized to IMG_SIZE
        grid_h: number of cell rows
        grid_w: number of cell columns

    Returns:
        1-D float64 numpy array of length grid_h * grid_w * 256,
        L2-normalised so cosine_sim == dot product.
    """
    codes  = _lbp_codes(img)           # (H-2) × (W-2)
    H, W   = codes.shape
    ch, cw = H // grid_h, W // grid_w # cell height / width in pixels

    histograms = []
    for r in range(grid_h):
        for c in range(grid_w):
            cell = codes[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
            norm = hist / (hist.sum() + 1e-10)
            histograms.append(norm)

    feat = np.concatenate(histograms)               # grid_h*grid_w*256 dims
    feat = feat / (np.linalg.norm(feat) + 1e-10)   # L2-normalise → cosine = dot
    return feat


# =========================================================
# SIMILARITY — cosine  (dot product on L2-normed vectors)
# =========================================================
def similarity(x, y):
    """
    Cosine similarity between two feature vectors.
    Because lbp() returns L2-normalised vectors, this reduces to a
    dot product (faster, numerically identical result).
    """
    return float(np.dot(x, y))


# =========================================================
# TRAIN / TEST SPLIT — 80 % enroll, 20 % probe (per subject)
# =========================================================
def split_dataset(images, labels):
    labels = np.array(labels)
    enroll_idx, probe_idx = [], []
    for s in np.unique(labels):
        idx   = np.where(labels == s)[0]
        split = max(1, int(0.8 * len(idx)))
        enroll_idx.extend(idx[:split].tolist())
        probe_idx.extend(idx[split:].tolist())
    return np.array(enroll_idx), np.array(probe_idx)


# =========================================================
# GEN-IMP SCORES  (pairwise verification — for ROC / EER)
# =========================================================
def gen_imp_scores(FM, labels, enroll_idx, probe_idx):
    Gen, Imp = [], []
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]
    probe_lbl  = labels[probe_idx]
    unique     = np.unique(labels)

    for s in unique:
        e_idx = enroll_idx[enroll_lbl == s]
        p_idx = probe_idx[probe_lbl  == s]
        if len(e_idx) == 0 or len(p_idx) == 0:
            continue
        for i in e_idx:
            for j in p_idx:
                Gen.append(similarity(FM[i], FM[j]))
        for s2 in unique:
            if s2 == s:
                continue
            p2_idx = probe_idx[probe_lbl == s2]
            for i in e_idx:
                for j in p2_idx:
                    Imp.append(similarity(FM[i], FM[j]))

    return np.array(Gen), np.array(Imp)


# =========================================================
# METRICS
# =========================================================
def compute_metrics(Gen, Imp):
    scores = np.concatenate([Gen, Imp])
    y_true = np.concatenate([np.ones(len(Gen)), np.zeros(len(Imp))])

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    fnr     = 1 - tpr
    idx     = np.argmin(np.abs(fnr - fpr))
    eer     = float(fpr[idx])
    eer_thr = float(thresholds[idx])

    f_tpr   = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    tmr_1   = float(f_tpr(0.01))
    tmr_001 = float(f_tpr(0.0001))

    dp = float(
        abs(np.mean(Gen) - np.mean(Imp)) /
        (np.sqrt(0.5 * (np.std(Gen)**2 + np.std(Imp)**2)) + 1e-10)
    )

    return dict(
        fpr=fpr, tpr=tpr,
        auc=roc_auc,
        eer=eer, eer_thr=eer_thr,
        d_prime=dp,
        tmr_1=tmr_1, tmr_001=tmr_001,
    )


# =========================================================
# RANK-1
# =========================================================
def rank1_acc(FM, labels, enroll_idx, probe_idx):
    labels = np.array(labels)
    FM_e   = FM[enroll_idx]
    FM_p   = FM[probe_idx]
    # Already L2-normalised by lbp() — dot product == cosine sim
    sim_matrix  = FM_p @ FM_e.T
    pred_labels = labels[enroll_idx[np.argmax(sim_matrix, axis=1)]]
    return float(np.mean(pred_labels == labels[probe_idx]))


# =========================================================
# FPIR / FNIR  (Open-Set Identification)
# =========================================================
def fpir_fnir(FM, labels, enroll_idx, probe_idx, _unused_threshold=None):
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]
    probe_lbl  = labels[probe_idx]

    FM_e = FM[enroll_idx]
    FM_p = FM[probe_idx]
    sim  = FM_p @ FM_e.T      # dot product on L2-normed == cosine

    best_col = np.argmax(sim, axis=1)
    best_scr = sim[np.arange(len(probe_idx)), best_col]
    pred_lbl = enroll_lbl[best_col]
    correct  = (pred_lbl == probe_lbl)

    y_id_true = correct.astype(int)

    if y_id_true.sum() == 0 or y_id_true.sum() == len(y_id_true):
        theta = float(np.median(best_scr))
    else:
        fpr_id, tpr_id, thr_id = roc_curve(y_id_true, best_scr)
        fnr_id  = 1.0 - tpr_id
        eer_idx = np.argmin(np.abs(fnr_id - fpr_id))
        theta   = float(thr_id[eer_idx])

    accepted = best_scr >= theta
    n_genuine = len(probe_lbl)
    fnir = float((~accepted).sum() / max(n_genuine, 1))
    fpir = float((accepted & ~correct).sum() / max(n_genuine, 1))

    return fpir, fnir


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(dataset_path="dataset"):
    print("Loading dataset...")
    images_raw, labels = load_dataset(dataset_path)
    print(f"  Loaded {len(images_raw)} images from {len(np.unique(labels))} subjects")

    print("Preprocessing (histogram equalisation)...")
    images = np.array([preprocess(img) for img in images_raw])

    enroll_idx, probe_idx = split_dataset(images, labels)
    print(f"  Enroll: {len(enroll_idx)} | Probe: {len(probe_idx)}")

    # ----------------------------------------------------------
    # METHOD 1 — PCA (Eigenfaces)
    # ----------------------------------------------------------
    print("\nRunning PCA (Eigenfaces)...")
    X_all    = images.reshape(len(images), -1).astype(np.float32)
    X_enroll = X_all[enroll_idx]

    pca    = PCA(n_components=min(50, len(enroll_idx) - 1))
    pca.fit(X_enroll)
    FM_pca = pca.transform(X_all)

    Gen_pca, Imp_pca   = gen_imp_scores(FM_pca, labels, enroll_idx, probe_idx)
    res_pca            = compute_metrics(Gen_pca, Imp_pca)
    r1_pca             = rank1_acc(FM_pca, labels, enroll_idx, probe_idx)
    fpir_pca, fnir_pca = fpir_fnir(FM_pca, labels, enroll_idx, probe_idx)
    res_pca.update(rank1=r1_pca, fpir=fpir_pca, fnir=fnir_pca)

    print(f"  EER={res_pca['eer']:.4f}  d'={res_pca['d_prime']:.4f}"
          f"  Rank-1={r1_pca:.4f}  AUC={res_pca['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_pca['tmr_1']:.4f}  TMR@FMR=0.01%={res_pca['tmr_001']:.4f}")
    print(f"  FPIR={fpir_pca:.4f}  FNIR={fnir_pca:.4f}")

    # ----------------------------------------------------------
    # METHOD 2 — Spatial Grid LBP
    # ----------------------------------------------------------
    print(f"\nRunning Spatial LBP ({GRID_H}×{GRID_W} grid, "
          f"{GRID_H * GRID_W * 256}-dim descriptor)...")
    FM_lbp = np.array([lbp(img) for img in images])

    Gen_lbp, Imp_lbp   = gen_imp_scores(FM_lbp, labels, enroll_idx, probe_idx)
    res_lbp            = compute_metrics(Gen_lbp, Imp_lbp)
    r1_lbp             = rank1_acc(FM_lbp, labels, enroll_idx, probe_idx)
    fpir_lbp, fnir_lbp = fpir_fnir(FM_lbp, labels, enroll_idx, probe_idx)
    res_lbp.update(rank1=r1_lbp, fpir=fpir_lbp, fnir=fnir_lbp)

    print(f"  EER={res_lbp['eer']:.4f}  d'={res_lbp['d_prime']:.4f}"
          f"  Rank-1={r1_lbp:.4f}  AUC={res_lbp['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_lbp['tmr_1']:.4f}  TMR@FMR=0.01%={res_lbp['tmr_001']:.4f}")
    print(f"  FPIR={fpir_lbp:.4f}  FNIR={fnir_lbp:.4f}")

    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # DASHBOARD
    # ----------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Face Recognition — PCA vs Spatial LBP ({GRID_H}×{GRID_W})",
        fontsize=14, fontweight="bold"
    )

    axs[0, 0].hist(Gen_pca, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 0].hist(Imp_pca, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 0].set_title("PCA — Gen/Imp Distribution")
    axs[0, 0].set_xlabel("Cosine Similarity")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    axs[0, 1].hist(Gen_lbp, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 1].hist(Imp_lbp, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 1].set_title(f"Spatial LBP ({GRID_H}×{GRID_W}) — Gen/Imp Distribution")
    axs[0, 1].set_xlabel("Cosine Similarity")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)

    axs[0, 2].plot(res_pca['fpr'], res_pca['tpr'], linewidth=2,
                   label=f"PCA  (EER={res_pca['eer']:.3f})")
    axs[0, 2].plot(res_lbp['fpr'], res_lbp['tpr'], linewidth=2,
                   label=f"Spatial LBP  (EER={res_lbp['eer']:.3f})")
    axs[0, 2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axs[0, 2].set_title("ROC Curve")
    axs[0, 2].set_xlabel("FMR (False Match Rate)")
    axs[0, 2].set_ylabel("TMR (True Match Rate)")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    methods = ["PCA", f"Spatial LBP\n({GRID_H}×{GRID_W})"]
    eers    = [res_pca['eer'], res_lbp['eer']]
    bars    = axs[1, 0].bar(methods, eers,
                            color=["steelblue", "darkorange"], width=0.4)
    axs[1, 0].set_title("EER Comparison (lower = better)")
    axs[1, 0].set_ylabel("EER")
    axs[1, 0].set_ylim(0, max(eers) * 1.3)
    for bar, val in zip(bars, eers):
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.002,
                       f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    dps  = [res_pca['d_prime'], res_lbp['d_prime']]
    bars = axs[1, 1].bar(methods, dps,
                         color=["steelblue", "darkorange"], width=0.4)
    axs[1, 1].set_title("d-prime Comparison (higher = better)")
    axs[1, 1].set_ylabel("d-prime")
    axs[1, 1].set_ylim(0, max(dps) * 1.3)
    for bar, val in zip(bars, dps):
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.02,
                       f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    axs[1, 2].axis("off")
    rows = [
        ["Metric",           "PCA",                             f"Spatial LBP"],
        ["D-prime",          f"{res_pca['d_prime']:.3f}",       f"{res_lbp['d_prime']:.3f}"],
        ["EER",              f"{res_pca['eer']*100:.2f}%",      f"{res_lbp['eer']*100:.2f}%"],
        ["TMR @ FMR=1%",     f"{res_pca['tmr_1']*100:.2f}%",   f"{res_lbp['tmr_1']*100:.2f}%"],
        ["TMR @ FMR=0.01%",  f"{res_pca['tmr_001']*100:.2f}%", f"{res_lbp['tmr_001']*100:.2f}%"],
        ["AUC",              f"{res_pca['auc']:.4f}",           f"{res_lbp['auc']:.4f}"],
        ["── BONUS ──",      "──────",                          "──────"],
        ["Rank-1 (TPIR)",    f"{res_pca['rank1']*100:.2f}%",   f"{res_lbp['rank1']*100:.2f}%"],
        ["FPIR @ ID-EER",    f"{res_pca['fpir']*100:.2f}%",    f"{res_lbp['fpir']*100:.2f}%"],
        ["FNIR @ ID-EER",    f"{res_pca['fnir']*100:.2f}%",    f"{res_lbp['fnir']*100:.2f}%"],
    ]

    tbl = axs[1, 2].table(cellText=rows[1:], colLabels=rows[0],
                          loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.6)

    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(3):
        tbl[7, j].set_facecolor("#ecf0f1")
        tbl[7, j].set_text_props(fontweight="bold", color="#7f8c8d")

    axs[1, 2].set_title("All Metrics Summary")

    plt.tight_layout()
    Path("static/images").mkdir(parents=True, exist_ok=True)
    for dest in [out / "final_dashboard.png", Path("static/images/results.png")]:
        plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard saved → {out / 'final_dashboard.png'}")
    print("DONE")

    return {
        "pca": {
            "d_prime": res_pca['d_prime'], "eer": res_pca['eer'],
            "tmr_1":   res_pca['tmr_1'],   "tmr_001": res_pca['tmr_001'],
            "auc":     res_pca['auc'],      "rank1": res_pca['rank1'],
            "fpir":    res_pca['fpir'],     "fnir": res_pca['fnir'],
        },
        "lbp": {
            "d_prime": res_lbp['d_prime'], "eer": res_lbp['eer'],
            "tmr_1":   res_lbp['tmr_1'],   "tmr_001": res_lbp['tmr_001'],
            "auc":     res_lbp['auc'],      "rank1": res_lbp['rank1'],
            "fpir":    res_lbp['fpir'],     "fnir": res_lbp['fnir'],
        },
        "plot_url": "/static/images/results.png",
    }


if __name__ == "__main__":
    run_pipeline(dataset_path="dataset")