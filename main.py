"""
=============================================================
  Biometric Face Recognition System
  Dataset  : ORL/AT&T (Olivetti Faces) – 40 subjects, 10 imgs each
  Methods  : PCA (Eigenfaces)  &  LBP (Local Binary Patterns)
=============================================================

DESIGN NOTES
------------
PCA  – holistic / appearance-based method.
       Known weakness  : sensitive to illumination variation.
       Mitigation      : histogram-equalisation pre-processing
                         + PCA fitted ONLY on enroll set (no data leakage).

LBP  – texture / local-feature method.
       Uses relative pixel comparisons → inherently illumination-robust.
       Vectorised NumPy implementation (~100× faster than pixel loops).
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
# LOAD DATASET  (reads .pgm files directly)
# =========================================================
def load_dataset(path, size=(100, 100)):
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
# Brings PCA closer to LBP's illumination robustness.
# =========================================================
def preprocess(img):
    return cv2.equalizeHist(img)


# =========================================================
# LBP FEATURE — Vectorised NumPy (~100x faster than loops)
# =========================================================
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


# =========================================================
# SIMILARITY
# =========================================================
def similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)


# =========================================================
# TRAIN / TEST SPLIT — 80% enroll, 20% probe (per subject)
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
# GEN-IMP SCORES  (uses separate enroll / probe sets)
# =========================================================
def gen_imp_scores(FM, labels, enroll_idx, probe_idx):
    Gen, Imp = [], []
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]
    probe_lbl  = labels[probe_idx]
    unique     = np.unique(labels)

    for s in unique:
        e_idx  = enroll_idx[enroll_lbl == s]
        p_idx  = probe_idx[probe_lbl  == s]
        if len(e_idx) == 0 or len(p_idx) == 0:
            continue
        # Genuine
        for i in e_idx:
            for j in p_idx:
                Gen.append(similarity(FM[i], FM[j]))
        # Impostor
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

    # Interpolated TMR at fixed FMR operating points
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
# RANK-1 — Vectorised (no Python loops, no self-match)
# =========================================================
def rank1_acc(FM, labels, enroll_idx, probe_idx):
    labels = np.array(labels)
    FM_e   = FM[enroll_idx]
    FM_p   = FM[probe_idx]
    norm_e = FM_e / (np.linalg.norm(FM_e, axis=1, keepdims=True) + 1e-10)
    norm_p = FM_p / (np.linalg.norm(FM_p, axis=1, keepdims=True) + 1e-10)
    sim_matrix  = norm_p @ norm_e.T
    pred_labels = labels[enroll_idx[np.argmax(sim_matrix, axis=1)]]
    return float(np.mean(pred_labels == labels[probe_idx]))


# =========================================================
# FPIR / FNIR at EER threshold — Vectorised
# =========================================================
def fpir_fnir(FM, labels, enroll_idx, probe_idx, threshold):
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]
    probe_lbl  = labels[probe_idx]

    FM_e   = FM[enroll_idx]
    FM_p   = FM[probe_idx]
    norm_e = FM_e / (np.linalg.norm(FM_e, axis=1, keepdims=True) + 1e-10)
    norm_p = FM_p / (np.linalg.norm(FM_p, axis=1, keepdims=True) + 1e-10)
    sim    = norm_p @ norm_e.T

    best_idx = np.argmax(sim, axis=1)
    best_scr = sim[np.arange(len(probe_idx)), best_idx]
    pred_lbl = enroll_lbl[best_idx]

    correct  = pred_lbl == probe_lbl
    accepted = best_scr >= threshold

    fpir = float((accepted & ~correct).sum() / len(probe_lbl))
    fnir = float((~accepted).sum()           / len(probe_lbl))
    return fpir, fnir


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(dataset_path="dataset"):
    print("Loading dataset...")
    images_raw, labels = load_dataset(dataset_path)
    print(f"  Loaded {len(images_raw)} images from {len(np.unique(labels))} subjects")

    # Histogram equalisation
    print("Preprocessing (histogram equalisation)...")
    images = np.array([preprocess(img) for img in images_raw])

    # 80 / 20 split
    enroll_idx, probe_idx = split_dataset(images, labels)
    print(f"  Enroll: {len(enroll_idx)} | Probe: {len(probe_idx)}")

    # ----------------------------------------------------------
    # METHOD 1 — PCA (Eigenfaces)
    # PCA fitted ONLY on enroll set → no data leakage into probe.
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
    fpir_pca, fnir_pca = fpir_fnir(FM_pca, labels, enroll_idx, probe_idx, res_pca['eer_thr'])
    res_pca.update(rank1=r1_pca, fpir=fpir_pca, fnir=fnir_pca)

    print(f"  EER={res_pca['eer']:.4f}  d'={res_pca['d_prime']:.4f}"
          f"  Rank-1={r1_pca:.4f}  AUC={res_pca['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_pca['tmr_1']:.4f}  TMR@FMR=0.01%={res_pca['tmr_001']:.4f}")

    # ----------------------------------------------------------
    # METHOD 2 — LBP (Local Binary Patterns)
    # Relative pixel comparisons → illumination-robust by design.
    # ----------------------------------------------------------
    print("\nRunning LBP...")
    FM_lbp = np.array([lbp(img) for img in images])

    Gen_lbp, Imp_lbp   = gen_imp_scores(FM_lbp, labels, enroll_idx, probe_idx)
    res_lbp            = compute_metrics(Gen_lbp, Imp_lbp)
    r1_lbp             = rank1_acc(FM_lbp, labels, enroll_idx, probe_idx)
    fpir_lbp, fnir_lbp = fpir_fnir(FM_lbp, labels, enroll_idx, probe_idx, res_lbp['eer_thr'])
    res_lbp.update(rank1=r1_lbp, fpir=fpir_lbp, fnir=fnir_lbp)

    print(f"  EER={res_lbp['eer']:.4f}  d'={res_lbp['d_prime']:.4f}"
          f"  Rank-1={r1_lbp:.4f}  AUC={res_lbp['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_lbp['tmr_1']:.4f}  TMR@FMR=0.01%={res_lbp['tmr_001']:.4f}")

    # Output folder (dashboard only — no CSV or TXT)
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # DASHBOARD  (2 × 3 grid)
    # ----------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Face Recognition — PCA vs LBP", fontsize=14, fontweight="bold")

    # ── Gen/Imp – PCA ─────────────────────────────────────────
    axs[0, 0].hist(Gen_pca, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 0].hist(Imp_pca, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 0].set_title("PCA — Gen/Imp Distribution")
    axs[0, 0].set_xlabel("Cosine Similarity")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    # ── Gen/Imp – LBP ─────────────────────────────────────────
    axs[0, 1].hist(Gen_lbp, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 1].hist(Imp_lbp, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 1].set_title("LBP — Gen/Imp Distribution")
    axs[0, 1].set_xlabel("Cosine Similarity")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)

    # ── ROC Curve ─────────────────────────────────────────────
    axs[0, 2].plot(res_pca['fpr'], res_pca['tpr'], linewidth=2,
                   label=f"PCA  (EER={res_pca['eer']:.3f})")
    axs[0, 2].plot(res_lbp['fpr'], res_lbp['tpr'], linewidth=2,
                   label=f"LBP  (EER={res_lbp['eer']:.3f})")
    axs[0, 2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axs[0, 2].set_title("ROC Curve")
    axs[0, 2].set_xlabel("FMR (False Match Rate)")
    axs[0, 2].set_ylabel("TMR (True Match Rate)")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # ── EER Bar Chart ─────────────────────────────────────────
    methods = ["PCA", "LBP"]
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

    # ── d-prime Bar Chart ─────────────────────────────────────
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

    # ── Styled Metrics Table (File 2 GUI style) ───────────────
    axs[1, 2].axis("off")

    rows = [
        ["Metric",           "PCA",                             "LBP"],
        ["D-prime",          f"{res_pca['d_prime']:.3f}",       f"{res_lbp['d_prime']:.3f}"],
        ["EER",              f"{res_pca['eer']*100:.2f}%",      f"{res_lbp['eer']*100:.2f}%"],
        ["TMR @ FMR=1%",     f"{res_pca['tmr_1']*100:.2f}%",   f"{res_lbp['tmr_1']*100:.2f}%"],
        ["TMR @ FMR=0.01%",  f"{res_pca['tmr_001']*100:.2f}%", f"{res_lbp['tmr_001']*100:.2f}%"],
        ["AUC",              f"{res_pca['auc']:.4f}",           f"{res_lbp['auc']:.4f}"],
        ["── BONUS ──",      "──────",                          "──────"],
        ["Rank-1 (TPIR)",    f"{res_pca['rank1']*100:.2f}%",   f"{res_lbp['rank1']*100:.2f}%"],
        ["FPIR @ EER-thr",   f"{res_pca['fpir']*100:.2f}%",    f"{res_lbp['fpir']*100:.2f}%"],
        ["FNIR @ EER-thr",   f"{res_pca['fnir']*100:.2f}%",    f"{res_lbp['fnir']*100:.2f}%"],
    ]

    tbl = axs[1, 2].table(cellText=rows[1:], colLabels=rows[0],
                          loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.6)

    # Header row — dark navy
    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Bonus separator row (index 7 in rendered table = rows[7])
    for j in range(3):
        tbl[7, j].set_facecolor("#ecf0f1")
        tbl[7, j].set_text_props(fontweight="bold", color="#7f8c8d")

    axs[1, 2].set_title("All Metrics Summary")

    plt.tight_layout()

    # Save to output/ AND static/images/ (for GUI / Flask route)
    Path("static/images").mkdir(parents=True, exist_ok=True)
    for dest in [out / "final_dashboard.png", Path("static/images/results.png")]:
        plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard saved → {out / 'final_dashboard.png'}")
    print("DONE ✔")

    # ----------------------------------------------------------
    # RETURN structured dict consumed by GUI / Flask route
    # ----------------------------------------------------------
    return {
        "pca": {
            "d_prime": res_pca['d_prime'],
            "eer":     res_pca['eer'],
            "tmr_1":   res_pca['tmr_1'],
            "tmr_001": res_pca['tmr_001'],
            "auc":     res_pca['auc'],
            "rank1":   res_pca['rank1'],
            "fpir":    res_pca['fpir'],
            "fnir":    res_pca['fnir'],
        },
        "lbp": {
            "d_prime": res_lbp['d_prime'],
            "eer":     res_lbp['eer'],
            "tmr_1":   res_lbp['tmr_1'],
            "tmr_001": res_lbp['tmr_001'],
            "auc":     res_lbp['auc'],
            "rank1":   res_lbp['rank1'],
            "fpir":    res_lbp['fpir'],
            "fnir":    res_lbp['fnir'],
        },
        "plot_url": "/static/images/results.png",
    }


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_pipeline(dataset_path="dataset")