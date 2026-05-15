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
       5x5 spatial grid (1475-dim uniform LBP) — identical to enrollment.py.

FPIR / FNIR NOTE  (Open-Set Identification)
--------------------------------------------
Previous version used a closed-set split (all 40 subjects enrolled).
This meant there were no true impostors, making FPIR trivially 0.

Fixed version uses an open-set split:
  - 30 subjects (75%) → enrolled in gallery
  - 10 subjects (25%) → held out as unknown impostors (never enrolled)
  - Of each enrolled subject's 10 images:
      60% → gallery  (enroll)
      40% → genuine probes

FNIR = P(genuine probe rejected)   = rejected genuines  / total genuines
FPIR = P(impostor probe accepted)  = accepted impostors  / total impostors

Threshold theta is calibrated at EER on the combined genuine+impostor
identification score distribution.
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

from enrollment import preprocess, lbp, IMG_SIZE


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
# OPEN-SET SPLIT
# 75% of subjects enrolled, 25% held out as unknown impostors
# =========================================================
def split_dataset_openset(images, labels, enroll_ratio=0.6, known_ratio=0.75):
    """
    Open-set split:
      - known_ratio  % of subjects  → enrolled (gallery)
      - remaining    % of subjects  → unknown impostors (never enrolled)
      - enroll_ratio % of known subject images → gallery
      - rest of known images        → genuine probes

    With ORL (40 subjects, 10 images each):
      known_ratio=0.75  → 30 enrolled, 10 impostors
      enroll_ratio=0.60 → 6 gallery images, 4 genuine probe images per subject

    Returns
    -------
    enroll_idx   : indices into images/labels array for gallery
    probe_idx    : indices for genuine probes  (known subjects, not in gallery)
    impostor_idx : indices for impostor probes (unknown subjects, all images)
    """
    labels    = np.array(labels)
    subjects  = np.unique(labels)
    n_known   = max(1, int(len(subjects) * known_ratio))

    known_subjects   = subjects[:n_known]
    unknown_subjects = subjects[n_known:]

    enroll_idx, probe_idx, impostor_idx = [], [], []

    for s in known_subjects:
        idx   = np.where(labels == s)[0]
        split = max(1, int(enroll_ratio * len(idx)))
        enroll_idx.extend(idx[:split].tolist())
        probe_idx.extend(idx[split:].tolist())

    for s in unknown_subjects:
        idx = np.where(labels == s)[0]
        impostor_idx.extend(idx.tolist())

    return (
        np.array(enroll_idx),
        np.array(probe_idx),
        np.array(impostor_idx),
    )


# =========================================================
# SIMILARITY — cosine
# =========================================================
def similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)


# =========================================================
# GEN-IMP SCORES  (pairwise verification scores)
# Used for ROC / AUC / EER / d' curves only.
# =========================================================
def gen_imp_scores(FM, labels, enroll_idx, probe_idx):
    Gen, Imp   = [], []
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]
    probe_lbl  = labels[probe_idx]
    unique     = np.unique(labels[np.concatenate([enroll_idx, probe_idx])])

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
# METRICS  (EER, AUC, d', TMR@FMR)
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
# RANK-1 — Vectorised
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
# OPEN-SET FPIR / FNIR
# =========================================================
def fpir_fnir_openset(FM, labels, enroll_idx, probe_idx, impostor_idx):
    """
    True open-set FPIR / FNIR.

    Genuine probes  : known subjects with images NOT in gallery
                      → system should accept them (score >= theta)
    Impostor probes : unknown subjects never enrolled
                      → system should reject ALL of them (score < theta)

    FNIR = genuine probes rejected  / total genuine probes
    FPIR = impostor probes accepted / total impostor probes

    Threshold theta calibrated at EER on the combined identification
    score distribution (genuine best-match scores vs impostor best-match
    scores), NOT on pairwise verification scores.
    """
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]

    FM_e   = FM[enroll_idx]
    norm_e = FM_e / (np.linalg.norm(FM_e, axis=1, keepdims=True) + 1e-10)

    def _best_match_scores(idx_set):
        """Return (best_score, predicted_label) for each probe in idx_set."""
        FM_p     = FM[idx_set]
        norm_p   = FM_p / (np.linalg.norm(FM_p, axis=1, keepdims=True) + 1e-10)
        sim      = norm_p @ norm_e.T
        best_col = np.argmax(sim, axis=1)
        best_scr = sim[np.arange(len(idx_set)), best_col]
        pred_lbl = enroll_lbl[best_col]
        return best_scr, pred_lbl

    # Genuine probes — known subjects, images not in gallery
    gen_scores, gen_pred = _best_match_scores(probe_idx)
    gen_correct          = (gen_pred == labels[probe_idx])

    # Impostor probes — unknown subjects, never enrolled
    imp_scores, _        = _best_match_scores(impostor_idx)
    # Impostors can never be "correct" — they have no gallery entry
    imp_correct          = np.zeros(len(impostor_idx), dtype=bool)

    # ── Calibrate threshold on identification scores ──────────────────
    # y=1 → genuine correct match, y=0 → impostor (or misclassified genuine)
    all_scores  = np.concatenate([gen_scores,              imp_scores])
    all_correct = np.concatenate([gen_correct.astype(int), imp_correct.astype(int)])

    if all_correct.sum() == 0 or all_correct.sum() == len(all_correct):
        theta = float(np.median(all_scores))
    else:
        fpr_c, tpr_c, thr_c = roc_curve(all_correct, all_scores)
        fnr_c   = 1.0 - tpr_c
        eer_idx = np.argmin(np.abs(fnr_c - fpr_c))
        theta   = float(thr_c[eer_idx])

    # ── FNIR ─────────────────────────────────────────────────────────
    gen_accepted = gen_scores >= theta
    fnir = float((~gen_accepted).sum() / max(len(probe_idx), 1))

    # ── FPIR ─────────────────────────────────────────────────────────
    imp_accepted = imp_scores >= theta
    fpir = float(imp_accepted.sum() / max(len(impostor_idx), 1))

    return fpir, fnir, theta


# =========================================================
# DET CURVE  (FPIR vs FNIR across all thresholds)
# =========================================================
def det_curve(FM, labels, enroll_idx, probe_idx, impostor_idx, n_points=300):
    """
    Sweep threshold 0→1 and record FPIR / FNIR at each point.
    Returns arrays suitable for plotting a DET curve.
    """
    labels     = np.array(labels)
    enroll_lbl = labels[enroll_idx]

    FM_e   = FM[enroll_idx]
    norm_e = FM_e / (np.linalg.norm(FM_e, axis=1, keepdims=True) + 1e-10)

    def _scores(idx_set):
        FM_p   = FM[idx_set]
        norm_p = FM_p / (np.linalg.norm(FM_p, axis=1, keepdims=True) + 1e-10)
        sim    = norm_p @ norm_e.T
        best   = np.argmax(sim, axis=1)
        return sim[np.arange(len(idx_set)), best]

    gen_scores = _scores(probe_idx)
    imp_scores = _scores(impostor_idx)

    thresholds = np.linspace(0.0, 1.0, n_points)
    fpirs = np.array([(imp_scores >= t).mean() for t in thresholds])
    fnirs = np.array([(gen_scores <  t).mean() for t in thresholds])

    return fpirs, fnirs, thresholds


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(dataset_path="dataset"):
    print("Loading dataset...")
    images_raw, labels = load_dataset(dataset_path)
    print(f"  Loaded {len(images_raw)} images from {len(np.unique(labels))} subjects")

    print("Preprocessing (gamma + CLAHE + blur + oval mask)...")
    images = np.array([preprocess(img) for img in images_raw])

    # ── Open-set split ────────────────────────────────────────────────
    enroll_idx, probe_idx, impostor_idx = split_dataset_openset(
        images, labels,
        enroll_ratio=0.60,   # 6 of 10 images per subject → gallery
        known_ratio=0.75,    # 30 subjects enrolled, 10 unknowns → impostors
    )
    labels = np.array(labels)
    print(f"  Enrolled subjects : {len(np.unique(labels[enroll_idx]))}")
    print(f"  Genuine probes    : {len(probe_idx)}")
    print(f"  Impostor probes   : {len(impostor_idx)}")

    # ------------------------------------------------------------------
    # METHOD 1 — PCA (Eigenfaces)
    # ------------------------------------------------------------------
    print("\nRunning PCA (Eigenfaces)...")
    X_all    = images.reshape(len(images), -1).astype(np.float32)
    X_enroll = X_all[enroll_idx]

    pca    = PCA(n_components=min(50, len(enroll_idx) - 1))
    pca.fit(X_enroll)
    FM_pca = pca.transform(X_all)

    Gen_pca, Imp_pca              = gen_imp_scores(FM_pca, labels, enroll_idx, probe_idx)
    res_pca                       = compute_metrics(Gen_pca, Imp_pca)
    r1_pca                        = rank1_acc(FM_pca, labels, enroll_idx, probe_idx)
    fpir_pca, fnir_pca, thr_pca  = fpir_fnir_openset(FM_pca, labels, enroll_idx, probe_idx, impostor_idx)
    res_pca.update(rank1=r1_pca, fpir=fpir_pca, fnir=fnir_pca)

    print(f"  EER={res_pca['eer']:.4f}  d'={res_pca['d_prime']:.4f}"
          f"  Rank-1={r1_pca:.4f}  AUC={res_pca['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_pca['tmr_1']:.4f}  TMR@FMR=0.01%={res_pca['tmr_001']:.4f}")
    print(f"  FPIR={fpir_pca:.4f}  FNIR={fnir_pca:.4f}  theta={thr_pca:.4f}")

    # ------------------------------------------------------------------
    # METHOD 2 — LBP (Local Binary Patterns)
    # ------------------------------------------------------------------
    print("\nRunning LBP...")
    FM_lbp = np.array([lbp(img) for img in images])

    Gen_lbp, Imp_lbp              = gen_imp_scores(FM_lbp, labels, enroll_idx, probe_idx)
    res_lbp                       = compute_metrics(Gen_lbp, Imp_lbp)
    r1_lbp                        = rank1_acc(FM_lbp, labels, enroll_idx, probe_idx)
    fpir_lbp, fnir_lbp, thr_lbp  = fpir_fnir_openset(FM_lbp, labels, enroll_idx, probe_idx, impostor_idx)
    res_lbp.update(rank1=r1_lbp, fpir=fpir_lbp, fnir=fnir_lbp)

    print(f"  EER={res_lbp['eer']:.4f}  d'={res_lbp['d_prime']:.4f}"
          f"  Rank-1={r1_lbp:.4f}  AUC={res_lbp['auc']:.4f}")
    print(f"  TMR@FMR=1%={res_lbp['tmr_1']:.4f}  TMR@FMR=0.01%={res_lbp['tmr_001']:.4f}")
    print(f"  FPIR={fpir_lbp:.4f}  FNIR={fnir_lbp:.4f}  theta={thr_lbp:.4f}")

    # ------------------------------------------------------------------
    # CALIBRATED THRESHOLD RECOMMENDATIONS
    # ------------------------------------------------------------------
    eer_thr = res_lbp['eer_thr']
    print("\n--- CALIBRATED THRESHOLDS (copy into enrollment.py) ---")
    print(f"  LBP EER threshold               : {eer_thr:.4f}")
    print(f"  Suggested VERIFY_THRESHOLD      : {eer_thr - 0.02:.4f}")
    print(f"  Suggested DUPLICATE_THRESHOLD   : {eer_thr + 0.03:.4f}")

    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DET CURVES
    # ------------------------------------------------------------------
    fpirs_pca, fnirs_pca, _ = det_curve(FM_pca, labels, enroll_idx, probe_idx, impostor_idx)
    fpirs_lbp, fnirs_lbp, _ = det_curve(FM_lbp, labels, enroll_idx, probe_idx, impostor_idx)

    # ------------------------------------------------------------------
    # DASHBOARD  (3 x 3 grid)
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        "Face Recognition — PCA vs LBP  (Open-Set Evaluation)",
        fontsize=14, fontweight="bold"
    )

    # Row 0 — Gen/Imp distributions
    axs[0, 0].hist(Gen_pca, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 0].hist(Imp_pca, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 0].set_title("PCA — Gen/Imp Distribution")
    axs[0, 0].set_xlabel("Cosine Similarity")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].legend(); axs[0, 0].grid(alpha=0.3)

    axs[0, 1].hist(Gen_lbp, bins=50, alpha=0.6, label="Genuine",
                   density=True, color="steelblue")
    axs[0, 1].hist(Imp_lbp, bins=50, alpha=0.6, label="Impostor",
                   density=True, color="darkorange")
    axs[0, 1].set_title("LBP — Gen/Imp Distribution")
    axs[0, 1].set_xlabel("Cosine Similarity")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend(); axs[0, 1].grid(alpha=0.3)

    # Row 0 col 2 — ROC
    axs[0, 2].plot(res_pca['fpr'], res_pca['tpr'], linewidth=2,
                   label=f"PCA  (EER={res_pca['eer']:.3f})")
    axs[0, 2].plot(res_lbp['fpr'], res_lbp['tpr'], linewidth=2,
                   label=f"LBP  (EER={res_lbp['eer']:.3f})")
    axs[0, 2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axs[0, 2].set_title("ROC Curve (Verification)")
    axs[0, 2].set_xlabel("FMR (False Match Rate)")
    axs[0, 2].set_ylabel("TMR (True Match Rate)")
    axs[0, 2].legend(); axs[0, 2].grid(True, alpha=0.3)

    # Row 1 — DET curve + EER bar + d' bar
    axs[1, 0].plot(fpirs_pca * 100, fnirs_pca * 100, linewidth=2,
                   label=f"PCA  (FPIR={fpir_pca*100:.1f}%  FNIR={fnir_pca*100:.1f}%)")
    axs[1, 0].plot(fpirs_lbp * 100, fnirs_lbp * 100, linewidth=2,
                   label=f"LBP  (FPIR={fpir_lbp*100:.1f}%  FNIR={fnir_lbp*100:.1f}%)")
    axs[1, 0].scatter([fpir_pca * 100], [fnir_pca * 100], zorder=5, s=80)
    axs[1, 0].scatter([fpir_lbp * 100], [fnir_lbp * 100], zorder=5, s=80)
    axs[1, 0].set_title("DET Curve — Open-Set  (EER operating point marked)")
    axs[1, 0].set_xlabel("FPIR %  (False Positive Identification Rate)")
    axs[1, 0].set_ylabel("FNIR %  (False Negative Identification Rate)")
    axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)

    methods = ["PCA", "LBP"]
    eers    = [res_pca['eer'], res_lbp['eer']]
    bars    = axs[1, 1].bar(methods, eers,
                            color=["steelblue", "darkorange"], width=0.4)
    axs[1, 1].set_title("EER Comparison (lower = better)")
    axs[1, 1].set_ylabel("EER")
    axs[1, 1].set_ylim(0, max(eers) * 1.3)
    for bar, val in zip(bars, eers):
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.002,
                       f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    dps  = [res_pca['d_prime'], res_lbp['d_prime']]
    bars = axs[1, 2].bar(methods, dps,
                         color=["steelblue", "darkorange"], width=0.4)
    axs[1, 2].set_title("d-prime Comparison (higher = better)")
    axs[1, 2].set_ylabel("d-prime")
    axs[1, 2].set_ylim(0, max(dps) * 1.3)
    for bar, val in zip(bars, dps):
        axs[1, 2].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.02,
                       f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    # Row 2 — FPIR/FNIR bar + summary table + split info
    x      = np.arange(2)
    width  = 0.35
    fpirs_ = [fpir_pca * 100, fpir_lbp * 100]
    fnirs_ = [fnir_pca * 100, fnir_lbp * 100]
    b1 = axs[2, 0].bar(x - width/2, fpirs_, width, label="FPIR %", color="crimson",   alpha=0.8)
    b2 = axs[2, 0].bar(x + width/2, fnirs_, width, label="FNIR %", color="steelblue", alpha=0.8)
    axs[2, 0].set_title("FPIR & FNIR @ EER threshold  (Open-Set)")
    axs[2, 0].set_xticks(x); axs[2, 0].set_xticklabels(methods)
    axs[2, 0].set_ylabel("%")
    axs[2, 0].legend(); axs[2, 0].grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        axs[2, 0].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.3,
                       f"{bar.get_height():.1f}%",
                       ha="center", va="bottom", fontsize=8, fontweight="bold")

    axs[2, 1].axis("off")
    rows = [
        ["Metric",           "PCA",                              "LBP"],
        ["D-prime",          f"{res_pca['d_prime']:.3f}",        f"{res_lbp['d_prime']:.3f}"],
        ["EER",              f"{res_pca['eer']*100:.2f}%",       f"{res_lbp['eer']*100:.2f}%"],
        ["TMR @ FMR=1%",     f"{res_pca['tmr_1']*100:.2f}%",    f"{res_lbp['tmr_1']*100:.2f}%"],
        ["TMR @ FMR=0.01%",  f"{res_pca['tmr_001']*100:.2f}%",  f"{res_lbp['tmr_001']*100:.2f}%"],
        ["AUC",              f"{res_pca['auc']:.4f}",            f"{res_lbp['auc']:.4f}"],
        ["── Open-Set ──",   "──────",                           "──────"],
        ["Rank-1 (TPIR)",    f"{res_pca['rank1']*100:.2f}%",    f"{res_lbp['rank1']*100:.2f}%"],
        ["FPIR @ EER",       f"{fpir_pca*100:.2f}%",            f"{fpir_lbp*100:.2f}%"],
        ["FNIR @ EER",       f"{fnir_pca*100:.2f}%",            f"{fnir_lbp*100:.2f}%"],
        ["ID Threshold",     f"{thr_pca:.4f}",                  f"{thr_lbp:.4f}"],
    ]

    tbl = axs[2, 1].table(cellText=rows[1:], colLabels=rows[0],
                          loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.5)
    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(3):
        tbl[7, j].set_facecolor("#ecf0f1")
        tbl[7, j].set_text_props(fontweight="bold", color="#7f8c8d")
    axs[2, 1].set_title("All Metrics Summary")

    axs[2, 2].axis("off")
    split_info = (
        f"Evaluation Protocol\n"
        f"{'─'*32}\n"
        f"Dataset     : ORL/AT&T\n"
        f"Subjects    : {len(np.unique(labels))} total\n\n"
        f"Enrolled    : {len(np.unique(labels[enroll_idx]))} subjects\n"
        f"  Gallery   : {len(enroll_idx)} images\n\n"
        f"Genuine probes\n"
        f"  Subjects  : {len(np.unique(labels[probe_idx]))}\n"
        f"  Images    : {len(probe_idx)}\n\n"
        f"Impostor probes (unknown)\n"
        f"  Subjects  : {len(np.unique(labels[impostor_idx]))}\n"
        f"  Images    : {len(impostor_idx)}\n\n"
        f"Thresholds @ EER\n"
        f"  PCA       : {thr_pca:.4f}\n"
        f"  LBP       : {thr_lbp:.4f}"
    )
    axs[2, 2].text(
        0.05, 0.95, split_info,
        transform=axs[2, 2].transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8),
    )
    axs[2, 2].set_title("Protocol Summary")

    plt.tight_layout()

    Path("static/images").mkdir(parents=True, exist_ok=True)
    for dest in [out / "final_dashboard.png", Path("static/images/results.png")]:
        plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard saved → {out / 'final_dashboard.png'}")
    print("DONE")

    return {
        "pca": {
            "d_prime": res_pca['d_prime'],
            "eer":     res_pca['eer'],
            "tmr_1":   res_pca['tmr_1'],
            "tmr_001": res_pca['tmr_001'],
            "auc":     res_pca['auc'],
            "rank1":   res_pca['rank1'],
            "fpir":    fpir_pca,
            "fnir":    fnir_pca,
        },
        "lbp": {
            "d_prime": res_lbp['d_prime'],
            "eer":     res_lbp['eer'],
            "tmr_1":   res_lbp['tmr_1'],
            "tmr_001": res_lbp['tmr_001'],
            "auc":     res_lbp['auc'],
            "rank1":   res_lbp['rank1'],
            "fpir":    fpir_lbp,
            "fnir":    fnir_lbp,
        },
        "plot_url": "/static/images/results.png",
        "calibration": {
            "lbp_eer_threshold":   round(eer_thr, 4),
            "suggested_verify":    round(eer_thr - 0.02, 4),
            "suggested_duplicate": round(eer_thr + 0.03, 4),
        },
    }


if __name__ == "__main__":
    run_pipeline(dataset_path="dataset")