"""
=============================================================
  Biometric Face Recognition System
  Dataset  : ORL/AT&T (Olivetti Faces) – 40 subjects, 10 imgs each
  Methods  : PCA (Eigenfaces)  &  LBP (Local Binary Patterns)
  Author   : Generated Project
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from skimage.feature import local_binary_pattern
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Biometric Face Recognition – ORL/AT&T Dataset")
print("=" * 60)

print("\n[1] Loading dataset …")
# ORL/AT&T (Olivetti) – 40 subjects × 10 images = 400 total
# Dataset is loaded from locally generated numpy arrays
images = np.load('faces_images.npy')   # (400, 64, 64)  float32 [0,1]
labels = np.load('faces_labels.npy')   # (400,)          int 0–39

N_SUBJECTS      = 40
N_PER_SUBJECT   = 10
IMG_H, IMG_W    = 64, 64

print(f"    Subjects      : {N_SUBJECTS}")
print(f"    Images/subject: {N_PER_SUBJECT}")
print(f"    Image size    : {IMG_H} × {IMG_W} (grayscale)")

# ─────────────────────────────────────────────────────────────
# 2.  80 / 20 SPLIT  (per subject)
# ─────────────────────────────────────────────────────────────
print("\n[2] Splitting data (80% train / 20% test per subject) …")

train_imgs, train_lbl = [], []
test_imgs,  test_lbl  = [], []

for s in range(N_SUBJECTS):
    subj_imgs = images[labels == s]          # 10 images
    n_train   = int(0.8 * N_PER_SUBJECT)    # 8
    train_imgs.extend(subj_imgs[:n_train])
    train_lbl.extend([s] * n_train)
    test_imgs.extend(subj_imgs[n_train:])   # 2
    test_lbl.extend([s] * (N_PER_SUBJECT - n_train))

train_imgs = np.array(train_imgs)           # (320, 64, 64)
train_lbl  = np.array(train_lbl)
test_imgs  = np.array(test_imgs)            # (80,  64, 64)
test_lbl   = np.array(test_lbl)

print(f"    Train : {len(train_imgs)} images  |  Test : {len(test_imgs)} images")

# ─────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n[3] Preprocessing …")

def preprocess(imgs):
    """Flatten → zero-mean / unit-variance normalisation per image."""
    flat = imgs.reshape(len(imgs), -1).astype(np.float64)
    mu   = flat.mean(axis=1, keepdims=True)
    sig  = flat.std(axis=1,  keepdims=True) + 1e-8
    return (flat - mu) / sig

train_flat = preprocess(train_imgs)
test_flat  = preprocess(test_imgs)
print("    Applied: flatten → zero-mean, unit-variance normalisation")

# ─────────────────────────────────────────────────────────────
# 4A. FEATURE EXTRACTION – METHOD 1: PCA  (Eigenfaces)
# ─────────────────────────────────────────────────────────────
print("\n[4a] Feature Extraction – Method 1: PCA (Eigenfaces) …")

N_COMPONENTS = 100
pca       = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)
train_pca = pca.fit_transform(train_flat)
test_pca  = pca.transform(test_flat)

explained = pca.explained_variance_ratio_.sum() * 100
print(f"    Components        : {N_COMPONENTS}")
print(f"    Variance explained: {explained:.1f}%")

# ─────────────────────────────────────────────────────────────
# 4B. FEATURE EXTRACTION – METHOD 2: LBP
# ─────────────────────────────────────────────────────────────
print("\n[4b] Feature Extraction – Method 2: LBP …")

def extract_lbp(imgs, radius=2, n_points=16):
    feats = []
    for img in imgs:
        lbp  = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(),
                               bins=n_points + 2,
                               range=(0, n_points + 2),
                               density=True)
        feats.append(hist)
    return np.array(feats)

train_lbp = extract_lbp(train_imgs)
test_lbp  = extract_lbp(test_imgs)
print(f"    Radius  : 2  |  Points: 16")
print(f"    Feature : {train_lbp.shape[1]}-dim histogram")

# ─────────────────────────────────────────────────────────────
# 5.  MATCHING  (cosine similarity)
# ─────────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return normalize(a) @ normalize(b).T   # (|a|, |b|)

# ─────────────────────────────────────────────────────────────
# 6.  GENUINE & IMPOSTOR SCORES
# ─────────────────────────────────────────────────────────────
print("\n[5+6] Matching & building score distributions …")

def get_scores(tr_feats, te_feats, tr_lbl, te_lbl):
    sim = cosine_sim(te_feats, tr_feats)
    gen, imp = [], []
    for i, tl in enumerate(te_lbl):
        for j, rl in enumerate(tr_lbl):
            (gen if tl == rl else imp).append(sim[i, j])
    return np.array(gen), np.array(imp)

gen_pca, imp_pca = get_scores(train_pca, test_pca, train_lbl, test_lbl)
gen_lbp, imp_lbp = get_scores(train_lbp, test_lbp, train_lbl, test_lbl)

print(f"    PCA – Genuine: {len(gen_pca):,}  |  Impostor: {len(imp_pca):,}")
print(f"    LBP – Genuine: {len(gen_lbp):,}  |  Impostor: {len(imp_lbp):,}")

# ─────────────────────────────────────────────────────────────
# 7.  METRICS  (D-prime, EER, TMR@FMR)
# ─────────────────────────────────────────────────────────────
print("\n[7] Computing verification metrics …")

def compute_metrics(gen, imp, name):
    # D-prime
    mu_g, mu_i = gen.mean(), imp.mean()
    sg,   si   = gen.std(),  imp.std()
    d_prime    = abs(mu_g - mu_i) / np.sqrt(0.5 * (sg**2 + si**2))

    # ROC
    scores = np.concatenate([gen, imp])
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # EER  (where FPR ≈ FNR)
    fnr     = 1 - tpr
    idx     = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[idx] + fnr[idx]) / 2
    eer_thr = thresholds[idx]

    # TMR at FMR = 1%  and  FMR = 0.01%
    f_tpr = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    tmr_1   = float(f_tpr(0.01))
    tmr_001 = float(f_tpr(0.0001))

    print(f"\n    [{name}]")
    print(f"      D-prime          : {d_prime:.4f}")
    print(f"      EER              : {eer*100:.2f}%  (threshold={eer_thr:.4f})")
    print(f"      TMR @ FMR=1%     : {tmr_1*100:.2f}%")
    print(f"      TMR @ FMR=0.01%  : {tmr_001*100:.2f}%")
    print(f"      AUC              : {roc_auc:.4f}")

    return dict(fpr=fpr, tpr=tpr, auc=roc_auc,
                d_prime=d_prime, eer=eer, eer_thr=eer_thr,
                tmr_1=tmr_1, tmr_001=tmr_001)

res_pca = compute_metrics(gen_pca, imp_pca, "PCA – Eigenfaces")
res_lbp = compute_metrics(gen_lbp, imp_lbp, "LBP")

# ─────────────────────────────────────────────────────────────
# 8.  IDENTIFICATION  (BONUS +20%)
#     Rank-1 accuracy, FPIR, FNIR
# ─────────────────────────────────────────────────────────────
print("\n[8] BONUS – Identification metrics (Rank-1, FPIR, FNIR) …")

def identification(tr_feats, te_feats, tr_lbl, te_lbl, threshold, name):
    sim   = cosine_sim(te_feats, tr_feats)   # (n_test, n_train)
    total = len(te_lbl)

    rank1  = 0
    tp = fp = fn = 0

    for i, true_lbl in enumerate(te_lbl):
        row            = sim[i]
        best_idx       = int(np.argmax(row))
        best_score     = row[best_idx]
        predicted_lbl  = tr_lbl[best_idx]

        if predicted_lbl == true_lbl:
            rank1 += 1

        if best_score >= threshold:          # system ACCEPTS
            if predicted_lbl == true_lbl:
                tp += 1
            else:
                fp += 1
        else:                                # system REJECTS
            fn += 1

    rank1_acc = rank1 / total
    fpir      = fp   / total
    fnir      = fn   / total

    print(f"\n    [{name}]")
    print(f"      Rank-1 Accuracy (TPIR) : {rank1_acc*100:.2f}%")
    print(f"      FPIR @ EER threshold   : {fpir*100:.2f}%")
    print(f"      FNIR @ EER threshold   : {fnir*100:.2f}%")

    return rank1_acc, fpir, fnir

r1_pca, fpir_pca, fnir_pca = identification(
    train_pca, test_pca, train_lbl, test_lbl,
    res_pca['eer_thr'], "PCA – Eigenfaces")

r1_lbp, fpir_lbp, fnir_lbp = identification(
    train_lbp, test_lbp, train_lbl, test_lbl,
    res_lbp['eer_thr'], "LBP")

# ─────────────────────────────────────────────────────────────
# 9.  PLOTS  (Gen-Imp  ×2 + ROC + metrics table)
# ─────────────────────────────────────────────────────────────
print("\n[9] Generating plots …")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Biometric Face Recognition – ORL/AT&T Dataset\n'
             'PCA (Eigenfaces) vs LBP',
             fontsize=14, fontweight='bold', y=1.01)

# ── Gen/Imp – PCA ──────────────────────────────────────────
ax = axes[0, 0]
ax.hist(imp_pca, bins=80, alpha=0.65, color='#e74c3c', density=True, label='Impostor')
ax.hist(gen_pca, bins=80, alpha=0.65, color='#27ae60', density=True, label='Genuine')
ax.set_title('Genuine / Impostor Distribution – PCA', fontsize=11)
ax.set_xlabel('Cosine Similarity Score')
ax.set_ylabel('Density')
ax.legend(); ax.grid(alpha=0.3)

# ── Gen/Imp – LBP ──────────────────────────────────────────
ax = axes[0, 1]
ax.hist(imp_lbp, bins=80, alpha=0.65, color='#e74c3c', density=True, label='Impostor')
ax.hist(gen_lbp, bins=80, alpha=0.65, color='#27ae60', density=True, label='Genuine')
ax.set_title('Genuine / Impostor Distribution – LBP', fontsize=11)
ax.set_xlabel('Cosine Similarity Score')
ax.set_ylabel('Density')
ax.legend(); ax.grid(alpha=0.3)

# ── ROC Curve ──────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(res_pca['fpr'], res_pca['tpr'], color='#2980b9', lw=2,
        label=f"PCA  AUC={res_pca['auc']:.3f}  EER={res_pca['eer']*100:.1f}%")
ax.plot(res_lbp['fpr'], res_lbp['tpr'], color='#e67e22', lw=2, ls='--',
        label=f"LBP  AUC={res_lbp['auc']:.3f}  EER={res_lbp['eer']*100:.1f}%")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.25)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.set_xlabel('FMR  (False Match Rate)')
ax.set_ylabel('TMR  (True Match Rate = 1 – FRR)')
ax.set_title('ROC Curve', fontsize=11)
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)

# ── Metrics Table ──────────────────────────────────────────
ax = axes[1, 1]
ax.axis('off')

rows = [
    ['Metric',            'PCA',                       'LBP'],
    ['D-prime',           f"{res_pca['d_prime']:.3f}", f"{res_lbp['d_prime']:.3f}"],
    ['EER',               f"{res_pca['eer']*100:.2f}%", f"{res_lbp['eer']*100:.2f}%"],
    ['TMR @ FMR=1%',      f"{res_pca['tmr_1']*100:.2f}%", f"{res_lbp['tmr_1']*100:.2f}%"],
    ['TMR @ FMR=0.01%',   f"{res_pca['tmr_001']*100:.2f}%", f"{res_lbp['tmr_001']*100:.2f}%"],
    ['AUC',               f"{res_pca['auc']:.4f}",    f"{res_lbp['auc']:.4f}"],
    ['── BONUS ──',       '──────',                    '──────'],
    ['Rank-1 (TPIR)',     f"{r1_pca*100:.2f}%",        f"{r1_lbp*100:.2f}%"],
    ['FPIR @ EER-thr',    f"{fpir_pca*100:.2f}%",      f"{fpir_lbp*100:.2f}%"],
    ['FNIR @ EER-thr',    f"{fnir_pca*100:.2f}%",      f"{fnir_lbp*100:.2f}%"],
]

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.15, 1.7)

for j in range(3):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for j in range(3):
    tbl[6, j].set_facecolor('#ecf0f1')
    tbl[6, j].set_text_props(fontweight='bold', color='#7f8c8d')

ax.set_title('Metrics Summary', fontsize=11, pad=15)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("    Saved: results.png")

print("\n" + "=" * 60)
print("  ✅  Project complete!")
print("=" * 60)
