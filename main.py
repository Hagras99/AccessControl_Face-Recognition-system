"""
=============================================================
  Biometric Face Recognition System
  Dataset  : ORL/AT&T (Olivetti Faces) – 40 subjects, 10 imgs each
  Methods  : PCA (Eigenfaces)  &  LBP (Local Binary Patterns)
  Author   : Generated Project
=============================================================
"""

import os
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

def preprocess(imgs):
    """Flatten → zero-mean / unit-variance normalisation per image."""
    flat = imgs.reshape(len(imgs), -1).astype(np.float64)
    mu   = flat.mean(axis=1, keepdims=True)
    sig  = flat.std(axis=1,  keepdims=True) + 1e-8
    return (flat - mu) / sig

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

def cosine_sim(a, b):
    return normalize(a) @ normalize(b).T   # (|a|, |b|)

def get_scores(tr_feats, te_feats, tr_lbl, te_lbl):
    sim = cosine_sim(te_feats, tr_feats)
    gen, imp = [], []
    for i, tl in enumerate(te_lbl):
        for j, rl in enumerate(tr_lbl):
            (gen if tl == rl else imp).append(sim[i, j])
    return np.array(gen), np.array(imp)

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

    return dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=float(roc_auc),
                d_prime=float(d_prime), eer=float(eer), eer_thr=float(eer_thr),
                tmr_1=float(tmr_1), tmr_001=float(tmr_001))

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

    return float(rank1_acc), float(fpir), float(fnir)


def run_pipeline():
    images = np.load('faces_images.npy')   
    labels = np.load('faces_labels.npy')   

    N_SUBJECTS      = 40
    N_PER_SUBJECT   = 10

    train_imgs, train_lbl = [], []
    test_imgs,  test_lbl  = [], []

    for s in range(N_SUBJECTS):
        subj_imgs = images[labels == s]          
        n_train   = int(0.8 * N_PER_SUBJECT)    
        train_imgs.extend(subj_imgs[:n_train])
        train_lbl.extend([s] * n_train)
        test_imgs.extend(subj_imgs[n_train:])   
        test_lbl.extend([s] * (N_PER_SUBJECT - n_train))

    train_imgs = np.array(train_imgs)           
    train_lbl  = np.array(train_lbl)
    test_imgs  = np.array(test_imgs)            
    test_lbl   = np.array(test_lbl)

    train_flat = preprocess(train_imgs)
    test_flat  = preprocess(test_imgs)

    N_COMPONENTS = 100
    pca       = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)
    train_pca = pca.fit_transform(train_flat)
    test_pca  = pca.transform(test_flat)

    train_lbp = extract_lbp(train_imgs)
    test_lbp  = extract_lbp(test_imgs)

    gen_pca, imp_pca = get_scores(train_pca, test_pca, train_lbl, test_lbl)
    gen_lbp, imp_lbp = get_scores(train_lbp, test_lbp, train_lbl, test_lbl)

    res_pca = compute_metrics(gen_pca, imp_pca, "PCA")
    res_lbp = compute_metrics(gen_lbp, imp_lbp, "LBP")

    r1_pca, fpir_pca, fnir_pca = identification(
        train_pca, test_pca, train_lbl, test_lbl,
        res_pca['eer_thr'], "PCA")

    r1_lbp, fpir_lbp, fnir_lbp = identification(
        train_lbp, test_lbp, train_lbl, test_lbl,
        res_lbp['eer_thr'], "LBP")
        
    res_pca['rank1'] = r1_pca
    res_pca['fpir'] = fpir_pca
    res_pca['fnir'] = fnir_pca
    
    res_lbp['rank1'] = r1_lbp
    res_lbp['fpir'] = fpir_lbp
    res_lbp['fnir'] = fnir_lbp

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Biometric Face Recognition – ORL/AT&T Dataset\nPCA vs LBP', fontsize=14, fontweight='bold', y=1.01)

    # ── Gen/Imp – PCA 
    ax = axes[0, 0]
    ax.hist(imp_pca, bins=80, alpha=0.65, color='#e74c3c', density=True, label='Impostor')
    ax.hist(gen_pca, bins=80, alpha=0.65, color='#27ae60', density=True, label='Genuine')
    ax.set_title('Genuine / Impostor Distribution – PCA', fontsize=11)
    ax.set_xlabel('Cosine Similarity Score')
    ax.set_ylabel('Density')
    ax.legend(); ax.grid(alpha=0.3)

    # ── Gen/Imp – LBP
    ax = axes[0, 1]
    ax.hist(imp_lbp, bins=80, alpha=0.65, color='#e74c3c', density=True, label='Impostor')
    ax.hist(gen_lbp, bins=80, alpha=0.65, color='#27ae60', density=True, label='Genuine')
    ax.set_title('Genuine / Impostor Distribution – LBP', fontsize=11)
    ax.set_xlabel('Cosine Similarity Score')
    ax.set_ylabel('Density')
    ax.legend(); ax.grid(alpha=0.3)

    # ── ROC Curve
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

    # ── Metrics Table
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
        ['Rank-1 (TPIR)',     f"{res_pca['rank1']*100:.2f}%",        f"{res_lbp['rank1']*100:.2f}%"],
        ['FPIR @ EER-thr',    f"{res_pca['fpir']*100:.2f}%",      f"{res_lbp['fpir']*100:.2f}%"],
        ['FNIR @ EER-thr',    f"{res_pca['fnir']*100:.2f}%",      f"{res_lbp['fnir']*100:.2f}%"],
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
    
    os.makedirs('static/images', exist_ok=True)
    plot_path = 'static/images/results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "pca": {
            "d_prime": res_pca['d_prime'],
            "eer": res_pca['eer'],
            "tmr_1": res_pca['tmr_1'],
            "tmr_001": res_pca['tmr_001'],
            "auc": res_pca['auc'],
            "rank1": res_pca['rank1'],
            "fpir": res_pca['fpir'],
            "fnir": res_pca['fnir']
        },
        "lbp": {
            "d_prime": res_lbp['d_prime'],
            "eer": res_lbp['eer'],
            "tmr_1": res_lbp['tmr_1'],
            "tmr_001": res_lbp['tmr_001'],
            "auc": res_lbp['auc'],
            "rank1": res_lbp['rank1'],
            "fpir": res_lbp['fpir'],
            "fnir": res_lbp['fnir']
        },
        "plot_url": f"/static/images/results.png"
    }

if __name__ == "__main__":
    results = run_pipeline()
    print("Metrics extracted successfully. Plot saved to static/images/results.png.")
