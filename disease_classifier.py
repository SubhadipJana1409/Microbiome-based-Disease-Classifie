"""
================================================================
Day 18 — Microbiome-based Disease Classifier (Random Forest)
Author  : Subhadip Jana | #30DaysOfBioinformatics
Dataset : IBD microbiome — Gevers 2014 style (CD vs Control)
          447 samples × 37 gut taxa (relative abundance)

Pipeline:
  1.  Data loading & EDA
  2.  Alpha diversity (Shannon, Chao1, Simpson)
  3.  CLR normalisation (compositional-aware)
  4.  Train/test split (stratified 80/20)
  5.  Random Forest + hyperparameter tuning (GridSearchCV)
  6.  Evaluation — ROC-AUC, PR, Confusion Matrix
  7.  Feature importance (Gini + Permutation)
  8.  Taxa differential abundance (Mann-Whitney + BH-FDR)
  9.  Combined dashboard + separate panels for key plots
================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro
from scipy.stats import entropy as sci_entropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV,
                                      learning_curve)
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              average_precision_score, precision_recall_curve,
                              f1_score, accuracy_score)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import warnings, os
warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("outputs/panels", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("="*62)
print("Day 18 — Microbiome Disease Classifier | IBD | RF")
print("="*62)

otu  = pd.read_csv("data/otu_table.csv",  index_col=0)
meta = pd.read_csv("data/metadata.csv")
meta = meta.set_index("SampleID")

# Align
common   = otu.index.intersection(meta.index)
otu      = otu.loc[common]
meta     = meta.loc[common]
y_raw    = (meta["diagnosis"] == "CD").astype(int).values
taxa     = otu.columns.tolist()
taxa_short = [t.replace("_"," ").replace(" prausnitzii","").replace(" unclassified","")
              for t in taxa]

print(f"\n✅ Loaded: {otu.shape[0]} samples × {otu.shape[1]} taxa")
print(f"   Control (HC): {(y_raw==0).sum()}")
print(f"   CD (IBD)    : {(y_raw==1).sum()}")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ALPHA DIVERSITY
# ═══════════════════════════════════════════════════════════════
print("\n📊 Computing alpha diversity...")

def shannon(row):
    p = row[row>0]; p = p/p.sum()
    return -np.sum(p*np.log(p))

def simpson(row):
    p = row[row>0]; p = p/p.sum()
    return 1 - np.sum(p**2)

def chao1(row):
    n1 = (row==1).sum(); n2 = max((row==2).sum(),1)
    obs = (row>0).sum()
    return obs + n1**2/(2*n2)

# Work with counts proxy (multiply relative by 1000)
counts_proxy = (otu * 1000).round().astype(int)
shannon_div  = otu.apply(shannon, axis=1)
simpson_div  = otu.apply(simpson, axis=1)
chao1_div    = counts_proxy.apply(chao1, axis=1)

meta["shannon"] = shannon_div
meta["simpson"] = simpson_div
meta["chao1"]   = chao1_div
meta["richness"]= (otu>0).sum(axis=1)

for metric in ["shannon","simpson","chao1"]:
    hc  = meta.loc[meta.diagnosis=="Control", metric]
    ibd = meta.loc[meta.diagnosis=="CD",      metric]
    stat,p = mannwhitneyu(hc, ibd, alternative="two-sided")
    print(f"   {metric:10s}: HC={hc.mean():.3f}±{hc.std():.3f}  "
          f"CD={ibd.mean():.3f}±{ibd.std():.3f}  p={p:.2e}")

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CLR NORMALISATION
# ═══════════════════════════════════════════════════════════════
# Centered Log-Ratio — standard for compositional microbiome data
pseudo  = 1e-6
otu_clr = np.log(otu + pseudo)
otu_clr = otu_clr.sub(otu_clr.mean(axis=1), axis=0)  # subtract row geometric mean
X       = otu_clr.values
y       = y_raw

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════
X_tr,X_te,y_tr,y_te = train_test_split(X,y, test_size=0.2,
                                         stratify=y, random_state=42)
print(f"\n   Train={len(y_tr)} | Test={len(y_te)}")
print(f"   Train HC/CD: {(y_tr==0).sum()}/{(y_tr==1).sum()}")
print(f"   Test  HC/CD: {(y_te==0).sum()}/{(y_te==1).sum()}")

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — RANDOM FOREST + TUNING
# ═══════════════════════════════════════════════════════════════
print("\n🔧 Training Random Forest (GridSearchCV 5-fold)...")
param_grid = {
    "n_estimators" : [200, 300, 500],
    "max_depth"    : [None, 5, 10],
    "max_features" : ["sqrt", "log2"],
    "min_samples_leaf": [1, 2],
}
cv   = StratifiedKFold(5, shuffle=True, random_state=42)
grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"),
                    param_grid, cv=cv, scoring="roc_auc",
                    n_jobs=-1, verbose=0)
grid.fit(X_tr, y_tr)
model       = grid.best_estimator_
best_params = grid.best_params_
print(f"   Best params : {best_params}")
print(f"   Best CV AUC : {grid.best_score_:.4f}")

cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
cv_f1  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")
print(f"   CV AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"   CV F1  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION
# ═══════════════════════════════════════════════════════════════
y_prob = model.predict_proba(X_te)[:,1]
fpr_c,tpr_c,thresholds = roc_curve(y_te, y_prob)
opt_thr = thresholds[np.argmax(tpr_c - fpr_c)]
y_pred  = (y_prob >= opt_thr).astype(int)

auc   = roc_auc_score(y_te, y_prob)
ap    = average_precision_score(y_te, y_prob)
f1    = f1_score(y_te, y_pred)
acc   = accuracy_score(y_te, y_pred)
cm    = confusion_matrix(y_te, y_pred)
tn,fp,fn,tp = cm.ravel()
sens  = tp/(tp+fn); spec=tn/(tn+fp)
prec_c,rec_c,_ = precision_recall_curve(y_te, y_prob)

print(f"\n📊 Test: AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  "
      f"Sens={sens:.4f}  Spec={spec:.4f}")

# GBT comparison
gbt = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.1, random_state=42)
gbt.fit(X_tr,y_tr)
gbt_prob = gbt.predict_proba(X_te)[:,1]
gbt_auc  = roc_auc_score(y_te, gbt_prob)
gbt_fpr,gbt_tpr,_ = roc_curve(y_te, gbt_prob)

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
print("\n🔍 Feature importances...")
gini_imp  = pd.Series(model.feature_importances_, index=taxa_short).sort_values()
perm      = permutation_importance(model, X_te, y_te,
                                    n_repeats=50, random_state=42,
                                    scoring="roc_auc")
perm_mean = pd.Series(perm.importances_mean, index=taxa_short).sort_values()
perm_std  = pd.Series(perm.importances_std,  index=taxa_short)

print(f"   Top 5 taxa by permutation importance:")
for t in perm_mean.sort_values(ascending=False).index[:5]:
    print(f"   {t:35s}: {perm_mean[t]:.4f} ± {perm_std[t]:.4f}")

# Learning curve
tr_sizes,tr_scores,val_scores = learning_curve(
    model, X, y, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1,1.0,8), random_state=42)

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — DIFFERENTIAL ABUNDANCE
# ═══════════════════════════════════════════════════════════════
print("\n🦠 Differential abundance (Mann-Whitney + BH-FDR)...")
hc_idx  = np.where(y==0)[0]
cd_idx  = np.where(y==1)[0]
da_res  = []
for i, t in enumerate(taxa_short):
    hc_vals = X[hc_idx, i]
    cd_vals = X[cd_idx, i]
    stat, p = mannwhitneyu(hc_vals, cd_vals, alternative="two-sided")
    lfc     = cd_vals.mean() - hc_vals.mean()   # CLR difference
    da_res.append({"taxa": t, "lfc_clr": lfc, "pvalue": p})

da_df   = pd.DataFrame(da_res).sort_values("pvalue")
# BH-FDR
n = len(da_df); da_df = da_df.reset_index(drop=True)
da_df["rank"] = da_df.index + 1
da_df["padj"] = (da_df["pvalue"] * n / da_df["rank"]).clip(upper=1)
for i in range(n-2,-1,-1):
    da_df.loc[i,"padj"] = min(da_df.loc[i,"padj"], da_df.loc[i+1,"padj"])
da_df["sig"] = da_df["padj"] < 0.05
print(f"   Significant taxa (FDR<0.05): {da_df.sig.sum()}")

# Save outputs
pd.DataFrame({"true":y_te,"pred":y_pred,"prob":y_prob}
             ).to_csv("outputs/predictions.csv", index=False)
da_df.to_csv("outputs/differential_abundance.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# COLOURS & HELPERS
# ═══════════════════════════════════════════════════════════════
C_HC   = "#3498DB"  # healthy/control
C_CD   = "#E74C3C"  # CD/IBD
C_RAND = "#95A5A6"
SUBTITLE = "Gut Microbiome | IBD Study | Crohn's Disease vs Healthy Control"

def save_panel(fig, name):
    fig.savefig(f"outputs/panels/{name}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {name}.png")

# ═══════════════════════════════════════════════════════════════
# COMBINED DASHBOARD  3×3
# ═══════════════════════════════════════════════════════════════
print("\n🎨 Combined dashboard...")
fig = plt.figure(figsize=(27, 23))
fig.suptitle(
    "Day 18 — Microbiome-based Disease Classifier  |  Random Forest\n"
    f"{SUBTITLE}\n"
    f"n=447 samples  ·  37 taxa  ·  "
    f"ROC-AUC={auc:.3f}  ·  F1={f1:.3f}  ·  "
    f"Sensitivity={sens:.3f}  ·  Specificity={spec:.3f}",
    fontsize=14, fontweight="bold", y=0.998)
gs = GridSpec(3,3,figure=fig,hspace=0.44,wspace=0.40,
              left=0.07,right=0.96,top=0.95,bottom=0.05)

# P1 — Alpha diversity boxplots
ax1 = fig.add_subplot(gs[0,0])
metrics_plot = ["shannon","chao1","simpson"]
positions    = [1,2,3]
for k, metric in enumerate(metrics_plot):
    hc_vals  = meta.loc[meta.diagnosis=="Control", metric].values
    cd_vals  = meta.loc[meta.diagnosis=="CD",      metric].values
    bp_hc = ax1.boxplot(hc_vals, positions=[k*3+1], widths=0.7,
                         patch_artist=True,
                         boxprops=dict(facecolor=C_HC, alpha=0.7),
                         medianprops=dict(color="black", lw=2),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"),
                         flierprops=dict(marker="o", markersize=3,
                                         color=C_HC, alpha=0.4),
                         showfliers=True)
    bp_cd = ax1.boxplot(cd_vals, positions=[k*3+2], widths=0.7,
                         patch_artist=True,
                         boxprops=dict(facecolor=C_CD, alpha=0.7),
                         medianprops=dict(color="black", lw=2),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"),
                         flierprops=dict(marker="o", markersize=3,
                                         color=C_CD, alpha=0.4),
                         showfliers=True)
ax1.set_xticks([1.5, 4.5, 7.5])
ax1.set_xticklabels(["Shannon","Chao1","Simpson"], fontsize=10)
ax1.set_ylabel("Diversity index", fontsize=10)
ax1.set_title("Alpha Diversity\nControl vs Crohn's Disease",
              fontweight="bold", fontsize=11)
ax1.legend(handles=[mpatches.Patch(color=C_HC,label="Control"),
                    mpatches.Patch(color=C_CD,label="CD")],
           fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

# P2 — ROC Curve
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(fpr_c, tpr_c, color=C_CD, lw=2.5, label=f"RF   AUC={auc:.3f}")
ax2.plot(gbt_fpr,gbt_tpr,color=C_HC,lw=2.0,linestyle="--",
         label=f"GBT  AUC={gbt_auc:.3f}")
ax2.plot([0,1],[0,1],color=C_RAND,lw=1.2,linestyle=":",label="Random")
ax2.fill_between(fpr_c,tpr_c,alpha=0.08,color=C_CD)
ax2.set_xlabel("False Positive Rate",fontsize=10)
ax2.set_ylabel("True Positive Rate", fontsize=10)
ax2.set_title("ROC Curve — RF vs GBT",fontweight="bold",fontsize=11)
ax2.legend(fontsize=9,loc="lower right")
ax2.spines[["top","right"]].set_visible(False)

# P3 — PR Curve
ax3 = fig.add_subplot(gs[0,2])
ax3.plot(rec_c,prec_c,color=C_CD,lw=2.5,label=f"RF  AP={ap:.3f}")
ax3.axhline(y_te.mean(),color=C_RAND,lw=1.2,
            linestyle=":",label=f"Baseline={y_te.mean():.2f}")
ax3.fill_between(rec_c,prec_c,alpha=0.08,color=C_CD)
ax3.set_xlabel("Recall",fontsize=10); ax3.set_ylabel("Precision",fontsize=10)
ax3.set_title("Precision-Recall Curve",fontweight="bold",fontsize=11)
ax3.legend(fontsize=9)
ax3.spines[["top","right"]].set_visible(False)

# P4 — Confusion matrix
ax4 = fig.add_subplot(gs[1,0])
cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]*100
sns.heatmap(cm_pct,annot=False,cmap="Blues",ax=ax4,linewidths=1.5,
            linecolor="white",xticklabels=["Pred HC","Pred CD"],
            yticklabels=["True HC","True CD"],
            cbar_kws={"label":"%","shrink":0.7})
for i in range(2):
    for j in range(2):
        n_val=cm[i,j]; pct=cm_pct[i,j]
        col="white" if pct>55 else "black"
        ax4.text(j+0.5,i+0.38,str(n_val),ha="center",va="center",
                 fontsize=16,fontweight="bold",color=col)
        ax4.text(j+0.5,i+0.62,f"({pct:.1f}%)",ha="center",va="center",
                 fontsize=10,color=col)
ax4.set_title(f"Confusion Matrix\n(threshold={opt_thr:.3f})",
              fontweight="bold",fontsize=11)
ax4.set_xlabel("Predicted",fontsize=10); ax4.set_ylabel("Actual",fontsize=10)

# P5 — Top 10 Gini importance
ax5 = fig.add_subplot(gs[1,1])
top_g  = gini_imp.tail(10)
cols5  = [C_CD if v>=top_g.median() else C_HC for v in top_g.values]
b5 = ax5.barh(range(len(top_g)),top_g.values,color=cols5,
               edgecolor="black",linewidth=0.4,alpha=0.88,height=0.65)
ax5.set_yticks(range(len(top_g)))
ax5.set_yticklabels(top_g.index,fontsize=9.5)
for bar,val in zip(b5,top_g.values):
    ax5.text(bar.get_width()+0.0005,bar.get_y()+bar.get_height()/2,
             f"{val:.4f}",va="center",fontsize=9,fontweight="bold")
ax5.set_xlabel("Gini Importance",fontsize=10)
ax5.set_title("Top 10 Taxa — Gini Importance",fontweight="bold",fontsize=11)
ax5.spines[["top","right"]].set_visible(False)

# P6 — Top 10 Permutation importance
ax6 = fig.add_subplot(gs[1,2])
top_p = perm_mean.tail(10); top_ps=perm_std[top_p.index]
cols6 = [C_CD if v>0 else C_RAND for v in top_p.values]
ax6.barh(range(len(top_p)),top_p.values,xerr=top_ps.values,
          color=cols6,edgecolor="black",linewidth=0.4,alpha=0.88,height=0.65,
          error_kw=dict(ecolor="#444",lw=1.2,capsize=3))
ax6.set_yticks(range(len(top_p)))
ax6.set_yticklabels(top_p.index,fontsize=9.5)
ax6.axvline(0,color="black",lw=1.2)
ax6.set_xlabel("AUC decrease (50 repeats)",fontsize=10)
ax6.set_title("Top 10 Taxa — Permutation Importance",
              fontweight="bold",fontsize=11)
ax6.spines[["top","right"]].set_visible(False)

# P7 — Differential abundance volcano
ax7 = fig.add_subplot(gs[2,0])
nlp  = -np.log10(da_df["pvalue"].clip(1e-300))
cols7= np.where((da_df["lfc_clr"]>0)&(da_df["sig"]),C_CD,
        np.where((da_df["lfc_clr"]<0)&(da_df["sig"]),C_HC,C_RAND))
ax7.scatter(da_df["lfc_clr"],nlp,c=cols7,s=60,alpha=0.85,
            edgecolors="white",linewidths=0.5,zorder=3)
ax7.axvline(0,color="black",lw=1.2)
ax7.axhline(-np.log10(0.05),color="gray",lw=1.2,
            linestyle="--",label="p=0.05")
ax7.set_xlabel("CLR difference (CD − Control)",fontsize=10)
ax7.set_ylabel("-log₁₀(p-value)",fontsize=10)
ax7.set_title("Differential Abundance\nVolcano Plot (CLR)",
              fontweight="bold",fontsize=11)
# Label top significant taxa
for _,row in da_df[da_df.sig].head(8).iterrows():
    nlp_val = -np.log10(max(row.pvalue,1e-300))
    ax7.text(row.lfc_clr+0.02, nlp_val+0.05, row.taxa,
             fontsize=7.5, fontweight="bold",
             color=C_CD if row.lfc_clr>0 else C_HC)
ax7.legend(fontsize=9)
ax7.spines[["top","right"]].set_visible(False)

# P8 — Probability distribution
ax8 = fig.add_subplot(gs[2,1])
bins8 = np.linspace(0,1,30)
ax8.hist(y_prob[y_te==0],bins=bins8,color=C_HC,alpha=0.75,
         label=f"Control (n={(y_te==0).sum()})",edgecolor="white",lw=0.4)
ax8.hist(y_prob[y_te==1],bins=bins8,color=C_CD,alpha=0.75,
         label=f"CD (n={(y_te==1).sum()})",edgecolor="white",lw=0.4)
ax8.axvline(opt_thr,color="black",lw=1.8,linestyle="--",
            label=f"Threshold={opt_thr:.3f}")
ax8.set_xlabel("Predicted probability of CD",fontsize=10)
ax8.set_ylabel("Count",fontsize=10)
ax8.set_title("Predicted Probability Distribution",
              fontweight="bold",fontsize=11)
ax8.legend(fontsize=9); ax8.spines[["top","right"]].set_visible(False)

# P9 — Summary table
ax9 = fig.add_subplot(gs[2,2]); ax9.axis("off")
rows9 = [
    ["Dataset",          "IBD Microbiome (Gevers 2014 style)"],
    ["Samples",          f"447 (225 HC, 222 CD)"],
    ["Taxa (features)",  "37 gut genera"],
    ["Normalisation",    "CLR (compositional)"],
    ["Model",            "Random Forest"],
    ["n_estimators",     str(best_params["n_estimators"])],
    ["max_depth",        str(best_params["max_depth"])],
    ["max_features",     str(best_params["max_features"])],
    ["CV AUC (5-fold)",  f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}"],
    ["Test ROC-AUC",     f"{auc:.4f}"],
    ["Avg Precision",    f"{ap:.4f}"],
    ["F1 Score",         f"{f1:.4f}"],
    ["Sensitivity",      f"{sens:.4f}"],
    ["Specificity",      f"{spec:.4f}"],
    ["Sig. DA taxa",     str(da_df.sig.sum())],
    ["Opt. threshold",   f"{opt_thr:.4f} (Youden's J)"],
    ["Top taxon",        perm_mean.sort_values().index[-1]],
]
tbl = ax9.table(cellText=rows9,colLabels=["Metric","Value"],
                cellLoc="left",loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.85,1.72)
for j in range(2):
    tbl[(0,j)].set_facecolor("#2C3E50")
    tbl[(0,j)].set_text_props(color="white",fontweight="bold")
for row,col in {1:"#D5F5E3",10:"#FADBD8",11:"#D6EAF8",
                12:"#FEF9E7",13:"#FDEDEC",14:"#EAF2FF"}.items():
    for j in range(2): tbl[(row,j)].set_facecolor(col)
for i in range(1,len(rows9)+1,2):
    if i not in {1,10,11,12,13,14}:
        for j in range(2): tbl[(i,j)].set_facecolor("#F5F6FA")
ax9.set_title("Model Summary",fontweight="bold",fontsize=11,pad=12)

fig.savefig("outputs/disease_classifier_dashboard.png",dpi=150,
            bbox_inches="tight",facecolor="white")
plt.close(fig)
print("  ✅ Combined dashboard saved")

# ═══════════════════════════════════════════════════════════════
# SEPARATE PANELS
# ═══════════════════════════════════════════════════════════════
print("\n🎨 Separate panels...")

# SEP1 — Permutation importance all taxa
fig,ax = plt.subplots(figsize=(13,11))
sp  = perm_mean.sort_values()
cs  = [C_CD if v>0 else C_RAND for v in sp.values]
bars= ax.barh(range(len(sp)),sp.values,xerr=perm_std[sp.index].values,
               color=cs,edgecolor="black",linewidth=0.5,alpha=0.88,height=0.65,
               error_kw=dict(ecolor="#444",lw=1.5,capsize=4))
ax.set_yticks(range(len(sp)))
ax.set_yticklabels(sp.index,fontsize=11,fontweight="bold")
for bar,(feat,val) in zip(bars,sp.items()):
    std=perm_std[feat]
    ax.text(max(val+std+0.0005,0.0005),bar.get_y()+bar.get_height()/2,
            f"{val:.4f} ± {std:.4f}",va="center",fontsize=9.5,fontweight="bold")
ax.axvline(0,color="black",lw=1.5)
ax.set_xlabel("Mean decrease in ROC-AUC (50 permutation repeats)",fontsize=12)
ax.set_title("Permutation Importance — All 37 Taxa\n"
             "Which gut bacteria best discriminate CD from healthy?\n"
             f"{SUBTITLE}",fontweight="bold",fontsize=13)
ax.legend(handles=[mpatches.Patch(color=C_CD,label="CD-enriched taxa"),
                   mpatches.Patch(color=C_RAND,label="Low/no importance")],
          fontsize=10)
ax.spines[["top","right"]].set_visible(False)
save_panel(fig,"SEP1_permutation_importance")

# SEP2 — Differential abundance barplot
fig,ax = plt.subplots(figsize=(14,10))
da_sig = da_df[da_df.sig].sort_values("lfc_clr")
cols_da= [C_CD if v>0 else C_HC for v in da_sig["lfc_clr"]]
bars2  = ax.barh(range(len(da_sig)),da_sig["lfc_clr"].values,
                  color=cols_da,edgecolor="black",linewidth=0.4,
                  alpha=0.88,height=0.65)
ax.set_yticks(range(len(da_sig)))
ax.set_yticklabels(da_sig["taxa"].values,fontsize=11,fontweight="bold")
for i,lbl in enumerate(ax.get_yticklabels()):
    lbl.set_color(C_CD if da_sig["lfc_clr"].values[i]>0 else C_HC)
for bar,(_,row) in zip(bars2,da_sig.iterrows()):
    x = row.lfc_clr
    padj_str = f"p={row.padj:.3f}"
    ax.text(x+0.02 if x>0 else x-0.02,
            bar.get_y()+bar.get_height()/2,
            padj_str,va="center",
            ha="left" if x>0 else "right",
            fontsize=8,color="#555555")
ax.axvline(0,color="black",lw=1.8)
ax.set_xlabel("CLR difference  (CD − Control)",fontsize=12)
ax.set_title("Differential Abundance — Significant Taxa (FDR<0.05)\n"
             "Red = enriched in CD  ·  Blue = depleted in CD\n"
             f"{SUBTITLE}",fontweight="bold",fontsize=13)
ax.legend(handles=[mpatches.Patch(color=C_CD,label="Higher in CD"),
                   mpatches.Patch(color=C_HC,label="Higher in Control")],
          fontsize=11)
ax.spines[["top","right"]].set_visible(False)
save_panel(fig,"SEP2_differential_abundance")

# SEP3 — ROC + PR
fig,axes = plt.subplots(1,2,figsize=(16,7))
axes[0].plot(fpr_c,tpr_c,color=C_CD,lw=2.5,label=f"RF   AUC={auc:.3f}")
axes[0].plot(gbt_fpr,gbt_tpr,color=C_HC,lw=2.0,linestyle="--",
             label=f"GBT  AUC={gbt_auc:.3f}")
axes[0].plot([0,1],[0,1],color=C_RAND,lw=1.2,linestyle=":",label="Random")
axes[0].fill_between(fpr_c,tpr_c,alpha=0.10,color=C_CD)
axes[0].set_xlabel("False Positive Rate",fontsize=12)
axes[0].set_ylabel("True Positive Rate",fontsize=12)
axes[0].set_title(f"ROC Curve | RF vs GBT\n{SUBTITLE}",
                  fontweight="bold",fontsize=12)
axes[0].legend(fontsize=11,loc="lower right")
axes[0].spines[["top","right"]].set_visible(False)
axes[1].plot(rec_c,prec_c,color=C_CD,lw=2.5,label=f"RF  AP={ap:.3f}")
axes[1].axhline(y_te.mean(),color=C_RAND,lw=1.2,linestyle=":",
                label=f"Baseline={y_te.mean():.2f}")
axes[1].fill_between(rec_c,prec_c,alpha=0.10,color=C_CD)
axes[1].set_xlabel("Recall",fontsize=12); axes[1].set_ylabel("Precision",fontsize=12)
axes[1].set_title(f"Precision-Recall Curve\n{SUBTITLE}",
                  fontweight="bold",fontsize=12)
axes[1].legend(fontsize=11)
axes[1].spines[["top","right"]].set_visible(False)
plt.tight_layout(pad=2.5)
save_panel(fig,"SEP3_ROC_PR_curves")

# SEP4 — Learning curve
fig,ax = plt.subplots(figsize=(12,7))
tr_m=tr_scores.mean(axis=1); tr_s=tr_scores.std(axis=1)
va_m=val_scores.mean(axis=1); va_s=val_scores.std(axis=1)
ax.plot(tr_sizes,tr_m,color=C_CD,lw=2.5,marker="o",markersize=7,
        label="Training AUC")
ax.fill_between(tr_sizes,tr_m-tr_s,tr_m+tr_s,alpha=0.12,color=C_CD)
ax.plot(tr_sizes,va_m,color=C_HC,lw=2.5,marker="s",markersize=7,
        linestyle="--",label="Validation AUC (CV)")
ax.fill_between(tr_sizes,va_m-va_s,va_m+va_s,alpha=0.12,color=C_HC)
ax.set_xlabel("Training samples",fontsize=12)
ax.set_ylabel("ROC-AUC",fontsize=12)
ax.set_title("Learning Curve — Random Forest\n"
             f"{SUBTITLE}",fontweight="bold",fontsize=13)
ax.legend(fontsize=11); ax.spines[["top","right"]].set_visible(False)
save_panel(fig,"SEP4_learning_curve")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("✅  Day 18 COMPLETE")
print(f"{'='*62}")
print(f"  Dashboard        : outputs/disease_classifier_dashboard.png")
print(f"  Separate panels  : outputs/panels/ (4 files)")
print(f"  Predictions CSV  : outputs/predictions.csv")
print(f"  DA results       : outputs/differential_abundance.csv")
print(f"\n  ROC-AUC     : {auc:.4f}")
print(f"  F1 Score    : {f1:.4f}")
print(f"  Sensitivity : {sens:.4f}")
print(f"  Specificity : {spec:.4f}")
print(f"  Top taxon   : {perm_mean.sort_values().index[-1]}")
