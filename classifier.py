"""
================================================================
Day 18 — Microbiome-based Disease Classifier (Random Forest)
Author  : Subhadip Jana | #30DaysOfBioinformatics
Dataset : Gevers et al. 2014, Cell Host & Microbe
          IBD (Crohn's Disease) vs Healthy Control
          447 samples | 37 OTUs | Real published taxa
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
from scipy.spatial.distance import braycurtis
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV, learning_curve)
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              average_precision_score, precision_recall_curve, f1_score)
from sklearn.inspection import permutation_importance
import warnings, os
warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("outputs/panels", exist_ok=True)

print("="*62)
print("Day 18 — Microbiome Disease Classifier | IBD vs Control")
print("="*62)

# ── Load ──────────────────────────────────────────────────────
otu  = pd.read_csv("data/otu_table.csv",  index_col=0)
meta = pd.read_csv("data/metadata.csv").set_index("SampleID")
meta["label"] = (meta["diagnosis"] == "CD").astype(int)
otu  = otu.loc[meta.index]

def genus_label(col):
    for p in reversed(col.split(";")):
        g = p.replace("g__","").replace("s__","").strip()
        if g and g not in ["","uncultured","unidentified"]: return g[:18]
    return col
FEAT_LABELS = [genus_label(c) for c in otu.columns]

print(f"\n✅ {otu.shape[0]} samples x {otu.shape[1]} OTUs")
print(f"   Control={( meta.diagnosis=='Control').sum()}  CD={( meta.diagnosis=='CD').sum()}")

# ── Features ─────────────────────────────────────────────────
rel     = otu.div(otu.sum(axis=1), axis=0)
pseudo  = 0.5
geo_m   = np.exp(np.log(otu + pseudo).mean(axis=1))
clr_mat = np.log((otu + pseudo).div(geo_m, axis=0))
clr_df  = pd.DataFrame(clr_mat.values, index=otu.index, columns=FEAT_LABELS)

def shannon(r): p=r[r>0]/r.sum(); return -np.sum(p*np.log(p))
def simpson(r): p=r/r.sum();       return 1-np.sum(p**2)
def chao1(r):
    obs=(r>0).sum(); f1=(r==1).sum(); f2=(r==2).sum()
    return obs+f1**2/(2*f2+1) if f2>0 else float(obs+f1)

alpha = pd.DataFrame({
    "Shannon":otu.apply(shannon,axis=1), "Simpson":otu.apply(simpson,axis=1),
    "Chao1":otu.apply(chao1,axis=1),     "N_OTUs":(otu>0).sum(axis=1),
    "SeqDepth":otu.sum(axis=1)}, index=otu.index)

X_full = pd.concat([clr_df, alpha], axis=1)
y      = meta["label"].values
print(f"   Total features: {X_full.shape[1]}  (CLR={clr_df.shape[1]} + alpha=5)")

# ── PCoA ─────────────────────────────────────────────────────
print("\n🔍 PCoA (Bray-Curtis)...")
n=len(rel); arr=rel.values; D=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n): D[i,j]=D[j,i]=braycurtis(arr[i],arr[j])
H=np.eye(n)-np.ones((n,n))/n; B=-0.5*H@(D**2)@H
ev,ec=np.linalg.eigh(B); idx=np.argsort(ev)[::-1]; ev,ec=ev[idx],ec[:,idx]
pm=ev>0; coords=ec[:,pm]*np.sqrt(np.abs(ev[pm])); varexp=ev[pm]/ev[pm].sum()*100
print(f"   PC1={varexp[0]:.1f}%  PC2={varexp[1]:.1f}%")

# ── Random Forest ────────────────────────────────────────────
print("\n🌲 Training Random Forest (GridSearchCV)...")
X=X_full.values
X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
param_grid={"n_estimators":[200,300,500],"max_depth":[None,5,10],
            "max_features":["sqrt","log2"],"min_samples_split":[2,5]}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
grid=GridSearchCV(RandomForestClassifier(random_state=42,class_weight="balanced"),
                  param_grid,cv=cv,scoring="roc_auc",n_jobs=-1)
grid.fit(X_tr,y_tr); model=grid.best_estimator_
cv_auc=cross_val_score(model,X_tr,y_tr,cv=cv,scoring="roc_auc")
cv_f1 =cross_val_score(model,X_tr,y_tr,cv=cv,scoring="f1")
print(f"   Best: {grid.best_params_}")
print(f"   CV AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}  F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}")

# ── Evaluation ───────────────────────────────────────────────
y_prob=model.predict_proba(X_te)[:,1]
fpr_c,tpr_c,thrs=roc_curve(y_te,y_prob)
opt_thr=thrs[np.argmax(tpr_c-fpr_c)]; y_pred=(y_prob>=opt_thr).astype(int)
auc=roc_auc_score(y_te,y_prob); ap=average_precision_score(y_te,y_prob)
f1=f1_score(y_te,y_pred); cm=confusion_matrix(y_te,y_pred)
tn,fp,fn,tp=cm.ravel(); sens=tp/(tp+fn); spec=tn/(tn+fp)
ppv=tp/(tp+fp) if tp+fp>0 else 0; npv=tn/(tn+fn) if tn+fn>0 else 0
prec_c,rec_c,_=precision_recall_curve(y_te,y_prob)
print(f"\n📊 Test: AUC={auc:.4f}  F1={f1:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

# ── Feature importance ───────────────────────────────────────
all_feats=list(X_full.columns)
gini_s=pd.Series(model.feature_importances_,index=all_feats).sort_values()
perm=permutation_importance(model,X_te,y_te,n_repeats=50,random_state=42,scoring="roc_auc")
perm_m=pd.Series(perm.importances_mean,index=all_feats).sort_values()
perm_s=pd.Series(perm.importances_std, index=all_feats)
print(f"\n   Top 5 (permutation):")
for f in perm_m.sort_values(ascending=False).index[:5]:
    print(f"   {f:25s}: {perm_m[f]:.4f}±{perm_s[f]:.4f}")

gbt=GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,random_state=42)
gbt.fit(X_tr,y_tr); gbt_prob=gbt.predict_proba(X_te)[:,1]
gbt_auc=roc_auc_score(y_te,gbt_prob); gbt_fpr,gbt_tpr,_=roc_curve(y_te,gbt_prob)

tr_sz,tr_sc,va_sc=learning_curve(model,X,y,cv=5,scoring="roc_auc",
                                  train_sizes=np.linspace(0.1,1.0,8),random_state=42)

ctrl_alpha=alpha[meta.diagnosis=="Control"]; cd_alpha=alpha[meta.diagnosis=="CD"]
_,pval_sh=mannwhitneyu(ctrl_alpha["Shannon"],cd_alpha["Shannon"])

lfc_dict={}
for col,lbl in zip(otu.columns,FEAT_LABELS):
    cm_v=rel[meta.diagnosis=="Control"][col].mean()+1e-6
    cd_v=rel[meta.diagnosis=="CD"][col].mean()+1e-6
    lfc_dict[lbl]=np.log2(cd_v/cm_v)
lfc_ser=pd.Series(lfc_dict).sort_values()

pd.DataFrame({"true":y_te,"pred":y_pred,"prob":y_prob}).to_csv("outputs/predictions.csv",index=False)
pd.DataFrame({"feature":all_feats,"gini":model.feature_importances_,"permutation":perm_m.values}
             ).sort_values("permutation",ascending=False).to_csv("outputs/feature_importance.csv",index=False)

C_CTRL="#3498DB"; C_IBD="#E74C3C"; C_RAND="#95A5A6"
SUBTITLE="IBD (Crohn's) vs Control | Gevers 2014 | 447 samples"

def save_panel(fig,name):
    fig.savefig(f"outputs/panels/{name}.png",dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  ✅ {name}.png")

# ════════════════════════════════════════════════════════════
# COMBINED DASHBOARD  3x3
# ════════════════════════════════════════════════════════════
print("\n🎨 Combined dashboard...")
fig=plt.figure(figsize=(27,23))
fig.suptitle(f"Day 18 — Microbiome Disease Classifier  |  Random Forest\n{SUBTITLE}\n"
             f"37 OTUs · CLR normalised · ROC-AUC={auc:.3f} · F1={f1:.3f} · "
             f"Sensitivity={sens:.3f} · Specificity={spec:.3f}",
             fontsize=14,fontweight="bold",y=0.998)
gs=GridSpec(3,3,figure=fig,hspace=0.44,wspace=0.40,left=0.07,right=0.96,top=0.95,bottom=0.05)

# P1 PCoA
ax1=fig.add_subplot(gs[0,0])
for lbl,col,name in [(0,C_CTRL,"Control"),(1,C_IBD,"CD (IBD)")]:
    idx=y==lbl
    ax1.scatter(coords[idx,0],coords[idx,1],c=col,s=22,alpha=0.60,linewidths=0,
                label=f"{name} (n={idx.sum()})")
ax1.set_xlabel(f"PC1 ({varexp[0]:.1f}%)",fontsize=10); ax1.set_ylabel(f"PC2 ({varexp[1]:.1f}%)",fontsize=10)
ax1.set_title("PCoA — Bray-Curtis\nMicrobiome separation",fontweight="bold",fontsize=11)
ax1.legend(fontsize=9); ax1.spines[["top","right"]].set_visible(False)

# P2 Shannon violin
ax2=fig.add_subplot(gs[0,1])
ctrl_sh=ctrl_alpha["Shannon"].values; cd_sh=cd_alpha["Shannon"].values
parts=ax2.violinplot([ctrl_sh,cd_sh],positions=[0,1],showmedians=True)
for pc,col in zip(parts["bodies"],[C_CTRL,C_IBD]): pc.set_facecolor(col); pc.set_alpha(0.6)
ax2.set_xticks([0,1]); ax2.set_xticklabels(["Control","CD/IBD"],fontsize=11)
for i,(data,col) in enumerate([(ctrl_sh,C_CTRL),(cd_sh,C_IBD)]):
    ax2.scatter(np.full(len(data),i)+np.random.uniform(-0.08,0.08,len(data)),
                data,c=col,s=8,alpha=0.5,zorder=3)
ax2.set_ylabel("Shannon Diversity",fontsize=10)
ax2.set_title(f"Alpha Diversity (Shannon)\np={pval_sh:.2e}",fontweight="bold",fontsize=11)
ax2.spines[["top","right"]].set_visible(False)

# P3 log2FC bar
ax3=fig.add_subplot(gs[0,2])
plot_lfc=pd.concat([lfc_ser.head(5),lfc_ser.tail(5)])
cols3=[C_IBD if v>0 else C_CTRL for v in plot_lfc.values]
ax3.barh(range(len(plot_lfc)),plot_lfc.values,color=cols3,
         edgecolor="black",linewidth=0.4,alpha=0.88,height=0.65)
ax3.set_yticks(range(len(plot_lfc))); ax3.set_yticklabels(plot_lfc.index,fontsize=10,fontweight="bold")
ax3.axvline(0,color="black",lw=1.5)
ax3.set_xlabel("log₂FC (CD / Control)",fontsize=10)
ax3.set_title("Top Discriminating Taxa\nRed=↑CD  Blue=↑Control",fontweight="bold",fontsize=11)
ax3.spines[["top","right"]].set_visible(False)

# P4 Confusion matrix
ax4=fig.add_subplot(gs[1,0])
cm_pct=cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]*100
sns.heatmap(cm_pct,annot=False,cmap="Blues",ax=ax4,linewidths=1.5,linecolor="white",
            xticklabels=["Pred Ctrl","Pred CD"],yticklabels=["True Ctrl","True CD"],
            cbar_kws={"label":"%","shrink":0.7})
for i in range(2):
    for j in range(2):
        n=cm[i,j]; pct=cm_pct[i,j]; col="white" if pct>55 else "black"
        ax4.text(j+0.5,i+0.38,str(n),ha="center",va="center",fontsize=16,fontweight="bold",color=col)
        ax4.text(j+0.5,i+0.62,f"({pct:.1f}%)",ha="center",va="center",fontsize=10,color=col)
ax4.set_title(f"Confusion Matrix (thr={opt_thr:.3f})",fontweight="bold",fontsize=11)
ax4.set_xlabel("Predicted",fontsize=10); ax4.set_ylabel("Actual",fontsize=10)

# P5 ROC
ax5=fig.add_subplot(gs[1,1])
ax5.plot(fpr_c,tpr_c,color=C_IBD,lw=2.5,label=f"RF   AUC={auc:.3f}")
ax5.plot(gbt_fpr,gbt_tpr,color=C_CTRL,lw=2.0,linestyle="--",label=f"GBT  AUC={gbt_auc:.3f}")
ax5.plot([0,1],[0,1],color=C_RAND,lw=1.2,linestyle=":",label="Random")
ax5.fill_between(fpr_c,tpr_c,alpha=0.08,color=C_IBD)
ax5.set_xlabel("FPR",fontsize=10); ax5.set_ylabel("TPR",fontsize=10)
ax5.set_title("ROC Curve — RF vs GBT",fontweight="bold",fontsize=11)
ax5.legend(fontsize=9,loc="lower right"); ax5.spines[["top","right"]].set_visible(False)

# P6 Gini top 10
ax6=fig.add_subplot(gs[1,2])
top_g=gini_s.tail(10)
cols6=["#E74C3C" if v>=top_g.median() else "#3498DB" for v in top_g.values]
b6=ax6.barh(range(len(top_g)),top_g.values,color=cols6,edgecolor="black",
             linewidth=0.4,alpha=0.88,height=0.65)
ax6.set_yticks(range(len(top_g))); ax6.set_yticklabels(top_g.index,fontsize=10)
for bar,val in zip(b6,top_g.values):
    ax6.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,
             f"{val:.3f}",va="center",fontsize=9,fontweight="bold")
ax6.set_xlabel("Gini Importance",fontsize=10)
ax6.set_title("Top 10 — Gini Importance",fontweight="bold",fontsize=11)
ax6.spines[["top","right"]].set_visible(False)

# P7 PR
ax7=fig.add_subplot(gs[2,0])
ax7.plot(rec_c,prec_c,color=C_IBD,lw=2.5,label=f"RF  AP={ap:.3f}")
ax7.axhline(y_te.mean(),color=C_RAND,lw=1.2,linestyle=":",label=f"Baseline={y_te.mean():.2f}")
ax7.fill_between(rec_c,prec_c,alpha=0.08,color=C_IBD)
ax7.set_xlabel("Recall",fontsize=10); ax7.set_ylabel("Precision",fontsize=10)
ax7.set_title("Precision-Recall Curve",fontweight="bold",fontsize=11)
ax7.legend(fontsize=9); ax7.spines[["top","right"]].set_visible(False)

# P8 Prob dist
ax8=fig.add_subplot(gs[2,1]); bins8=np.linspace(0,1,30)
ax8.hist(y_prob[y_te==0],bins=bins8,color=C_CTRL,alpha=0.75,edgecolor="white",
         linewidth=0.4,label=f"Control (n={(y_te==0).sum()})")
ax8.hist(y_prob[y_te==1],bins=bins8,color=C_IBD,alpha=0.75,edgecolor="white",
         linewidth=0.4,label=f"CD/IBD (n={(y_te==1).sum()})")
ax8.axvline(opt_thr,color="black",lw=1.8,linestyle="--",label=f"Thr={opt_thr:.3f}")
ax8.set_xlabel("Predicted probability of CD/IBD",fontsize=10)
ax8.set_ylabel("Count",fontsize=10)
ax8.set_title("Predicted Probability Distribution",fontweight="bold",fontsize=11)
ax8.legend(fontsize=9); ax8.spines[["top","right"]].set_visible(False)

# P9 Summary
ax9=fig.add_subplot(gs[2,2]); ax9.axis("off")
rows=[
    ["Dataset",        "Gevers 2014, Cell Host Microbe"],
    ["Samples",        "447  (Control=225, CD=222)"],
    ["OTUs",           "37 curated IBD signature taxa"],
    ["Normalisation",  "CLR  (centred log-ratio)"],
    ["Features",       "42  (37 CLR + 5 alpha diversity)"],
    ["Model",          "Random Forest (balanced weights)"],
    ["n_estimators",   str(grid.best_params_["n_estimators"])],
    ["max_depth",      str(grid.best_params_["max_depth"])],
    ["CV AUC (5-fold)",f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}"],
    ["Test ROC-AUC",   f"{auc:.4f}"],
    ["Avg Precision",  f"{ap:.4f}"],
    ["F1 Score",       f"{f1:.4f}"],
    ["Sensitivity",    f"{sens:.4f}"],
    ["Specificity",    f"{spec:.4f}"],
    ["Threshold",      f"{opt_thr:.4f}  (Youden's J)"],
    ["Top OTU",        perm_m.sort_values().index[-1]],
    ["Shannon p-val",  f"{pval_sh:.2e}  (Mann-Whitney U)"],
]
tbl=ax9.table(cellText=rows,colLabels=["Metric","Value"],cellLoc="left",loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.9,1.68)
for j in range(2):
    tbl[(0,j)].set_facecolor("#2C3E50")
    tbl[(0,j)].set_text_props(color="white",fontweight="bold")
for row,col in {1:"#D5F5E3",10:"#FADBD8",11:"#D6EAF8",12:"#FEF9E7",13:"#FDEDEC",14:"#EAF2FF"}.items():
    for j in range(2): tbl[(row,j)].set_facecolor(col)
for i in range(1,len(rows)+1,2):
    if i not in {1,10,11,12,13,14}:
        for j in range(2): tbl[(i,j)].set_facecolor("#F5F6FA")
ax9.set_title("Model Summary",fontweight="bold",fontsize=11,pad=12)

fig.savefig("outputs/disease_classifier_dashboard.png",dpi=150,bbox_inches="tight",facecolor="white")
plt.close(fig); print("  ✅ Combined dashboard saved")

# ════════════════════════════════════════════════════════════
# SEPARATE PANELS
# ════════════════════════════════════════════════════════════
print("\n🎨 Separate panels...")

# SEP1 — Permutation importance all features
fig,ax=plt.subplots(figsize=(13,12))
sp=perm_m.sort_values(); cols_s=["#E74C3C" if v>0 else C_RAND for v in sp.values]
bars=ax.barh(range(len(sp)),sp.values,xerr=perm_s[sp.index].values,color=cols_s,
              edgecolor="black",linewidth=0.5,alpha=0.88,height=0.65,
              error_kw=dict(ecolor="#444",lw=1.5,capsize=4))
ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index,fontsize=10,fontweight="bold")
for bar,(feat,val) in zip(bars,sp.items()):
    std=perm_s[feat]
    ax.text(max(val+std+0.001,0.001),bar.get_y()+bar.get_height()/2,
            f"{val:.4f}",va="center",fontsize=8.5,fontweight="bold")
ax.axvline(0,color="black",lw=1.5)
ax.set_xlabel("Mean decrease in ROC-AUC (50 permutation repeats)",fontsize=12)
ax.set_title("Permutation Feature Importance — All 42 Features\n"
             "Which OTUs / diversity metrics best classify IBD vs Control?\n"
             f"{SUBTITLE}",fontweight="bold",fontsize=13)
ax.spines[["top","right"]].set_visible(False)
save_panel(fig,"SEP1_permutation_importance")

# SEP2 — ROC + PR
fig,axes=plt.subplots(1,2,figsize=(16,7))
axes[0].plot(fpr_c,tpr_c,color=C_IBD,lw=2.5,label=f"RF   AUC={auc:.3f}")
axes[0].plot(gbt_fpr,gbt_tpr,color=C_CTRL,lw=2.0,linestyle="--",label=f"GBT  AUC={gbt_auc:.3f}")
axes[0].plot([0,1],[0,1],color=C_RAND,lw=1.2,linestyle=":")
axes[0].fill_between(fpr_c,tpr_c,alpha=0.10,color=C_IBD)
axes[0].set_xlabel("FPR",fontsize=12); axes[0].set_ylabel("TPR",fontsize=12)
axes[0].set_title(f"ROC Curve | {SUBTITLE}",fontweight="bold",fontsize=12)
axes[0].legend(fontsize=11,loc="lower right"); axes[0].spines[["top","right"]].set_visible(False)
axes[1].plot(rec_c,prec_c,color=C_IBD,lw=2.5,label=f"RF  AP={ap:.3f}")
axes[1].axhline(y_te.mean(),color=C_RAND,lw=1.2,linestyle=":")
axes[1].fill_between(rec_c,prec_c,alpha=0.10,color=C_IBD)
axes[1].set_xlabel("Recall",fontsize=12); axes[1].set_ylabel("Precision",fontsize=12)
axes[1].set_title(f"Precision-Recall | {SUBTITLE}",fontweight="bold",fontsize=12)
axes[1].legend(fontsize=11); axes[1].spines[["top","right"]].set_visible(False)
plt.tight_layout(pad=2.5); save_panel(fig,"SEP2_ROC_PR_curves")

# SEP3 — Alpha diversity all 5 metrics
fig,axes=plt.subplots(1,5,figsize=(22,7))
for metric,ax in zip(alpha.columns,axes):
    cv2=ctrl_alpha[metric].values; id2=cd_alpha[metric].values
    _,pv=mannwhitneyu(cv2,id2)
    parts=ax.violinplot([cv2,id2],positions=[0,1],showmedians=True)
    for pc,col in zip(parts["bodies"],[C_CTRL,C_IBD]): pc.set_facecolor(col); pc.set_alpha(0.65)
    for ii,(data,col) in enumerate([(cv2,C_CTRL),(id2,C_IBD)]):
        ax.scatter(np.full(len(data),ii)+np.random.uniform(-0.08,0.08,len(data)),
                   data,c=col,s=6,alpha=0.5,zorder=3)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Ctrl","CD"],fontsize=10)
    ax.set_title(f"{metric}\np={pv:.2e}",fontweight="bold",fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"Alpha Diversity — All 5 Metrics | {SUBTITLE}",
             fontweight="bold",fontsize=13,y=1.02)
plt.tight_layout(pad=2.0); save_panel(fig,"SEP3_alpha_diversity")

# SEP4 — Learning curve
fig,ax=plt.subplots(figsize=(12,7))
tr_m=tr_sc.mean(axis=1); tr_s=tr_sc.std(axis=1)
va_m=va_sc.mean(axis=1); va_s=va_sc.std(axis=1)
ax.plot(tr_sz,tr_m,color=C_IBD,lw=2.5,marker="o",markersize=7,label="Training AUC")
ax.fill_between(tr_sz,tr_m-tr_s,tr_m+tr_s,alpha=0.12,color=C_IBD)
ax.plot(tr_sz,va_m,color=C_CTRL,lw=2.5,marker="s",markersize=7,
        linestyle="--",label="Validation AUC (5-fold CV)")
ax.fill_between(tr_sz,va_m-va_s,va_m+va_s,alpha=0.12,color=C_CTRL)
ax.set_xlabel("Training samples",fontsize=12); ax.set_ylabel("ROC-AUC",fontsize=12)
ax.set_title(f"Learning Curve — Random Forest\n{SUBTITLE}",fontweight="bold",fontsize=13)
ax.legend(fontsize=11); ax.spines[["top","right"]].set_visible(False)
save_panel(fig,"SEP4_learning_curve")

print(f"\n{'='*62}")
print("✅  Day 18 COMPLETE")
print(f"{'='*62}")
print(f"  Dashboard       : outputs/disease_classifier_dashboard.png")
print(f"  Separate panels : outputs/panels/ (4 files)")
print(f"  Predictions     : outputs/predictions.csv")
print(f"\n  ROC-AUC     : {auc:.4f}")
print(f"  F1 Score    : {f1:.4f}")
print(f"  Sensitivity : {sens:.4f}")
print(f"  Specificity : {spec:.4f}")
print(f"  Top feature : {perm_m.sort_values().index[-1]}")
