# Day 18 — Microbiome-based Disease Classifier (Random Forest)
### 🧬 30 Days × 30 Unique Projects | Subhadip Jana

**Dataset:** IBD gut microbiome — Gevers 2014 / HMP2 style  
**Task:** Classify Crohn's Disease vs Healthy Control from 16S OTU profiles  
**Model:** Random Forest + CLR normalisation + permutation importance

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Dataset | IBD Microbiome (Gevers 2014 style) |
| Samples | 447 (225 HC, 222 CD) |
| Taxa (features) | 37 gut genera |
| Normalisation | CLR (compositional-aware) |
| Model | Random Forest |
| CV ROC-AUC (5-fold) | 1.0000 ± 0.0000 |
| **Test ROC-AUC** | **1.0000** |
| **F1 Score** | **1.0000** |
| **Sensitivity** | **1.0000** |
| **Specificity** | **1.0000** |
| Significant DA taxa | 36 / 37 |
| Top discriminating taxon | Fusobacterium nucleatum |

---

## 🔬 Key Biological Findings

- **Fusobacterium nucleatum** — strongest classifier feature (enriched in CD), consistent with gut inflammation
- **Escherichia coli / Klebsiella** — Proteobacteria bloom in IBD (well-established)
- **Faecalibacterium prausnitzii** — depleted in CD (canonical anti-inflammatory biomarker)
- **Roseburia / Ruminococcus** — butyrate producers depleted in CD
- Alpha diversity: no significant difference (Shannon p=0.71) — classification driven by compositional shifts, not diversity per se

---

## 🖼️ Output Files

| File | Description |
|------|-------------|
| `outputs/disease_classifier_dashboard.png` | Combined 3×3 dashboard |
| `outputs/panels/SEP1_permutation_importance.png` | All 37 taxa — permutation importance |
| `outputs/panels/SEP2_differential_abundance.png` | Sig. DA taxa barplot |
| `outputs/panels/SEP3_ROC_PR_curves.png` | ROC + PR (RF vs GBT) |
| `outputs/panels/SEP4_learning_curve.png` | Learning curve |
| `outputs/predictions.csv` | Test set predictions |
| `outputs/differential_abundance.csv` | DA results (all taxa) |

---

## 🚀 How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy
python3 generate_data.py   # creates data/
python3 disease_classifier.py
```

---

## 📁 Project Structure

```
day18-disease-classifier/
├── generate_data.py
├── disease_classifier.py
├── README.md
├── data/
│   ├── otu_table.csv
│   └── metadata.csv
└── outputs/
    ├── disease_classifier_dashboard.png
    ├── predictions.csv
    ├── differential_abundance.csv
    └── panels/
        ├── SEP1_permutation_importance.png
        ├── SEP2_differential_abundance.png
        ├── SEP3_ROC_PR_curves.png
        └── SEP4_learning_curve.png
```

---

**#30DaysOfBioinformatics | Author: Subhadip Jana**
[GitHub](https://github.com/SubhadipJana1409) | [LinkedIn](https://linkedin.com/in/subhadip-jana1409)
