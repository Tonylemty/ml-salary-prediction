import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# ── 字型設定（支援中文）────────────────────────────────────────────────────────
font_candidates = [
    "Microsoft JhengHei", "Microsoft YaHei",
    "Arial Unicode MS", "DejaVu Sans",
]
available = {f.name for f in fm.fontManager.ttflist}
chosen_font = next((f for f in font_candidates if f in available), None)
if chosen_font:
    plt.rcParams["font.family"] = chosen_font
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("results/figures", exist_ok=True)

# ── 讀入資料 ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/processed_data.csv", encoding="utf-8-sig")
FEATURES = ["工作地區", "公司規模", "產業類別", "職務類別", "需求技能數量", "學歷要求"]
TARGET   = "薪資區間"
CLASSES  = [0, 1, 2]
CLASS_NAMES = ["低薪", "中薪", "高薪"]

X = df[FEATURES].values
y = df[TARGET].values

# ── 模型定義 ──────────────────────────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=None,
        class_weight="balanced", random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=1e8, max_iter=100,
        class_weight="balanced", random_state=42
    ),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ── 訓練與評估 ────────────────────────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"模型：{name}")

    y_pred      = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob      = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    y_bin       = label_binarize(y, classes=CLASSES)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
    auc_score = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC       : {auc_score:.4f}")

    results[name] = {
        "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-score": f1, "AUC": auc_score,
        "y_pred": y_pred, "y_prob": y_prob,
    }

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(3), yticks=range(3),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="預測標籤", ylabel="實際標籤",
        title=f"{name} — Confusion Matrix"
    )
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fname = name.replace(" ", "_")
    plt.savefig(f"results/figures/confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"  Confusion Matrix 已儲存")

# ── ROC Curve 比較圖 ──────────────────────────────────────────────────────────
colors = ["steelblue", "darkorange", "green"]
fig, ax = plt.subplots(figsize=(8, 6))
y_bin = label_binarize(y, classes=CLASSES)

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, roc_auc_val = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], res["y_prob"][:, i])
        roc_auc_val[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    mean_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=color, lw=2,
            label=f"{name} (AUC = {mean_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlim=[0, 1], ylim=[0, 1.05],
       xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curve 三模型比較")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/figures/roc_curve_comparison.png", dpi=150)
plt.close()
print("\nROC Curve 比較圖已儲存")

# ── 薪資分佈長條圖 ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = pd.Series(y).map({0: "低薪", 1: "中薪", 2: "高薪"}).value_counts()
counts = counts.reindex(["低薪", "中薪", "高薪"])
bars = ax.bar(counts.index, counts.values, color=["#4C9BE8", "#F0A500", "#E84C4C"])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(val), ha="center", va="bottom")
ax.set(title="薪資區間分佈", xlabel="薪資區間", ylabel="筆數")
plt.tight_layout()
plt.savefig("results/figures/salary_distribution.png", dpi=150)
plt.close()
print("薪資分佈圖已儲存")

# ── Feature Importance 橫條圖 ─────────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
importance = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
importance.plot(kind="barh", ax=ax, color="steelblue")
ax.set(title="Feature Importance（Random Forest）", xlabel="Importance")
plt.tight_layout()
plt.savefig("results/figures/feature_importance.png", dpi=150)
plt.close()
print("Feature Importance 圖已儲存")

# ── 模型比較長條圖 ────────────────────────────────────────────────────────────
model_names = list(results.keys())
acc_scores = [results[m]["Accuracy"] for m in model_names]
f1_scores  = [results[m]["F1-score"] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, acc_scores, width, label="Accuracy", color="steelblue")
bars2 = ax.bar(x + width/2, f1_scores,  width, label="F1-score", color="darkorange")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set(xticks=x, xticklabels=model_names, ylim=[0, 1.1],
       title="模型 Accuracy / F1-score 比較", ylabel="Score")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/model_comparison.png", dpi=150)
plt.close()
print("模型比較圖已儲存")

# ── 匯出結果表格 ──────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    m: {k: v for k, v in res.items() if k not in ["y_pred", "y_prob"]}
    for m, res in results.items()
}).T.round(4)
summary.to_csv("results/model_results.csv", encoding="utf-8-sig")
print(f"\n結果表格已儲存到 results/model_results.csv")
print(f"\n{'='*50}")
print(summary.to_string())
