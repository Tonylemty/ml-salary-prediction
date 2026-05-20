# 104 人力銀行職缺薪資區間預測系統
> Salary Range Prediction for 104 Job Listings Using Machine Learning

機器學習期末專案 ｜ Week 14 Proposal → Week 16 Final Report

---

## 專案簡介

本專案透過爬取 104 人力銀行的職缺資料，利用機器學習模型（Decision Tree、Random Forest、Logistic Regression）預測職缺的薪資區間（低薪 / 中薪 / 高薪），並建立互動式網站供使用者輸入求職條件即時獲得預測結果。

---

## 資料夾結構

```
ml-salary-prediction/
├── data/
│   ├── raw/              # 爬蟲直接輸出的原始資料（負責人：楊忠諭）
│   └── processed/        # 前處理後的乾淨資料（負責人：蔡東廷）
├── scraper/              # 爬蟲程式碼（負責人：楊忠諭）
├── models/               # 訓練好的模型與編碼器 .pkl 檔（負責人：蔡東廷）
├── notebooks/            # 資料前處理與模型訓練腳本（負責人：蔡東廷）
├── results/              # 評估指標與圖表輸出
├── website/              # 互動式預測網站（負責人：謝傑安）
├── requirements.txt      # Python 套件清單
└── README.md
```

---

## 組員分工

| 組員 | 負責範疇 | 對應資料夾 |
|------|----------|------------|
| 楊忠諭 | 爬蟲撰寫、資料收集、原始資料整理 | `scraper/`、`data/raw/` |
| 蔡東廷 | 資料前處理、特徵工程、模型訓練與評估 | `data/processed/`、`models/`、`notebooks/` |
| 謝傑安 | 互動式網站開發、視覺化圖表、PPT 製作 | `website/` |

---

## 分支策略

| 分支名稱 | 用途 | 負責人 |
|----------|------|--------|
| `main` | 穩定版本 | 全員 |
| `feature/scraper` | 爬蟲開發 | 楊忠諭 |
| `feature/preprocessing` | 資料前處理與特徵工程 | 蔡東廷 |
| `feature/website` | 網站介面開發 | 謝傑安 |

---

## 環境安裝

```bash
git clone https://github.com/Tonylemty/ml-salary-prediction.git
cd ml-salary-prediction
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 啟動網站

```bash
python website/app.py
```

瀏覽器開啟 `http://127.0.0.1:5000`

---

## 薪資分類標準

| 等級 | 月薪範圍（TWD） |
|------|----------------|
| 低薪 | < 40,000 元 |
| 中薪 | 40,000 – 55,000 元 |
| 高薪 | > 55,000 元 |

---

## 使用模型

- Decision Tree（scikit-learn DecisionTreeClassifier）
- Random Forest（scikit-learn RandomForestClassifier）
- Logistic Regression（scikit-learn LogisticRegression）

---

## 時程規劃

| 週次 | 工作項目 |
|------|----------|
| Week 14 | 提交 Proposal、完成爬蟲雛形 |
| Week 14–15 | 完成資料收集與前處理 |
| Week 15 | 完成模型訓練與評估 |
| Week 15 後半 | 完成網站與視覺化圖表 |
| Week 16 | 口頭報告與繳交所有項目 |
