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
│   ├── raw/              # 爬蟲直接輸出的原始資料（負責人：組員 A）
│   └── processed/        # 前處理後的乾淨資料（負責人：組員 B）
├── scraper/              # 爬蟲程式碼（負責人：組員 A）
├── models/               # WEKA 模型檔與訓練結果（負責人：組員 B）
├── notebooks/            # 資料分析、視覺化 Jupyter Notebook（負責人：組員 B）
├── website/              # 互動式預測網站（負責人：組員 C）
├── requirements.txt      # Python 套件清單
└── README.md
```

---

## 組員分工

| 組員 | 負責範疇 | 對應資料夾 |
|------|----------|------------|
| 組員 A（請填入姓名） | 爬蟲撰寫、資料收集、原始資料整理 | `scraper/`、`data/raw/` |
| 組員 B（請填入姓名） | 資料前處理、特徵工程、模型訓練與評估 | `data/processed/`、`models/`、`notebooks/` |
| 組員 C（請填入姓名） | 互動式網站開發、視覺化圖表、PPT 製作 | `website/` |

---

## 分支策略

| 分支名稱 | 用途 | 負責人 |
|----------|------|--------|
| `main` | 穩定版本，只接受 Pull Request 合併 | 全員 |
| `feature/scraper` | 爬蟲開發 | 組員 A |
| `feature/preprocessing` | 資料前處理與特徵工程 | 組員 B |
| `feature/website` | 網站介面開發 | 組員 C |

### 工作流程

```
# 1. 切換到自己的分支
git checkout feature/scraper   # 各自換成自己的分支名稱

# 2. 開發完一個功能後 commit
git add .
git commit -m "功能描述"

# 3. push 到遠端
git push origin feature/scraper

# 4. 到 GitHub 開 Pull Request，通知其他組員 review 後合併到 main
```

> **注意**：請勿直接 push 到 `main`，一律透過 Pull Request 合併。

---

## 環境安裝

```bash
git clone https://github.com/github帳號/ml-salary-prediction.git
cd ml-salary-prediction
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 薪資分類標準

| 等級 | 月薪範圍（TWD） |
|------|----------------|
| 低薪 | < 30,000 元 |
| 中薪 | 30,000 – 50,000 元 |
| 高薪 | > 50,000 元 |

---

## 使用模型

- Decision Tree（J48）
- Random Forest
- Logistic Regression

---

## 時程規劃

| 週次 | 工作項目 |
|------|----------|
| Week 14 | 提交 Proposal、完成爬蟲雛形 |
| Week 14–15 | 完成資料收集與前處理 |
| Week 15 | 完成模型訓練與評估 |
| Week 15 後半 | 完成網站與視覺化圖表 |
| Week 16 | 口頭報告與繳交所有項目 |
