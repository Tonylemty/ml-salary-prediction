# 104 人力銀行職缺薪資區間預測系統 — 技術說明文件

本文件詳細介紹專案中每個重要檔案的邏輯、技術與方法，供開發者理解整體架構與實作細節。

---

## 專案資料流程

```
104.com.tw API
      ↓
scraper/scraper.py          → scraper/data/raw/raw_data.csv
      ↓
notebooks/preprocessing.py  → data/processed/processed_data.csv
                             → models/encoder_*.pkl
                             → models/scaler_需求技能數量.pkl
      ↓
notebooks/train_models.py   → models/*.pkl
                             → results/figures/*.png
                             → results/model_results.csv
      ↓
website/app.py              ← 載入所有 .pkl 檔案
      ↓
website/templates/index.html → 使用者互動介面
```

---

## 1. `scraper/scraper.py` — 爬蟲資料收集

### 功能
透過 104 人力銀行的非公開 JSON API 批次爬取職缺資料，涵蓋 20 種職務關鍵字，每個關鍵字爬取 8 頁（共約 160 筆），最終去重後存為 CSV。

### 技術方法
- **HTTP 請求**：使用 `requests` 發送 GET 請求，設定 `User-Agent` 與 `Referer` Header 模擬瀏覽器行為，避免被伺服器拒絕
- **API 端點**：`https://www.104.com.tw/jobs/search/api/jobs`，透過 `keyword`、`page`、`pagesize` 等參數控制查詢
- **防爬蟲處理**：每頁請求後隨機延遲 1.0–2.0 秒（`time.sleep(random.uniform(1.0, 2.0))`），避免請求過快被封鎖

### 資料解析邏輯（`parse_job`）
| 欄位 | 來源 API 欄位 | 處理方式 |
|------|-------------|---------|
| 薪資 | `salaryLow`, `salaryHigh` | 若 `salaryHigh == 9999999`（面議）則取 `salaryLow`；否則取中位數 `(low + high) // 2` |
| 學歷要求 | `optionEdu`（代碼列表） | 取最高學歷代碼，對應 `EDU_MAP` 轉為文字 |
| 工作經驗 | `s9`（代碼） | 對應 `EXP_MAP` 轉為文字 |
| 需求技能數量 | `pcSkills`（技能列表） | 計算列表長度 `len(pcSkills)` |
| 職務類別 | — | 直接使用搜尋關鍵字（如「工程師」、「行銷」） |

### 關鍵設計決策
- **面議過濾**：`salaryLow == 0 AND salaryHigh == 0` 的職缺直接跳過，不納入資料集，因為無法轉為數值標籤
- **去重**：所有關鍵字爬完後執行 `df.drop_duplicates()`，避免同一職缺被多個關鍵字重複收錄

---

## 2. `notebooks/preprocessing.py` — 資料前處理

### 功能
將爬蟲原始資料清洗、轉換、編碼，輸出可供模型訓練的乾淨資料集，並儲存 Encoder 與 Scaler 供網站推理時使用。

### 處理流程（依序執行）

#### 步驟一：台灣地區篩選
```python
TAIWAN_CITIES = ["台北市", "臺北市", "新北市", ...]
df = df[df["工作地區"].apply(lambda x: any(str(x).startswith(c) for c in TAIWAN_CITIES))]
```
過濾所有非台灣地區的職缺，使用 `startswith` 比對縣市前綴，因為工作地區格式為「台北市信義區」等完整地址。

#### 步驟二：缺失值處理
- 刪除薪資為 0 的資料（`df["薪資"] > 0`）
- 類別型欄位缺失值填入該欄位眾數（`fillna(mode()[0])`）
- 缺失超過 3 欄的整筆刪除（`dropna(thresh=len(columns) - 3)`）

#### 步驟三：異常值清洗
- 薪資範圍限制：1,000 ≤ 薪資 ≤ 500,000
- 需求技能數量使用 **IQR 方法**：計算 Q1、Q3，過濾超出 `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` 的資料

#### 步驟四：薪資標籤轉換
| 薪資範圍 | 標籤 | 編碼 |
|---------|------|------|
| ≤ 40,000 | 低薪 | 0 |
| 40,001–55,000 | 中薪 | 1 |
| > 55,000 | 高薪 | 2 |

#### 步驟五：標籤編碼
- **有序型（Ordinal）**：學歷、工作經驗 → 手動定義順序對應字典（`edu_order`、`exp_order`），保留語義上的大小關係
- **名目型（Nominal）**：職務類別、工作地區、產業類別、上班時間 → `sklearn.LabelEncoder`，每個欄位獨立一個 Encoder 物件並存檔備用

#### 步驟六：特徵選擇（Feature Importance）
1. 訓練初步 Random Forest（100 棵樹）計算各特徵重要性
2. 刪除 Importance < 0.03 的特徵（上班時間 0.021、工作經驗 0.001）
3. 最終保留 6 個特徵：工作地區、公司規模、產業類別、職務類別、需求技能數量、學歷要求

#### 步驟七：正規化
- 對數值型欄位「需求技能數量」使用 `MinMaxScaler`，縮放至 [0, 1]
- Scaler 物件存檔（`scaler_需求技能數量.pkl`），確保網站推理時使用相同的縮放基準

### 輸出檔案
| 檔案 | 用途 |
|------|------|
| `data/processed/processed_data.csv` | 模型訓練用的乾淨資料 |
| `models/encoder_職務類別.pkl` | 推理時還原編碼 |
| `models/encoder_工作地區.pkl` | 推理時還原編碼 |
| `models/encoder_產業類別.pkl` | 推理時還原編碼 |
| `models/scaler_需求技能數量.pkl` | 推理時正規化輸入值 |

---

## 3. `notebooks/train_models.py` — 模型訓練與評估

### 功能
載入前處理後的資料，訓練三種分類模型，輸出評估指標、視覺化圖表，並儲存最終模型。

### 訓練設定

#### 交叉驗證
```python
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```
使用 **10-Fold 分層交叉驗證**（Stratified K-Fold），確保每個 Fold 的薪資區間比例與整體資料集一致，避免因類別不平衡造成評估偏差。

#### 模型參數
| 參數 | Decision Tree | Random Forest | Logistic Regression |
|------|-------------|--------------|-------------------|
| `max_depth` | None | None | — |
| `min_samples_leaf` | 2 | — | — |
| `n_estimators` | — | 100 | — |
| `C` | — | — | 1e8 |
| `max_iter` | — | — | 100 |
| `class_weight` | balanced | balanced | balanced |
| `random_state` | 42 | 42 | 42 |

- `class_weight="balanced"`：自動依各類別樣本數反比加權，應對類別不平衡（低薪 49.5% / 中薪 38.3% / 高薪 12.2%）
- `C=1e8`（Logistic Regression）：正則化幾乎不生效，近似無正則化，讓模型充分擬合資料

#### 評估指標計算方式
```python
y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
```
透過 `cross_val_predict` 收集每個 Fold 的預測結果後統一計算，所有指標（Accuracy、Precision、Recall、F1、AUC）均為 **10-Fold 平均值**，Precision / Recall / F1 使用 `average="weighted"`（加權平均，考慮各類別樣本數比例）。

### 圖表產出邏輯

| 圖表 | 技術 | 說明 |
|------|------|------|
| Confusion Matrix | `imshow` + 數字標注 | 3×3 熱力圖，正確預測集中在對角線 |
| ROC Curve | One-vs-Rest 策略 | 每個類別各算一條 FPR/TPR 曲線，取 macro 平均後繪製 |
| Feature Importance | 另訓練一個無 class_weight 的 RF | 橫條圖，依重要性排序 |
| 模型比較長條圖 | 並列長條圖 | Accuracy 與 F1-score 並排比較 |

### 最終模型儲存
訓練完成後以**完整資料集**重新 `fit` 一次（非 CV 版本），用 `joblib.dump` 存為 `.pkl`，供網站載入推理。

---

## 4. `website/app.py` — Flask 後端

### 功能
載入所有模型與編碼器，提供 REST API，並處理前端的預測請求與資料轉換。

### 啟動時載入
```python
MODELS   = { "random_forest": ..., "decision_tree": ..., "logistic_regression": ... }
ENCODERS = { "職務類別": ..., "工作地區": ..., "產業類別": ... }
SCALER   = joblib.load("scaler_需求技能數量.pkl")
```
所有模型與編碼器在啟動時一次性載入記憶體，避免每次請求重複讀取磁碟。

### 路由說明

#### `GET /` — 首頁
從 Encoder 的 `classes_` 屬性取得所有可用值，傳給 Jinja2 模板：
- `job_types`：所有職務類別
- `taiwan_areas`：依縣市分組的地區字典（見下方 `group_areas` 說明）
- `industry_groups`：依產業大類分組的字典（見下方 `group_industries` 說明）

#### `POST /predict` — 預測 API
**輸入**（JSON）：
```json
{
  "職務類別": "工程師",
  "工作地區": "台北市信義區",
  "產業類別": "電腦軟體服務業",
  "學歷要求": "大學",
  "公司規模": 500,
  "需求技能數量": 5,
  "model": "random_forest"
}
```

**特徵轉換流程**：
1. 地區、產業、職務 → `LabelEncoder.transform()`，轉為整數編碼
2. 學歷 → `EDU_ORDER` 字典查表，轉為有序整數（0–6）
3. 技能數量 → `MinMaxScaler.transform()`，正規化至 [0, 1]
4. 組成 numpy array，順序嚴格對應訓練時的 `FEATURES`：`["工作地區", "公司規模", "產業類別", "職務類別", "需求技能數量", "學歷要求"]`

**特徵重要性計算**：
- Random Forest / Decision Tree：直接取 `model.feature_importances_`
- Logistic Regression：取 `abs(model.coef_).mean(axis=0)` 後正規化，作為各特徵影響程度的估算值

**輸出**（JSON）：
```json
{
  "prediction": 2,
  "label": "高薪",
  "range": "NT$ 55,000 以上",
  "probabilities": { "低薪": 5.2, "中薪": 18.3, "高薪": 76.5 },
  "feature_importance": [
    { "label": "工作地區", "pct": 25.0 },
    ...
  ]
}
```

### 地區分組邏輯（`group_areas`）
原始地區資料格式為「台北市信義區」，需分兩層呈現（縣市 → 行政區）：
1. 對每個地區字串，從長到短比對 `PREFIX_TO_CITY` 字典中的前綴
2. 新竹縣市、嘉義縣市因涉及跨行政區合併，前綴保留完整字串作為 label
3. 依 `CITY_ORDER`（104 API 排序）輸出，確保下拉選單順序符合使用習慣

### 產業分組邏輯（`group_industries`）
使用關鍵字比對將約 100+ 種細項產業歸類到 15 個大類：
1. 對每個產業名稱，逐一比對 `INDUSTRY_MAPPING` 的關鍵字列表
2. 先比對到的大類優先（如「電腦軟體服務業」先命中「電子資訊」，不再往下比對）
3. 無法比對到的歸入「一般服務業」

---

## 5. `website/templates/index.html` — 前端 UI

### 功能
單頁應用程式，左側為輸入表單，右側為預測結果，透過 Fetch API 非同步呼叫後端。

### 串聯下拉選單邏輯

#### 工作地區（兩層）
```
城市下拉（citySelect）
    → 觸發 change 事件
    → 從 TAIWAN_AREAS[city] 取出行政區列表
    → 動態填入行政區下拉（districtSelect）並顯示
    → 隱藏欄位 areaValue 更新為選中值
```

#### 產業類別（兩層）
```
產業大類下拉（industryGroupSelect）
    → 觸發 change 事件
    → 從 INDUSTRY_GROUPS[group] 取出細項列表
    → 動態填入細項下拉（industrySelect）並顯示
    → 隱藏欄位 industryValue 更新為選中值
```

兩組下拉的資料（`TAIWAN_AREAS`、`INDUSTRY_GROUPS`）由後端 Jinja2 模板渲染時注入：
```javascript
const TAIWAN_AREAS    = {{ taiwan_areas | tojson }};
const INDUSTRY_GROUPS = {{ industry_groups | tojson }};
```

### 預測結果呈現邏輯
1. 送出表單 → `fetch("/predict", { method: "POST", body: JSON.stringify(data) })`
2. 收到回應後：
   - 更新薪資標籤 Badge（顏色：低薪紅 / 中薪琥珀 / 高薪綠）
   - 更新信心度：顯示預測類別的機率百分比
   - 機率長條圖：依各類別機率設定 `width`，CSS `transition` 產生動畫效果
   - 特徵影響程度：使用後端回傳的 `feature_importance`，依序延遲 80ms 逐條動畫展開，第一條（影響最大）標示「最大影響」標籤

---

## 6. 模型檔案說明（`models/`）

| 檔案 | 類型 | 說明 |
|------|------|------|
| `random_forest.pkl` | RandomForestClassifier | 以完整資料集訓練的最終模型 |
| `decision_tree.pkl` | DecisionTreeClassifier | 以完整資料集訓練的最終模型 |
| `logistic_regression.pkl` | LogisticRegression | 以完整資料集訓練的最終模型 |
| `encoder_職務類別.pkl` | LabelEncoder | 職務類別字串 → 整數 |
| `encoder_工作地區.pkl` | LabelEncoder | 工作地區字串 → 整數 |
| `encoder_產業類別.pkl` | LabelEncoder | 產業類別字串 → 整數 |
| `scaler_需求技能數量.pkl` | MinMaxScaler | 技能數量 → [0, 1] |

> **注意**：推理時使用的 Encoder 和 Scaler 必須與訓練時相同的物件，不可重新 fit，否則編碼對應關係會不一致。

---

## 7. 結果檔案說明（`results/`）

| 檔案 | 說明 |
|------|------|
| `model_results.csv` | 三個模型的 Accuracy / Precision / Recall / F1 / AUC 數值表 |
| `figures/salary_distribution.png` | 薪資區間分佈長條圖（低薪 1113 / 中薪 862 / 高薪 275） |
| `figures/feature_importance.png` | Random Forest 6 個特徵重要性橫條圖 |
| `figures/confusion_matrix_*.png` | 三個模型各自的混淆矩陣熱力圖 |
| `figures/roc_curve_comparison.png` | 三個模型的 ROC Curve 比較圖（One-vs-Rest） |
| `figures/model_comparison.png` | 三個模型 Accuracy / F1-score 並列比較長條圖 |

---

## 8. 重要技術決策摘要

| 決策 | 說明 |
|------|------|
| 只保留台灣地區資料 | 海外職缺薪資結構、幣別與台灣差異大，納入會引入不相關雜訊 |
| 刪除工作經驗特徵 | 資料中大多數填「不拘」，變異性極低（Importance = 0.001），幾乎無區分力 |
| 刪除上班時間特徵 | 與薪資關聯性低（Importance = 0.021），低於門檻 0.03 |
| 使用 class_weight="balanced" | 高薪樣本僅 12.2%，不平衡設定避免模型偏向多數類 |
| 10-Fold Stratified CV | 確保每個 Fold 的類別分佈一致，評估結果更可靠 |
| 使用 Weighted Average 評估指標 | 多類別分類問題，Weighted Average 考慮各類別樣本數比例，比 Macro 更能反映實際表現 |
| LabelEncoder 獨立儲存 | 每個欄位使用獨立 Encoder 物件，避免混淆，推理時精確還原 |
