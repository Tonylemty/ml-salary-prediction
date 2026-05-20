import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

INPUT_PATH  = "scraper/data/raw/raw_data.csv"
OUTPUT_PATH = "data/processed/processed_data.csv"

os.makedirs("data/processed", exist_ok=True)

TAIWAN_CITIES = [
    "台北市", "臺北市", "新北市", "桃園市", "台中市", "臺中市",
    "台南市", "臺南市", "高雄市", "基隆市", "新竹市", "嘉義市",
    "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣",
    "屏東縣", "宜蘭縣", "花蓮縣", "台東縣", "臺東縣", "澎湖縣",
    "金門縣", "連江縣",
]

# ── 讀入資料 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
print(f"原始資料筆數：{len(df)}")

# 只保留台灣地區
df = df[df["工作地區"].apply(lambda x: any(str(x).startswith(c) for c in TAIWAN_CITIES))]
print(f"篩選台灣地區後：{len(df)} 筆")

# ── 步驟一：缺失值處理 ────────────────────────────────────────────────────────
df = df[df["薪資"] > 0]                        # 移除薪資為 0 的資料
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])  # 類別型欄位填眾數
df = df.dropna(thresh=len(df.columns) - 3)       # 缺失超過 3 欄的整筆刪除
print(f"缺失值處理後：{len(df)} 筆")

# ── 步驟二：異常值清洗 ────────────────────────────────────────────────────────
df = df[(df["薪資"] >= 1000) & (df["薪資"] <= 500000)]

# 需求技能數量 IQR 清洗
Q1 = df["需求技能數量"].quantile(0.25)
Q3 = df["需求技能數量"].quantile(0.75)
IQR = Q3 - Q1
df = df[df["需求技能數量"].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]
print(f"異常值清洗後：{len(df)} 筆")

# ── 步驟三：薪資標籤轉換 ──────────────────────────────────────────────────────
def salary_label(salary):
    if salary <= 40000:
        return "低薪"
    elif salary <= 55000:
        return "中薪"
    else:
        return "高薪"

df["薪資區間"] = df["薪資"].apply(salary_label)
print(f"薪資分佈：\n{df['薪資區間'].value_counts()}")

# ── 步驟四：標籤編碼 ──────────────────────────────────────────────────────────
# 有序型屬性（Ordinal）
edu_order    = {"不拘": 0, "高中以下": 1, "高中職": 2, "專科": 3, "大學": 4, "碩士": 5, "博士": 6}
exp_order    = {"不拘": 0, "1年以下": 1, "1–3年": 2, "3–5年": 3, "5–10年": 4, "10年以上": 5}

df["學歷要求"] = df["學歷要求"].map(edu_order).fillna(0).astype(int)
df["工作經驗"] = df["工作經驗"].map(exp_order).fillna(0).astype(int)

# 名目型屬性（Nominal）
nominal_cols = ["職務類別", "工作地區", "產業類別", "上班時間"]
encoders = {}
for col in nominal_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 目標標籤
target_order = {"低薪": 0, "中薪": 1, "高薪": 2}
df["薪資區間"] = df["薪資區間"].map(target_order)

# ── 步驟五：特徵選擇 ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["職務類別", "工作地區", "學歷要求", "工作經驗", "公司規模", "產業類別", "上班時間", "需求技能數量"]
TARGET   = "薪資區間"

X = df[FEATURES]
y = df[TARGET]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\nFeature Importance：\n{importance}")

# 保留 importance >= 0.03 的特徵
selected = importance[importance >= 0.03].index.tolist()
print(f"\n保留特徵：{selected}")

# ── 步驟六：正規化 ────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
if "需求技能數量" in selected:
    df["需求技能數量"] = scaler.fit_transform(df[["需求技能數量"]])

# ── 儲存 Encoders 與 Scaler（供網站使用）────────────────────────────────────────
os.makedirs("models", exist_ok=True)
for col, enc in encoders.items():
    joblib.dump(enc, f"models/encoder_{col}.pkl")
joblib.dump(scaler, "models/scaler_需求技能數量.pkl")
print("Encoders 與 Scaler 已儲存到 models/")

# ── 儲存結果 ──────────────────────────────────────────────────────────────────
output_cols = selected + [TARGET]
df[output_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n完成！共儲存 {len(df)} 筆資料到 {OUTPUT_PATH}")
print(f"最終欄位：{output_cols}")
