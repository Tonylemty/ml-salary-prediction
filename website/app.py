import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_file

WEBSITE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBSITE_DIR)
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")

app = Flask(__name__)

MODELS = {}
for key, fname in [
    ("random_forest",       "random_forest.pkl"),
    ("decision_tree",       "decision_tree.pkl"),
    ("logistic_regression", "logistic_regression.pkl"),
]:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        MODELS[key] = joblib.load(path)

ENCODERS = {}
for col in ["職務類別", "工作地區", "產業類別"]:
    path = os.path.join(MODEL_DIR, f"encoder_{col}.pkl")
    if os.path.exists(path):
        ENCODERS[col] = joblib.load(path)

scaler_path = os.path.join(MODEL_DIR, "scaler_需求技能數量.pkl")
SCALER = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

EDU_ORDER = {"不拘": 0, "高中以下": 1, "高中職": 2, "專科": 3, "大學": 4, "碩士": 5, "博士": 6}

SALARY_LABELS = {0: "低薪", 1: "中薪", 2: "高薪"}
SALARY_RANGES = {0: "NT$ 40,000 以下", 1: "NT$ 40,000 - 55,000", 2: "NT$ 55,000 以上"}

FEATURES = ["工作地區", "公司規模", "產業類別", "職務類別", "需求技能數量", "學歷要求"]

# 產業大類順序（對應圖表）與關鍵字對應（先比對到的優先）
INDUSTRY_MAPPING = [
    ("電子資訊／軟體／半導體相關業",  ["電腦", "軟體", "半導體", "積體電路", "光電", "晶圓", "網路", "資訊", "通訊"]),
    ("一般製造業",                  ["製造", "加工", "機械", "化工", "化學", "塑膠", "橡膠", "鋼鐵", "金屬",
                                     "紡織", "印刷", "造紙", "玻璃", "陶瓷", "汽車", "自行車", "光學",
                                     "電機", "鞋類", "工業"]),
    ("批發／零售／傳直銷業",         ["批發", "零售", "傳直銷", "百貨", "超市", "便利商店"]),
    ("住宿／餐飲服務業",             ["餐", "飲料店", "住宿", "旅館", "飯店", "咖啡"]),
    ("法律／會計／顧問／研發／設計業",["法律", "會計", "顧問", "廣告", "設計", "研發", "公關", "市調"]),
    ("建築營造及不動產相關業",        ["建築", "營造", "不動產", "建材", "室內裝修", "房地產"]),
    ("金融投顧及保險業",             ["銀行", "金融", "保險", "證券", "投資", "期貨", "信託", "租賃"]),
    ("醫療保健及環境衛生業",         ["醫療", "醫院", "藥", "環保", "衛生", "健康", "生技", "生物"]),
    ("運輸物流及倉儲",               ["運輸", "物流", "倉儲", "航空", "海運", "快遞", "貨運", "港埠"]),
    ("旅遊／休閒／運動業",           ["旅遊", "旅行", "休閒", "運動", "健身", "娛樂"]),
    ("大眾傳播相關業",               ["傳播", "廣播", "電視", "電影", "出版", "媒體", "雜誌"]),
    ("文教相關業",                   ["教育", "學校", "補習", "文教", "學術", "培訓"]),
    ("農林漁牧水電資源業",           ["農", "林", "漁", "牧", "水電", "能源", "礦", "石油"]),
    ("政治宗教及社福相關業",         ["政府", "公部門", "社福", "宗教", "非營利", "協會", "財團法人", "社團法人"]),
    ("一般服務業",                   []),  # 其餘歸入此類
]


def group_industries(industries):
    """依 INDUSTRY_MAPPING 將產業細項歸類到大類。"""
    groups = {cat: [] for cat, _ in INDUSTRY_MAPPING}
    for industry in sorted(industries):
        matched = "一般服務業"
        for cat, keywords in INDUSTRY_MAPPING:
            if not keywords:
                break
            if any(kw in industry for kw in keywords):
                matched = cat
                break
        groups[matched].append({"value": industry, "label": industry})
    return {cat: items for cat, items in groups.items() if items}

# 資料中的前綴 → 顯示用縣市名稱（104 API 順序）
PREFIX_TO_CITY = {
    "台北市": "台北市", "臺北市": "台北市",
    "新北市": "新北市",
    "宜蘭縣": "宜蘭縣",
    "基隆市": "基隆市",
    "桃園市": "桃園市",
    "新竹市": "新竹縣市",   # 新竹市歸入新竹縣市群組
    "新竹縣": "新竹縣市",
    "苗栗縣": "苗栗縣",
    "台中市": "台中市", "臺中市": "台中市",
    "彰化縣": "彰化縣",
    "南投縣": "南投縣",
    "雲林縣": "雲林縣",
    "嘉義市": "嘉義縣市",   # 嘉義市歸入嘉義縣市群組
    "嘉義縣": "嘉義縣市",
    "台南市": "台南市", "臺南市": "台南市",
    "高雄市": "高雄市",
    "屏東縣": "屏東縣",
    "台東縣": "台東縣", "臺東縣": "台東縣",
    "花蓮縣": "花蓮縣",
    "澎湖縣": "澎湖縣",
    "金門縣": "金門縣",
    "連江縣": "連江縣",
}

CITY_ORDER = [
    "台北市", "新北市", "宜蘭縣", "基隆市", "桃園市",
    "新竹縣市", "苗栗縣", "台中市", "彰化縣", "南投縣", "雲林縣",
    "嘉義縣市", "台南市", "高雄市", "屏東縣", "台東縣", "花蓮縣",
    "澎湖縣", "金門縣", "連江縣",
]

# 複合縣市的前綴：保留原始字串當 label（避免混淆新竹市 vs 新竹縣）
COMPOUND_PREFIXES = {"新竹市", "新竹縣", "嘉義市", "嘉義縣"}


def group_areas(areas):
    """將地區清單依 CITY_ORDER 順序分組為 {顯示縣市: [{value, label}]}。"""
    groups = {}
    for area in sorted(areas):
        matched_prefix = next(
            (p for p in sorted(PREFIX_TO_CITY, key=len, reverse=True) if area.startswith(p)),
            None
        )
        if not matched_prefix:
            continue
        city = PREFIX_TO_CITY[matched_prefix]
        if matched_prefix in COMPOUND_PREFIXES:
            label = area  # 保留完整字串（新竹市東區、新竹縣竹北市）
        else:
            district = area[len(matched_prefix):]
            label = district if district else area
        groups.setdefault(city, []).append({"value": area, "label": label})

    # 依 CITY_ORDER 排序，只回傳資料中有出現的縣市
    return {city: groups[city] for city in CITY_ORDER if city in groups}


@app.route("/")
def index():
    areas_raw    = ENCODERS["工作地區"].classes_.tolist() if "工作地區" in ENCODERS else []
    taiwan_areas = group_areas(areas_raw)

    industries_raw  = ENCODERS["產業類別"].classes_.tolist() if "產業類別" in ENCODERS else []
    industry_groups = group_industries(industries_raw)

    return render_template("index.html",
        job_types       = ENCODERS["職務類別"].classes_.tolist() if "職務類別" in ENCODERS else [],
        taiwan_areas    = taiwan_areas,
        industry_groups = industry_groups,
        edu_levels      = list(EDU_ORDER.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        area_enc     = int(ENCODERS["工作地區"].transform([data["工作地區"]])[0])
        company_size = int(data["公司規模"])
        industry_enc = int(ENCODERS["產業類別"].transform([data["產業類別"]])[0])
        job_enc      = int(ENCODERS["職務類別"].transform([data["職務類別"]])[0])
        skill_raw    = float(data["需求技能數量"])
        skill_norm   = float(SCALER.transform([[skill_raw]])[0][0]) if SCALER else skill_raw
        edu_enc      = EDU_ORDER.get(data["學歷要求"], 0)

        # 順序需與 train_models.py 的 FEATURES 一致：
        # ["工作地區", "公司規模", "產業類別", "職務類別", "需求技能數量", "學歷要求"]
        features = np.array([[area_enc, company_size, industry_enc, job_enc, skill_norm, edu_enc]])

        model = MODELS.get(data.get("model", "random_forest"), MODELS.get("random_forest"))
        pred  = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()

        # 計算各模型的特徵重要性
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # Logistic Regression：取各類別係數絕對值的平均，再正規化
            importances = np.abs(model.coef_).mean(axis=0)
            importances = importances / importances.sum()

        feat_importance = [
            {"label": FEATURES[i], "pct": round(float(importances[i]) * 100, 1)}
            for i in range(len(FEATURES))
        ]
        feat_importance.sort(key=lambda x: x["pct"], reverse=True)

        return jsonify({
            "prediction": pred,
            "label": SALARY_LABELS[pred],
            "range": SALARY_RANGES[pred],
            "probabilities": {
                "低薪": round(proba[0] * 100, 1),
                "中薪": round(proba[1] * 100, 1),
                "高薪": round(proba[2] * 100, 1),
            },
            "feature_importance": feat_importance,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/figures/<filename>")
def figure(filename):
    path = os.path.join(FIGURES_DIR, os.path.basename(filename))
    return send_file(path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
