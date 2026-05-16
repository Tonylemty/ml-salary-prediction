import requests
import pandas as pd
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.104.com.tw/jobs/search/",
}

KEYWORDS = [
    "工程師", "軟體", "硬體", "機械", "電機", "化工",
    "行銷", "業務", "財務", "會計", "人資", "企劃",
    "UI設計", "平面設計", "室內設計",
    "護理師", "藥師",
    "教師", "廚師", "記者",
]

PAGES_PER_KEYWORD = 8
PAGE_SIZE = 20

EDU_MAP = {
    1: "不拘",
    2: "高中以下",
    3: "高中職",
    4: "專科",
    5: "大學",
    6: "碩士",
    7: "博士",
}

JOB_TYPE_MAP = {
    1: "日班",
    2: "夜班",
    3: "日夜輪班",
    4: "假日班",
    8: "彈性",
}

EXP_MAP = {
    1: "不拘",
    3: "1年以下",
    4: "1–3年",
    5: "3–5年",
    6: "5–10年",
    7: "10年以上",
}


def fetch_jobs(keyword, page):
    url = "https://www.104.com.tw/jobs/search/api/jobs"
    params = {
        "jobsource": "index_s",
        "keyword": keyword,
        "mode": "s",
        "order": 15,
        "page": page,
        "pagesize": PAGE_SIZE,
    }
    try:
        res = requests.get(url, headers=HEADERS, params=params, timeout=10)
        res.raise_for_status()
        return res.json().get("data", [])
    except Exception as e:
        print(f"  [錯誤] keyword={keyword} page={page}: {e}")
        return []


def parse_job(job, keyword):
    salary_low  = job.get("salaryLow",  0)
    salary_high = job.get("salaryHigh", 0)

    # 面議或薪資為 0 → 跳過
    if salary_low == 0 and salary_high == 0:
        return None
    if salary_high == 9999999:
        salary = salary_low
    else:
        salary = (salary_low + salary_high) // 2

    edu_codes = job.get("optionEdu", [])
    edu = EDU_MAP.get(max(edu_codes), "不拘") if edu_codes else "不拘"

    return {
        "職務類別":     keyword,
        "工作地區":     job.get("jobAddrNoDesc", ""),
        "學歷要求":     edu,
        "工作經驗":     EXP_MAP.get(job.get("s9", [1])[0], "不拘") if job.get("s9") else "不拘",
        "公司規模":     job.get("employeeCount", 0),
        "產業類別":     job.get("coIndustryDesc", ""),
        "上班時間":     JOB_TYPE_MAP.get(job.get("jobType", 1), "日班"),
        "需求技能數量": len(job.get("pcSkills", [])),
        "薪資":         salary,
        "職缺名稱":     job.get("jobName", ""),
        "公司名稱":     job.get("custName", ""),
    }


def main():
    all_records = []

    for keyword in KEYWORDS:
        print(f"爬取關鍵字：{keyword}")
        for page in range(1, PAGES_PER_KEYWORD + 1):
            jobs = fetch_jobs(keyword, page)
            if not jobs:
                break
            for job in jobs:
                record = parse_job(job, keyword)
                if record:
                    all_records.append(record)
            print(f"  第 {page} 頁，目前共 {len(all_records)} 筆")
            time.sleep(random.uniform(1.0, 2.0))   # 避免請求過快被封鎖

    import os
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(all_records)
    df.drop_duplicates(inplace=True)
    df.to_csv("data/raw/raw_data.csv", index=False, encoding="utf-8-sig")
    print(f"\n完成！共儲存 {len(df)} 筆資料到 data/raw/raw_data.csv")


if __name__ == "__main__":
    main()
