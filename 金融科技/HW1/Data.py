import pandas as pd
import glob, os, csv, re

FOLDER = "./data"
FILES  = sorted(glob.glob(os.path.join(FOLDER, "Daily_*.csv")))

def find_header_and_encoding(path):
    """回傳 (header_row_index, encoding)：找包含『商品代號』的那一列"""
    for enc in ("cp950", "big5", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for i, row in enumerate(csv.reader(f)):
                    if row and any("商品代號" in (str(c) if c is not None else "") for c in row):
                        return i, enc
        except UnicodeDecodeError:
            pass
    return None, None

def normalize_hhmmss(x: str) -> str:
    """僅取數字並補到 6 碼（如 '3210' -> '003210'）"""
    s = re.sub(r"\D", "", str(x))
    return s.zfill(6)[:6] if s else ""

def read_one(path):
    hdr, enc = find_header_and_encoding(path)
    if hdr is None:
        print(f"找不到表頭：{os.path.basename(path)}")
        return None

    df = pd.read_csv(path, encoding=enc, header=hdr, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    if "商品代號" not in df.columns:
        print(f"欄位缺少『商品代號』：{os.path.basename(path)}")
        return None

    # 只留 TX
    df["商品代號"] = df["商品代號"].astype(str).str.replace("\u3000", " ").str.strip()
    df = df[df["商品代號"] == "TX"]
    if df.empty:
        print(f"（{os.path.basename(path)} 無 TX 筆數）")
        return None

    # ✅ 只保留「202509」開頭，且排除跨月（如 202509/202510）
    if "到期月份(週別)" in df.columns:
        df["到期月份(週別)"] = df["到期月份(週別)"].astype(str).str.strip()
        df = df[df["到期月份(週別)"].str.startswith("202509")]
        df = df[~df["到期月份(週別)"].str.contains("/", na=False)]
        if df.empty:
            print(f"（{os.path.basename(path)} 無 202509 月份的 TX 筆數）")
            return None
    else:
        print(f"缺少『到期月份(週別)』欄：{os.path.basename(path)}")
        return None

    # ✅ 日盤 08:45:00 ~ 13:45:00
    if "成交時間" not in df.columns:
        print(f"缺少『成交時間』欄：{os.path.basename(path)}")
        return None
    df["成交時間"] = df["成交時間"].map(normalize_hhmmss)
    day_mask = df["成交時間"].between("084500", "134500", inclusive="both")
    df = df[day_mask]
    if df.empty:
        print(f"（{os.path.basename(path)} TX 但無日盤時段筆數）")
        return None

    # 只保留常用欄位（存在才留）
    keep = [c for c in ["成交日期","商品代號","到期月份(週別)","成交時間",
                        "成交價格","成交數量(B+S)","近月價格","遠月價格","開盤集合競價"]
            if c in df.columns]
    df = df[keep].copy()
    df["source_file"] = os.path.basename(path)
    return df

# ======== 主程式 ========
all_parts = []
for fp in FILES:
    part = read_one(fp)
    if part is not None and not part.empty:
        print(f"✔ {os.path.basename(fp)}：{len(part)} rows (TX, 202509, 日盤)")
        all_parts.append(part)

if not all_parts:
    print("沒有讀到任何 TX 的 202509 日盤筆數，請確認欄名/編碼/時間欄內容。")
else:
    out = pd.concat(all_parts, ignore_index=True, sort=False)

    # 再保險一次時間與月份篩（避免個別檔案漏網）
    out["成交時間"] = out["成交時間"].astype(str).str.zfill(6).str[:6]
    out = out[out["成交時間"].between("084500", "134500", inclusive="both")]
    out["到期月份(週別)"] = out["到期月份(週別)"].astype(str).str.strip()
    out = out[out["到期月份(週別)"].str.startswith("202509")]
    out = out[~out["到期月份(週別)"].str.contains("/", na=False)]

    # 去重
    subset = [c for c in ["成交日期","商品代號","到期月份(週別)","成交時間","成交價格"] if c in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset)

    out.to_csv("TX.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"\n✅ 輸出：TX.csv；日盤總筆數 {len(out)}（TX、202509、排除跨月）")
