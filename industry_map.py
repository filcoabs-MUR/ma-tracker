#!/usr/bin/env python3
"""
industry_map.py — 美股行業地圖（每日盤後更新均線）

讀取：
  industries.json             行業定義（商業模式、擴張判斷、代表股、ETF），手動維護
  industry_map_template.html  頁面樣板（含 /*__DATA__*/ 佔位）
輸出：
  output/industry_map.html    完整頁面（GitHub Pages 發布為 industry.html）
  output/industry_ma.csv      各行業 50MA 數據
  output/industry_ma_history.csv  每日累積

計算：每檔 距20MA%、距50MA%、50MA 方向（近 5 日）、距60MA%、60MA 方向
用法：python industry_map.py [--open]
僅為數據整理，不構成投資建議。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
DEF_PATH = BASE / "industries.json"
TPL_PATH = BASE / "industry_map_template.html"


# ───────────────────────── 抓資料 ─────────────────────────
def _closes(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in lv0 and "Close" in raw[t]:
                c = raw[t]["Close"].dropna()
                if len(c):
                    out[t] = c
    elif len(tickers) == 1 and "Close" in raw:
        c = raw["Close"].dropna()
        if len(c):
            out[tickers[0]] = c
    return out


def fetch(tickers: list[str]) -> dict[str, pd.Series]:
    data: dict[str, pd.Series] = {}
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        raw = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
        data.update(_closes(raw, chunk))
        time.sleep(1)
    for t in [t for t in tickers if t not in data]:  # 個別重試
        for _ in range(2):
            try:
                r = yf.download(t, period="1y", interval="1d", auto_adjust=False, progress=False)
                if isinstance(r.columns, pd.MultiIndex):
                    r.columns = r.columns.get_level_values(0)
                c = r["Close"].dropna()
                if len(c):
                    data[t] = c
                    break
            except Exception:
                pass
            time.sleep(1.5)
    return data


def stats(c: pd.Series) -> dict | None:
    c = c.astype(float)
    if len(c) < 70:
        return None
    # 最後一筆過舊（下市/併購）就略過
    last = pd.Timestamp(c.index[-1])
    last = last.tz_convert(None) if last.tzinfo else last
    if (pd.Timestamp.now() - last).days > 10:
        return None
    p = c.iloc[-1]
    ma20, ma50, ma60 = (c.rolling(n).mean() for n in (20, 50, 60))
    return {
        "date": c.index[-1].strftime("%Y-%m-%d"),
        "vs20": round((p / ma20.iloc[-1] - 1) * 100, 1),
        "vs50": round((p / ma50.iloc[-1] - 1) * 100, 1),
        "up50": int(ma50.iloc[-1] > ma50.iloc[-6]),
        "vs60": round((p / ma60.iloc[-1] - 1) * 100, 1),
        "up60": ma60.iloc[-1] > ma60.iloc[-6],
    }


# ───────────────────────── 主程式 ─────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    spec = json.loads(DEF_PATH.read_text(encoding="utf-8"))
    sectors, inds = spec["sectors"], spec["industries"]
    sec_etf = {s["key"]: s["etf"] for s in sectors}
    for ind in inds:
        ind["etf"] = ind.get("etf") or sec_etf[ind["sector"]]

    tickers = sorted({"SPY", *sec_etf.values(), *(i["etf"] for i in inds),
                      *(t for i in inds for t in i["tickers"])})
    print(f"抓取 {len(tickers)} 檔 ...")
    px = fetch(tickers)
    S = {t: s for t, c in px.items() if (s := stats(c))}
    failed = [t for t in tickers if t not in S]
    if "SPY" not in S:
        print("SPY 抓取失敗，停止輸出")
        return 1
    asof = S["SPY"]["date"]

    def etf_tuple(e: str) -> tuple[float, str]:
        s = S.get(e)
        return (s["vs60"], "↑" if s["up60"] else "↓") if s else (0.0, "–")

    data = {
        "asof": asof,
        "sectors": [[s["key"], s["zh"], s["en"], s["etf"], *etf_tuple(s["etf"])] for s in sectors],
        "D": [[i["sector"], i["zh"], i["en"], i["status"], i["structure"], i["model"],
               i["driver"], i["thesis"], i["etf"], *etf_tuple(i["etf"]),
               " ".join(t for t in i["tickers"] if t in S)] for i in inds],
        "m50": {t: [s["vs20"], s["vs50"], s["up50"]] for t, s in S.items()},
    }

    tpl = TPL_PATH.read_text(encoding="utf-8")
    if "/*__DATA__*/null" not in tpl:
        print("樣板缺少 /*__DATA__*/null 佔位")
        return 1
    html = tpl.replace("/*__DATA__*/null",
                       json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "industry_map.html").write_text(html, encoding="utf-8")

    # CSV + 歷史
    sec_zh = {s["key"]: s["zh"] for s in sectors}
    rows = []
    for i in inds:
        v = [S[t]["vs50"] for t in i["tickers"] if t in S]
        e = S.get(i["etf"])
        rows.append({
            "date": asof, "sector": sec_zh[i["sector"]], "industry": i["zh"],
            "status": i["status"], "avg_vs50": round(sum(v) / len(v), 1) if v else None,
            "above50": sum(x > 0 for x in v), "n": len(v),
            "etf": i["etf"], "etf_vs50": e["vs50"] if e else None,
        })
    df = pd.DataFrame(rows).sort_values("avg_vs50", ascending=False)
    df.to_csv(OUT / "industry_ma.csv", index=False, encoding="utf-8-sig")
    hp = OUT / "industry_ma_history.csv"
    h = pd.concat([pd.read_csv(hp), df], ignore_index=True) if hp.exists() else df
    h.drop_duplicates(["date", "industry"], keep="last").to_csv(hp, index=False, encoding="utf-8-sig")

    up = int((df["avg_vs50"] > 0).sum())
    print(f"\n資料日 {asof}｜站上50MA 行業 {up}/{len(df)}")
    pd.set_option("display.unicode.east_asian_width", True)
    print(df[["sector", "industry", "avg_vs50", "above50", "n"]].head(8).to_string(index=False))
    print("…")
    print(df[["sector", "industry", "avg_vs50", "above50", "n"]].tail(8).to_string(index=False))
    if failed:
        print(f"\n抓取失敗或資料過舊（已略過）：{', '.join(failed)}")
    print(f"\nHTML：{OUT / 'industry_map.html'}")
    if args.open:
        webbrowser.open((OUT / "industry_map.html").as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
