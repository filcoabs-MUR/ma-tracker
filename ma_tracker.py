#!/usr/bin/env python3
"""
ma_tracker.py — 主題股均線 / 賣壓追蹤（每日盤後更新）

每檔計算：
  收盤、距20MA%、距60MA%、20MA/60MA 方向(近5日)、距一年高點%、20日漲跌%、
  RSI14、近25日出貨日數、量比(近5日均量/前50日均量)
賣壓判讀：
  🟢輕  站上60MA
  🟠中  跌破60MA，但60MA仍上揚（或跌幅未達-30%）
  🔴重  跌破60MA + 60MA下彎 + 距一年高點跌逾30%
進場條件(建倉規則)：站上60MA 且 RSI14 < 70 → ✅

輸出（皆在 ./output/）：
  latest.html          手機好讀的分類報表
  latest.csv           本次結果
  daily/ma_YYYY-MM-DD.csv
  history.csv          歷史累積（run_date+ticker 去重）
  alerts_YYYY-MM-DD.txt 新站上 / 新跌破 60MA 清單

用法：
  python ma_tracker.py           # 跑全部
  python ma_tracker.py --open    # 跑完自動開啟 HTML
僅為數據整理，不構成投資建議。
"""
from __future__ import annotations

import argparse
import html
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
DAILY = OUT / "daily"

# ───────────────────────── 追蹤清單（分類） ─────────────────────────
WATCHLIST: dict[str, dict[str, str]] = {
    "指數": {
        "VOO": "S&P500 VOO", "^NDX": "那斯達克100", "^SOX": "費半", "^TWII": "加權指數",
    },
    "CPU": {
        "AMD": "AMD", "INTC": "Intel", "ARM": "Arm",
    },
    "CPU載板": {
        "4062.T": "Ibiden", "3037.TW": "欣興", "3189.TW": "景碩", "ATS.VI": "AT&S",
    },
    "記憶體/儲存": {
        "MU": "Micron", "STX": "Seagate",
    },
    "光通訊(美)": {
        "LITE": "Lumentum", "COHR": "Coherent", "AXTI": "AXT", "GLW": "康寧",
        "TSEM": "Tower Semi", "CIEN": "Ciena", "CRDO": "Credo",
        "SMTC": "Semtech", "MXL": "MaxLinear", "NOK": "Nokia",
    },
    "光通訊/連接(台)": {
        "3533.TW": "嘉澤", "6715.TW": "嘉基", "3665.TW": "貿聯-KY",
        "3081.TWO": "聯亞", "6442.TW": "光聖", "2455.TW": "全新",
    },
    "AI晶片/ASIC": {
        "NVDA": "Nvidia", "AVGO": "Broadcom", "MRVL": "Marvell", "2454.TW": "聯發科",
    },
    "晶圓代工/設備/測試": {
        "TSM": "台積電ADR", "2330.TW": "台積電", "ASML": "ASML", "AMAT": "Applied Mat",
        "ONTO": "Onto", "TER": "Teradyne", "KEYS": "Keysight",
    },
    "電力/VPD": {
        "VRT": "Vertiv", "BE": "Bloom Energy", "2308.TW": "台達電", "VICR": "Vicor",
    },
    "伺服器/EMS": {
        "CLS": "Celestica",
    },
    "軟體/資安/Agent": {
        "MSFT": "Microsoft", "PLTR": "Palantir", "NET": "Cloudflare",
        "CRWD": "CrowdStrike", "PANW": "Palo Alto",
    },
    "大型科技": {
        "AAPL": "Apple", "AMZN": "Amazon", "GOOG": "Google", "META": "Meta", "TSLA": "Tesla",
    },
}
# 註：Samtec 為未上市公司，無法追蹤。

# ───────────────────────── 抓資料 ─────────────────────────
def _split(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in lv0:
                df = raw[t].dropna(subset=["Close"])
                if not df.empty:
                    out[t] = df
    elif len(tickers) == 1:
        df = raw.dropna(subset=["Close"])
        if not df.empty:
            out[tickers[0]] = df
    return out


def fetch_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                      auto_adjust=False, threads=True, progress=False)
    data = _split(raw, tickers)
    missing = [t for t in tickers if t not in data]
    for t in missing:  # 個別重試
        for _ in range(2):
            try:
                r = yf.download(t, period="1y", interval="1d", auto_adjust=False,
                                progress=False)
                if isinstance(r.columns, pd.MultiIndex):
                    r.columns = r.columns.get_level_values(0)
                r = r.dropna(subset=["Close"])
                if not r.empty:
                    data[t] = r
                    break
            except Exception:
                pass
            time.sleep(1.5)
    return data

# ───────────────────────── 指標 ─────────────────────────
def rsi14(c: pd.Series) -> float:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def metrics(df: pd.DataFrame) -> dict | None:
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float).fillna(0) if "Volume" in df else pd.Series(0, index=c.index)
    if len(c) < 70:
        return None
    p = c.iloc[-1]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    m20, m60 = ma20.iloc[-1], ma60.iloc[-1]
    hi = c.max()

    dist = 0
    for i in range(len(c) - 25, len(c)):
        if c.iloc[i] < c.iloc[i - 1] * 0.998 and v.iloc[i] > v.iloc[i - 1]:
            dist += 1
    v5, v50 = v.iloc[-5:].mean(), v.iloc[-55:-5].mean()

    vs20 = (p / m20 - 1) * 100
    vs60 = (p / m60 - 1) * 100
    up20 = m20 > ma20.iloc[-6]
    up60 = m60 > ma60.iloc[-6]
    dd = (p / hi - 1) * 100
    r = rsi14(c)

    if vs60 > 0:
        rating = "🟢輕"
    elif (not up60) and dd <= -30:
        rating = "🔴重"
    else:
        rating = "🟠中"

    return {
        "date": c.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(p), 2),
        "vs20": round(vs20, 1),
        "vs60": round(vs60, 1),
        "ma20_dir": "↑" if up20 else "↓",
        "ma60_dir": "↑" if up60 else "↓",
        "dd_1y": round(dd, 1),
        "ret20": round((p / c.iloc[-21] - 1) * 100, 1),
        "rsi": round(r, 1),
        "dist_days": dist,
        "vol_ratio": round(v5 / v50, 2) if v50 > 0 else np.nan,
        "above20": vs20 > 0,
        "above60": vs60 > 0,
        "rating": rating,
        "entry_ok": "✅" if (vs60 > 0 and r < 70) else "",
    }

# ───────────────────────── 報表 ─────────────────────────
def build_table(results: pd.DataFrame) -> pd.DataFrame:
    order = {k: i for i, k in enumerate(WATCHLIST)}
    results["_o"] = results["category"].map(order)
    return results.sort_values(["_o", "vs60"], ascending=[True, False]).drop(columns="_o")


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["category"] != "指數"].groupby("category", sort=False)
    s = pd.DataFrame({
        "檔數": g.size(),
        "站上60MA": g["above60"].sum().astype(int),
        "站上20MA": g["above20"].sum().astype(int),
        "平均距60MA%": g["vs60"].mean().round(1),
        "🔴重": g["rating"].apply(lambda x: (x == "🔴重").sum()),
    })
    return s.sort_values("平均距60MA%", ascending=False)


def diff_alerts(cur: pd.DataFrame, prev_path: Path) -> list[str]:
    if not prev_path.exists():
        return []
    prev = pd.read_csv(prev_path).set_index("ticker")
    msgs = []
    for _, r in cur.iterrows():
        t = r["ticker"]
        if t not in prev.index or r["date"] == prev.loc[t, "date"]:
            continue
        was = bool(prev.loc[t, "above60"])
        if r["above60"] and not was:
            msgs.append(f"⬆️ 新站上60MA  {r['name']}({t})  距60MA {r['vs60']:+.1f}%")
        elif (not r["above60"]) and was:
            msgs.append(f"⬇️ 新跌破60MA  {r['name']}({t})  距60MA {r['vs60']:+.1f}%")
    return msgs


def _cls(v: float) -> str:
    return "pos" if v > 0 else "neg"


def write_html(df: pd.DataFrame, summ: pd.DataFrame, alerts: list[str],
               failed: list[str], path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>均線追蹤 {now}</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e5e5e5;--pos:#c62828;--neg:#1b7f3b;--card:#f7f7f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--mut:#999;--line:#333;--card:#1e1e1e;--pos:#ef5350;--neg:#66bb6a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,"PingFang TC","Microsoft JhengHei",sans-serif;margin:0;padding:12px 16px}}
h1{{font-size:18px;margin:4px 0}} h2{{font-size:15px;margin:18px 0 6px;border-left:4px solid var(--mut);padding-left:8px}}
.mut{{color:var(--mut);font-size:12px}} .wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:var(--bg)}}
th{{font-size:12px;color:var(--mut);font-weight:600}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
.card{{background:var(--card);border-radius:8px;padding:8px 10px;margin:8px 0}}
</style></head><body>
<h1>主題股均線追蹤</h1><div class="mut">更新 {now}｜紅=在均線上 綠=在均線下（台股慣例）｜僅數據整理，非投資建議</div>"""]

    if alerts:
        parts.append('<div class="card"><b>今日變化</b><br>' +
                     "<br>".join(html.escape(a) for a in alerts) + "</div>")

    parts.append("<h2>類股強弱</h2><div class='wrap'><table><tr><th>分類</th>"
                 "<th>檔數</th><th>站上60</th><th>站上20</th><th>平均距60%</th><th>🔴</th></tr>")
    for cat, r in summ.iterrows():
        parts.append(f"<tr><td>{html.escape(cat)}</td><td>{r['檔數']}</td>"
                     f"<td>{r['站上60MA']}</td><td>{r['站上20MA']}</td>"
                     f"<td class='{_cls(r['平均距60MA%'])}'>{r['平均距60MA%']:+.1f}</td>"
                     f"<td>{r['🔴重']}</td></tr>")
    parts.append("</table></div>")

    for cat in WATCHLIST:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        parts.append(f"<h2>{html.escape(cat)}</h2><div class='wrap'><table><tr>"
                     "<th>股票</th><th>收盤</th><th>距20%</th><th>距60%</th><th>60MA</th>"
                     "<th>距高%</th><th>20日%</th><th>RSI</th><th>出貨</th><th>量比</th>"
                     "<th>賣壓</th><th>進場</th></tr>")
        for _, r in sub.iterrows():
            parts.append(
                f"<tr><td>{html.escape(r['name'])}<br><span class='mut'>{html.escape(r['ticker'])}</span></td>"
                f"<td>{r['close']:,}</td>"
                f"<td class='{_cls(r['vs20'])}'>{r['vs20']:+.1f}</td>"
                f"<td class='{_cls(r['vs60'])}'>{r['vs60']:+.1f}</td>"
                f"<td>{r['ma60_dir']}</td><td>{r['dd_1y']:.0f}</td>"
                f"<td class='{_cls(r['ret20'])}'>{r['ret20']:+.1f}</td>"
                f"<td>{r['rsi']:.0f}</td><td>{r['dist_days']}</td><td>{r['vol_ratio']}</td>"
                f"<td>{r['rating']}</td><td>{r['entry_ok']}</td></tr>")
        parts.append("</table></div>")

    parts.append("<p class='mut'>出貨＝近25日收黑且量增的天數；量比＝近5日均量/前50日均量；"
                 "進場✅＝站上60MA且RSI&lt;70。資料：Yahoo Finance。</p>")
    if failed:
        parts.append(f"<p class='mut'>抓取失敗：{html.escape(', '.join(failed))}</p>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")

# ───────────────────────── 主程式 ─────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="跑完開啟 HTML")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    DAILY.mkdir(exist_ok=True)

    meta = [(cat, t, n) for cat, d in WATCHLIST.items() for t, n in d.items()]
    tickers = [t for _, t, _ in meta]
    print(f"抓取 {len(tickers)} 檔 ...")
    data = fetch_all(tickers)

    rows, failed = [], []
    for cat, t, n in meta:
        m = metrics(data[t]) if t in data else None
        if m is None:
            failed.append(t)
            continue
        rows.append({"category": cat, "ticker": t, "name": n, **m})
    if not rows:
        print("全部抓取失敗，請檢查網路或 yfinance 版本")
        return 1

    df = build_table(pd.DataFrame(rows))
    summ = category_summary(df)

    today = datetime.now().strftime("%Y-%m-%d")
    latest_csv = OUT / "latest.csv"
    alerts = diff_alerts(df, latest_csv)

    df.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    df.to_csv(DAILY / f"ma_{today}.csv", index=False, encoding="utf-8-sig")

    hist_path = OUT / "history.csv"
    h = df.assign(run_date=today)
    if hist_path.exists():
        h = pd.concat([pd.read_csv(hist_path), h], ignore_index=True)
    h.drop_duplicates(["run_date", "ticker"], keep="last").to_csv(
        hist_path, index=False, encoding="utf-8-sig")

    if alerts:
        (OUT / f"alerts_{today}.txt").write_text("\n".join(alerts), encoding="utf-8")

    html_path = OUT / "latest.html"
    write_html(df, summ, alerts, failed, html_path)

    # console
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n== 類股強弱 ==")
    print(summ.to_string())
    print("\n== 站上60MA ==")
    print(df[df["above60"] & (df["category"] != "指數")][["name", "vs20", "vs60", "rsi", "entry_ok"]]
          .to_string(index=False))
    if alerts:
        print("\n== 今日變化 ==")
        print("\n".join(alerts))
    if failed:
        print(f"\n抓取失敗：{', '.join(failed)}")
    print(f"\nHTML：{html_path}")

    if args.open:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
