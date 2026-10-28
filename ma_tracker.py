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
篩選條件：站上60MA 且 RSI14 < 70 → ✅（欄位 entry_ok）

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
    "載板": {
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
        "PLTR": "Palantir", "NET": "Cloudflare", "CRWD": "CrowdStrike", "PANW": "Palo Alto",
    },
    "大型科技CSP": {
        "MSFT": "Microsoft", "AAPL": "Apple", "AMZN": "Amazon", "GOOG": "Google", "META": "Meta",
        "TSLA": "Tesla",
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
    return "pos" if v > 0 else "neg" if v < 0 else ""


def _num(v, fmt: str = "{:+.1f}") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _bar(v: float, scale: float = 30.0) -> str:
    """以 0 為中心的小長條，±scale% 封頂"""
    w = min(abs(v), scale) / scale * 50
    side = "r" if v >= 0 else "l"
    return f"<i class='bar'><i class='{side} {_cls(v)}' style='width:{w:.1f}%'></i></i>"


RATING_PILL = {"🟢輕": ("lt", "賣壓輕"), "🟠中": ("md", "賣壓中"), "🔴重": ("hv", "賣壓重")}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#16181d;--mut:#6b7280;--line:#e7e9ee;--soft:#f1f3f6;
--pos:#d6312f;--neg:#16a34a;--acc:#2563eb;--amb:#b45309;--ambbg:#fef3c7;--vio:#7c3aed;--viobg:#ede9fe;--accbg:#dbeafe}
@media(prefers-color-scheme:dark){:root{--bg:#0e1013;--card:#171a1f;--fg:#e8eaee;--mut:#8b93a1;--line:#262a31;--soft:#1e2228;
--pos:#f05252;--neg:#34d399;--acc:#60a5fa;--amb:#fbbf24;--ambbg:#3a2c0d;--vio:#a78bfa;--viobg:#2a2140;--accbg:#172a4a}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:20px 16px 48px}
.num,.px,.m,.stat b{font-variant-numeric:tabular-nums}
header h1{font-size:22px;margin:0;letter-spacing:.02em}
header .sub{color:var(--mut);font-size:13px;margin-top:4px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}.acc{color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-top:14px}
h2{font-size:15px;margin:0 0 10px;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
h2 small{font-weight:400;color:var(--mut);font-size:12px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.stat span{display:block;color:var(--mut);font-size:12px}.stat b{font-size:22px}
.stat b small{font-size:13px;color:var(--mut);font-weight:500}
.idx{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.ix{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.ix .t{font-size:12px;color:var(--mut)}.ix .c{font-size:17px;font-weight:600;margin:2px 0}
.ix .d{font-size:12px;display:flex;gap:10px;flex-wrap:wrap}.ix .d span{white-space:nowrap}
.alert{border-color:var(--acc);background:var(--accbg)}
.alert ul{margin:0;padding-left:18px}
.sec{display:grid;grid-template-columns:minmax(120px,1.3fr) 100px minmax(120px,2fr) 60px;gap:6px 12px;align-items:center}
.sec>div{padding:5px 0;border-bottom:1px solid var(--line)}
.sec .hd{color:var(--mut);font-size:12px;border-bottom:1px solid var(--line)}
.sec a{color:inherit;text-decoration:none}.sec a:hover{text-decoration:underline}
.kn{display:flex;align-items:center;gap:6px;justify-content:flex-end}
.dots{display:flex;gap:2px}.dots i{width:7px;height:7px;border-radius:50%;background:var(--line)}
.dots i.on{background:var(--pos)}
.bar{position:relative;display:block;height:8px;background:var(--soft);border-radius:4px;min-width:60px}
.bar::after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--mut);opacity:.5}
.bar i{position:absolute;top:0;bottom:0;border-radius:4px}
.bar i.r{left:50%}.bar i.l{right:50%}
.bar i.pos{background:var(--pos)}.bar i.neg{background:var(--neg)}
.right{text-align:right}
nav.chips{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0 8px;margin-top:8px;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}
nav.chips::-webkit-scrollbar{display:none}
.chip{flex:none;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:5px 12px;font-size:13px;cursor:pointer;text-decoration:none;font-family:inherit}
.chip.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.chip.sep{border:none;background:none;padding:5px 2px;color:var(--line);cursor:default}
.rows .row{display:grid;grid-template-columns:minmax(130px,1.6fr) 84px 58px 120px 40px 52px 58px 40px 40px 46px 132px;gap:0 8px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}
.rows .row:last-child{border-bottom:none}
.rows .hdr{color:var(--mut);font-size:12px;padding-top:0}
.ms{display:contents}
.m,.px{text-align:right}
.m60{display:flex;align-items:center;gap:6px;justify-content:flex-end}.m60 .bar{flex:1}
.nm b{display:block;font-weight:600}.nm span{color:var(--mut);font-size:12px}
.pill{display:inline-block;font-size:12px;padding:1px 8px;border-radius:999px;white-space:nowrap}
.pill.lt{background:var(--soft);color:var(--mut)}
.pill.md{background:var(--ambbg);color:var(--amb)}
.pill.hv{background:var(--viobg);color:var(--vio);font-weight:600}
.pill.ok{background:var(--accbg);color:var(--acc);font-weight:600}
.st{display:flex;gap:4px;justify-content:flex-end;flex-wrap:wrap}
details{margin-top:18px;color:var(--mut);font-size:13px}
details summary{cursor:pointer}
section.cat{scroll-margin-top:56px}
footer{margin-top:14px;color:var(--mut);font-size:12px}
footer a{color:var(--acc)}
@media(max-width:760px){
 .stats,.idx{grid-template-columns:repeat(2,1fr)}
 .sec{grid-template-columns:minmax(0,1.3fr) 44px minmax(0,1fr) 44px;gap:4px 8px;font-size:13px}
 .dots{display:none}
 .sec .hd{white-space:nowrap;font-size:11px}
 .ix .d{font-size:11px;gap:6px}
 .rows .hdr{display:none}
 .rows .row{grid-template-columns:1fr auto;gap:6px 8px;padding:10px 0}
 .nm{grid-column:1;grid-row:1/3}.px{grid-column:2;grid-row:1;font-size:15px;font-weight:600}
 .ms{display:grid;grid-column:1/-1;grid-row:3;grid-template-columns:repeat(4,1fr);gap:4px 8px}
 .m{text-align:left;font-size:13px}
 .m::before{content:attr(data-l);display:block;color:var(--mut);font-size:11px}
 .m60{display:block}.m60 .bar{margin-top:3px;min-width:0}
 .st{grid-column:2;grid-row:2}
}
"""

JS = """
const btns=document.querySelectorAll('[data-f]');
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.classList.toggle('on',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('section.cat').forEach(s=>{
    let n=0;
    s.querySelectorAll('.row[data-t]').forEach(r=>{
      const ok=f==='all'||r.dataset[f]==='1';r.hidden=!ok;if(ok)n++;});
    s.hidden=!n;
  });
});
"""


def write_html(df: pd.DataFrame, summ: pd.DataFrame, alerts: list[str],
               failed: list[str], path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    esc = html.escape
    stocks = df[df["category"] != "指數"]
    n_all, n60 = len(stocks), int(stocks["above60"].sum())
    n_ok = int((stocks["entry_ok"] == "✅").sum())
    n_hv = int((stocks["rating"] == "🔴重").sum())
    data_date = df["date"].max()

    parts = [f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>主題股均線追蹤</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>主題股均線追蹤</h1>
<div class="sub">資料日 {data_date}｜更新 {now}（台北）｜紅＝在均線上、綠＝在均線下</div></header>
<div class="stats">
<div class="stat"><span>站上 60MA</span><b>{n60}<small> / {n_all}</small></b></div>
<div class="stat"><span>符合條件</span><b class="acc">{n_ok}</b></div>
<div class="stat"><span>賣壓重</span><b>{n_hv}</b></div>
<div class="stat"><span>今日變化</span><b>{len(alerts)}</b></div>
</div>"""]

    idx = df[df["category"] == "指數"]
    if not idx.empty:
        parts.append('<div class="idx">')
        for _, r in idx.iterrows():
            parts.append(
                f"<div class='ix'><div class='t'>{esc(r['name'])}</div>"
                f"<div class='c'>{r['close']:,.2f}</div><div class='d'>"
                f"<span>距60 <b class='{_cls(r['vs60'])}'>{r['vs60']:+.1f}%</b></span>"
                f"<span>20日 <b class='{_cls(r['ret20'])}'>{r['ret20']:+.1f}%</b></span></div></div>")
        parts.append("</div>")

    if alerts:
        parts.append('<div class="card alert"><h2>今日變化</h2><ul>' +
                     "".join(f"<li>{esc(a)}</li>" for a in alerts) + "</ul></div>")

    cats = [c for c in WATCHLIST if c != "指數" and (df["category"] == c).any()]
    parts.append("<div class='card'><h2>類股強弱 <small>依平均距60MA排序</small></h2><div class='sec'>"
                 "<div class='hd'>分類</div><div class='hd right'>站上60</div>"
                 "<div class='hd'>平均距60MA</div><div class='hd right'>%</div>")
    for cat, r in summ.iterrows():
        k, n, avg = int(r["站上60MA"]), int(r["檔數"]), float(r["平均距60MA%"])
        dots = "".join(f"<i class='{'on' if i < k else ''}'></i>" for i in range(n))
        parts.append(
            f"<div><a href='#c{cats.index(cat)}'>{esc(cat)}</a></div>"
            f"<div class='kn'><span class='dots'>{dots}</span><span class='mut num'>{k}/{n}</span></div>"
            f"<div>{_bar(avg)}</div><div class='right num {_cls(avg)}'>{avg:+.1f}</div>")
    parts.append("</div></div>")

    parts.append("<nav class='chips'>"
                 "<button class='chip on' data-f='all'>全部</button>"
                 "<button class='chip' data-f='ok'>符合條件</button>"
                 "<button class='chip' data-f='up'>站上60</button>"
                 "<button class='chip' data-f='hv'>賣壓重</button><span class='chip sep'>|</span>" +
                 "".join(f"<a class='chip' href='#c{i}'>{esc(c)}</a>" for i, c in enumerate(cats)) +
                 "</nav>")

    hdr = ("<div class='row hdr'><div>股票</div><div class='px'>收盤</div>"
           "<div class='m'>距20%</div><div class='m'>距60%</div><div class='m'>60MA</div>"
           "<div class='m'>距高%</div><div class='m'>20日%</div><div class='m'>RSI</div>"
           "<div class='m'>出貨</div><div class='m'>量比</div><div class='st'>狀態</div></div>")
    for i, cat in enumerate(cats):
        sub = df[df["category"] == cat]
        k = int(sub["above60"].sum())
        parts.append(f"<section class='cat card' id='c{i}'><h2>{esc(cat)} "
                     f"<small>站上60 {k}/{len(sub)}｜平均距60 {sub['vs60'].mean():+.1f}%</small></h2>"
                     f"<div class='rows'>{hdr}")
        for _, r in sub.iterrows():
            cls, label = RATING_PILL.get(r["rating"], ("lt", r["rating"]))
            ok = r["entry_ok"] == "✅"
            pills = (f"<span class='pill {cls}'>{label}</span>" +
                     ("<span class='pill ok'>符合條件</span>" if ok else ""))
            parts.append(
                f"<div class='row' data-t='1' data-ok='{int(ok)}' data-up='{int(r['above60'])}' "
                f"data-hv='{int(cls == 'hv')}'>"
                f"<div class='nm'><b>{esc(r['name'])}</b>"
                f"{'' if r['name'] == r['ticker'] else '<span>' + esc(r['ticker']) + '</span>'}</div>"
                f"<div class='px'>{r['close']:,.2f}</div><div class='ms'>"
                f"<div class='m {_cls(r['vs20'])}' data-l='距20MA'>{r['vs20']:+.1f}</div>"
                f"<div class='m m60' data-l='距60MA'><span class='v {_cls(r['vs60'])}'>{r['vs60']:+.1f}</span>{_bar(r['vs60'])}</div>"
                f"<div class='m' data-l='60MA'>{r['ma60_dir']}</div>"
                f"<div class='m' data-l='距高'>{r['dd_1y']:.0f}</div>"
                f"<div class='m {_cls(r['ret20'])}' data-l='20日'>{r['ret20']:+.1f}</div>"
                f"<div class='m' data-l='RSI'>{r['rsi']:.0f}</div>"
                f"<div class='m' data-l='出貨'>{int(r['dist_days'])}</div>"
                f"<div class='m' data-l='量比'>{_num(r['vol_ratio'], '{:.2f}')}</div></div>"
                f"<div class='st'>{pills}</div></div>")
        parts.append("</div></section>")

    parts.append("""<details><summary>指標說明</summary><p>
距20%／距60%：收盤相對 20／60 日均線的距離。60MA：近 5 日均線方向。距高%：距一年高點。<br>
出貨：近 25 日收黑且量增的天數。量比：近 5 日均量 ÷ 前 50 日均量。<br>
賣壓輕＝站上 60MA；賣壓中＝跌破 60MA；賣壓重＝跌破 60MA 且 60MA 下彎、距一年高點跌逾 30%。<br>
符合條件＝站上 60MA 且 RSI14 &lt; 70，僅為篩選條件，非買賣建議。</p></details>""")
    if failed:
        parts.append(f"<p class='mut'>抓取失敗：{esc(', '.join(failed))}</p>")
    parts.append("<footer>資料：Yahoo Finance｜<a href='latest.csv'>下載 CSV</a>｜"
                 "<a href='https://github.com/filcoabs-MUR/ma-tracker/blob/main/output/history.csv'>歷史紀錄</a>"
                 "｜僅數據整理，非投資建議</footer>")
    parts.append(f"</div><script>{JS}</script></body></html>")
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
