#!/usr/bin/env python3
"""台指（加權指數）年線反彈情境追蹤 — 固定腳本。

規則固定寫在這裡，AI 或排程只負責「提供資料、執行、讀結果」，不重寫計算。

劇本判定（優先順序 C > B > A > 進行中）：
  C 續跌   ：最新收盤 < 錨點低點（預設 2026-11-20 低點 35,739）
  B V 型   ：最近 3 日收盤皆站上各自當日年線（MA240），
             且外資台指期淨空單較錨點日減少 >= 10,000 口
             （未提供籌碼 → B_PRICE_ONLY：價格達標、籌碼未確認）
  A 2018型 ：本波反彈中曾「最高價觸及年線 ±1% 但收盤站不上」，
             或「收盤站上年線後 3 日內跌回」，且最新收盤 < 年線
  ABOVE    ：收盤站上年線但未滿 3 日
  RALLY    ：以上皆非（反彈進行中／整理中）

Minervini / O'Neil 落底計數：
  - 樞紐低點 = 錨點日（含）之後的最低「最低價」；被跌破即重設
  - 反彈第 1 天 = 樞紐低點之後第一根收漲日
  - 追價確認日（FTD）= 反彈第 4～7 天，漲幅 >= 1.5% 且成交量大於前一日

用法：
  python taiex_rebound.py                       # 用 yfinance 抓 ^TWII
  python taiex_rebound.py --csv daily.csv       # 用自備日線（date,open,high,low,close,volume）
  python taiex_rebound.py --oi-anchor 105000 --oi-today 93000

輸出：output/taiex_rebound.json、output/taiex_rebound.md
僅為數據與規則判讀，不構成投資建議。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 固定參數 -------------------------------------------------------------
ANCHOR_DATE = "2026-11-20"
ANCHOR_LOW = 35739.0           # 11/20 盤中低點（TWSE）
PRIOR_LOW = 37837.0            # 11/6 前低
SWING_HIGH = 49968.0           # 10/6 高點
SUPPORT_ZONE = (33000.0, 34000.0)
MA_WINDOWS = (20, 60, 240)
YEARLINE = 240
TOUCH_BAND = 0.01              # 觸及年線 ±1%
HOLD_DAYS = 3                  # 站穩天數 / 假突破判定天數
OI_DROP_REQUIRED = 10_000      # 外資淨空單需減少口數
FTD_MIN_PCT = 1.5
FTD_DAY_RANGE = (4, 7)
CHART_DAYS = 80                # 網頁圖表顯示天數
# 劇本示意路徑：(錨點後第 k 個交易日, 點位)，只在錨點後 25 個交易日內顯示
SCENARIO_PATHS = {
    "A": [(0, 36033), (7, 39400), (17, 35600), (25, 37600)],
    "B": [(0, 36033), (6, 39500), (12, 41300), (18, 40400), (25, 42500)],
    "C": [(0, 36033), (4, 37300), (13, 33500), (25, 34600)],
}

SCENARIO_TEXT = {
    "C": "C 續跌：收盤跌破錨點低點",
    "B": "B V型：站穩年線 3 日且外資空單明顯回補",
    "B_PRICE_ONLY": "B（價格達標）：站穩年線 3 日，外資籌碼未確認",
    "A": "A 2018型：反彈至年線遇壓",
    "ABOVE": "站上年線（未滿 3 日）",
    "RALLY": "反彈進行中／整理中",
}


@dataclass
class Result:
    as_of: str
    close: float
    change_pct: float
    ma20: float
    ma60: float
    ma240: float
    ma240_slope5: float
    dist_ma240: float
    dist_ma240_pct: float
    bias_ma20_pct: float
    scenario: str
    scenario_text: str
    reason: str
    pivot_low: float
    pivot_date: str
    rally_day: int | None
    rally_day1: str | None
    ftd_today: bool
    ftd_date: str | None
    volume_available: bool
    oi_anchor: int | None
    oi_today: int | None
    oi_change: int | None
    levels: dict


# ---- 資料 -----------------------------------------------------------------
def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "turnover_billion" in df.columns and "volume" not in df.columns:
        df["volume"] = df["turnover_billion"]
    df["date"] = pd.to_datetime(df["date"])
    return normalize(df.set_index("date"))


def fetch_yf(period: str = "2y") -> pd.DataFrame:
    import yfinance as yf  # 延遲匯入，測試不需要網路

    raw = yf.download("^TWII", period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    return normalize(raw)


# ---- 證交所 API（GitHub Actions 用；英文版日期為西元）---------------------------
TWSE_OHLC = "https://www.twse.com.tw/rwd/en/TAIEX/MI_5MINS_HIST?date={ym}01&response=json"
TWSE_VALUE = "https://www.twse.com.tw/rwd/en/afterTrading/FMTQIK?date={ym}01&response=json"


def _num(x) -> float:
    return float(str(x).replace(",", "").strip())


def _date(x: str) -> pd.Timestamp:
    y, m, d = str(x).strip().split("/")
    y = int(y)
    return pd.Timestamp(y + 1911 if y < 1911 else y, int(m), int(d))


def _col(fields: list[str], *keys: str) -> int:
    for i, f in enumerate(fields):
        fl = str(f).lower()
        if any(k in fl for k in keys):
            return i
    raise ValueError(f"證交所欄位找不到 {keys}：{fields}")


def parse_twse_ohlc(js: dict) -> pd.DataFrame:
    if js.get("stat") != "OK" or not js.get("data"):
        return pd.DataFrame()
    f = js["fields"]
    io, ih, il, ic = (_col(f, "open", "開盤"), _col(f, "high", "最高"),
                      _col(f, "low", "最低"), _col(f, "clos", "收盤"))
    rows = [(_date(r[0]), _num(r[io]), _num(r[ih]), _num(r[il]), _num(r[ic])) for r in js["data"]]
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"]).set_index("date")


def parse_twse_value(js: dict) -> pd.Series:
    if js.get("stat") != "OK" or not js.get("data"):
        return pd.Series(dtype=float)
    iv = _col(js["fields"], "trade value", "成交金額")
    return pd.Series({_date(r[0]): _num(r[iv]) / 1e8 for r in js["data"]}, name="volume")


def fetch_twse(months: int = 14, get=None, pause: float = 1.5, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """抓加權指數日線（開高低收）＋成交金額（億），約 14 個月以滿足年線。get 可注入以便測試。"""
    import time

    if get is None:
        import requests

        sess = requests.Session()
        sess.headers["User-Agent"] = "Mozilla/5.0 (ma-tracker)"

        def get(url):
            r = sess.get(url, timeout=20)
            r.raise_for_status()
            return r.json()

    today = pd.Timestamp.today().normalize() if today is None else today
    ohlc, val = [], []
    for k in range(months - 1, -1, -1):
        ym = (today - pd.DateOffset(months=k)).strftime("%Y%m")
        ohlc.append(parse_twse_ohlc(get(TWSE_OHLC.format(ym=ym))))
        if pause:
            time.sleep(pause)
        val.append(parse_twse_value(get(TWSE_VALUE.format(ym=ym))))
        if pause:
            time.sleep(pause)
    df = pd.concat([d for d in ohlc if not d.empty])
    df["volume"] = pd.concat([v for v in val if not v.empty]).reindex(df.index)
    return normalize(df)


# ---- FinMind（證交所 API 在海外主機失敗時的主要備援；不需 token）------------------
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def parse_finmind(js: dict) -> pd.DataFrame:
    if js.get("msg") != "success" or not js.get("data"):
        raise ValueError(f"FinMind 無資料：{js.get('msg')}")
    df = pd.DataFrame(js["data"])
    df = df.rename(columns={"max": "high", "min": "low"})
    df["volume"] = df["Trading_money"].astype(float) / 1e8  # 元 → 億
    df["date"] = pd.to_datetime(df["date"])
    return normalize(df.set_index("date"))


def fetch_finmind(days: int = 450, get=None, today: pd.Timestamp | None = None) -> pd.DataFrame:
    import os

    today = pd.Timestamp.today().normalize() if today is None else today
    params = {"dataset": "TaiwanStockPrice", "data_id": "TAIEX",
              "start_date": (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")}
    if os.environ.get("FINMIND_TOKEN"):
        params["token"] = os.environ["FINMIND_TOKEN"]
    if get is None:
        import requests

        def get(p):
            r = requests.get(FINMIND_URL, params=p, timeout=30)
            r.raise_for_status()
            return r.json()
    return parse_finmind(get(params))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"日線缺少欄位：{missing}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    out = df[need + ["volume"]].astype(float).sort_index()
    out = out[~out.index.duplicated(keep="last")].dropna(subset=["close"])
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


# ---- 計算 -----------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < YEARLINE:
        raise ValueError(f"資料只有 {len(df)} 根，年線需要至少 {YEARLINE} 根")
    out = df.copy()
    for n in MA_WINDOWS:
        out[f"ma{n}"] = out["close"].rolling(n, min_periods=n).mean()
    out["chg_pct"] = out["close"].pct_change() * 100
    return out


def rally_state(df: pd.DataFrame, anchor_date: str = ANCHOR_DATE) -> dict:
    """樞紐低點、反彈天數與追價確認日。df 需含 low/close/volume/chg_pct。"""
    since = df.loc[df.index >= pd.Timestamp(anchor_date)]
    if since.empty:
        raise ValueError(f"資料沒有涵蓋錨點日 {anchor_date} 之後")
    pivot_date = since["low"].idxmin()
    pivot_low = float(since.loc[pivot_date, "low"])

    after = since.loc[since.index > pivot_date]
    up = after[after["chg_pct"] > 0]
    if up.empty:
        return dict(pivot_low=pivot_low, pivot_date=pivot_date, rally_day=None,
                    rally_day1=None, ftd_today=False, ftd_date=None)

    day1 = up.index[0]
    rally = since.loc[since.index >= day1]
    days = np.arange(1, len(rally) + 1)
    vol_up = rally["volume"] > df["volume"].shift(1).reindex(rally.index)
    ftd_mask = (
        (days >= FTD_DAY_RANGE[0]) & (days <= FTD_DAY_RANGE[1])
        & (rally["chg_pct"] >= FTD_MIN_PCT).to_numpy()
        & vol_up.fillna(False).to_numpy()
    )
    ftd_dates = rally.index[ftd_mask]
    return dict(
        pivot_low=pivot_low,
        pivot_date=pivot_date,
        rally_day=int(days[-1]),
        rally_day1=day1,
        ftd_today=bool(ftd_mask[-1]),
        ftd_date=ftd_dates[0] if len(ftd_dates) else None,
    )


def classify(df: pd.DataFrame, rs: dict, oi_anchor: int | None = None,
             oi_today: int | None = None, anchor_low: float = ANCHOR_LOW,
             anchor_date: str = ANCHOR_DATE) -> tuple[str, str]:
    last = df.iloc[-1]
    close, ma = float(last["close"]), float(last[f"ma{YEARLINE}"])

    if close < anchor_low:
        return "C", f"收盤 {close:,.0f} 跌破錨點低點 {anchor_low:,.0f}"

    above = df["close"] > df[f"ma{YEARLINE}"]
    tail = above.iloc[-HOLD_DAYS:]
    if len(tail) == HOLD_DAYS and tail.all():
        if oi_anchor is not None and oi_today is not None:
            drop = oi_anchor - oi_today
            if drop >= OI_DROP_REQUIRED:
                return "B", f"連 {HOLD_DAYS} 日站上年線，外資淨空單減少 {drop:,} 口"
            return "B_PRICE_ONLY", f"連 {HOLD_DAYS} 日站上年線，但外資淨空單只減少 {drop:,} 口"
        return "B_PRICE_ONLY", f"連 {HOLD_DAYS} 日站上年線，未提供外資籌碼"

    # A 只看本波反彈（反彈第 1 天起；尚未反彈則看錨點日起）
    start = rs["rally_day1"] if rs["rally_day1"] is not None else pd.Timestamp(anchor_date)
    seg = df.loc[df.index >= start]
    seg_above = above.loc[seg.index]
    touched_fail = (seg["high"] >= seg[f"ma{YEARLINE}"] * (1 - TOUCH_BAND)) & ~seg_above
    false_break = False
    vals = seg_above.to_numpy()
    for i in range(1, len(vals)):
        if vals[i - 1] and not vals[i]:
            run = 0
            j = i - 1
            while j >= 0 and vals[j]:
                run += 1
                j -= 1
            if run < HOLD_DAYS:
                false_break = True
    if close < ma and (touched_fail.any() or false_break):
        why = "收盤站上年線後 3 日內跌回" if false_break else "最高價觸及年線但收盤站不上"
        return "A", why

    if close >= ma:
        n = 0
        for v in reversed(above.to_numpy()):
            if not v:
                break
            n += 1
        return "ABOVE", f"站上年線第 {n} 日"
    return "RALLY", f"距年線還有 {ma - close:,.0f} 點"


def evaluate(df: pd.DataFrame, oi_anchor: int | None = None, oi_today: int | None = None,
             anchor_date: str | None = None, anchor_low: float | None = None) -> Result:
    anchor_date = ANCHOR_DATE if anchor_date is None else anchor_date
    anchor_low = ANCHOR_LOW if anchor_low is None else anchor_low
    d = add_indicators(df)
    if np.isnan(d[f"ma{YEARLINE}"].iloc[-1]):
        raise ValueError("最新一日年線無法計算")
    rs = rally_state(d, anchor_date)
    scen, reason = classify(d, rs, oi_anchor, oi_today, anchor_low, anchor_date)
    last = d.iloc[-1]
    ma240 = float(last["ma240"])
    fmt = lambda t: None if t is None else pd.Timestamp(t).strftime("%Y-%m-%d")
    fib382 = anchor_low + (SWING_HIGH - anchor_low) * 0.382
    return Result(
        as_of=fmt(d.index[-1]),
        close=round(float(last["close"]), 2),
        change_pct=round(float(last["chg_pct"]), 2),
        ma20=round(float(last["ma20"]), 0),
        ma60=round(float(last["ma60"]), 0),
        ma240=round(ma240, 0),
        ma240_slope5=round(ma240 - float(d["ma240"].iloc[-6]), 0),
        dist_ma240=round(ma240 - float(last["close"]), 0),
        dist_ma240_pct=round((ma240 / float(last["close"]) - 1) * 100, 2),
        bias_ma20_pct=round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2),
        scenario=scen,
        scenario_text=SCENARIO_TEXT[scen],
        reason=reason,
        pivot_low=rs["pivot_low"],
        pivot_date=fmt(rs["pivot_date"]),
        rally_day=rs["rally_day"],
        rally_day1=fmt(rs["rally_day1"]),
        ftd_today=rs["ftd_today"],
        ftd_date=fmt(rs["ftd_date"]),
        volume_available=bool(d["volume"].iloc[-2:].notna().all() and (d["volume"].iloc[-2:] > 0).all()),
        oi_anchor=oi_anchor,
        oi_today=oi_today,
        oi_change=None if oi_anchor is None or oi_today is None else oi_today - oi_anchor,
        levels={
            "anchor_low": anchor_low,
            "prior_low": PRIOR_LOW,
            "fib_0382": round(fib382, 0),
            "support_zone": list(SUPPORT_ZONE),
        },
    )


# ---- 輸出 -----------------------------------------------------------------
def to_markdown(r: Result, prev_scenario: str | None = None) -> str:
    changed = prev_scenario is not None and prev_scenario != r.scenario
    flag = "【劇本變化】" if (changed or r.ftd_today) else ""
    oi = "未取得" if r.oi_today is None else f"{r.oi_today:,} 口（較錨點 {r.oi_change:+,}）"
    rally = "尚未出現收漲日" if r.rally_day is None else f"第 {r.rally_day} 天（自 {r.rally_day1}）"
    ftd = "是" if r.ftd_today else ("否" if r.ftd_date is None else f"否（本波已於 {r.ftd_date} 出現）")
    if not r.volume_available:
        ftd += "；量能資料缺，無法判定"
    lv = r.levels
    return "\n".join([
        f"{flag}{r.as_of}｜{r.scenario_text}｜{r.reason}",
        "",
        "| 項目 | 數值 |",
        "|---|---|",
        f"| 收盤 | {r.close:,.0f}（{r.change_pct:+.2f}%） |",
        f"| 年線 MA240 | {r.ma240:,.0f}（5日 {r.ma240_slope5:+,.0f}） |",
        f"| 距年線 | {r.dist_ma240:+,.0f} 點（{r.dist_ma240_pct:+.2f}%） |",
        f"| MA20／MA60 | {r.ma20:,.0f}／{r.ma60:,.0f} |",
        f"| MA20 乖離 | {r.bias_ma20_pct:+.2f}% |",
        "",
        "| 籌碼與落底 | 數值 |",
        "|---|---|",
        f"| 外資台指期淨空單 | {oi} |",
        f"| 樞紐低點 | {r.pivot_low:,.0f}（{r.pivot_date}） |",
        f"| 反彈天數 | {rally} |",
        f"| 追價確認日 | {ftd} |",
        "",
        f"關鍵價位：年線 {r.ma240:,.0f}／0.382 回撤 {lv['fib_0382']:,.0f}／"
        f"前低 {lv['prior_low']:,.0f}／錨點低 {lv['anchor_low']:,.0f}／"
        f"支撐區 {lv['support_zone'][0]:,.0f}～{lv['support_zone'][1]:,.0f}",
        "",
        "僅為數據與規則判讀，不構成投資建議。",
    ])


def chart_series(df: pd.DataFrame, n: int = CHART_DAYS) -> dict:
    d = add_indicators(df).tail(n)
    r = lambda col: [None if pd.isna(v) else round(float(v), 0) for v in d[col]]
    return {"dates": [t.strftime("%Y-%m-%d") for t in d.index], "close": r("close"),
            "ma20": r("ma20"), "ma60": r("ma60"), "ma240": r("ma240")}


SVG_STYLE = {
    "close": ("var(--fg)", 2.2, ""), "ma20": ("#eda100", 1.4, ""), "ma60": ("#e87ba4", 1.4, ""),
    "ma240": ("#1baf7a", 2.4, ""), "A": ("#2563eb", 1.8, "6 4"), "B": ("#16a34a", 2.0, "2 3"),
    "C": ("#d6312f", 1.8, "8 4"),
}


def _interp(pts: list[tuple[int, float]]) -> dict[int, float]:
    out = {}
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        for k in range(a, b + 1):
            out[k] = va + (vb - va) * (k - a) / (b - a)
    return out


def to_svg_chart(d: dict, anchor_date: str | None = None) -> str:
    """指數＋均線＋劇本示意線的 inline SVG（不需外部 JS，深色模式沿用頁面變數）。"""
    s = d.get("series")
    if not s or not s.get("dates"):
        return ""
    anchor_date = ANCHOR_DATE if anchor_date is None else anchor_date
    dates, H = s["dates"], len(s["dates"])
    ai = dates.index(anchor_date) if anchor_date in dates else None
    after = H - 1 - ai if ai is not None else None
    future = 0
    scen = {}
    if ai is not None and after <= 25:
        future = 25 - after
        for k, pts in SCENARIO_PATHS.items():
            scen[k] = {ai + off: v for off, v in _interp(pts).items()}
    N = H + future
    lv = d["levels"]
    hlines = [(lv["fib_0382"], "0.382"), (lv["prior_low"], "前低"), (lv["anchor_low"], "錨點")]
    vals = [v for k in ("close", "ma20", "ma60", "ma240") for v in s[k] if v is not None]
    vals += [v for m in scen.values() for v in m.values()] + [v for v, _ in hlines]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 100
    lo, hi = lo - pad, hi + pad
    W, Ht, L, R, T, B = 680, 300, 8, 118, 12, 26
    X = lambda i: L + (W - L - R) * i / max(N - 1, 1)
    Y = lambda v: T + (Ht - T - B) * (hi - v) / (hi - lo)

    def line(pts: dict[int, float], key: str) -> str:
        color, w, dash = SVG_STYLE[key]
        xy = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in sorted(pts.items()) if v is not None)
        if not xy:
            return ""
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return (f"<polyline points='{xy}' fill='none' stroke='{color}' stroke-width='{w}'"
                f" stroke-linejoin='round' stroke-linecap='round'{da}/>")

    g = []
    for v, t in hlines:
        if lo < v < hi:
            g.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='var(--mut)'"
                     f" stroke-opacity='.5' stroke-dasharray='3 3'/>"
                     f"<text x='{W - R + 4}' y='{Y(v) + 5:.1f}' font-size='15' fill='var(--mut)'>{t} {v:,.0f}</text>")
    if ai is not None:
        g.append(f"<line x1='{X(ai):.1f}' x2='{X(ai):.1f}' y1='{T}' y2='{Ht - B}' stroke='var(--mut)' stroke-opacity='.35'/>")
    for k in ("ma60", "ma20", "ma240"):
        g.append(line(dict(enumerate(s[k])), k))
    for k in ("A", "B", "C"):
        if k in scen:
            g.append(line(scen[k], k))
    g.append(line(dict(enumerate(s["close"])), "close"))
    last = s["close"][-1]
    g.append(f"<circle cx='{X(H - 1):.1f}' cy='{Y(last):.1f}' r='3.5' fill='var(--fg)' stroke='var(--card)' stroke-width='2'/>")
    ma = s["ma240"][-1]
    if ma is not None:
        g.append(f"<text x='{W - R + 4}' y='{Y(ma) + 5:.1f}' font-size='15' fill='#1baf7a'>年線 {ma:,.0f}</text>")
    ticks = {0: dates[0][5:], H - 1: dates[-1][5:]}
    if ai is not None and 6 < ai < H - 7:
        ticks[ai] = dates[ai][5:]
    if future:
        ticks[N - 1] = f"+{future}日"
    for i, t in ticks.items():
        anchor = "start" if i == 0 else "end" if i == N - 1 else "middle"
        g.append(f"<text x='{X(i):.1f}' y='{Ht - 6}' font-size='15' fill='var(--mut)' text-anchor='{anchor}'>{t}</text>")

    def key(color, dash, label, w=2):
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return (f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:10px;white-space:nowrap'>"
                f"<svg width='18' height='6'><line x1='0' x2='18' y1='3' y2='3' stroke='{color}' stroke-width='{w}'{da}/></svg>{label}</span>")
    legend = ("<div class='fn' style='margin-top:6px'>" + key("var(--fg)", "", "指數") + key("#eda100", "", "MA20")
              + key("#e87ba4", "", "MA60") + key("#1baf7a", "", "年線 MA240", 3)
              + (key("#2563eb", "6 4", "A 遇壓回測") + key("#16a34a", "2 3", "B V型") + key("#d6312f", "8 4", "C 續跌") if scen else "")
              + "</div>")
    return (legend + f"<svg viewBox='0 0 {W} {Ht}' width='100%' role='img' "
            f"aria-label='加權指數近 {H} 日與 MA20、MA60、年線走勢' style='display:block;margin-top:4px'>"
            + "".join(g) + "</svg>"
            + ("<p class='fn' style='margin-top:2px'>劇本線為 11/20 當天畫的示意路徑，不是預測；錨點後 25 個交易日自動移除。</p>" if scen else ""))


SOURCE_NOTE = {
    "finmind": "資料：FinMind（證交所開放資料，含成交金額）；網頁未含外資期貨籌碼，B 劇本最多判到「價格達標」，完整判讀見盤後排程。",
    "twse": "資料：證交所（含成交金額）；網頁未含外資期貨籌碼，B 劇本最多判到「價格達標」，完整判讀見盤後排程。",
    "csv": "資料：自備日線。",
    "yahoo": "資料：Yahoo ^TWII（證交所與 FinMind 當日都失敗時的備援，無量能與籌碼）；完整判讀以盤後排程為準。",
}

PILL = {"C": "hv", "B": "ok", "ABOVE": "ok", "B_PRICE_ONLY": "md", "A": "md", "RALLY": "lt"}


def to_html_card(d: dict) -> str:
    """給 ma_tracker 網頁用的卡片（沿用 ma_tracker 的 CSS：card／idx／ix／pill／pos／neg）。"""
    import html as _h

    e = _h.escape
    cls = lambda v: "pos" if v > 0 else "neg" if v < 0 else ""
    lv = d["levels"]
    flag = "<span class='pill ok'>追價確認日</span> " if d.get("ftd_today") else ""
    rally = "尚未出現收漲日" if d.get("rally_day") is None else f"第 {d['rally_day']} 天"
    ftd = "是" if d.get("ftd_today") else (f"已於 {e(d['ftd_date'])}" if d.get("ftd_date") else "否")
    if not d.get("volume_available", True):
        ftd += "（缺量能）"
    oi = "未取得" if d.get("oi_today") is None else f"{d['oi_today']:,}（{d['oi_change']:+,}）"
    tile = lambda t, c, sub: (f"<div class='ix'><div class='t'>{t}</div><div class='c'>{c}</div>"
                              f"<div class='d'>{sub}</div></div>")
    return (
        f"<div class='card' id='taiex'><h2>台指年線反彈追蹤 <small>{e(d['as_of'])}</small></h2>"
        f"<p>{flag}<span class='pill {PILL.get(d['scenario'], 'lt')}'>{e(d['scenario_text'])}</span> "
        f"<span class='mut'>{e(d['reason'])}</span></p>"
        + to_svg_chart(d)
        + "<div class='idx'>"
        + tile("加權指數", f"{d['close']:,.0f}",
               f"<span>今日 <b class='{cls(d['change_pct'])}'>{d['change_pct']:+.2f}%</b></span>")
        + tile("年線 MA240", f"{d['ma240']:,.0f}",
               f"<span>距年線 <b>{d['dist_ma240']:+,.0f}</b>（{d['dist_ma240_pct']:+.2f}%）</span>")
        + tile("MA20 乖離", f"<span class='{cls(d['bias_ma20_pct'])}'>{d['bias_ma20_pct']:+.2f}%</span>",
               f"<span>MA20 {d['ma20']:,.0f}／MA60 {d['ma60']:,.0f}</span>")
        + tile("反彈天數", rally, f"<span>追價確認 {ftd}</span>")
        + "</div>"
        f"<p class='fn'>外資台指期淨空單：{oi}｜樞紐低點 {d['pivot_low']:,.0f}（{e(d['pivot_date'])}）<br>"
        f"關鍵價位：年線 {d['ma240']:,.0f}／0.382 回撤 {lv['fib_0382']:,.0f}／前低 {lv['prior_low']:,.0f}／"
        f"錨點低 {lv['anchor_low']:,.0f}／支撐區 {lv['support_zone'][0]:,.0f}～{lv['support_zone'][1]:,.0f}<br>"
        "劇本：A 年線遇壓回測｜B 站穩年線 3 日且外資空單減少 1 萬口｜C 跌破錨點低點。"
        + SOURCE_NOTE.get(d.get("source"), SOURCE_NOTE["yahoo"])
        + "僅數據整理，非投資建議。</p></div>"
    )


def load_auto(source: str = "auto") -> tuple[pd.DataFrame, str]:
    """依序嘗試證交所 → FinMind → Yahoo；指定來源時只用該來源（失敗就報錯）。"""
    fetchers = {"twse": lambda: fetch_twse(), "finmind": lambda: fetch_finmind(), "yahoo": lambda: fetch_yf()}
    order = ["twse", "finmind", "yahoo"] if source == "auto" else [source]
    last = None
    for name in order:
        try:
            return fetchers[name](), name
        except Exception as ex:
            last = ex
            print(f"{name} 失敗：{type(ex).__name__} {ex}")
    raise RuntimeError(f"所有資料來源都失敗：{last}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="台指年線反彈情境追蹤")
    ap.add_argument("--csv", help="自備日線 CSV（date,open,high,low,close,volume 或 turnover_billion）")
    ap.add_argument("--oi-anchor", type=int, help="錨點日外資台指期淨空單（口，正數）")
    ap.add_argument("--oi-today", type=int, help="今日外資台指期淨空單（口，正數）")
    ap.add_argument("--source", default="auto", choices=["auto", "twse", "finmind", "yahoo"],
                    help="資料來源；auto = 證交所 → FinMind → Yahoo")
    ap.add_argument("--out", default="output", help="輸出資料夾")
    ap.add_argument("--hy", type=float, help="美國高收益債 OAS（%，FRED BAMLH0A0HYM2）；僅私人通知用")
    ap.add_argument("--hy-20d", type=float, help="20 個交易日前的 HY OAS（%）")
    ap.add_argument("--ccc", type=float, help="CCC 級 OAS（%，FRED BAMLH0A3HYC）")
    a = ap.parse_args(argv)

    if a.csv:
        df, source = load_csv(a.csv), "csv"
    else:
        df, source = load_auto(a.source)
    r = evaluate(df, a.oi_anchor, a.oi_today)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "taiex_rebound.json"
    prev = None
    if jpath.exists():
        try:
            old = json.loads(jpath.read_text(encoding="utf-8"))
            if old.get("as_of") != r.as_of:
                prev = old.get("scenario")
            else:
                prev = old.get("prev_scenario")
        except (json.JSONDecodeError, OSError):
            prev = None
    payload = asdict(r) | {"prev_scenario": prev, "series": chart_series(df), "source": source}
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(r, prev)
    if a.hy is not None:  # ICE 數據僅供個人使用：只進 md／json，不進網頁卡片
        from credit_spread import assess, to_markdown as credit_md

        c = assess(a.hy, a.hy_20d, a.ccc)
        payload["credit"] = asdict(c)
        jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md = md.replace("\n\n僅為數據與規則判讀", "\n\n" + credit_md(c) + "\n\n僅為數據與規則判讀")
    (out / "taiex_rebound.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
