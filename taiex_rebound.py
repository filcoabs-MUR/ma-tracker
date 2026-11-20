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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="台指年線反彈情境追蹤")
    ap.add_argument("--csv", help="自備日線 CSV（date,open,high,low,close,volume 或 turnover_billion）")
    ap.add_argument("--oi-anchor", type=int, help="錨點日外資台指期淨空單（口，正數）")
    ap.add_argument("--oi-today", type=int, help="今日外資台指期淨空單（口，正數）")
    ap.add_argument("--out", default="output", help="輸出資料夾")
    a = ap.parse_args(argv)

    df = load_csv(a.csv) if a.csv else fetch_yf()
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
    payload = asdict(r) | {"prev_scenario": prev}
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(r, prev)
    (out / "taiex_rebound.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
