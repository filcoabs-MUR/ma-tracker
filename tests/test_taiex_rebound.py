"""taiex_rebound.py 的規則測試（全部用合成資料，不需網路）。

前段 250 日固定 40,000 點 → 年線約 40,000；錨點日低點 35,000。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import taiex_rebound as tr  # noqa: E402

ANCHOR_LOW = 35000.0


def build(post: list[tuple[float, float, float, float]], vols: list[float] | None = None,
          base: float = 40000.0, n_base: int = 250):
    """post：錨點日（含）之後每日 (high, low, close, volume_or_None)。回傳 (df, anchor_date)。"""
    n = n_base + len(post)
    idx = pd.bdate_range("2025-01-01", periods=n)
    rows = [(base, base + 50, base - 50, base, 1000.0)] * n_base
    prev = base
    for h, l, c, v in post:
        rows.append((prev, h, l, c, 1000.0 if v is None else v))
        prev = c
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])
    return df, idx[n_base].strftime("%Y-%m-%d")


def ev(post, oi_anchor=None, oi_today=None):
    df, anchor = build(post)
    return tr.evaluate(df, oi_anchor, oi_today, anchor_date=anchor, anchor_low=ANCHOR_LOW)


# ---- 均線 ------------------------------------------------------------------
def test_yearline_is_240_not_200():
    """今天實際踩過的錯：把 MA200 當年線。上升趨勢中兩者差很多。"""
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = np.linspace(30000, 45000, 300)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1.0}, index=idx)
    d = tr.add_indicators(df)
    assert d["ma240"].iloc[-1] == pytest.approx(close[-240:].mean())
    assert abs(d["ma240"].iloc[-1] - close[-200:].mean()) > 500


def test_not_enough_data_raises():
    idx = pd.bdate_range("2024-01-01", periods=239)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    with pytest.raises(ValueError, match="年線"):
        tr.add_indicators(df)


# ---- 劇本 ------------------------------------------------------------------
def test_c_break_anchor_low():
    r = ev([(36500, 35000, 35500, None), (35600, 34500, 34800, None)])
    assert r.scenario == "C"


def test_rally_in_progress():
    r = ev([(36500, 35000, 35500, None), (36800, 35400, 36600, None), (37600, 36500, 37500, None)])
    assert r.scenario == "RALLY"
    assert r.rally_day == 2
    assert "距年線" in r.reason


def test_a_touch_and_fail():
    r = ev([(36500, 35000, 35500, None), (37000, 35500, 36900, None),
            (39800, 37000, 38500, None), (38800, 37800, 38000, None)])
    assert r.scenario == "A"
    assert "觸及" in r.reason


def test_a_false_breakout():
    r = ev([(36500, 35000, 35500, None), (38000, 35500, 37900, None),
            (40500, 38000, 40400, None), (40600, 39000, 39200, None)])
    assert r.scenario == "A"
    assert "跌回" in r.reason


def test_above_less_than_3_days():
    r = ev([(36500, 35000, 35500, None), (38000, 35500, 37900, None),
            (40600, 38000, 40500, None), (40800, 40200, 40600, None)])
    assert r.scenario == "ABOVE"
    assert "第 2 日" in r.reason


ABOVE3 = [(36500, 35000, 35500, None), (38000, 35500, 37900, None), (40600, 38000, 40500, None),
          (40800, 40200, 40600, None), (41000, 40400, 40800, None)]


def test_b_needs_oi_drop():
    assert ev(ABOVE3, oi_anchor=100_000, oi_today=88_000).scenario == "B"
    assert ev(ABOVE3, oi_anchor=100_000, oi_today=95_000).scenario == "B_PRICE_ONLY"


def test_b_price_only_without_oi():
    assert ev(ABOVE3).scenario == "B_PRICE_ONLY"


# ---- 反彈天數與追價確認日 -----------------------------------------------------
def test_rally_resets_on_new_low():
    """樞紐低點被跌破，反彈天數要重算。"""
    post = [(36500, 35200, 35500, None), (36500, 35500, 36400, None), (36600, 35800, 36000, None),
            (36100, 35100, 35300, None), (36000, 35200, 35900, None)]
    df, anchor = build(post)
    r = tr.evaluate(df, anchor_date=anchor, anchor_low=34000)
    assert r.pivot_low == 35100
    assert r.rally_day == 1


def _ftd_post(day4_chg_pct: float, day4_vol: float):
    p = [(36500, 35000, 35500, 900), (36000, 35400, 35800, 900), (36100, 35700, 35900, 900),
         (36200, 35800, 36000, 1000)]
    c = 36000 * (1 + day4_chg_pct / 100)
    p.append((c + 50, 36000, c, day4_vol))
    return p


def test_ftd_on_day4_with_volume():
    r = ev(_ftd_post(2.0, 1500))
    assert r.rally_day == 4
    assert r.ftd_today is True


def test_no_ftd_when_volume_down():
    assert ev(_ftd_post(2.0, 800)).ftd_today is False


def test_no_ftd_when_gain_too_small():
    assert ev(_ftd_post(1.0, 1500)).ftd_today is False


def test_no_ftd_on_day3():
    p = [(36500, 35000, 35500, 900), (36000, 35400, 35800, 900), (36100, 35700, 35900, 1000),
         (36700, 35900, 36600, 1500)]
    r = ev(p)
    assert r.rally_day == 3 and r.ftd_today is False


# ---- 輸出 ------------------------------------------------------------------
def test_markdown_and_flag(tmp_path):
    df, anchor = build(_ftd_post(2.0, 1500))
    r = tr.evaluate(df, anchor_date=anchor, anchor_low=ANCHOR_LOW)
    md = tr.to_markdown(r)
    assert md.startswith("【劇本變化】")
    assert "未取得" in md
    for bad in ("買進", "賣出", "建議加碼"):
        assert bad not in md


def test_cli_writes_json(tmp_path):
    df, anchor = build(ABOVE3)
    csv = tmp_path / "d.csv"
    df.rename_axis("date").reset_index().to_csv(csv, index=False)
    old_date, old_low = tr.ANCHOR_DATE, tr.ANCHOR_LOW
    tr.ANCHOR_DATE, tr.ANCHOR_LOW = anchor, ANCHOR_LOW
    try:
        r = tr.evaluate(tr.load_csv(csv), anchor_date=anchor, anchor_low=ANCHOR_LOW)
        assert r.scenario == "B_PRICE_ONLY"
        assert tr.main(["--csv", str(csv), "--out", str(tmp_path / "o")]) == 0
    finally:
        tr.ANCHOR_DATE, tr.ANCHOR_LOW = old_date, old_low
    data = json.loads((tmp_path / "o" / "taiex_rebound.json").read_text(encoding="utf-8"))
    assert data["as_of"] == df.index[-1].strftime("%Y-%m-%d")
    assert (tmp_path / "o" / "taiex_rebound.md").exists()


def test_fib_0382_is_rebound_from_low():
    """真實數據抓到的錯：0.382 回撤要從低點往上算（49,968→35,739 應為 41,174）。"""
    df, anchor = build([(36500, 35000, 35500, None)])
    r = tr.evaluate(df, anchor_date=anchor, anchor_low=35739.0)
    assert r.levels["fib_0382"] == pytest.approx(35739 + (49968 - 35739) * 0.382, abs=1)


def test_html_card_renders_and_escapes():
    df, anchor = build(ABOVE3)
    r = tr.evaluate(df, anchor_date=anchor, anchor_low=ANCHOR_LOW)
    d = json.loads(json.dumps(tr.asdict(r)))
    d["reason"] = "<script>x</script>"
    h = tr.to_html_card(d)
    assert "id='taiex'" in h and "年線 MA240" in h
    assert "<script>" not in h and "&lt;script&gt;" in h
    assert "pill md" in h  # B_PRICE_ONLY
    for bad in ("買進", "賣出"):
        assert bad not in h
