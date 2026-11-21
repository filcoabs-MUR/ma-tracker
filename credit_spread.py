#!/usr/bin/env python3
"""美國高收益債信用利差（ICE BofA HY OAS）— 判斷「估值修正」還是「信用／金融危機」。

固定規則，數值由呼叫端提供（FRED：BAMLH0A0HYM2 整體、BAMLH0A3HYC CCC 級）。
ICE 數據僅限個人內部使用：只放進私人通知，不寫進公開網頁或 repo。

等級（取「水準」與「20 日擴大速度」兩者中較嚴重者）：
  0 正常      HY < 4.5%                     → 股市下跌偏估值修正
  1 警戒      4.5% ≤ HY < 6.0%，或 20 日擴大 ≥ 100bp
  2 信用壓力  6.0% ≤ HY < 8.0%，或 20 日擴大 ≥ 200bp
  3 危機等級  HY ≥ 8.0%（2008、2020 年 3 月等級）
附加旗標：
  底層裂縫    CCC ≥ 10% 且整體 HY 仍 < 4.5%（壓力集中在最差信用，尚未擴散）
參考：長期平均約 5.2%；金融海嘯後「非恐慌」區間約 3.25%～5.25%。
僅為數據與規則判讀，不構成投資建議。
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = (4.5, 6.0, 8.0)          # %
SPEED_BP = (100, 200)             # 20 日擴大 bp → 等級 1、2
CCC_CRACK = 10.0                  # %
LONG_RUN_AVG = 5.2

NAMES = {0: "正常", 1: "警戒", 2: "信用壓力", 3: "危機等級"}
VERDICT = {
    0: "信用市場平穩，股市下跌偏「估值修正」",
    1: "信用開始緊張，留意是否從估值修正轉向信用事件",
    2: "信用壓力明顯，下跌已不只是估值問題",
    3: "達金融危機等級的信用利差",
}


@dataclass
class Credit:
    hy: float
    hy_20d: float | None
    ccc: float | None
    change_bp: int | None
    level: int
    name: str
    verdict: str
    ccc_crack: bool
    reason: str


def assess(hy: float, hy_20d: float | None = None, ccc: float | None = None) -> Credit:
    if not 0 < hy < 30:
        raise ValueError(f"HY OAS 應以百分比輸入（例如 3.52），收到 {hy}")
    by_level = sum(hy >= t for t in LEVELS)
    change = None if hy_20d is None else round((hy - hy_20d) * 100)
    by_speed = 0 if change is None else sum(change >= t for t in SPEED_BP)
    level = max(by_level, by_speed)
    crack = ccc is not None and ccc >= CCC_CRACK and hy < LEVELS[0]

    why = [f"HY {hy:.2f}%（長期平均約 {LONG_RUN_AVG}%）"]
    if change is not None:
        why.append(f"20 日 {change:+d}bp")
    if by_speed > by_level:
        why.append("擴大速度觸發升級")
    if crack:
        why.append(f"CCC {ccc:.2f}% 已達 {CCC_CRACK:.0f}%，壓力集中在最差信用")
    return Credit(hy=hy, hy_20d=hy_20d, ccc=ccc, change_bp=change, level=level, name=NAMES[level],
                  verdict=VERDICT[level], ccc_crack=crack, reason="；".join(why))


def to_markdown(c: Credit) -> str:
    chg = "未提供" if c.change_bp is None else f"{c.change_bp:+d}bp"
    ccc = "未提供" if c.ccc is None else f"{c.ccc:.2f}%" + ("（底層裂縫）" if c.ccc_crack else "")
    return "\n".join([
        f"信用利差：等級 {c.level}「{c.name}」— {c.verdict}",
        "",
        "| 美國高收益債 OAS | 數值 |",
        "|---|---|",
        f"| 整體 HY | {c.hy:.2f}% |",
        f"| 20 日變化 | {chg} |",
        f"| CCC 級 | {ccc} |",
        f"| 門檻 | 4.5% 警戒／6% 信用壓力／8% 危機 |",
        "",
        f"判讀依據：{c.reason}",
    ])
