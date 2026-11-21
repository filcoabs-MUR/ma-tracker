"""credit_spread.py 規則測試。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import credit_spread as cs  # noqa: E402
import taiex_rebound as tr  # noqa: E402
from test_taiex_rebound import ABOVE3, ANCHOR_LOW, build  # noqa: E402


@pytest.mark.parametrize("hy,level", [(3.52, 0), (4.49, 0), (4.5, 1), (6.0, 2), (7.99, 2), (8.0, 3), (21.8, 3)])
def test_level_by_spread(hy, level):
    assert cs.assess(hy).level == level


def test_speed_escalates_even_when_level_low():
    c = cs.assess(4.2, hy_20d=3.1)  # +110bp
    assert c.level == 1 and "速度" in c.reason
    assert cs.assess(4.4, hy_20d=2.3).level == 2  # +210bp


def test_speed_never_downgrades():
    assert cs.assess(8.5, hy_20d=9.5).level == 3


def test_ccc_crack_only_when_headline_calm():
    assert cs.assess(3.52, ccc=10.0).ccc_crack is True
    assert cs.assess(5.0, ccc=12.0).ccc_crack is False
    assert cs.assess(3.52, ccc=9.9).ccc_crack is False


def test_rejects_basis_points_input():
    """常見錯誤：把 352（bp）當成百分比傳進來。"""
    with pytest.raises(ValueError, match="百分比"):
        cs.assess(352)


def test_cli_adds_credit_to_private_md_but_not_card(tmp_path, monkeypatch):
    df, anchor = build(ABOVE3)
    csv = tmp_path / "d.csv"
    df.rename_axis("date").reset_index().to_csv(csv, index=False)
    monkeypatch.setattr(tr, "ANCHOR_DATE", anchor)
    monkeypatch.setattr(tr, "ANCHOR_LOW", ANCHOR_LOW)
    out = tmp_path / "o"
    assert tr.main(["--csv", str(csv), "--out", str(out), "--hy", "3.52", "--hy-20d", "3.10", "--ccc", "10.1"]) == 0
    md = (out / "taiex_rebound.md").read_text(encoding="utf-8")
    assert "信用利差：等級 0" in md and "底層裂縫" in md
    assert md.rstrip().endswith("不構成投資建議。")
    d = json.loads((out / "taiex_rebound.json").read_text(encoding="utf-8"))
    assert d["credit"]["change_bp"] == 42
    assert "信用利差" not in tr.to_html_card(d) and "3.52" not in tr.to_html_card(d)
