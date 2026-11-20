# MA Tracker

主題股均線 / 賣壓追蹤報表，由 GitHub Actions 每日自動更新（台北 07:30、14:45）。

- 網頁：https://filcoabs-mur.github.io/ma-tracker/
- 歷史紀錄：`output/history.csv`

本機執行：`pip install -r requirements.txt && python ma_tracker.py --open`

## 台指年線反彈追蹤（`taiex_rebound.py`）

固定規則腳本：年線 MA240、劇本 A／B／C 判定、Minervini 反彈天數與追價確認日。輸出 `output/taiex_rebound.md`、`output/taiex_rebound.json`。

```bash
python -m pytest -q tests                                  # 規則測試
python taiex_rebound.py                                    # yfinance 抓 ^TWII
python taiex_rebound.py --csv daily.csv --oi-anchor 99438 --oi-today 90000
```

規則與參數都寫在檔案開頭，要改規則請先改測試。

僅為數據整理，不構成投資建議。資料來源：Yahoo Finance。
