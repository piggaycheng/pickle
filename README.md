# Pickle

本專案目前的主要入口是本地的 Streamlit 標記介面。

## Requirements

- Python `>=3.11`
- `uv`

## Install

```bash
uv sync
```

## Start The UI Server

在 repo 根目錄執行：

```bash
uv run streamlit run streamlit_app.py
```

Streamlit 預設會開在：

```text
http://localhost:8501
```

## First-Time Project Setup

UI 啟動後會從 `datasets/` 讀 project。若目前沒有任何 project，畫面會顯示 `No projects found.`。

先用一支本地影片建立 project 並跑一次 pose extraction：

```bash
uv run pose_detector.py <path-to-video> --project-id <project-id>
```

範例：

```bash
uv run pose_detector.py videos/input.mp4 --project-id smoke-pose-001
```

這會建立：

```text
datasets/<project-id>/
```

然後就可以重新打開 UI，在側邊欄選這個 project。

## Optional Data Root Override

預設 data root 是 repo 下的 `datasets/`。若要改位置，可先設定 `PICKLE_DATA_DIR`：

```powershell
$env:PICKLE_DATA_DIR = "D:\\path\\to\\datasets"
uv run streamlit run streamlit_app.py
```

## Dataset Export Outputs

在 Streamlit UI 裡按 `Export Dataset` 會把目前可信任的 annotations 輸出到：

```text
datasets/<project-id>/exports/
```

主要輸出包含：

- `exports/latest/timeline.csv`：給人看的最新標註時間軸。
- `exports/latest/training_examples.jsonl`：給模型訓練用的最新 structured metadata。
- `exports/latest/export_manifest.json`：最新 export 的摘要。
- `exports/latest/clips/`：只有勾選 `Extract clips with ffmpeg` 時才會產生，內含每筆 annotation 對應的短影片片段。
- `exports/runs/<export-id>/`：每次 export 的歷史版本，方便之後比較不同資料集輸出。

UI 會讀 `exports/latest/` 來顯示 `Exported Clips`，並用 read-only 的 `Export History` 面板列出 `exports/runs/` 裡的歷史版本。若要清掉目前 UI 顯示的 export，在 UI 裡按 `Clear export outputs`。這會刪除 `exports/latest/` 的 outputs 和舊版 root-level export outputs，但會保留 `exports/runs/` 的歷史版本，也不會刪除 `artifacts/`。

`artifacts/` 保留 replayable jobs 的來源紀錄和中間結果，例如 pose extraction、hit candidates、job manifests。這些資料用來 debug、重建 queue、追蹤資料來源；清除 export 不應該破壞它們。

## Other Useful Commands

下載 YouTube 影片：

```bash
uv run video_downloader.py <youtube-url>
```

跑測試：

```bash
uv run pytest
```
