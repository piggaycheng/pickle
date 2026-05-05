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

## Other Useful Commands

下載 YouTube 影片：

```bash
uv run video_downloader.py <youtube-url>
```

跑測試：

```bash
uv run pytest
```
