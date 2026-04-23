import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

def download_model(model_path):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

def process_video(input_path, output_path):
    model_path = "tasks/pose_landmarker.task"
    download_model(model_path)

    # 檢查輸入檔案
    if not os.path.exists(input_path):
        print(f"Error: Input video file '{input_path}' not found.")
        return

    # 開啟影片
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    # 設定影片寫入器 (使用 avc1/H.264 編碼，VS Code 支援度較佳)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 初始化 Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    print(f"Processing video: {input_path}")
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 轉換為 MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 計算時間戳 (ms)
            timestamp_ms = int(1000 * frame_index / fps)
            
            # 偵測姿態
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 繪製結果 (手動繪製，因為 Tasks API 不包含舊的 drawing_utils)
            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            out.write(frame)
            frame_index += 1

    cap.release()
    out.release()
    print(f"Processing complete. Result saved to: {output_path}")

if __name__ == "__main__":
    process_video("videos/input.mp4", "outputs/output.mp4")
