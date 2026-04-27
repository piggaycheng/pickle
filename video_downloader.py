import yt_dlp
import os

def download_video(url, output_path='videos'):
    """
    使用 yt-dlp 下載影片。
    
    Args:
        url (str): 影片的 URL。
        output_path (str): 儲存影片的目錄。預設為當前目錄。
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"正在從 {url} 下載影片...")
            ydl.download([url])
            print("下載完成！")
    except Exception as e:
        print(f"下載失敗: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        download_video(video_url)
    else:
        print("用法: uv run video_downloader.py [影片網址]")
