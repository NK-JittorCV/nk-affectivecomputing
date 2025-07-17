import os
import cv2
from moviepy.editor import VideoFileClip

def extract_video_frames_and_audio(video_path, output_frame_dir, output_audio_path):
    """
    将视频分解为帧图像和音频（mp3格式）

    参数:
    - video_path: 原始视频路径
    - output_frame_dir: 输出帧图像的文件夹路径
    - output_audio_path: 输出音频的保存路径 (.mp3)

    返回:
    - 成功信息或异常信息
    """

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    # 创建保存帧图像的文件夹
    os.makedirs(output_frame_dir, exist_ok=True)

    # ---------------------- 提取图像帧 ----------------------
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    '''
    print("[INFO] 开始提取帧图像...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_frame_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"[INFO] 提取完成，共保存 {frame_idx} 帧到 {output_frame_dir}")
    '''

    # ---------------------- 提取音频 ----------------------
    print("[INFO] 开始提取音频...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec='mp3')
        print(f"[INFO] 音频已保存到 {output_audio_path}")
    except Exception as e:
        print(f"[ERROR] 提取音频失败: {e}")

video_path = "/home/ubuntu/zzq/movie.mp4"
output_frame_dir = "/home/ubuntu/zzq/Affective_Computing/src/CTEN/data/video"
output_audio_path = "/home/ubuntu/zzq/Affective_Computing/src/CTEN/data/mp3/1.mp3"

extract_video_frames_and_audio(video_path, output_frame_dir, output_audio_path)
