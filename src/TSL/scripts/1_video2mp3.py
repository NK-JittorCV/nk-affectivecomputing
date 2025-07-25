# from __future__ import print_function, division
# import os
# import sys
# import subprocess
#
# def class_process(dir_path, dst_dir_path):
#     src_class_path = dir_path
#     if not os.path.isdir(src_class_path):
#         return
#     dst_class_path = dst_dir_path
#     if not os.path.exists(dst_class_path):
#         os.makedirs(dst_class_path)
#     for file_name in os.listdir(src_class_path):
#         if '.mp4' not in file_name:
#             continue
#         name, ext = os.path.splitext(file_name)
#         music_file_name = name + '.mp3'
#         video_file_path = os.path.join(src_class_path, file_name)
#         music_file_path = os.path.join(dst_class_path, music_file_name)
#         cmd = 'ffmpeg -i \"{}\" \"{}\"'.format(video_file_path, music_file_path)
#         print(cmd)
#         subprocess.call(cmd, shell=True)
#         print('\n')
#
# if __name__ == "__main__":
#
#     dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/vid/test"
#     dst_dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/audio/test"
#
#     class_process(dir_path, dst_dir_path)


from __future__ import print_function, division
import os
import sys
import subprocess

def extract_audio(video_file_path, dst_dir_path):
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    name, ext = os.path.splitext(os.path.basename(video_file_path))
    music_file_name = name + '.mp3'
    music_file_path = os.path.join(dst_dir_path, music_file_name)

    cmd = 'ffmpeg -i \"{}\" \"{}\"'.format(video_file_path, music_file_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

def class_process(input_path, dst_dir_path):
    if os.path.isdir(input_path):
        # 处理目录下所有 mp4 文件
        for file_name in os.listdir(input_path):
            if '.mp4' not in file_name.lower():
                continue
            video_file_path = os.path.join(input_path, file_name)
            extract_audio(video_file_path, dst_dir_path)
    elif os.path.isfile(input_path):
        # 处理单个 mp4 文件
        extract_audio(input_path, dst_dir_path)
    else:
        print("Input path is neither a directory nor a file:", input_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    # input_path = "/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88.mp4"
    # dst_dir_path="/home/ubuntu/TSL_jittor/tmp"

    class_process(input_path, dst_dir_path)
