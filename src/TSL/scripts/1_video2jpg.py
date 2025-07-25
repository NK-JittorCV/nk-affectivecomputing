# from __future__ import print_function, division
# import os
# import sys
# import subprocess
#
# def class_process(dir_path, dst_dir_path):
#     if not os.path.exists(dst_dir_path):
#         os.mkdir(dst_dir_path)
#     for file_name in os.listdir(dir_path):
#         if '.mp4' not in file_name:
#             continue
#         name, ext = os.path.splitext(file_name)
#         dst_directory_path = os.path.join(dst_dir_path, name)
#         video_file_path = os.path.join(dir_path, file_name)
#         try:
#             if os.path.exists(dst_directory_path):
#                 if not os.path.exists(os.path.join(dst_directory_path, 'image00001.jpg')):
#                     subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
#                     print('remove {}'.format(dst_directory_path))
#                     os.makedirs(dst_directory_path)
#                 else:
#                     continue
#             else:
#                 os.mkdir(dst_directory_path)
#         except:
#             print(dst_directory_path)
#             continue
#         cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/%06d.jpg\"'.format(video_file_path, dst_directory_path)
#         print(cmd)
#         subprocess.call(cmd, shell=True)
#         print('\n')
#
# if __name__ == "__main__":
#     dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/vid/test"  # avi directory
#     dst_dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/pic/test"  # jpg directory
#     class_process(dir_path, dst_dir_path)

from __future__ import print_function, division
import os
import sys
import subprocess

def process_video(video_file_path, dst_dir_path):
    name, ext = os.path.splitext(os.path.basename(video_file_path))
    dst_directory_path = os.path.join(dst_dir_path, name)

    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'image00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.makedirs(dst_directory_path)
            else:
                return
        else:
            os.mkdir(dst_directory_path)
    except Exception as e:
        print("Error creating directory:", dst_directory_path, e)
        return

    cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/%06d.jpg\"'.format(video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

def class_process(input_path, dst_dir_path):
    if os.path.isdir(input_path):
        # 处理目录下所有mp4文件
        for file_name in os.listdir(input_path):
            if '.mp4' not in file_name.lower():
                continue
            video_file_path = os.path.join(input_path, file_name)
            process_video(video_file_path, dst_dir_path)
    elif os.path.isfile(input_path):
        # 处理单个文件
        process_video(input_path, dst_dir_path)
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
