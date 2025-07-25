# import os
# import cv2
# import json
#
# root = '/home/ubuntu/data/sentiment/temporal/our_1219_full/vid'
# fps={}
#
# for split in os.listdir(root):
#     for vid in os.listdir(os.path.join(root,split)):
#         v = cv2.VideoCapture(os.path.join(root,split,vid))
#         cur_fps = v.get(cv2.CAP_PROP_FPS)
#         name, _ = os.path.splitext(vid)
#         fps[name]=cur_fps
# with open("/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/fps_dict.json","w+") as f:
#     json.dump(fps,f)

import os
import cv2
import json
import sys

# 从命令行获取输入路径
# input_path = sys.argv[1]
input_path = "/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88.mp4"
fps = {}

if os.path.isdir(input_path):
    # 输入是目录
    for split in os.listdir(input_path):
        split_path = os.path.join(input_path, split)
        if os.path.isdir(split_path):
            for vid in os.listdir(split_path):
                video_path = os.path.join(split_path, vid)
                v = cv2.VideoCapture(video_path)
                cur_fps = v.get(cv2.CAP_PROP_FPS)
                name, _ = os.path.splitext(vid)
                fps[name] = cur_fps
                v.release()
    # 保存到 fps_dict.json
    with open("fps_dict.json", "w+") as f:
        json.dump(fps, f)
    print("Saved fps_dict.json for directory:", input_path)

elif os.path.isfile(input_path):
    # 输入是单个文件
    v = cv2.VideoCapture(input_path)
    cur_fps = v.get(cv2.CAP_PROP_FPS)
    name, _ = os.path.splitext(os.path.basename(input_path))
    fps[name] = cur_fps
    v.release()
    # 保存到 fps.json
    with open("/home/ubuntu/TSL_jittor/tmp/fps.json", "w+") as f:
        json.dump(fps, f)
    print(f"Saved fps.json for file: {input_path} (fps: {cur_fps})")

else:
    print("Input path is neither a directory nor a file. Please check your input.")
