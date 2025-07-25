# from __future__ import print_function, division
# import os, shutil
# import sys
# import subprocess
# import json
# import librosa
# import numpy as np
# from tqdm import tqdm
# def class_process(img_path, dir_path, dst_dir_path):
#     src_class_path = dir_path
#     if not os.path.isdir(src_class_path):
#         return
#     dst_class_path = dst_dir_path
#     if os.path.exists(dst_class_path):
#         shutil.rmtree(dst_class_path)
#     os.makedirs(dst_class_path)
#     fpss = json.load(open('/home/ubuntu/sentiment/LACP_solution/dataset/VideoSenti/fps_dict.json'))
#     std = min(fpss.values())
#     for file_name in tqdm(os.listdir(src_class_path)):
#         if '.mp3' not in file_name:
#             continue
#         name, ext = os.path.splitext(file_name)
#         mfcc_file_name = name + '.npy'
#         music_file_path = os.path.join(src_class_path, file_name)
#         mfcc_file_path = os.path.join(dst_class_path, mfcc_file_name)
#         fps = fpss[name]
#         y, sr = librosa.load(music_file_path, sr=int(fps/std*44100/4.323882352941176*4/1.9992360580595874*2))
#         #log mfcc
#         melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
#         logmelspec = librosa.power_to_db(melspectrogram)
#         print(sr,logmelspec.shape[1]-2*len(os.listdir(os.path.join(img_path,name))))
#         np.save(mfcc_file_path,logmelspec.T)
#
#
# if __name__ == "__main__":
#     split='test'
#     img_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/pic/"+split
#     dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/audio/"+split
#     dst_dir_path = "/home/ubuntu/data/sentiment/temporal/our_1219_full/features/"+split+"/logmfcc"
#
#     class_process(img_path, dir_path, dst_dir_path)


from __future__ import print_function, division
import os, shutil
import sys
import subprocess
import json
import librosa
import numpy as np
from tqdm import tqdm

def process_single_file(img_path, file_path, dst_dir_path, fpss, std):
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    name, ext = os.path.splitext(os.path.basename(file_path))
    mfcc_file_name = name + '.npy'
    mfcc_file_path = os.path.join(dst_dir_path, mfcc_file_name)
    fps = fpss.get(name, None)
    if fps is None:
        print(f"FPS for {name} not found, skipping.")
        return

    y, sr = librosa.load(file_path, sr=int(fps/std*44100/4.323882352941176*4/1.9992360580595874*2))
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
    logmelspec = librosa.power_to_db(melspectrogram)

    img_dir = os.path.join(img_path, name)
    if os.path.exists(img_dir):
        diff = logmelspec.shape[1] - 2 * len(os.listdir(img_dir))
        print(sr, diff)
    else:
        print(f"Image directory {img_dir} not found.")

    np.save(mfcc_file_path, logmelspec.T)

def class_process(img_path, input_path, dst_dir_path):
    fpss = json.load(open('/home/ubuntu/TSL_jittor/dataset/VideoSenti/fps_dict.json'))
    std = min(fpss.values())

    if os.path.isfile(input_path):
        process_single_file(img_path, input_path, dst_dir_path, fpss, std)

    elif os.path.isdir(input_path):
        if os.path.exists(dst_dir_path):
            shutil.rmtree(dst_dir_path)
        os.makedirs(dst_dir_path)

        for file_name in tqdm(os.listdir(input_path)):
            if '.mp3' not in file_name.lower():
                continue
            file_path = os.path.join(input_path, file_name)
            process_single_file(img_path, file_path, dst_dir_path, fpss, std)

    else:
        print("Input path is neither a file nor a directory:", input_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <img_path> <input_path> <dst_dir_path> <split>")
        sys.exit(1)

    img_path = sys.argv[1]
    input_path = sys.argv[2]
    dst_dir_path = sys.argv[3]
    # img_path="/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88"
    # input_path = "/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88.mp3"
    # dst_dir_path = "/home/ubuntu/TSL_jittor/tmp"
    class_process(img_path, input_path, dst_dir_path)
