video_path="/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88.mp4"
output_path="/home/ubuntu/TSL_jittor/tmp/"
img_path="/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88"
audio_path = "/home/ubuntu/TSL_jittor/tmp/40_VideoEmotion8_Anger_88.mp3"

mfcc_dst_dir_path = "/home/ubuntu/TSL_jittor/tmp/mfcc"
rgb_dst_dir_path = "/home/ubuntu/TSL_jittor/tmp/rgb"

rgb_path = "/home/ubuntu/TSL_jittor/tmp/rgb/40_VideoEmotion8_Anger_88.npy"
mfcc_path = "/home/ubuntu/TSL_jittor/tmp/mfcc/40_VideoEmotion8_Anger_88.npy"

# 获取帧率
python ./scripts/0_data_fps.py ${video_path}

# 视频预处理成帧和音频分离
python ./scripts/1_video2jpg.py ${video_path} ${output_path}
python ./scripts/1_video2mp3.py ${video_path} ${output_path}

# mfcc提取音频特征
python ./scripts/2_mp3audio2mfcc.py ${img_path} ${audio_path} ${mfcc_dst_dir_path}

# 使用I3D提取视频特征,我们使用的预训练模型和代码均来自于 https://github.com/piergiaj/pytorch-i3d
python ./pytorch-i3d/extract_features.py --root ${img_path} --save_dir ${rgb_dst_dir_path}

# inference
python main_inference.py --rgb_path ${rgb_path} --mfcc_path ${mfcc_path}