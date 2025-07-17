import os
import random
from opts import parse_opts
from core.model import generate_vaaerase_model, generate_visual_Erase_model
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy,batch_augment
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_data_loader, get_test_set,get_ve8
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train import train_epoch
import time
from tqdm import tqdm
import numpy as np
#from torch.cuda import device_count
from collections import OrderedDict
from tensorboardX import SummaryWriter
from collections.abc import Iterable
import jittor.dataset as jd
import jittor as jt
from jittor.dataset import Dataset
from jittor import mpi
import cv2
from moviepy.editor import VideoFileClip

def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies2 = AverageMeter()
    end_time = time.time()
    print(data_loader)
    for i, data_item in tqdm(enumerate(data_loader)):
        visual, target, audio, visualization_item, batch_size,video_item = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with jt.no_grad():
            output1, loss1, gamma1 = run_model(opt, [visual, target, audio], model, criterion, i, print_attention=False)
            predicted_class =  jt.argmax(output1.data, 1)[0] 
            print('预测的视频情感类别是:',predicted_class)

def get_single_video_test_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    
    return get_ve8(opt, 'validation', transforms)


import os
import cv2
from moviepy.editor import VideoFileClip

def extract_video_frames_and_audio(video_path, output_frame_root, output_audio_root, video_id="1"):
    """
    将视频分解为帧图像和音频（mp3格式）

    参数:
    - video_path: 原始视频路径
    - output_frame_root: 帧图像的根目录（会保存到 output_frame_root/test/video_id）
    - output_audio_root: 音频的根目录（会保存为 output_audio_root/test/video_id.mp3）
    - video_id: 当前视频的编号，决定帧文件夹名和音频文件名，默认 "1"

    返回:
    - 成功信息或异常信息
    """

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    # 组装输出路径
    output_frame_dir = os.path.join(output_frame_root, "test", video_id)
    output_audio_path = os.path.join(output_audio_root, "test", f"{video_id}.mp3")

    # 创建输出目录
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    # ---------------------- 提取图像帧 ----------------------
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    print(f"[INFO] 开始提取帧图像到 {output_frame_dir} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_frame_dir, f"{frame_idx:06d}.jpg")  # 六位数命名
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"[INFO] 提取完成，共保存 {frame_idx} 帧")

    # ---------------------- 提取音频 ----------------------
    print(f"[INFO] 开始提取音频到 {output_audio_path} ...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec='mp3')
        print(f"[INFO] 音频已保存到 {output_audio_path}")
    except Exception as e:
        print(f"[ERROR] 提取音频失败: {e}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

def remove_prefix(state_dict):
    return OrderedDict({k.replace("module.", ""): v for k, v in state_dict.items()})
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_len = len(dataset1) + len(dataset2)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < len(self.dataset1):
            return self.dataset1[index]
        else:
            return self.dataset2[index - len(self.dataset1)]

def main(opt):
    set_random_seed(42)
    jt.flags.use_cuda = 1 
    opt.device_ids = list([0,1])
    local2global_path(opt)
    model, parameters = generate_vaaerase_model(opt)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)
    writer = SummaryWriter(logdir=opt.log_path)
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    validation_data = get_single_video_test_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False, num_workers=1)

    ckpt_path=opt.checkpoint_path
    checkpoint = jt.load(ckpt_path)
    model.load_state_dict(remove_prefix(checkpoint))
    acc = val_epoch(1, val_loader, model, criterion, opt, writer, optimizer)
    writer.close()


if __name__ == "__main__":
    opt = parse_opts()
    
    extract_video_frames_and_audio(
        video_path=opt.video_path,
        output_frame_dir=opt.output_frame_dir,
        output_audio_path=opt.output_audio_dir
    )
    


    # 第三步：将外部上传的帧图像路径/音频路径，覆盖 opt 中默认的路径
    opt.video_path = opt.output_frame_dir
    opt.audio_path = opt.output_audio_dir

    # 可以检查一下是否正确
    print(f"[INFO] 使用帧图像路径: {opt.video_path}")
    print(f"[INFO] 使用音频路径: {opt.audio_path}")
    main(opt)



