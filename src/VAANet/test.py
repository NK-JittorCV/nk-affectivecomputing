'''
import os
from opts import parse_opts
from core.model import generate_vaaerase_model, generate_visual_Erase_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_data_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train import train_epoch
from validation import val_epoch
import time
import torch
#from torch.cuda import device_count
from tensorboardX import SummaryWriter
from collections.abc import Iterable
import jittor as jt
from jittor import mpi
#from torch.utils.data import DataLoader
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
import os
from core.model import generate_vaaerase_model, generate_visual_Erase_model
import time
from datasets.ve8 import VE8Dataset
from tensorboardX import SummaryWriter
#from torch.utils.data import DataLoader

def get_ve8(video_path,audio_path,annotation_path, subset, transforms):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(video_path,
                      audio_path,
                      annotation_path,
                      subset,
                      30,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      need_audio=False)

def get_validation_set(dataset,video_path,audio_path,annotation_path, spatial_transform, temporal_transform, target_transform):
    if dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(video_path,audio_path,annotation_path, 'validation', transforms)
    else:
        raise Exception

def test_vaanet(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Test our model'.format(epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with jt.no_grad():
            output, loss = run_model(opt, [visual, target, audio], model, criterion, i)
        acc = calculate_accuracy(output, target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    save_file_path = os.path.join(opt.ckpt_path, 'save_{}_{:.4f}.pth'.format(epoch,accuracies.avg))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    jt.save(states, save_file_path)

if __name__ == "__main__":
    opt = parse_opts()
    opt.device_ids = list([0,1])
    model, parameters = generate_visual_Erase_model(opt)
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    writer = SummaryWriter(logdir=opt.log_path)
    print(opt.result_path)
    checkpoint = jt.load("/home/ubuntu/zzq/CTEN_jittor/ve8.pth")
    model.load_state_dict(checkpoint)
    #dataset='ve8'
    #video_path="/home/ubuntu7/wlj/dataset/Youtube-8-jpg"
    #audio_path="VideoEmotion8--mp3"
    #annotation_path="/home/ubuntu7/wlj/code/VAANet-master/data/k_fold/ve8_05.json"
    test_vaanet(1, val_loader, model, criterion, opt, writer, optimizer=None)
'''

import os
from opts import parse_opts
#from core.model import generate_vaaerase_model, generate_visual_Erase_model
from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_data_loader, get_test_set
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train import train_epoch
from validation import val_epoch
import time
#from torch.cuda import device_count
from collections import OrderedDict
from tensorboardX import SummaryWriter
from collections.abc import Iterable
import jittor.dataset as jd
import jittor as jt
from jittor.dataset import Dataset
from jittor import mpi
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

def main():
    jt.flags.use_cuda = 1 
    opt = parse_opts()
    opt.device_ids = list([0,1])
    local2global_path(opt)
    model, parameters = generate_model(opt)

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)

    # train
    spatial_transform = get_spatial_transform(opt,'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=False)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False, num_workers=1)

    #test
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    test_data = get_test_set(opt, spatial_transform, temporal_transform, target_transform)

    print('len_training_data:',len(training_data))
    print('len_validation_data:',len(validation_data))

    combined_data = CombinedDataset(training_data, validation_data)
    combined_loader = get_data_loader(opt, combined_data, shuffle=False ,num_workers=1)

    state_dict = jt.load("/home/ubuntu/wwc/zzq/VAANet/result/result_20250526_144152/model_012.pth")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 去除 'module.' 前缀
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    acc = val_epoch(1, combined_loader, model, criterion, opt, writer, optimizer)
    writer.close()


if __name__ == "__main__":
    main()



