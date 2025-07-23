import os
from opts import parse_opts
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



