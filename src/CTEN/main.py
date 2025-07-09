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
#from torch.cuda import device_count
from tensorboardX import SummaryWriter
from collections.abc import Iterable
import jittor as jt
from jittor import mpi
def main():
    jt.flags.use_cuda = 1 
    opt = parse_opts()
    opt.device_ids = list([0,1])
    local2global_path(opt)
    model, parameters = generate_vaaerase_model(opt)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)
    #print(optimizer)
    writer = SummaryWriter(logdir=opt.log_path)
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)

    # train
    spatial_transform = get_spatial_transform(opt,'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    #print('=========================================',len(training_data))
    train_loader = get_data_loader(opt, training_data, shuffle=True,num_workers=1)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False, num_workers=1)
    his_acc = -1
    for i in range(1, opt.n_epochs + 1):
        start_time=time.time()
        train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names, writer)
        end_time=time.time()
        train_time=end_time-start_time
        print('epoch_time:',train_time)
        acc = val_epoch(i, val_loader, model, criterion, opt, writer, optimizer)
        if acc>his_acc:
            his_acc = acc
            svpath = os.path.join(opt.result_path, 'model_'+str(i).zfill(3)+'.pth')
            print(svpath)
            #torch.save(model.state_dict(), svpath)
        print('History Acc:', his_acc)
    writer.close()


if __name__ == "__main__":
    main()
