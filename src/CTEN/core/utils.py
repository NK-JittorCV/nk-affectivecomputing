import os
import datetime
import shutil
import jittor as jt
from transforms.temporal import TSN
from transforms.spatial import Preprocessing
from datasets.ve8 import get_default_video_loader
import numpy as np



def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.audio_path = os.path.join(opt.root_path, opt.audio_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        if opt.debug:
            opt.result_path = "debug"
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.expr_name == '':
            now = datetime.datetime.now()
            now = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            os.mkdir(opt.result_path)
        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exceptioncd

def get_spatial_transform(opt, mode):
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=True, center=False)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    else:
        raise Exception

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def process_data_item(opt, data_item):
    visual, target, audio, visualization_item, video, n_frames = data_item
    target = target.cuda()
    visual = visual.cuda()
    audio = audio.cuda()
    batch = visual.size(0)
    return visual, target, audio, visualization_item, batch, {'video':video, 'n_frames':n_frames}

def run_model(opt, inputs, model, criterion, i=0, print_attention=False, period=30, return_attention=False):
    visual, target, audio = inputs
    outputs, gamma = model([visual, audio])
    loss = criterion(outputs, target)
    return outputs,loss,gamma

def run_model_inf(opt, inputs, model, i=0, print_attention=False, period=30, return_attention=False):
    visual, _, audio = inputs
    outputs, gamma = model([visual, audio])
    return outputs,gamma

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.equal(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size

def get_new_indices(frame_indices):
    pass

def batch_augment(video_item,erase_index,opt,visual):
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    spatial_transform = get_spatial_transform(opt, 'train')
    loadder=get_default_video_loader()
    seq_len=opt.seq_len
    for i in range(len(video_item['video'])):
        frame_indices = []
        video_path=video_item['video'][i]
        n_frames = video_item['n_frames'][i]
        #print(type(n_frames))
        n_frames=n_frames.item()
        segment_duration = n_frames // seq_len
        #erase_index[i] = erase_index[i].bool()
        index=(jt.where(erase_index[i]==True)[0])
        for ind in range(len(index)):
            t=index[ind].detach().cpu().item()
            '''
            segment_duration=segment_duration.detach().cpu()
            max_len=min(segment_duration*(t+1),n_frames+1)
            max_len = min(jt.float32(segment_duration) * jt.float32(t + 1), n_frames + 1)
            test=list(range(t*segment_duration+1, max_len+1))
            '''
            #segment_duration = segment_duration.detach().cpu().item()  # 确保是 Python 数值
            max_len = min(int(segment_duration * (t + 1)), n_frames + 1)  # 保证类型一致
            test = list(range(t * int(segment_duration) + 1, max_len + 1))  # 确保范围计算正确

            frame_indices.extend(test)
        if len(index)==0 or len(frame_indices) == 0:
            frame_indices=list(range(1,n_frames+1))
        snippets_frame_idx = None
        try:
            if temporal_transform(frame_indices) is not None:
                snippets_frame_idx = temporal_transform(frame_indices)
            else:
                print(video_path,n_frames)
        except:
            print(video_path,n_frames)
        snippets = []
        if snippets_frame_idx:
            for snippet_frame_idx in snippets_frame_idx:
                snippet =loadder(video_path, snippet_frame_idx)
                snippets.append(snippet)
            spatial_transform.randomize_parameters()
            snippets_transformed = []
            for snippet in snippets:
                snippet = [spatial_transform(img) for img in snippet]
                snippet= [jt.array(tensor.numpy(), dtype=jt.float32) for tensor in snippet]
                #print('<<<<<<<<<<<>>>>>>>>>>>>',type(snippet[0]))
                snippet = jt.stack(snippet, 0).transpose(0, 1).transpose(2, 3)
                snippets_transformed.append(snippet)
            snippets = snippets_transformed
            #print(snippets[0].shape)
            snippets = jt.stack(snippets, 0)
            visual[i]=snippets
    return visual

def batch_augment2(video_item,erase_index,opt,visual):
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    spatial_transform = get_spatial_transform(opt, 'train')
    loadder=get_default_video_loader()
    seq_len=opt.seq_len
    for i in range(len(video_item['video'])):
        frame_indices = []
        video_path=video_item['video'][i]
        n_frames = video_item['n_frames'][i]
        segment_duration = n_frames // seq_len

        index=(jt.where(erase_index[i]==True)[0])
        for ind in range(len(index)):
            t=index[ind].detach().cpu().item()
            #segment_duration=segment_duration.detach().cpu()
            max_len = min(int(segment_duration * (t + 1)), n_frames + 1)  # 保证类型一致
            test = list(range(t * int(segment_duration) + 1, max_len + 1))  # 确保范围计算正确
            frame_indices.extend(test)
        if len(index)==0 or len(frame_indices) == 0:
            frame_indices=list(range(1,n_frames+1))
        snippets_frame_idx = temporal_transform(frame_indices)
        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet =loadder(video_path, snippet_frame_idx)
            snippets.append(snippet)
        spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [spatial_transform(img) for img in snippet]
            snippet= [jt.array(tensor.numpy(), dtype=jt.float32) for tensor in snippet]
            snippet = jt.stack(snippet, 0).transpose(0, 1).transpose(2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = jt.stack(snippets, 0)
        visual[i]=snippets
    return visual

def batch_random_erase(video_item,opt,visual):
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    spatial_transform = get_spatial_transform(opt, 'train')
    loadder = get_default_video_loader()
    seq_len = opt.seq_len
    for i in range(len(video_item['video'])):
        frame_indices = []
        video_path = video_item['video'][i]
        n_frames = video_item['n_frames'][i]
        segment_duration = n_frames // seq_len
        erase_index = jt.from_numpy(np.random.randint(2,size=(32,16)))
        index = (jt.where(erase_index[i] == True)[0])
        for ind in range(len(index)):
            t = index[ind].detach().cpu()
            segment_duration = segment_duration.detach().cpu()
            max_len = min(segment_duration * (t + 1), n_frames + 1)
            test = list(range(t * segment_duration + 1, max_len + 1))
            frame_indices.extend(test)
        if len(index) == 0 or len(frame_indices) == 0:
            frame_indices = list(range(1, n_frames + 1))
        snippets_frame_idx = temporal_transform(frame_indices)
        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = loadder(video_path, snippet_frame_idx)
            snippets.append(snippet)
        spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [spatial_transform(img) for img in snippet]
            snippet= [jt.array(tensor.numpy(), dtype=jt.float32) for tensor in snippet]
            snippet = jt.stack(snippet, 0).transpose(0, 1).transpose(2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = jt.stack(snippets, 0)
        visual[i] = snippets
    return visual
