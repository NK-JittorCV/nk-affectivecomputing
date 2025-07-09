from datasets.ve8 import VE8Dataset
#from torch.utils.data import DataLoader
from jittor.dataset import DataLoader
def get_ve8(opt, subset, transforms):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(opt.video_path,
                      opt.audio_path,
                      opt.annotation_path,
                      subset,
                      opt.fps,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      need_audio=True)

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(opt, 'training', transforms)
    else:
        raise Exception

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(opt, 'validation', transforms)
    else:
        raise Exception

def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(opt, 'test', transforms)
    else:
        raise Exception

def get_data_loader(opt, dataset, shuffle,num_workers=None):
    batch_size = opt.batch_size
    workers = opt.n_threads
    '''
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=opt.dl
    )
    '''
    return DataLoader(
        dataset,
        batch_size=8
    )
