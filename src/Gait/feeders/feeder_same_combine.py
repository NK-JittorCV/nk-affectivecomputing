import numpy as np
import pickle
from jittor.dataset import Dataset  
import sys
sys.path.extend(['../'])
from feeders import tools
import jittor as jt

class Feeder(Dataset):
    def __init__(self, data_m_path, data_p_path, label_path,feature_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        super().__init__()

        self.debug = debug
        self.data_m_path = data_m_path
        self.data_p_path = data_p_path
        self.label_path = label_path
        self.feature_path=feature_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.load_data()
        if normalization:
            self.get_mean_map()
        self.set_attrs(total_len=len(self.label),batch_size=32, shuffle=True)

    def load_data(self):
        # data: N C V T M

        if '.npy' not in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            self.label = np.load(self.label_path, allow_pickle=True)
            self.sample_name = None
        # load data

        self.data_m = np.load(self.data_m_path)
        self.data_p = np.load(self.data_p_path)
        self.feature=np.load(self.feature_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data_m = self.data_m[0:100]
            self.data_p = self.data_p[0:100]
            self.sample_name = self.sample_name[0:100]
            self.feature=self.feature[0:100]

    def get_mean_map(self):
        data_m = self.data_m
        N, C, T, V, M = data_m.shape
        self.mean_map_m = data_m.mean(axis=0)
        self.std_map_m = data_m.std(axis=0)+1e-6

        data_p = self.data_p
        N, C, T, V, M = data_p.shape
        self.mean_map_p=data_p.mean(axis=0)
        self.std_map_p=data_p.std(axis=0)+1e-6

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        data_numpy_m = self.data_m[index]
        data_numpy_p = self.data_p[index]

        label = self.label[index]
        feature_numpy = self.feature[index]
        data_numpy_m = jt.array(data_numpy_m)
        data_numpy_p = jt.array(data_numpy_p)
        feature_numpy = jt.array(feature_numpy)
        label = jt.array(label)

        if self.normalization:
            data_numpy_m = (data_numpy_m - self.mean_map_m) / self.std_map_m
            data_numpy_p = (data_numpy_p - self.mean_map_p) / self.std_map_p
        if self.random_shift:
            data_numpy_m = tools.random_shift(data_numpy_m)
            data_numpy_p = tools.random_shift(data_numpy_p)
        if self.random_choose:
            data_numpy_m = tools.random_choose(data_numpy_m, self.window_size)
            data_numpy_p = tools.random_choose(data_numpy_p, self.window_size)
        elif self.window_size > 0:
            data_numpy_m = tools.auto_pading(data_numpy_m, self.window_size)
            data_numpy_p = tools.auto_pading(data_numpy_p, self.window_size)
        if self.random_move:
            data_numpy_m = tools.random_move(data_numpy_m)
            data_numpy_p = tools.random_move(data_numpy_p)
        return data_numpy_m, data_numpy_p, label,feature_numpy, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
