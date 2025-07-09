from __future__ import print_function
import jittor as jt
import argparse
import inspect
import os
import pickle
import random
import shutil
from jittor.dataset import DataLoader
from GPUtil import GPUtil
from collections import OrderedDict
import numpy as np
import jittor.nn as nn
import jittor.optim as optim
import yaml
from tensorboardX import SummaryWriter

from tqdm import tqdm
import time

jt.flags.use_cuda = 1  
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
def get_gpu_memory():
    gpus = GPUtil.getGPUs()
    return gpus[1].memoryUsed  
def init_seed(seed):
    jt.set_seed(seed)  
    np.random.seed(seed)  
    random.seed(seed) 


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
        
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./src/BPM_GCN/config/EGait_journal/train_diff_combine_double_score_fagg.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 2],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=20,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    
    parser.add_argument('--train_ratio', default=0.9)
    parser.add_argument('--val_ratio', default=0.0)
    parser.add_argument('--test_ratio', default=0.1)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch_size', type=int, default=2, help='training batch size')

    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--save_model', default=False)
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        if arg.phase == 'train':
            self.save_arg()
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = 'n'
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            dataset1 = Feeder(**self.arg.train_feeder_args)

            self.data_loader['train'] = DataLoader(
                dataset= dataset1 ,
                batch_size=self.arg.batch_size,
                num_workers=8,
                shuffle=True,
                drop_last=True
            )
        self.data_loader['test'] = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True
        )

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args)
        self.loss = nn.CrossEntropyLoss()
        self.loss2=nn.MSELoss()

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = jt.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_parameters(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_parameters(state)



    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()


        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            self.optimizer.lr = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, memory_usages,forward_time,save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
        train_class_total_num = np.array([0, 0, 0, 0])
        train_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        time_all = 0
        count_batchsize = 0
        for batch_idx, (data_m,data_p, label,feature, index) in tqdm(enumerate(loader),total= int(len(loader)/self.arg.batch_size)):
            label = label.reshape(-1)
            if len(label.size()) > 1:
                train_mode = 'MLL'
            else:
                train_mode = 'SLL'

            self.global_step += 1
            data_m = data_m.float().stop_grad()
            data_p = data_p.float().stop_grad()
            label = label.long().stop_grad()
            if train_mode == 'MLL':
                label = label.float()
            feature = feature.float().stop_grad()

            timer['dataloader'] += self.split_time()

            # forward
            begin_time = time.time()

            output_p,output2,output_m  = self.model(data_p,data_m)

            output=(output_m+output_p)/2

            if train_mode == 'MLL': 
                output_p = jt.sigmoid(output_p)
                output_m = jt.sigmoid(output_m)

            loss1_m=self.loss(output_m,label)
            loss1_p = self.loss(output_p, label)

            loss2=self.loss2(output2,feature)
            loss =loss1_m+loss1_p +loss2

            # backward
            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()
            end_time = time.time()
            time_all += end_time - begin_time
            count_batchsize += 1

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            if train_mode == 'SLL':
                predict_label = jt.argmax(output, dim=1)
                total_acc += jt.sum(jt.array(predict_label == label.data).float())
                cnt += label.shape[0]
                trues = label.numpy().tolist()
                for idx, lb in enumerate(predict_label[0]):
                    train_class_total_num[trues[idx]]+=1
                    train_class_true_num[trues[idx]] += int(lb==trues[idx])
            else:
                total_acc += jt.sum(jt.round(output) == label)
                cnt += label.numel()
                class_total_num = jt.sum(jt.round(output) == 1, dim=0) 
                class_true_num = jt.sum((jt.round(output) == label) & (label == 1), dim=0)  

                for idx in range(len(class_total_num)): train_class_total_num[idx] += class_total_num[idx]
                for idx in range(len(class_true_num)): train_class_true_num[idx] += class_true_num[idx]


            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_1m', loss1_m.numpy(), self.global_step)
            self.train_writer.add_scalar('loss_1p', loss1_p.numpy(), self.global_step)
            self.train_writer.add_scalar('loss_2', loss2.numpy(), self.global_step)

            # statistics
            self.lr = self.optimizer.lr
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            timer['statistics'] += self.split_time()

        current_memory = get_gpu_memory()  
        memory_usages.append(current_memory)

        time_average = time_all / count_batchsize
        forward_time.append(time_average)
        tmp_time = "epoch: " + str(epoch) + ":Time for forward:" + str(time_average) + '\n'
        with open('./src/BPM_GCN/logs/train/time_need.txt', 'a') as f:
            f.write(tmp_time)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))
        
        print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(train_class_true_num[0]*1.0/train_class_total_num[0],
                                                            train_class_true_num[1]*1.0/train_class_total_num[1],
                                                            train_class_true_num[2]*1.0/train_class_total_num[2],
                                                            train_class_true_num[3]*1.0/train_class_total_num[3]))
        print('Train Accuracy: {: .2f}%'.format(100 * total_acc*1.0 / cnt))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        test_class_total_num = np.array([0, 0, 0, 0])
        test_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            for batch_idx, (data_m,data_p, label,feature, index) in tqdm(enumerate(self.data_loader[ln]), total= int(len(self.data_loader[ln])/self.arg.batch_size)):
                label=label.reshape(-1)
                if len(label.size()) > 1: test_mode = 'MLL'
                else: test_mode = 'SLL'

                if test_mode == 'MLL': label = label.float()

                output_p, output2,output_m = self.model(data_p,data_m)
                output=(output_m+output_p)/2

                if test_mode == 'MLL': output = jt.sigmoid(output)

                loss = self.loss(output, label)
                loss_value.append(loss.data.item())

                predict_label = jt.argmax(output, 1)
                step += 1

                if test_mode == 'SLL':
                    trues = label.numpy().tolist()
                    for idx, lb in enumerate(predict_label[0]):
                        test_class_total_num[int(trues[idx])]+=1
                        test_class_true_num[int(trues[idx])] += int(lb==trues[idx])
                    total_acc += jt.sum(jt.array(predict_label == label.data))

                    cnt += label.shape[0]
                else:
                    total_acc += jt.sum(jt.round(output) == label)    
                    cnt += label.numel()
                    class_total_num = jt.sum(jt.round(output) == 1, dim=0)
                    class_true_num = jt.sum((jt.round(output) == label) & (label == 1), dim=0)

                    for idx in range(len(class_total_num)): test_class_total_num[idx] += class_total_num[idx]
                    for idx in range(len(class_true_num)): test_class_true_num[idx] += class_true_num[idx]

                if wrong_file is not None or result_file is not None:
                    predict = predict_label[0]
                    true = label.numpy()
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            loss = np.mean(loss_value)
            accuracy = total_acc * 1.0 / cnt
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                if self.arg.save_model:
                    jt.save(self.model.state_dict(), os.path.join(self.arg.model_saved_name, "model_epoch_{}_best.pth".format(epoch)))


            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))

            print('Top1: {:.2f}%'.format(accuracy*100))
            self.print_log('Best acc: {:.2f}%'.format(self.best_acc*100))
            print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(test_class_true_num[0] * 1.0 / test_class_total_num[0],
                                                                test_class_true_num[1] * 1.0 / test_class_total_num[1],
                                                                test_class_true_num[2] * 1.0 / test_class_total_num[2],
                                                                test_class_true_num[3] * 1.0 / test_class_total_num[3]))


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            memory_usages = []  
            forward_time = []  
            time_all = 0
            count_iters = 0

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                save_model=False
                start = time.time()
                self.train(epoch, memory_usages, forward_time, save_model=save_model)
                end = time.time()
                time_all += end - start
                count_iters += 1
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            self.print_log('best accuracy: {}'.format(self.best_acc))
            with open('./src/BPM_GCN/logs/train/memory_usages.txt', 'a') as file:
                for item in memory_usages:
                    file.write(f"{item}\n")

            with open('./src/BPM_GCN/logs/train/forward_time_need.txt', 'a') as file:
                for item in forward_time:
                    file.write(f"{item}\n")

            tmp_time = "Time for one epoch::" + str(time_all / count_iters) + '\n'

            with open('./src/BPM_GCN/logs/train/time_need.txt', 'a') as f:
                f.write(tmp_time)

            with open('./src/BPM_GCN/logs/train/time_need.txt', 'a') as f:
                f.write(f"Average memory usage：{sum(memory_usages) / len(memory_usages)}\n")
                f.write(f"Average forward time：{sum(forward_time) / len(forward_time)}\n")

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '/wrong.txt'
                rf = self.arg.model_saved_name + '/right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
