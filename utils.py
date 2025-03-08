import random
import torch
import os
import time

import numpy as np
import pprint as pprint

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.constants import micro
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_precision(logits, label, label_list, args):
    index = torch.argmax(logits, dim=1).cpu()  # 获取每个样本的预测类别

    pred = torch.div(torch.Tensor(label_list[index]), args.num_actions, rounding_mode='floor')

    precision = precision_score(label.cpu(), pred.cpu(), average='macro')

    return precision


def count_recall(logits, label, label_list, args):
    index = torch.argmax(logits, dim=1).cpu()  # 获取每个样本的预测类别

    pred = torch.div(torch.Tensor(label_list[index]), args.num_actions, rounding_mode='floor')

    recall = recall_score(label.cpu(), pred.cpu(), average='macro')

    return recall


def count_f1_score(logits, label, label_list, args):
    index = torch.argmax(logits, dim=1).cpu()  # 获取每个样本的预测类别

    pred = torch.div(torch.Tensor(label_list[index]), args.num_actions, rounding_mode='floor')

    f1_score_ = f1_score(label.cpu(), pred.cpu(), average='macro')

    return f1_score_


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    # return accuracy_score(label.cpu(), pred.cpu())
    # pred = pred // 6
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_final(logits, label_list, label, args):
    index = torch.argmax(logits, dim=1).cpu()
    pred = torch.div(torch.Tensor(label_list[index]),args.num_actions, rounding_mode='floor')
    if torch.cuda.is_available():
        pred = pred.cuda()
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def encode_by_label(label, args):
    action_label = label % args.num_actions
    action_index = action_label
    index = torch.unique(action_label)
    for i, item in enumerate(index):
        ind = torch.nonzero(action_label == item).squeeze()
        action_index[ind] = i

    subject_label = torch.div(label, args.num_actions, rounding_mode='floor')
    subject_index = subject_label
    index = torch.unique(subject_label)
    for i, item in enumerate(index):
        ind = torch.nonzero(subject_label == item).squeeze()
        subject_index[ind] = i
    return subject_index, action_index
