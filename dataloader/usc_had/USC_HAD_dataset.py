import torch.utils.data as data

from sklearn.model_selection import train_test_split
import scipy.io as sio
import re
from dataloader.data_utils import *

import os

class USCHADDataSet(data.Dataset):
    baseurl = 'dataset/USC-HAD/'
    train_data_path = 'train/InertialSignals/'
    train_subject_path = 'train/subject_train.txt'
    train_action_path = 'train/y_train.txt'
    test_data_path = 'test/InertialSignals/'
    test_subject_path = 'test/subject_test.txt'
    test_action_path = 'test/y_test.txt'

    def __init__(self, root, train=True, transform=None,
                 download=False, index=None, base_sess=True):

        self.root = root
        self.train = train  # training set or test set

        self.transform = transform

        path = os.path.join(self.root,self.baseurl)
        self.data, self.targets, self.actions = self.load_data(path)

        # encode targets
        self.targets_encoder = self.targets * 6 + self.actions

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.targets_encoder, test_size=0.3,
                                                            random_state=0)

        if train:
            self.data, self.targets_encoder = self.SelectfromDefault(x_train, y_train, index)
        else:
            self.data, self.targets_encoder = self.SelectfromDefault(x_test, y_test, index)

        if base_sess:
            self.targets = self.targets_encoder // 6
        else:
            self.targets = self.targets_encoder

    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []

        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.targets[idx]

        return data, labels

    def __len__(self):
        return len(self.data)

    def load_data(self,path):
        data = []
        target = []
        action = []
        dirs = os.listdir(path)
        pattern = re.compile(r'a[1-6]t')
        for dir in dirs:
            sub_path = os.path.join(path,dir)
            if os.path.isdir(sub_path):
                files = os.listdir(sub_path)
                for file in files:
                    if re.match(pattern,file):
                        fileName = os.path.join(sub_path, file)
                        rawdata = sio.loadmat(fileName)
                        splited_data,splited_target,splited_action = self.split_data(rawdata)
                        data.extend(splited_data)
                        target.extend(splited_target)
                        action.extend(splited_action)
        data = np.array(data,dtype=np.float32).transpose(0,2,1)
        target = np.array(target,dtype=np.int64).reshape(-1)
        action = np.array(action,dtype=np.int64).reshape(-1)
        return data,target,action


    def split_data(self, rawdata, lenths=128, overlap=0.5):
        data = rawdata['sensor_readings']
        data_list = []
        l = data.shape[0]
        for i in range(0, l-lenths, int(lenths * (1 - overlap))):
            d = data[i:i+lenths]
            data_list.append(d)
        target_list = [int(rawdata['subject'])-1]*len(data_list)
        action_list = [int(rawdata['activity_number'])-1]*len(data_list)
        return data_list,target_list,action_list
