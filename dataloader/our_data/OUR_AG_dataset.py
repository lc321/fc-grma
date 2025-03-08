import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import re

import os

class OURAGDataSet(data.Dataset):
    baseurl = 'dataset/our_data/'
    act_type = {'walk': 0,
                'upstairs': 1,
                'downstairs': 2}

    def __init__(self, root, train=True, transform=None,index=None, base_sess=True):

        self.root = root
        self.train = train  # training set or test set

        self.transform = transform

        path = os.path.join(self.root,self.baseurl)
        self.data, self.targets, self.actions = self.load_data(path)

        # side=left datatype=imu
        self.SelectByData_type(self.data)

        # encode targets
        self.targets_encoder = self.targets * 3 + self.actions

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.targets_encoder, test_size=0.3,
                                                            random_state=0)


        if index is not None:
            if train:
                self.data, self.targets_encoder = self.SelectfromDefault(x_train, y_train, index)
            else:
                self.data, self.targets_encoder = self.SelectfromDefault(x_test, y_test, index)

        if base_sess:
            self.targets = self.targets_encoder // 3
        else:
            self.targets = self.targets_encoder

    def SelectFromAction(self, action_index, index):
        index_tmp = []

        for i in action_index:
            ind_cl = np.where(i == (index % 3))[0]
            if index_tmp == []:
                index_tmp = index[ind_cl]
            else:
                index_tmp = np.hstack((index_tmp, index[ind_cl])).flatten()

        return index_tmp

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
        if self.transform:
            data = self.transform(data)
        return data, self.targets[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self,path):
        data = []
        target = []
        action = []
        dirs = os.listdir(path)
        pattern = re.compile(r'.npy$')
        for dir in dirs:
            sub_path = os.path.join(path,dir)
            if os.path.isdir(sub_path):
                files = os.listdir(sub_path)
                for file in files:
                    fileName = os.path.join(sub_path, file)
                    rawdata = np.load(fileName)
                    splited_data = self.split_data(rawdata)
                    data.extend(splited_data)
                    target.extend([int(dir)-1]*len(splited_data))
                    action_index = int(re.sub(pattern,'',file))
                    if action_index == 1:
                        action.extend([self.act_type['walk']]*len(splited_data))
                    elif action_index % 2 != 0:
                        action.extend([self.act_type['upstairs']]*len(splited_data))
                    elif action_index % 2 == 0:
                        action.extend([self.act_type['downstairs']]*len(splited_data))
        data = np.array(data,dtype=np.float32)
        target = np.array(target,dtype=np.int64).reshape(-1)
        action = np.array(action,dtype=np.int64).reshape(-1)
        return data,target,action


    def split_data(self, rawdata, lenths=128, overlap=0.5):
        data = rawdata
        data_list = []
        l = data.shape[1]
        for i in range(0, l-lenths, int(lenths * (1 - overlap))):
            d = data[:,i:i+lenths]
            data_list.append(d)
        return data_list
    
    def SelectByData_type(self,data,side='L',data_type='imu'):
        if side == 'all':
            if data_type == 'all':
                self.data = data
                print(self.data[0].shape, 'all')
            elif data_type == 'acc':
                self.data = np.vstack((data[:,0:3,:],data[:,16:19,:]))
                # self.data = [np.vstack((i[0:3, :], i[16:19, :])) for i in data]
                print(self.data[0].shape, 'acc')
                # print(self.data[0].shape)
            elif data_type == 'gyr':
                self.data = np.vstack((data[:, 3:6, :], data[:, 19:22, :]))
                # self.data = [np.vstack((i[3:6, :], i[19:22, :])) for i in data]
                print(self.data[0].shape, 'gyr')
            elif data_type == 'force':
                self.data = np.vstack((data[:, 6:16, :], data[:, 22:32, :]))
                # self.data = [np.vstack((i[6:16, :], i[22:32, :])) for i in data]
                print(self.data[0].shape, 'force')
            elif data_type == 'imu':
                self.data = np.vstack((data[:, 0:6, :], data[:, 16:22, :]))
                # self.data = [np.vstack((i[0:6, :], i[16:22, :])) for i in data]
                print(self.data[0].shape, 'imu')
        elif side == 'L':
            if data_type == 'all':
                self.data = data[:,0:16,:]
                # self.data = [i[0:16, :] for i in data]
                print(self.data[0].shape, 'all')
            elif data_type == 'acc':
                self.data = data[:, 0:3, :]
                # self.data = [i[0:3, :] for i in data]
                print(self.data[0].shape, 'acc')
                # print(self.data[0].shape)
            elif data_type == 'gyr':
                self.data = data[:, 3:6, :]
                # self.data = [i[3:6, :] for i in data]
                print(self.data[0].shape, 'gyr')
            elif data_type == 'force':
                self.data = data[:, 6:16, :]
                # self.data = [i[6:16, :] for i in data]
                print(self.data[0].shape, 'force')
            elif data_type == 'imu':
                self.data = data[:, 0:6, :]
                # self.data = [i[0:6, :] for i in data]
                print(self.data[0].shape, 'imu')
        elif side == 'R':
            if data_type == 'all':
                self.data = data[:, 16:32, :]
                # self.data = [i[16:32, :] for i in data]
                print(self.data[0].shape, 'all')
            elif data_type == 'acc':
                self.data = data[:, 16:19, :]
                # self.data = [i[16:19, :] for i in data]
                print(self.data[0].shape, 'acc')
                # print(self.data[0].shape)
            elif data_type == 'gyr':
                self.data = data[:, 19:22, :]
                # self.data = [i[19:22, :] for i in data]
                print(self.data[0].shape, 'gyr')
            elif data_type == 'force':
                self.data = data[:, 22:32, :]
                # self.data = [i[22:32, :] for i in data]
                print(self.data[0].shape, 'force')
            elif data_type == 'imu':
                self.data = data[:, 16:22, :]
                # self.data = [i[16:22, :] for i in data]
                print(self.data[0].shape, 'imu')