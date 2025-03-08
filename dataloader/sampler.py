import torch
import numpy as np
import copy

class TrueClassesSampler():

    def __init__(self, label, n_batch,n_act, n_cls, n_per, mode='base'):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_act = n_act
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.label = label
        self.index = np.unique(label)
        self.m_ind = []  # the data index of each class
        for i in self.index:
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            t_cls = len(self.m_ind) // self.n_act
            true_classes = torch.randperm(t_cls)[:self.n_cls]
            classes = true_classes.unsqueeze(1) * self.n_act + torch.arange(self.n_act)
            classes = classes.flatten()
            idx = torch.randperm(classes.nelement())
            classes = classes[idx]
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, mode='base'):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.label = label
        self.index = np.unique(label)
        self.m_ind = []  # the data index of each class
        for i in self.index:
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
