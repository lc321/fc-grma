import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'our-ag':
        import dataloader.our_data.OUR_AG_dataset as Dataset
        args.base_class = 36
        args.num_actions = 3
        args.num_classes=60
        args.way = 9
        args.shot = 5
        args.sessions = 9
        args.Dataset=Dataset
        args.action_index = np.array([0, 1, 2])
    if args.dataset == 'usc-had':
        import dataloader.usc_had.USC_HAD_dataset as Dataset
        args.base_class = 6
        args.num_actions = 6
        args.num_classes=14
        args.way = 6
        args.shot = 5
        args.sessions = 9
        args.Dataset=Dataset
        args.action_index = np.array([0,1,2,3,4,5])
    return args

def get_base_testset(args):
    if args.dataset == 'usc-had':
        class_index = get_base_classes(args)
        testset = args.Dataset.USCHADDataSet(root=args.dataroot, train=False, download=False,
                                              index=class_index, base_sess=False)
    if args.dataset == 'our-ag':
        class_index = get_base_classes(args)
        testset = args.Dataset.OURAGDataSet(root=args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=False)
    return testset

def get_new_testset(args,session):
    if args.dataset == 'usc-had':
        class_new = get_session_class_uci(args,session,train=False)
        testset = args.Dataset.USCHADDataSet(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'our-ag':
        class_new = get_session_class_uci(args, session, train=False)
        testset = args.Dataset.OURAGDataSet(root=args.dataroot, train=False, download=False,
                                         index=class_new, base_sess=False)
    return testset

def get_dataloader(args,session):
    if session == 0:
        if args.project == 'pre-train':
            trainset, trainloader, testloader = get_pre_train_dataloader(args)
        elif args.project == 'meta-train':
            trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_pre_train_dataloader(args):
    if args.dataset == 'usc-had':
        class_index = get_base_classes(args)
        trainset = args.Dataset.USCHADDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)
        testset = args.Dataset.USCHADDataSet(root=args.dataroot, train=False, download=False,
                                              index=class_index, base_sess=False)
    if args.dataset == 'our-ag':
        class_index = get_base_classes(args)
        trainset = args.Dataset.OURAGDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)
        testset = args.Dataset.OURAGDataSet(root=args.dataroot, train=False, download=False,
                                              index=class_index, base_sess=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'usc-had':
        class_index = get_base_classes(args)
        trainset = args.Dataset.USCHADDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)
        testset = args.Dataset.USCHADDataSet(root=args.dataroot, train=False, download=False,
                                              index=class_index, base_sess=False)
    if args.dataset == 'our-ag':
        class_index = get_base_classes(args)
        trainset = args.Dataset.OURAGDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)
        testset = args.Dataset.OURAGDataSet(root=args.dataroot, train=False, download=False,
                                              index=class_index, base_sess=False)

    train_sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                      args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_index = np.arange(args.base_class + (session - 1) * args.way, args.base_class + session * args.way)
    if args.dataset == 'usc-had':
        class_index = get_session_class_uci(args,session)
        print("class_index",class_index)
        trainset = args.Dataset.USCHADDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)
    if args.dataset == 'our-ag':
        class_index = get_session_class_uci(args,session)
        trainset = args.Dataset.OURAGDataSet(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=False)

    train_sampler = CategoriesSampler(trainset.targets, args.inc_episode, args.episode_way,
                                args.episode_shot, mode='new')

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'usc-had':
        class_new = get_session_class_uci(args,session,train=False)
        testset = args.Dataset.USCHADDataSet(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'our-ag':
        class_new = get_session_class_uci(args, session, train=False)
        testset = args.Dataset.OURAGDataSet(root=args.dataroot, train=False, download=False,
                                         index=class_new, base_sess=False)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    class_list = SelectFromAction(args,class_list)
    class_list.sort()
    return class_list

def get_base_classes(args):
    class_index = np.arange(args.base_class * args.num_actions)
    class_index = SelectFromAction(args, class_index)
    class_index.sort()
    return class_index

def get_session_class_uci(args,session,train=True):
    if train:
        class_index = np.arange(args.base_class * args.num_actions + (session-1) * args.way, args.base_class * args.num_actions + session * args.way)
    else:
        class_index = np.arange(args.base_class * args.num_actions + session * args.way)
    class_index= SelectFromAction(args, class_index)
    class_index.sort()
    return class_index

def SelectFromAction(args,index):
    index_tmp = []

    for i in args.action_index:
        ind_cl = np.where(i == (index % 6))[0]
        if len(index_tmp) == 0:
            index_tmp = index[ind_cl]
        else:
            index_tmp = np.hstack((index_tmp, index[ind_cl])).flatten()

    return index_tmp
