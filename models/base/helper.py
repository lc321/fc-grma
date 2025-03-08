# import new Network name here and add in model_class args
import torch.cuda

from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
from dataloader.data_utils import *
from sklearn.preprocessing import LabelEncoder
from utils import encode_by_label

lamda = 0.9
le = LabelEncoder()
def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    train_class_index = get_base_classes(args)
    le.fit(train_class_index)
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]

        logits = model(data)
        logits = logits[:, train_class_index]

        label = le.transform(train_label.cpu())
        label = torch.LongTensor(label)
        if torch.cuda.is_available():
            label = label.cuda()

        loss = F.cross_entropy(logits, label)
        label_list = train_class_index
        true_label = torch.div(train_label, args.num_actions, rounding_mode='floor')
        acc = count_acc_final(logits, label_list, true_label, args)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def base_test(model, testloader, epoch, args):
    vscore = []
    all_logits = []
    all_labels = []

    test_class = get_base_classes(args)
    le.fit(test_class)
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]
            logits = model(data)
            logits = logits[:, test_class]

            label = le.transform(test_label.cpu())
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                label = label.cuda()

            loss = F.cross_entropy(logits, label)
            label_list = test_class
            true_label = torch.div(test_label, args.num_actions, rounding_mode='floor')
            acc = count_acc_final(logits, label_list, true_label, args)

            vl.add(loss.item())
            va.add(acc)

            all_logits.extend(logits)
            all_labels.extend(true_label)

        vl = vl.item()
        va = va.item()

        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)

        vprecision = count_precision(all_logits, all_labels, test_class, args)
        vrecall = count_recall(all_logits, all_labels, test_class, args)
        vf1_score = count_f1_score(all_logits, all_labels, test_class, args)
        vscore.extend((vprecision, vrecall, vf1_score))

    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, vscore


def proto_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()

    tl1 = Averager()
    tl2 = Averager()

    tqdm_gen = tqdm(trainloader)

    label = torch.arange(args.episode_way).repeat(args.episode_query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)

    for i, batch in enumerate(tqdm_gen, 1):
        data, true_label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]

        k = args.episode_way * args.episode_shot

        model.module.mode = 'encoder'
        data = model(data)

        proto, query = data[:k], data[k:]
        proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
        query = query.view(args.episode_query, args.episode_way, query.shape[-1])

        proto = proto.mean(0)
        proto_all = proto.unsqueeze(0).unsqueeze(0)
        query = query.unsqueeze(0)
        subject_index,action_index = encode_by_label(true_label[:args.episode_way], args)
        logits0,logits1,logits2,logits3 = model.module._forward_t(proto_all,subject_index,action_index, query)

        loss1 = F.cross_entropy(logits1, label)
        loss2 = F.cross_entropy(logits2, label)
        logits = logits3
        total_loss = loss1+loss2
        label_list = true_label[:args.episode_way]
        true_label = torch.div(true_label[k:], args.num_actions, rounding_mode='floor')
        acc = count_acc_final(logits, label_list, true_label, args)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl1.add(loss1.item())
        tl2.add(loss2.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    tl1 = tl1.item()
    tl2 = tl2.item()
    ta = ta.item()
    return tl, ta, tl1, tl2

def test(model, testloader, args):
    vscore = []
    all_logits = []
    all_labels = []

    test_labels = get_base_classes(args)
    le.fit(test_labels)
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]

            model.module.mode = 'encoder'
            query = model(data)
            query = query.unsqueeze(0).unsqueeze(0)

            proto_all = model.module.fc2.weight[test_labels, :].detach()
            proto_all = proto_all.unsqueeze(0).unsqueeze(0)
            subject_index, action_index = encode_by_label(torch.LongTensor(test_labels), args)
            logits0,logits1, logits2, logits3 = model.module._forward_t(proto_all, subject_index, action_index, query)

            label = le.transform(test_label.cpu())
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                label = label.cuda()
            loss1 = F.cross_entropy(logits1, label)
            loss2 = F.cross_entropy(logits2, label)
            logits = logits3
            loss = loss1+loss2
            true_label = torch.div(test_label, args.num_actions, rounding_mode='floor')
            acc = count_acc_final(logits, test_labels, true_label, args)
            vl.add(loss.item())
            va.add(acc)

            all_logits.extend(logits)
            all_labels.extend(true_label)

        vl = vl.item()
        va = va.item()

        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)

        vprecision = count_precision(all_logits, all_labels, test_labels, args)
        vrecall = count_recall(all_logits, all_labels, test_labels, args)
        vf1_score = count_f1_score(all_logits, all_labels, test_labels, args)
        vscore.extend((vprecision, vrecall, vf1_score))

    return vl, va, vscore


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=0, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_all_list = []
    base_classes = get_base_classes(args)
    for class_index in base_classes:
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_all_list.append(embedding_this)

    proto_all_list = torch.stack(proto_all_list, dim=0)
    if torch.cuda.is_available():
        proto_all_list = proto_all_list.cuda()
    model.module.fc2.weight.data[base_classes] = proto_all_list

    return model

def incr_test(model, testloader, args, session):
    vscore = []
    all_logits = []
    all_labels = []

    test_class_index = get_session_class_uci(args,session,train=False)
    le.fit(test_class_index)
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]

            model.module.mode = 'encoder'
            query = model(data)
            query = query.unsqueeze(0).unsqueeze(0)

            proto_all = model.module.fc2.weight[test_class_index, :].detach()
            proto_all = proto_all.unsqueeze(0).unsqueeze(0)
            subject_index, action_index = encode_by_label(torch.LongTensor(test_class_index), args)
            logits0,logits1, logits2, logits3 = model.module._forward_t(proto_all, subject_index, action_index, query)

            label = le.transform(test_label.cpu())
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                label = label.cuda()

            loss1 = F.cross_entropy(logits1, label)
            loss2 = F.cross_entropy(logits2, label)

            loss = loss1+loss2
            logits = logits3

            true_label = torch.div(test_label,args.num_actions,rounding_mode='floor')
            acc = count_acc_final(logits, test_class_index, true_label, args)
            vl.add(loss.item())
            va.add(acc)

            all_logits.extend(logits)
            all_labels.extend(true_label)

        vl = vl.item()
        va = va.item()

        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)

        vprecision = count_precision(all_logits, all_labels, test_class_index, args)
        vrecall = count_recall(all_logits, all_labels, test_class_index, args)
        vf1_score = count_f1_score(all_logits, all_labels, test_class_index, args)
        vscore.extend((vprecision, vrecall, vf1_score))

    return vl, va, vscore

def incr_base_test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            print(torch.unique(test_label))
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va
