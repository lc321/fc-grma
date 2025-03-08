from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.args.save_path = self.args.save_path + '/%s/' % self.args.dataset + '%s/' % args.project
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            if torch.cuda.is_available():
                self.best_model_dict = torch.load(self.args.model_dir)['params']
            else:
                self.best_model_dict = torch.load(self.args.model_dir, map_location='cpu')['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_optimizer_meta(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.self_attn.parameters(), 'lr': self.args.lrg1},
                                     {'params': self.model.module.attn.parameters(), 'lr': self.args.lrg2}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def pre_train(self, optimizer, scheduler, epoch, trainloader, testloader, args, session, result_list):
        start_time = time.time()

        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
        # test model with all seen class
        tsl, tsa, _ = base_test(self.model, testloader, epoch, args)

        # save better model
        if (tsa * 100) >= self.trlog['max_acc'][session]:
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_acc_epoch'] = epoch
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            # save backdone
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('********A better model is found!!**********')
            print('Saving model to :%s' % save_model_dir)
        print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                           self.trlog['max_acc'][session]))

        self.trlog['train_loss'].append(tl)
        self.trlog['train_acc'].append(ta)
        self.trlog['test_loss'].append(tsl)
        self.trlog['test_acc'].append(tsa)
        lrc = scheduler.get_last_lr()[0]
        result_list.append(
            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                epoch, lrc, tl, ta, tsl, tsa))
        print('This epoch takes %d seconds' % (time.time() - start_time),
              '\nstill need around %.2f mins to finish this session' % (
                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
        scheduler.step()

    def meta_train(self, optimizer, scheduler, epoch, args, session, result_list):
        start_time = time.time()
        train_set, trainloader, testloader = get_base_dataloader(args)

        # train base sess
        self.model.eval()
        tl, ta, tl1, tl2 = proto_train(self.model, trainloader, optimizer, scheduler, epoch, args)

        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

        self.model.module.mode = 'avg_cos'
        tsl, tsa, vscore = test(self.model, testloader, args)
        if (tsa * 100) >= self.trlog['max_acc'][session]:
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_precision'][session] = float('%.3f' % (vscore[0] * 100))
            self.trlog['max_recall'][session] = float('%.3f' % (vscore[1] * 100))
            self.trlog['max_f1_score'][session] = float('%.3f' % (vscore[2] * 100))
            self.trlog['max_acc_epoch'] = epoch
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('********A better model is found!!**********')
            print('Saving model to :%s' % save_model_dir)
        print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                           self.trlog['max_acc'][session]))
        self.trlog['test_loss'].append(tsl)
        self.trlog['test_acc'].append(tsa)
        lrc = scheduler.get_last_lr()[0]
        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f,training_loss1:%.5f,training_loss2:%.5f' % (
            epoch, lrc, tl, ta, tsl, tsa, tl1, tl2))
        result_list.append(
            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                epoch, lrc, tl, ta, tsl, tsa))

        self.trlog['train_loss'].append(tl)
        self.trlog['train_acc'].append(ta)

        print('This epoch takes %d seconds' % (time.time() - start_time),
              '\nstill need around %.2f mins to finish' % (
                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
        scheduler.step()

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label

                if args.project == 'base':
                    print("======= pre-train =========")
                    optimizer, scheduler = self.get_optimizer_base()
                    train_set, trainloader, testloader = get_pre_train_dataloader(args)
                    for epoch in range(args.epochs_base):
                        self.pre_train(optimizer, scheduler, epoch, trainloader, testloader, args, session,
                                  result_list)
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                    if not args.not_data_init:
                        self.model.load_state_dict(self.best_model_dict)
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                        best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        torch.save(dict(params=self.model.state_dict()), best_model_dir)

                        self.model.module.mode = 'avg_cos'
                        tsl, tsa, vscore = base_test(self.model, testloader, 0, args)
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_precision'][session] = float('%.3f' % (vscore[0] * 100))
                        self.trlog['max_recall'][session] = float('%.3f' % (vscore[1] * 100))
                        self.trlog['max_f1_score'][session] = float('%.3f' % (vscore[2] * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                if args.project == 'meta':
                    print("======= meta-train =========")
                    optimizer, scheduler = self.get_optimizer_meta()
                    self.model = self.update_param(self.model, self.best_model_dict)
                    for epoch in range(args.epochs_base):
                        self.meta_train(optimizer, scheduler, epoch, args, session,
                                        result_list)
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                    # always replace fc with avg mean
                    train_set, trainloader, testloader = get_base_dataloader(args)
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, vscore = test(self.model, testloader, args)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    self.trlog['max_precision'][session] = float('%.3f' % (vscore[0] * 100))
                    self.trlog['max_recall'][session] = float('%.3f' % (vscore[1] * 100))
                    self.trlog['max_f1_score'][session] = float('%.3f' % (vscore[2] * 100))
                    print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                train_set, trainloader, testloader = get_new_dataloader(args, session)
                print("new classes for this session:\n", np.unique(train_set.targets // args.num_actions))

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                class_list = np.unique(train_set.targets)
                self.model.module.update_fc(trainloader, class_list)

                tsl, tsa, vscore = incr_test(self.model, testloader, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_precision'][session] = float('%.3f' % (vscore[0] * 100))
                self.trlog['max_recall'][session] = float('%.3f' % (vscore[1] * 100))
                self.trlog['max_f1_score'][session] = float('%.3f' % (vscore[2] * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print("max_acc:", self.trlog['max_acc'])
        print("max_precision:", self.trlog['max_precision'])
        print("max_recall:", self.trlog['max_recall'])
        print("max_f1_score:", self.trlog['max_f1_score'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
