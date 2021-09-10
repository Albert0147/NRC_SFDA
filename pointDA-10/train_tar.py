import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_pointnet import Pointnet_cls
import Model
from dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
from torch.autograd import Variable
import time
import numpy as np
import os
import sys
import os.path as osp
import argparse
import pdb
import mmd
# from utils import *
import math
import warnings
import random
import shutil

#from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source',
                    '-s',
                    type=str,
                    help='source dataset',
                    default='scannet')
parser.add_argument('-target',
                    '-t',
                    type=str,
                    help='target dataset',
                    default='modelnet')
parser.add_argument('-batchsize',
                    '-b',
                    type=int,
                    help='batch size',
                    default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='7')
parser.add_argument('-epochs',
                    '-e',
                    type=int,
                    help='training epoch',
                    default=100)
parser.add_argument('-K', type=int, default=2)
parser.add_argument('-KK', type=int, default=2)
parser.add_argument('-models',
                    '-m',
                    type=str,
                    help='alignment model',
                    default='MDA')
parser.add_argument('-lr', type=float, help='learning rate',
                    default=0.00001)  #0.0001
parser.add_argument('-scaler',
                    type=float,
                    help='scaler of learning rate',
                    default=1.)
parser.add_argument('-weight',
                    type=float,
                    help='weight of src loss',
                    default=1.)
parser.add_argument('-datadir',
                    type=str,
                    help='directory of data',
                    default='./dataset/')
parser.add_argument('-tb_log_dir',
                    type=str,
                    help='directory of tb',
                    default='./logs')
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
#writer = SummaryWriter(log_dir=args.tb_log_dir)

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 10
dir_root = os.path.join(args.datadir, 'PointDA_data/')


# print(dir_root)
def main():
    print('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar:', args.target)

    # Data loading

    data_func = {
        'modelnet': Modelnet40_data,
        'scannet': Scannet_data_h5,
        'shapenet': Shapenet_data
    }

    source_train_dataset = data_func[args.source](pc_input_num=1024,
                                                  status='train',
                                                  aug=True,
                                                  pc_root=dir_root +
                                                  args.source)
    target_train_dataset1 = data_func[args.target](pc_input_num=1024,
                                                   status='train',
                                                   aug=True,
                                                   pc_root=dir_root +
                                                   args.target)
    source_test_dataset = data_func[args.source](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.source)
    target_test_dataset1 = data_func[args.target](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_train1 = len(target_train_dataset1)
    num_target_test1 = len(target_test_dataset1)

    source_train_dataloader = DataLoader(source_train_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)
    source_test_dataloader = DataLoader(source_test_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=2,
                                        drop_last=True)
    target_train_dataloader = DataLoader(target_train_dataset1,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)
    target_test_dataloader = DataLoader(target_test_dataset1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=2,
                                        drop_last=True)

    print(
        'num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d} '
        .format(num_source_train, num_source_test, num_target_test1))
    print('batch_size:', BATCH_SIZE)

    # Model

    model_f = Model.FE()
    model_f = model_f.to(device=device)

    model_c = Model.Pointnet_c()
    model_c = model_c.to(device=device)

    modelpath = args.output_dir + '/F_src.pt'
    model_f.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/C_src.pt'
    model_c.load_state_dict(torch.load(modelpath))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    remain_epoch = 50

    # Optimizer

    params = [{
        'params': v
    } for k, v in model_f.named_parameters() if 'pred_offset' not in k]

    optimizer_g = optim.Adam(params, lr=LR, weight_decay=weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g,
                                                         T_max=args.epochs +
                                                         remain_epoch)

    optimizer_c = optim.Adam([{
        'params': model_c.parameters()
    }],
                             lr=LR * 2,
                             weight_decay=weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c,
                                                         T_max=args.epochs +
                                                         remain_epoch)
    '''optimizer_dis = optim.Adam([{'params':model.g.parameters()},{'params':model.attention_s.parameters()},{'params':model.attention_t.parameters()}],
        lr=LR*args.scaler, weight_decay=weight_decay)
    lr_schedule_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=args.epochs+remain_epoch)'''
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
        if epoch > 0:
            if epoch <= 30:
                lr = args.lr * args.scaler * (0.5**(epoch // 5))
            else:
                lr = args.lr * args.scaler * (0.5**(epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            #writer.add_scalar('lr_dis', lr, epoch)

    def discrepancy(out1, out2):
        """discrepancy loss"""
        out = torch.mean(
            torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
        return out

    def make_variable(tensor, volatile=False):
        """Convert Tensor to Variable."""
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)

    with torch.no_grad():
        model_f.eval()
        model_c.eval()
        loss_total = 0
        correct_total = 0
        data_total = 0
        acc_class = torch.zeros(10, 1)
        acc_to_class = torch.zeros(10, 1)
        acc_to_all_class = torch.zeros(10, 10)

        correct_total = 0
        data_total = 0
        for batch_idx, (data, label, _) in enumerate(target_test_dataloader):
            data = data.to(device=device)
            label = label.to(device=device).long()
            output = model_c(model_f(data))
            loss = criterion(output, label)
            _, pred = torch.max(output, 1)
            correct_total += torch.sum(pred == label)
            data_total += data.size(0)

        #pred_loss = loss_total/data_total
        pred_acc_t = correct_total.double() / data_total

        logs = 'Target 1: before training [Acc_t: {:.4f}]'.format(pred_acc_t)
        print(logs)
        #writer.add_scalar('accs/target_test_acc', pred_acc_s, epoch)
        args.out_file.write(logs + '\n')
        args.out_file.flush()

    acc_init = 0
    start = True
    loader = target_train_dataloader
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 1024)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    model_f.eval()
    model_c.eval()
    with torch.no_grad():
        iter_test = iter(target_train_dataloader)
        for i in range(len(target_train_dataloader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = model_f(inputs)  # a^t
            #output_norm = F.normalize(output)
            output_norm = output
            outputs = model_c(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    for epoch in range(max_epoch):

        model_f.train()
        model_c.train()

        since_e = time.time()

        lr_schedule_g.step(epoch=epoch)
        lr_schedule_c.step(epoch=epoch)

        loss_total = 0
        loss_adv_total = 0
        loss_node_total = 0
        correct_total = 0
        data_total = 0
        data_t_total = 0
        cons = math.sin((epoch + 1) / max_epoch * math.pi / 2)

        # Training
        for batch_idx, batch_s in enumerate(target_train_dataloader):

            data, label, tar_idx = batch_s

            data = data.to(device=device)

            inputs_target = data.cuda()

            features_test = model_f(inputs_target)

            output = model_c(features_test)
            softmax_out = nn.Softmax(dim=1)(output)
            output_re = softmax_out.unsqueeze(1)

            with torch.no_grad():
                #output_f_norm = F.normalize(features_test)
                output_f_norm = features_test
                output_f_ = output_f_norm.cpu().detach().clone()

                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = softmax_out.detach().clone()

                distance = torch.cdist(output_f_, fea_bank, p=2)
                _, idx_near = torch.topk(distance,
                                         dim=-1,
                                         largest=False,
                                         k=args.K + 1)
                idx_near = idx_near[:, 1:]  #batch x K
                score_near = score_bank[idx_near]  #batch x K x C

                fea_near = fea_bank[idx_near]  #batch x K x num_dim
                fea_bank_re = fea_bank.unsqueeze(0).expand(
                    fea_near.shape[0], -1, -1)  # batch x n x dim
                #distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                distance_ = torch.cdist(fea_near, fea_bank_re)

                _, idx_near_near = torch.topk(
                    distance_, dim=-1, largest=False,
                    k=args.KK + 1)  # M near neighbors for each of above K ones
                idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
                match = (
                    idx_near_near == tar_idx_).sum(-1).float()  # batch x K
                weight = torch.where(
                    match > 0., match,
                    torch.ones_like(match).fill_(0.1))  # batch x K

                weight_kk = weight.unsqueeze(-1).expand(
                    -1, -1, args.KK)  # batch x K x M
                score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                #print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(
                    weight_kk.shape[0], -1)  # batch x KM

                score_near_kk = score_near_kk.contiguous().view(
                    score_near_kk.shape[0], -1,
                    args.class_num)  # batch x KM x C

            # nn of nn
            output_re = softmax_out.unsqueeze(1).expand(
                -1, args.K * args.KK, -1)  # batch x C x 1
            const = torch.mean(
                (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                 weight_kk.cuda()).sum(1))
            loss = torch.mean(const)  #* 0.5

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(
                -1, args.K, -1)  # batch x K x C

            loss += torch.mean(
                (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)
                 * weight.cuda()).sum(1))

            msoftmax = softmax_out.mean(dim=0)
            im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            loss += im_div

            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_g.step()
            optimizer_c.step()

        # Testing

        with torch.no_grad():
            model_f.eval()
            model_c.eval()
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(10, 1)
            acc_to_class = torch.zeros(10, 1)
            acc_to_all_class = torch.zeros(10, 10)

            correct_total = 0
            data_total = 0
            for batch_idx, (data, label,
                            _) in enumerate(target_test_dataloader):
                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model_c(model_f(data))
                #output = (pred1 + pred2)/2
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                #loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            #pred_loss = loss_total/data_total
            pred_acc_t = correct_total.double() / data_total

            logs = 'Target 1:{} [Acc_t: {:.4f}]'.format(epoch, pred_acc_t)
            print(logs)
            #writer.add_scalar('accs/target_test_acc', pred_acc_s, epoch)
            args.out_file.write(logs + '\n')
            args.out_file.flush()

        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(
            epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        print(' ')


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '9'
    args.class_num = 10
    '''args.K=2
    args.KK=2'''

    SEED = 2021
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    since = time.time()

    #task = ['scannet', 'modelnet', 'shapenet']
    args.output_dir = osp.join('./model', args.source + '2' + args.target)
    '''if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    task_s = args.source
    task.remove(task_s)
    task_all = [task_s + '2' + i for i in task]
    for task_sameS in task_all:
        path_task = os.getcwd() + '/' + 'model/' + task_sameS
        if not osp.exists(path_task):
            os.mkdir(path_task)'''

    args.out_file = open(osp.join(args.output_dir, 'tar.txt'), 'w')
    args.out_file.write('\n')
    args.out_file.flush()

    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_pass // 60, time_pass % 60))
