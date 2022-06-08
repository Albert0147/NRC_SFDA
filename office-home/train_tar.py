import argparse
import os, sys
sys.path.append('./')

import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import pickle
from utils import *
from torch import autograd



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def office_load_idx(args):
    train_bs = args.batch_size
    if args.home == True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'Art'
        elif ss == 'c':
            s = 'Clipart'
        elif ss == 'p':
            s = 'Product'
        elif ss == 'r':
            s = 'Real_World'

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'

        s_tr, s_ts = './data/office-home/{}.txt'.format(
            s), './data/office-home/{}.txt'.format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)
    
        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = './data/office-home/{}.txt'.format(
            t), './data/office-home/{}.txt'.format(t)
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict['source'])
        test_source = ImageList_idx(s_ts, transform=prep_dict['source'])
        train_target = ImageList_idx(open(t_tr).readlines(),
                                     transform=prep_dict['target'])
        test_target = ImageList_idx(open(t_ts).readlines(),
                                    transform=prep_dict['test'])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  #2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False)
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  #3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False)
    return dset_loaders



def train_target(args):
    dset_loaders = office_load_idx(args)
    ## set base network

    netF = network.ResNet_FE().cuda()
    oldC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'
    oldC.load_state_dict(torch.load(modelpath))

    optimizer = optim.SGD(
        [
            {
                'params': netF.feature_layers.parameters(),
                'lr': args.lr * .1  #1
            },
            {
                'params': netF.bottle.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netF.bn.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': oldC.parameters(),
                'lr': args.lr * 1  #10
            }
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)
    optimizer = op_copy(optimizer)

    acc_init = 0
    start = True
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netF.forward(inputs)  # a^t
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()
            #all_label = torch.cat((all_label, labels.float()), 0)
        #fea_bank = fea_bank.detach().cpu().numpy()
        #score_bank = score_bank.detach()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    oldC.train()

    while iter_num < max_iter:
        
        # comment this if on office-31
        if iter_num>0.5*max_iter:
            args.K = 5
            args.KK = 4

        #for epoch in range(args.max_epoch):
        netF.train()
        oldC.train()
        #iter_target = iter(dset_loaders["target"])
        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        
        # uncomment this if on office-31
        #lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_target = inputs_test.cuda()

        features_test = netF(inputs_target)

        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)
        output_re = softmax_out.unsqueeze(1)  # batch x 1 x num_class

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=args.K + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C
            #score_near=score_near.permute(0,2,1)

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,
                                                       -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near,
                                  fea_bank_re.permute(0, 2,
                                                      1))  # batch x K x n
            _, idx_near_near = torch.topk(
                distance_, dim=-1, largest=True,
                k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M

            
            #weight_kk[idx_near_near == tar_idx_] = 0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM
            weight_kk = weight_kk.fill_(0.1)
            score_near_kk = score_near_kk.contiguous().view(
                score_near_kk.shape[0], -1, args.class_num)  # batch x KM x C

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x KM x C
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1))
        loss = torch.mean(const)  #* 0.5

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C
        
        loss += torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
             weight.cuda()).sum(1))  #

        msoftmax = softmax_out.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss += im_div  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()

            #print("target")
            acc1, _ = cal_acc_(dset_loaders['test'], netF, oldC)  #1
            #acc_knn = cal_acc_knn(dset_loaders['test'], netF, oldC)  #1
            #print("source")
            log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                args.dset, iter_num, max_iter, acc1 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            if acc1 >= acc_init:
                acc_init = acc1
                best_netF = netF.state_dict()
                best_netC = oldC.state_dict()

                torch.save(best_netF, osp.join(args.output_dir, "F_submitted.pt"))
                torch.save(best_netC,
                           osp.join(args.output_dir, "C_submitted.pt"))

    #return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=40,
                        help="maximum epoch") # set to 50 on office-31
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--k',
                        type=int,
                        default=2,
                        help="number of neighborhoods")
    parser.add_argument('--dset', type=str, default='a2r')
    parser.add_argument('--choice', type=str, default='shot')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--K', type=int, default=4) # set to 2 on office-31 (or 3 on a2w)
    parser.add_argument('--KK', type=int, default=3) # set to 3 on office-31 (or 2 on a2w)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--par', type=float, default=0.1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='weight')  #trainingC_2
    parser.add_argument('--file', type=str, default='k33_nodecay')
    parser.add_argument('--home', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    #train_target(args)
    #if args.file=='cluster':
    train_target(args)
    
