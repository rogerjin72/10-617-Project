#!/usr/bin/env python3 -u

from __future__ import print_function

import csv
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import data_loader
from model import ResNet18

from collections import OrderedDict

from adversarial import FGSM
from dct_adversarial import DCTFGSM

from args import get_args_test
args = get_args_test()
use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()

def print_status(string):
    if args.local_rank % ngpus_per_node ==0:
        print(string)

if args.local_rank % ngpus_per_node == 0:
    print(torch.cuda.device_count())
    print('Using CUDA..')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print_status('==> Preparing data..')

trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

if args.dataset == 'cifar-10':
    num_outputs = 10
elif args.dataset == 'cifar-100':
    num_outputs = 100

if args.model == 'ResNet50':
    expansion = 4
else:
    expansion = 1

# Model
print_status('==> Building model..')
train_type  = args.train_type

model = ResNet18(10, True) 
model.fc = nn.Identity()
if args.dataset=='cifar-10':
    Linear = nn.Sequential(nn.Linear(512*expansion, 10))
elif args.dataset=='cifar-100':
    Linear = nn.Sequential(nn.Linear(512*expansion, 100))

checkpoint_ = torch.load(args.load_checkpoint)
new_state_dict = OrderedDict()
for k, v in checkpoint_['model'].items():
    name = k[7:]
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

linearcheckpoint_ = torch.load(args.load_checkpoint)
new_state_dict = OrderedDict()
for k, v in linearcheckpoint_['linear_layer'].items():
    name = k[7:]
    new_state_dict[name] = v
Linear.load_state_dict(new_state_dict)

criterion = nn.CrossEntropyLoss()

use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()
    model.cuda()
    Linear.cuda()
    print_status(torch.cuda.device_count())
    print_status('Using CUDA..')
    cudnn.benchmark = True

attack_info = 'epsilon_'+str(args.epsilon)+'_alpha_'+ str(args.alpha) + '_min_val_' + str(0.0) + '_max_val_' + str(1.0) + '_max_iters_' + str(args.k) + '_type_' + str(args.attack_type) + '_randomstart_' + str(args.random_start) + '_dct_' + str(args.dct)
print_status("Attack information...")
print_status(attack_info)

if args.dct:
    attacker = DCTFGSM(model, Linear, epsilon=args.epsilon, alpha=args.alpha, n=args.k)
else:
    attacker = FGSM(model, Linear, epsilon=args.epsilon, alpha=args.alpha, min_val=0.0, max_val=1.0, n=args.k)


def test(attacker):
    global best_acc

    model.eval()
    Linear.eval()

    test_clean_loss = 0
    test_adv_loss = 0
    clean_correct = 0
    adv_correct = 0
    clean_acc = 0
    total = 0

    embeds_ori = None
    embeds_adv = None
    labels = None
    for idx, (image, label) in enumerate(testloader):

        img = image.cuda()
        y = label.cuda()
        total += y.size(0)
        if 'ResNet18' in args.model:
            if args.epsilon==0.0314 or args.epsilon==0.047:
                out = Linear(model(img))
                _, predx = torch.max(out.data, 1)
                clean_loss = criterion(out, y)

                clean_correct += predx.eq(y.data).cpu().sum().item()
                
                clean_acc = 100.*clean_correct/total

                test_clean_loss += clean_loss.data
        
        adv_inputs = attacker.get_adversarial_example(imgs=img, labels=y, perturb=args.random_start)
        
        out = model(adv_inputs)
        out_ori = model(img)
        if embeds_adv is None:
            embeds_adv = out
            embeds_ori = out_ori
        else:
            embeds_adv = torch.cat((embeds_adv, out))
            embeds_ori = torch.cat((embeds_ori, out_ori))
            
        out = Linear(out)
        
        _, predx = torch.max(out.data, 1)
        adv_loss = criterion(out, y)

        adv_correct += predx.eq(y.data).cpu().sum().item()
        adv_acc = 100.*adv_correct/total

        test_adv_loss += adv_loss.data
        if args.local_rank % ngpus_per_node == 0:
            print(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f} , adv Loss {:.3f}, adv acc {:.3f}'.format(test_clean_loss/(idx+1), clean_acc, test_adv_loss/(idx+1), adv_acc))

    print ("Test accuracy: {0}/{1}".format(clean_acc, adv_acc))
    torch.save(embeds_adv, f'test_embeds/embeds_adv_{attack_info}_{args.load_checkpoint[11:]}.pt')
    torch.save(embeds_ori, f'test_embeds/embeds_ori_{attack_info}_{args.load_checkpoint[11:]}.pt')
    return (clean_acc, adv_acc)

test_acc, adv_acc = test(attacker)

if not os.path.isdir('results'):
    os.mkdir('results')

args.name += ('_Robust'+ args.train_type + '_' +args.model + '_' + args.dataset)
loginfo = 'results/log_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['random_start', 'attack_type','epsilon','k','adv_acc'])

if args.local_rank % ngpus_per_node == 0:
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([args.random_start, args.attack_type, args.epsilon, args.k, adv_acc])

