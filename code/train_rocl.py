# reference: https://github.com/Kim-Minseon/RoCL/blob/master/src/rocl_train.py

import data_loader
from model import ResNet18
from args import get_args
from adversarial import FGSM, InstanceAdversary
import torch
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
from loss import NT_xent_loss
from torchsummary import summary

args = get_args()
#load data
trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

#load model
if 'contrastive' in args.train_type or 'linear_eval' in args.train_type:
    print("Contrastive")
    contrastive_learning = True  
else:
    contrastive_learning = False
model = ResNet18(10, contrastive_learning)
summary(model.cuda(), (3, 32, 32))
print('ResNet18 is loading ...')

# to GPU
if args.use_cuda:
    model = model.cuda()
    print('Model is on GPU ...')

# attack
Rep = InstanceAdversary(model, epsilon=args.epsilon, alpha=args.alpha, min_val=args.min, max_val=args.max, n=args.k, temperature=args.temperature)

# Aggregating model parameter & projection parameter #
model_params = model.parameters()

# LARS optimizer from KAKAO-BRAIN github "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)

# from torchlars import LARS
optimizer = base_optimizer #LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# Cosine learning rate annealing (SGDR) & Learning rate warmup git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

import os
os.makedirs('results', exist_ok=True) # make checkpoint folder

args.name += (args.train_type + '_' +args.model + '_' + args.dataset + '_b' + str(args.batch_size)+'_l'+str(args.lamda))
loginfo = 'results/log_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')
print(loginfo)

##### Log file #####
import csv
with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'reg loss'])

print(args.name)

def train(epoch):
    model.train()
    scheduler_warmup.step()

    total_loss = 0
    reg_simloss = 0
    reg_loss = 0

    for batch_idx, (ori, inputs_1, inputs_2, label) in enumerate(trainloader):
        if args.use_cuda:
            ori, inputs_1, inputs_2 = ori.cuda(), inputs_1.cuda() ,inputs_2.cuda() # to GPU
        if args.attack_to=='original':
            attack_target = inputs_1
        else:
            attack_target = inputs_2

        if 'Rep' in args.advtrain_type :
            advinputs = Rep.get_adversarial_example(imgs = inputs_1, target = attack_target)
            adv_loss = Rep.get_adversarial_loss(imgs = inputs_1, target = attack_target, optimizer = optimizer) * 1.0 / args.lamda
            reg_loss += adv_loss.data

        if not (args.advtrain_type == 'None'):
            inputs = torch.cat((inputs_1, inputs_2, advinputs))
            slices = 3
        else:
            inputs = torch.cat((inputs_1, inputs_2))
            slices = 2
        
        outputs = model(inputs)
        # print(inputs, outputs)
        # similarity, gathered_outputs = pairwise_similarity(outputs, temperature=args.temperature, multi_gpu=multi_gpu, adv_type = args.advtrain_type) 
                
        simloss  = NT_xent_loss(outputs, temperature=args.temperature, slices=slices) #, args.advtrain_type)
        
        if not (args.advtrain_type=='None'):
            loss = simloss + adv_loss
        else:
            loss = simloss
        
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        reg_simloss += simloss.data
        
        optimizer.step()

        if 'Rep' in args.advtrain_type:
            print(batch_idx, len(trainloader),
                            'Loss: %.3f | SimLoss: %.3f | Adv: %.4f'
                            % (total_loss / (batch_idx + 1), reg_simloss / (batch_idx + 1), reg_loss / (batch_idx + 1)))
        else:
            print(batch_idx, len(trainloader),
                        'Loss: %.3f | Adv: %.4f'
                        % (total_loss/(batch_idx+1), reg_simloss/(batch_idx+1)))
        
    return (total_loss/batch_idx, reg_simloss/batch_idx)

##### Training #####
for epoch in range(0, args.epoch):
    train_loss, reg_loss = train(epoch)
    
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }

    save_name = './checkpoint/ckpt.t7' + args.name + '_' + str(args.seed)

    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, save_name)
    
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), reg_loss.item()])