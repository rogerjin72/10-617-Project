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

CHECKPOINT = 'rocl_checkpoint/checkpoint200'
checkpoint = torch.load(CHECKPOINT)
epoch = checkpoint['epoch']
model_state = checkpoint['model']
optimizer_state = checkpoint['optimizer_state']

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
