import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch RoCL training')
    parser.add_argument('--module',action='store_true')

    ##### arguments for RoCL #####
    parser.add_argument('--lamda', default=256, type=float)

    parser.add_argument('--regularize_to', default='other', type=str, help='original/other')
    parser.add_argument('--attack_to', default='other', type=str, help='original/other')

    parser.add_argument('--loss_type', type=str, default='sim', help='loss type for Rep')
    parser.add_argument('--advtrain_type', default='Rep', type=str, help='Rep/None')

    ##### arguments for Training Self-Sup #####
    parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_multiplier', default=15.0, type=float, help='learning rate multiplier')
    parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')

    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100')

    parser.add_argument('--load_checkpoint', default='./checkpoint/ckpt.t7one_task_0', type=str, help='PATH TO CHECKPOINT')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default='ResNet18', type=str,
                        help='model type ResNet18/ResNet50')

    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=1000, type=int, help='total epochs to run')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

    ##### arguments for distributted parallel #####
    parser.add_argument('--local_rank', type=int, default=0)   
    parser.add_argument('--ngpu', type=int, default=1)

    ##### arguments for PGD attack & Adversarial Training #####
    parser.add_argument('--min', type=float, default=0.0, help='min for cliping image')
    parser.add_argument('--max', type=float, default=1.0, help='max for cliping image')
    parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
    parser.add_argument('--epsilon', type=float, default=0.0314,
        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--alpha', type=float, default=0.007, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', type=int, default=7, help='maximum iteration when generating adversarial examples')
    parser.add_argument('--random_start', type=bool, default=True,
        help='True for PGD')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--use_cuda', type=bool, default=False, help='use cuda or not')
    
    args = parser.parse_args()

    return args