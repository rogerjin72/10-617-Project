#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 rocl_train.py --ngpu 1 \
                                   --batch-size=256 --model='ResNet18' --k=7 --loss_type='sim' --advtrain_type='Rep' \
                                   --attack_type='linf' --name='authors' --regularize_to='other' --attack_to='other' \
                                   --train_type='contrastive' --dataset='cifar-10'