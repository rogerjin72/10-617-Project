#!/bin/bash
python rocl_train.py --batch-size=256 --model='ResNet18' --k=7 --loss_type='sim' --advtrain_type='Rep' --attack_type='linf' \
                     --name=trial --regularize_to='other' --attack_to='other' --train_type='contrastive' --dataset='cifar-10'