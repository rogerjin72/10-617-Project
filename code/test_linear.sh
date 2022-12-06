#!/bin/bash
python linear_eval.py --ngpu 1 --batch-size=1024 \
      --train_type='linear_eval' --model=ResNet18 --epoch 150 --lr 0.01 --name training_1_1000_no_finetune \
      --load_checkpoint=checkpoint/ckpt.t7training_1contrastive_ResNet18_cifar-10_b256_l256_0 \
       --dataset=cifar-10 --seed=0 --adv_img=True
