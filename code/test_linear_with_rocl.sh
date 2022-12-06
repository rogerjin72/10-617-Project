#!/bin/bash
python linear_eval_with_rocl.py --ngpu 1 --batch-size=1024 \
      --train_type='linear_eval' --model=ResNet18 --epoch 50 --lr 0.01 --name rocl_init_adv_img_with_rocl \
      --load_checkpoint=checkpoint/ckpt.t7training_1contrastive_ResNet18_cifar-10_b256_l256_0 \
      --dataset=cifar-10 --seed=0 --adv_img=True --clean=True