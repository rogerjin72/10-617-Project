#!/bin/bash
python linear_eval.py --ngpu 1 --batch-size=1024 \
      --train_type='linear_eval' --model=ResNet18 --epoch 150 --lr 0.1 --name training_1 \
      --load_checkpoint=None \
      --clean=True --dataset=cifar-10 --seed=0 --finetune=True

python -m torch.distributed.launch --nproc_per_node=1 linear_eval.py --ngpu 1 --batch-size=1024 \
       --train_type='linear_eval' --model=ResNet18 --epoch 150 --lr 0.1 --name training_1 --load_checkpoint=None --clean=True --dataset=$6 --seed=$7