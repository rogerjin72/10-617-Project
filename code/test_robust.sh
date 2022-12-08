#!/bin/bash
python robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=ResNet18 \
                --load_checkpoint=checkpoint/ckpt.t7rocl_init_adv_img_with_rocl_rince_Evaluate_linear_eval_ResNet18_cifar-10_0 \
                --attack_type='linf' --epsilon=0.0314 --alpha=0.00314 --k=20 --dataset=cifar-10 --seed=0
