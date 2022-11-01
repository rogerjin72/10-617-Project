import torch
from data.cifar import CIFAR10
from torchvision import transforms

# reference: https://github.com/Kim-Minseon/RoCL/blob/master/src/data_loader.py

def get_dataset(args): #CIFAR-10
    ### color augmentation ###    
    color_jitter = transforms.ColorJitter(0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, \
                    0.2*args.color_jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p = 0.8)
    rnd_gray = transforms.RandomGrayscale(p = 0.2)
    
    transform_temp = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])
    
    if args.train_type =='contrastive':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transform_temp,
            ])
            transform_test = transform_train
    
    elif args.train_type=='linear_eval':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transform_temp,
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

    elif args.train_type == 'test':
        transform_train = transform_temp
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    train_dst   = CIFAR10(train=True, transform=transform_train, contrastive_learning=args.train_type)
    val_dst     = CIFAR10(train=False, transform=transform_test, contrastive_learning=args.train_type)
    
    if args.train_type=='contrastive':
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=False,
                shuffle=True,
            )

        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100,
                num_workers=args.num_workers,
                pin_memory=False,
                shuffle=False,
            )
        
        return train_loader, train_dst, val_loader, val_dst
    
    else:
        train_loader = torch.utils.data.DataLoader(train_dst,
                                                batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)
        val_batch = 100
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch,
                                                shuffle=False, num_workers=args.num_workers)

        return train_loader, train_dst, val_loader, val_dst
    
    