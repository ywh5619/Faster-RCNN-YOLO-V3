import sys
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

transform = transforms.Compose(
    [
        # transforms.Resize(cfg.INPUT.SIZE_TRAIN),
        # transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB), #
        # transforms.Pad(cfg.INPUT.PADDING),
        # transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        transforms.ToTensor()
    ]
)
train_set = tv.datasets.ImageFolder(root='C:/Users/单欣宇/Desktop/nndl-project-final-main/picture',
                                    transform=transform)
data_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)

to_pil_image = transforms.ToPILImage()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


for input, target in data_loader:
    # 方法1：Image.show()
    # transforms.ToPILImage()中有一句
    # npimage = np.transpose(pic.numpy(), (1, 2, 0))
    # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
    print(target)
    r = np.random.rand(1)  # 0-1 之间的小数 array([0.33473484])
    argbeta = 1.0
    prob = 1
    length = 80
    method = 'cutout'
    target_a = target
    target_b = target
    if method != 'baseline' and (argbeta > 0 and r < prob):
        if method == 'cutout':
            _, _, h, w = input.shape
            h = input.shape[2]
            w = input.shape[3]
            lam = 1 - (length ** 2 / (h * w))
            for _ in range(1):
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = 0.
        else:
            lam = np.random.beta(argbeta, argbeta)
            rand_index = torch.randperm(input.size()[0])
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            if method == 'cutmix':
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            else:  # method=mixup
                input = lam * input + (1 - lam) * input[rand_index, :, :]

    print(lam)
    print(target_a)
    print(target_b)
    print(input.shape)
    for i in range(len(target_b)):
        image = to_pil_image(input[i])
        image.show()
    break


