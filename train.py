# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import argparse  # 参数
import os  # 函数操作
import time  # 统计时间

import torch
import torch.nn as nn
import torch.optim as optim  # 优化器
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(epoch):
    start = time.time()  # 开始计算时间
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
#batch_index为（images，labels）索引
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()  # 梯度清零
        outputs = net(images)  # 计算预测值
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 损失向后传播
        optimizer.step()  # 更新网络参数

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        # 迭代步数

        last_layer = list(net.children())[-1]  # 遍历子层，module返回子层的子层
        for name, para in last_layer.named_parameters():
            if 'weight' in name:  # tensorboard添加标量
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],  # 长度为6的字典
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))  # \t空四个字符

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)  # layer根目录
        attr = attr[1:]  # .txt拓展名
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    # 添加直方图
    finish = time.time()  # 结束训练

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


# 不需要计算梯度
@torch.no_grad()
# 测试阶段
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()  # 测试阶段，主要影响batchnorm，dropout

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)  # 所在行最大值指标为preds
        correct += preds.eq(labels).sum()  # 正确的数目

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True    #打乱顺序
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    #动态学习率，milestones数组中元素的为epoch，在第epoch时，学习率*0.2
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:  # 拼接路径
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))  # 生成文件夹
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder（关卡文件夹） to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')  # 就是用于将预训练的参数权重加载到新的模型之中
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)  # 测试最优参数网络
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        # 将最近网络下载
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
#小epoch表示迭代参数，EPOCH=200
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:#预热结束，使用自动方式调节学习率
            train_scheduler.step(epoch)

        if args.resume:#设置断点，然后可重新训练
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)#评价网络

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:#每10个epoch保存一个权重文件
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
