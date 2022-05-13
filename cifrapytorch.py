import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用指定GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from train_eval_utils import *
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print(device)

# 训练
def main():
    """程序运行时，首先会创建model_path路径文件，该文件下有：
    一个与method有关的txt文件记录训练过程中的训练集、测试集的误差和准确率及学习率
    一个weigthts文件，里面再创建一个以方法命名的文件，存入训练过程的每轮权重
    一个logs文件，里面又创建了与方法有关的训练和测试集分别相关文件，分别存入训练过程中的tensorboard数据"""
    method = 'cutmix'  # baseline,mixup,cutout,cutmix
    model_path = "./originnet3"  # 模型训练路径
    load_weights = './weights/baseline/model-99.pth'  # 模型加载权重的路径
    # load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    ispcishow = True  # 是否显示增强过的数据图片
    save_weights = './weights/' + method
    if os.path.exists(save_weights) is False:
        os.makedirs(save_weights)
    # 实例化SummaryWriter对象，分成训练集与测试集
    train_tb_writer = SummaryWriter(log_dir="./logs/" + method + "_train")
    test_tb_writer = SummaryWriter(log_dir="./logs/" + method + "_test")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    # data preprocessing:
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    val_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  # 打乱顺序
    )

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    test_tb_writer.add_graph(net, init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)  # 指定步长多步下降

    print("Start Training, Resnet-18!")

    for epoch in range(1, settings.EPOCH + 1):

        """'train_one_epoch_data_augmentation'函数实现数据增强的三种方法，用'method'进行指定，默认为'baseline'
        此外还有mixup,cutout,cutmix三种方式。
        在选定为cutout时，有参数n_holes=1,length=16进行调整；
        在选定cutmix时，有参数argbeta=1.0进行调整
        为与原论文方法相同，三种方法中的增强概率分别设为了1.0,1.0,0.5，若有需要可以在原函数中进行修改"""
        mean_loss, train_acc = train_one_epoch_data_augmentation(model=net,
                                                                 optimizer=optimizer,
                                                                 data_loader=train_loader,
                                                                 device=device,
                                                                 epoch=epoch,
                                                                 method=method,
                                                                 picshow=ispcishow)

        scheduler.step()

        # 进行验证
        test_loss, test_acc, score_array, label_array = evaluate(model=net,
                                                                 data_loader=val_loader,
                                                                 device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["误差", "准确率", "误差", "准确率"]
        train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_pr_curve('pr曲线', label_array, score_array, epoch)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
        # 将训练指标存入文档
        with open('trainprocess' + method + '.txt', 'a') as f:
            f.write('%03d epoch |train loss: %.08f|test loss: %.08f|'
                    'train accuracy:%.8f |test accuracy:%.8f |learning rate :%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'train_loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), save_weights + "/{}_model-{}.pth".format(method, epoch))


if __name__ == '__main__':
    main()
