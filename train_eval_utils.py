import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


def train_one_epoch_data_augmentation(model, optimizer, data_loader, device, epoch,method='baseline',
                                      argbeta=1.0,prob=-1.0,n_holes=1,length=16,picshow=False):
    if method not in ['baseline','mixup','cutout','cutmix']:
        print('method error!')
    if prob<0 or prob>1:
        problist={'baseline':1.0,'mixup':1.0,'cutout':1.0,'cutmix':0.5}
        prob=problist[method]
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()
     #组合成新的图片，label
    for step, data in enumerate(data_loader):
        images, labels = data
        input=images.to(device)#device=torch.device("cpu"),使用的是CPU，将模型加载到cpu上
        target=labels.to(device)
        target_a = target
        target_b = target
        lam=1.0
        r=np.random.rand(1)
        if method!='baseline' and (argbeta>0 and r<prob):
            if method=='cutout':
                _, _, h, w = input.shape
                h = input.shape[2]
                w = input.shape[3]
                lam = 1 - (length ** 2 / (h * w))
                for _ in range(n_holes):
                    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                    input[:, :, bbx1:bbx2, bby1:bby2] = 0.
            else:
                lam = np.random.beta(argbeta, argbeta)
                rand_index = torch.randperm(input.size()[0]).to(device)
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                if method=='cutmix':
                    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                else:#method=mixup
                    input = lam * input + (1 - lam) * input[rand_index, :, :]
        output = model(input)
        loss = loss_function(output, target_a) * lam + loss_function(output, target_b) * (1. - lam)

        preds = torch.max(output, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()
        loss.backward()
        # mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        total_loss+=loss.detach()
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(total_loss.item()/(step+1), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    # mean_loss=mean_loss*step/num_samples
    mean_loss=total_loss/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

@torch.no_grad()#用有效集测试
def evaluate(model, data_loader, device):
    model.eval()
    score_list = []
    label_list = []
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
#    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        # pred=nn.Softmax(pred,dim=1)
        loss+=loss_function(pred,labels.to(device))
        m = nn.Softmax(dim=1)
        pred = m(pred)
        #print(perloss)
        #loss+=perloss
        pred,predlabel = torch.max(pred, dim=1)
        score_list.extend(pred.detach().cpu().numpy())
        label_right = torch.eq(predlabel, labels.to(device))
        # print(label_right)
        # print(preds)
        label_list.extend(label_right.detach().cpu().numpy())
        #l=torch.eq(pred, labels.to(device))
        #print(l)
        #print(images[l])
        #print(pred[l])
        #print(labels[l])

        sum_num += torch.eq(predlabel, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    # print(num_samples)
    loss = loss / num_samples
    score_array = np.array(score_list)
    label_array = np.array(label_list)

    return loss.item(),acc,score_array,label_array

@torch.no_grad()
def evaluate1(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    #data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss+=loss_function(pred,labels.to(device))
        #print(perloss)
        #loss+=perloss
        probs, pred = torch.max(torch.softmax(pred, dim=1), dim=1)
        #probs = torch.max(pred, dim=1)[0]
        #pred = torch.max(pred, dim=1)[1]

        l=torch.eq(pred, labels.to(device))
        l=[not i for i in l]
        falseimg=images[l]
        falselabel=labels[l ]
        falsepred=pred[l]
        class_indices={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        # import matplotlib.pyplot as plt
        num_imgs=10
        fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
        for i in range(num_imgs):
            # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
            ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

            # CHW -> HWC
            # npimg = images[i].cpu().numpy().transpose(1, 2, 0)

            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            # npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img=falseimg[i].cpu().numpy().transpose(1,2,0)
            img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255


            title = "{}\n{:.2f}%\n{}".format(
                class_indices[int(falsepred[i])],  # predict class
                probs[i] * 100,  # predict probability
                class_indices[int(falselabel[i])]  # true class
            )
            ax.set_title(title)
            plt.imshow(img.astype('uint8'))
            plt.axis('off')
        plt.show()
       #plt.axis('off')
        #plt.show(fig)

        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    loss = loss / num_samples

    return loss.item(),acc

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
#截取0到H之间的值
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



