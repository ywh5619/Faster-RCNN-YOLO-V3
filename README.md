一：以Resnet18为baseline，对比cutmix, cutout, mixup的性能

conf/gloabal_settings.py中设置了一些参数。


文件夹models保存了一些CNN模型

文件夹originnet3中data保存了CIFAR-100数据，logs为tensorboard可视化文件，weights保存了各种方法的运行数据

文件夹picture保存了经过cutout，mixup，cutmix的图片.

报告为该实验的实验报告

将 cifrapytorch.py 中的 method 分别设为 'baseline', 'cutmix', 'cutout', 'mixup'时，
在终端运行 python cifrapytorch.py  -net resnet18 -gpu; 可得到这四种方法的训练，测试数据，并保存
在 weights文件里.

二：使用tensorboard 可视化结果

将根目录设置在originnet3下, 在终端运行 tensorboard --logdir logs --port 6606 --host localhost


三：可视化经过cutout，cutmix，mixup数据加强的图片

将 picture.py 中得 method 分别设置为'cutmix', 'cutout', 'mixup'时，运行 Python picture.py 可得到 
经过’cutmix‘,'cutout', 'mixup'处理的图片.

