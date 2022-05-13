# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements

#experiment enviroument
- python3.9
- pytorch1.7.1+cu101
- tensorboard 2.2.2(optional)


## Usage

### 1. enter directory
```bash
cd pytorch-cifar100
```

### 3. run tensorbard(optional)
Install tensorboard
pip install tensorboard
mkdir runs
Run tensorboard
tensorboard --logdir runs --port 6006 --host localhost


### 4. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
python train.py -net resnet18 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.


### 5. test the model
Test the model using test.py
```bash
python test.py -net resnet18 -weights
checkpoint\resnet18\Friday_April_2022_09h_oom_53s\resnet18-20-regular.ph
```

#6.learning rate
python lr_finder.py
