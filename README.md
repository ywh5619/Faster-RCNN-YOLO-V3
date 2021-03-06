# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.








# ??????Resnet8


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











# ??????Cutout, cutmix, mixup??????

?????????Resnet18???baseline?????????cutmix, cutout, mixup?????????

conf/gloabal_settings.py???????????????????????????


?????????models???????????????CNN??????

?????????originnet3???data?????????CIFAR-100?????????logs???tensorboard??????????????????weights????????????????????????????????????

?????????picture???????????????cutout???mixup???cutmix?????????.

?????????????????????????????????

??? cifrapytorch.py ?????? method ???????????? 'baseline', 'cutmix', 'cutout', 'mixup'??????
??????????????? python cifrapytorch.py ??-net resnet18 -gpu; ????????????????????????????????????????????????????????????
??? weights?????????.

????????????tensorboard ???????????????

?????????????????????originnet3???, ??????????????? tensorboard --logdir logs --port 6606 --host localhost


?????????????????????cutout???cutmix???mixup?????????????????????

??? picture.py ?????? method ???????????????'cutmix', 'cutout', 'mixup'???????????? Python picture.py ????????? 
?????????cutmix???,'cutout', 'mixup'???????????????.
