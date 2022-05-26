# Accurate-and-Efficient-Soma-Reconstruction-in-a-Full-Adult-Fruit-Fly-Brain
## Introduction
This is the code of our article being reviewed.


## Enviromenent

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. .

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v3.1
```

## Implement the full brain data processing pipeline
### 1. Intra-block Segmentation
```shell
cd ./Full_Brain_Soma_Segmentation_Pipeline/full_data_process2.0_seg
python script/submit_task.py 
```
### 2. Inter-block Stitching
```shell
cd ./Full_Brain_Soma_Segmentation_Pipeline/full_data_process2.0_stitch
python script/submit_task.py 
```

## Train Localization Network
```shell
cd Localization
python train.py 
```

## Train Segmentation Network
```shell
cd Segmentation
python train.py 
```

## Inference on a 3D block
We provide our trained models, including the localization network and the segmentation network in the fold './trained_model'.
If you want to test the model on a small block, you can implement the following command
```shell
python inference.py
```
## Examples
We provide an example their corresponding segmentation result predicted by our trained model in the following Google Driver:
https://drive.google.com/drive/folders/1V1wtjCtubjfj7WV7hKSPNOKx9XNdRnh6?usp=sharing

The full annotated dataset will be updated soon!


