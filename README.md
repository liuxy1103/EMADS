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

<!-- ## Train Localization Network
```shell
cd Localization
python train.py 
```

## Train Segmentation Network
```shell
cd Segmentation
python train.py 
``` -->

## Inference on a 3D block
We provide our trained models, including the localization network and the segmentation network in the fold './trained_model'.
If you want to test the model on a small block, you can implement the following command
```shell
python inference.py
```
## Examples
We provide an example of their corresponding segmentation result predicted by our trained model in the following Google Driver:
https://drive.google.com/drive/folders/13DkerjQuPYudh-G_doKpOKCiq1C6cRiC?usp=sharing

### patch-level example
![image](https://user-images.githubusercontent.com/54794058/170446111-ea728ea2-269b-43bf-bf2c-bfdca0feafb2.png)
![image](https://user-images.githubusercontent.com/54794058/170446133-2ec054d0-a49d-45c9-97ba-8eb56a2ccd1e.png)

### block-level example
![image](https://user-images.githubusercontent.com/54794058/170445354-c628f1b2-9456-4a3d-90f0-6edd05c85566.png)
![image](https://user-images.githubusercontent.com/54794058/170445365-1d9e08a3-5d0c-40e0-92f6-25488dccaf37.png)


## Dataset
The annotated dataset EMADS will be released soon !


