# A Soma Segmentation Benchmark in Full Adult Fly Brain
**Accepted by CVPR-2023**

**Xiaoyu Liu**, Bo Hu, Mingxing Li, Wei Huang, Zhiwei Xiong*, and Yueyi Zhang 

University of Science and Technology of China (USTC), Hefei, China

Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, China

*Corresponding Author

## Abstract
Neuron reconstruction in a full adult fly brain from high-resolution electron microscopy (EM) data is regarded as a cornerstone for neuroscientists to explore how neurons inspire intelligence. As the central part of neurons, somas in the full brain indicate the origin of neurogenesis and neural functions. However, due to the absence of EM datasets specifically annotated for somas, existing deep learning-based neuron reconstruction methods cannot directly provide accurate soma distribution and morphology. Moreover, full brain neuron reconstruction remains extremely time-consuming due to the unprecedentedly large size of EM data.
In this paper, we develop an efficient soma reconstruction method for obtaining accurate soma distribution and morphology information in a full adult fly brain. 
To this end, we first make a high-resolution EM dataset with fine-grained 3D manual annotations on somas. Relying on this dataset, we propose an efficient, two-stage deep learning algorithm for predicting accurate locations and boundaries of 3D soma instances. Further, we deploy a parallelized, high-throughput data processing pipeline for executing the above algorithm on the full brain. Finally, we provide quantitative and qualitative benchmark comparisons on the testset to validate the superiority of the proposed method, as well as preliminary statistics of the reconstructed somas in the full adult fly brain from the biological perspective.


## Enviromenent

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows：

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v3.1
```

## Implement the full brain data processing pipeline
### 1. Full Brain Data Division
```shell
cd ./Full_Brain_Soma_Segmentation_Pipeline/full_data_process2.0_seg
python script/submit_task.py -tn=divide_block
```
### 2. Intra-block Segmentation
```shell
python script/submit_task.py 
```
### 3. Inter-block Stitching
```shell
cd ./Full_Brain_Soma_Segmentation_Pipeline/full_data_process2.0_stitch
python script/submit_task.py -tn=sort_ids
python scripts/submit_task.py -tn=stitching -sd=x0
python scripts/submit_task.py -tn=stitching -sd=x1
python scripts/submit_task.py -tn=stitching -sd=y0
python scripts/submit_task.py -tn=stitching -sd=y1
python scripts/submit_task.py -tn=stitching -sd=z0
python scripts/submit_task.py -tn=stitching -sd=z1
python scripts/submit_task.py -tn=concat
python scripts/submit_task.py -tn=global
python scripts/submit_task.py -tn=remap
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
We have released our trained models, including the localization network and the segmentation network in the fold './trained_model':


If you want to test the model on a block, you can implement the following command:
```shell
python inference.py
```
## Examples
We provide an example of their corresponding segmentation result predicted by our trained model in [[Examples](https://drive.google.com/drive/folders/13DkerjQuPYudh-G_doKpOKCiq1C6cRiC?usp=sharing)]



### patch-level example
![image](https://user-images.githubusercontent.com/54794058/170446111-ea728ea2-269b-43bf-bf2c-bfdca0feafb2.png)
![image](https://user-images.githubusercontent.com/54794058/170446133-2ec054d0-a49d-45c9-97ba-8eb56a2ccd1e.png)

### block-level example
![image](https://user-images.githubusercontent.com/54794058/170445354-c628f1b2-9456-4a3d-90f0-6edd05c85566.png)
![image](https://user-images.githubusercontent.com/54794058/170445365-1d9e08a3-5d0c-40e0-92f6-25488dccaf37.png)


## Visualization of our soma reconstruction 
![image](https://user-images.githubusercontent.com/54794058/224546913-34a85a35-9fa0-42f5-a2bb-29c53055fa6c.png)


## Soma statistics of the full brain
![image](https://user-images.githubusercontent.com/54794058/224547231-9589eb5e-8eb3-4f42-a2d3-251e2172ea10.png)





## Dataset
The annotated dataset EMADS is released and can be downloaded at 
[[Dataset](https://drive.google.com/drive/folders/1WLVaU3sGd8RQfwsBIBomZyNwl4m2D8pc?usp=share_link)]
### Overview of data annotation
![image](https://user-images.githubusercontent.com/54794058/224547553-d158ab49-b610-45ea-84cf-47f743ea73f3.png)


## Contact

If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).
