# Semantic_Human_Matting
The project is my reimplement of paper ([Semantatic Human Matting](https://arxiv.org/abs/1809.01354)) from Alibaba,  it proposes a new end-to-end scheme to predict human alpha from image. SHM is the first algorithm that learns to jointly fit both semantic information and high quality details with deep networks. 

One of the main contributions of the paper is that: ***A large scale high quality human matting dataset is created. It contains 35,513 unique human images with corresponding alpha mattes***. But, the dataset is not avaiable. 

I collected 6k+ images as my dataset of the project. The architecture of my network, which builded with mobilenet and shallow encoder-decoder net, is a light version compaired to original implement. 

# Requirements
- python3.5 / 3.6
- pytorch >= 0.4
- opencv-python

# Usage

Directory structure of the project:
```
Semantic_Human_Matting
│   README.md
│   train.py
│   train.sh
|   test_camera.py
|   test_camera.sh
└───model
│   │   M_Net.py
│   │   T_Net.py
│   │   network.py
└───data
    │   dataset.py
    │   gen_trimap.py
    |   gen_trimap.sh
    |   knn_matting.py
    |   knn_matting.sh
    └───image
    └───mask
    └───trimap
    └───alpha
```
follow these steps:
## Step 1: prepare dataset

```./data/train.txt``` contain image names according to 6k+ images(```./data/image```) and corresponding masks(```./data/mask```). 

Use ```./data/gen_trimap.sh``` to get trimaps of the masks.

Use ```./data/knn_matting.sh``` to get alpha mattes.

## Step 2: build network

![SHM](https://github.com/lizhengwei1992/Semantic_Human_Matting/raw/master/network.png)


- ***Trimap generation: T-Net***


  The T-Net plays the role of semantic segmentation. I use mobilenet_v2+unet as T-Net to predict trimap.

- ***Matting network: M-Net***


  The M-Net aims to capture detail information and generate alpha matte. I build M-Net same as the paper, but reduce channels of the original net.

## Step 3: train

Firstly, pre_train T-Net, use ```./train.sh``` as :

```
python3 train.py \
	--dataDir='./data' \
	--saveDir='./ckpt' \
	--trainData='human_matting_data' \
	--trainList='./data/train.txt' \
	--load='human_matting' \
	--nThreads=4 \
	--patch_size=320 \
	--train_batch=8 \
	--lr=1e-3 \
	--lrdecayType='keep' \
	--nEpochs=1000 \
	--save_epoch=1 \
	--train_phase='pre_train_t_net'

```
Then, train end to end, use ```./train.sh``` as:
```
python3 train.py \
	--dataDir='./data' \
	--saveDir='./ckpt' \
	--trainData='human_matting_data' \
	--trainList='./data/train.txt' \
	--load='human_matting' \
	--nThreads=4 \
	--patch_size=320 \
	--train_batch=8 \
	--lr=1e-4 \
	--lrdecayType='keep' \
	--nEpochs=2000 \
	--save_epoch=1 \
	--finetuning \
	--train_phase='end_to_end'

```
# Result 
 TODO



