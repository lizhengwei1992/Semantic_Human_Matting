# Semantic_Human_Matting
The project is my reimplement of paper ([Semantatic Human Matting](https://arxiv.org/abs/1809.01354)) from Alibaba,  it proposes a new end-to-end scheme to predict human alpha from image. SHM is the first algorithm that learns to jointly fit both semantic information and high quality details with deep networks. 

One of the main contributions of the paper is that: ***A large scale high quality human matting dataset is created. It contains 35,513 unique human images with corresponding alpha mattes***. But, the dataset is not avaiable. 


I collected 6k+ images as my dataset of the project. Worth noting that, the architecture of my network, which builded with mobilenet and shallow encoder-decoder net, is a light version compaired to original implement. 

# update 2019/04/08 
:thumbsup: :thumbsup: The company [***爱分割***](http://www.aisegment.com) shared their [dataset](https://github.com/aisegmentcn/matting_human_datasets) recently !


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

## Step 1: prepare dataset

```./data/train.txt``` contain image names according to 6k+ images(```./data/image```) and corresponding masks(```./data/mask```). 

Use ```./data/gen_trimap.sh``` to get trimaps of the masks.

Use ```./data/knn_matting.sh``` to get alpha mattes(it will take long time...).

## Step 2: build network

![SHM](https://github.com/lizhengwei1992/Semantic_Human_Matting/raw/master/network.png)


- ***Trimap generation: T-Net***


  The T-Net plays the role of semantic segmentation. I use mobilenet_v2+unet as T-Net to predict trimap.

- ***Matting network: M-Net***


  The M-Net aims to capture detail information and generate alpha matte. I build M-Net same as the paper, but reduce channels of the original net.
  
- ***Fusion Module***

  Probabilistic estimation of alpha matte can be written as <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;_{p}&space;=&space;F_{s}&space;&plus;&space;U_{s}\alpha&space;_{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;_{p}&space;=&space;F_{s}&space;&plus;&space;U_{s}\alpha&space;_{r}" title="\alpha _{p} = F_{s} + U_{s}\alpha _{r}" /></a>


## Step 3: build loss 

The overall prediction loss for alpha_p at each pixel is <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L_{p}&space;=&space;\gamma\left&space;\|&space;\alpha&space;_{p}&space;-&space;\alpha&space;_{g}&space;\right&space;\|_{1}&space;&plus;&space;\left&space;(&space;1-\gamma&space;\right&space;)\left&space;\|&space;c_{p}&space;-&space;c_{g}&space;\right&space;\|_{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L_{p}&space;=&space;\gamma\left&space;\|&space;\alpha&space;_{p}&space;-&space;\alpha&space;_{g}&space;\right&space;\|_{1}&space;&plus;&space;\left&space;(&space;1-\gamma&space;\right&space;)\left&space;\|&space;c_{p}&space;-&space;c_{g}&space;\right&space;\|_{1}" title="L_{p} = \gamma\left \| \alpha _{p} - \alpha _{g} \right \|_{1} + \left ( 1-\gamma \right )\left \| c_{p} - c_{g} \right \|_{1}" /></a>

The total loss is <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L&space;=&space;L_{p}&space;&plus;&space;\lambda&space;L_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L&space;=&space;L_{p}&space;&plus;&space;\lambda&space;L_{t}" title="L = L_{p} + \lambda L_{t}" /></a>

Read papers for more details, and my codes for two loss functions:
```
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img
    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()
    L_p = 0.5*L_alpha + 0.5*L_composition
```




## Step 4: train

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
# Test
  
  run ```./test_camera.sh```




