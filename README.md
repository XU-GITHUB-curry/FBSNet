 # FBSNet

This repository is an official PyTorch implementation of our paper"FBSNet: A Fast Bilateral Symmetrical Network for Real-Time Semantic Segmentation".  Accepted by IEEE TRANSACTIONS ON MULTIMEDIA, 2022. (IF: 6.513)

[Paper](https://arxiv.org/abs/2109.00699v1) | [Code](https://github.com/XU-GITHUB-curry/FBSNet)



## Installation

```
cuda == 10.2
Python == 3.6.4
Pytorch == 1.8.0+cu101

# clone this repository
git clone https://github.com/XU-GITHUB-curry/FBSNet.git
```



## Datasets

We used Cityscapes dataset and CamVid dataset to train our model.  

- You can download cityscapes dataset from [here](https://www.cityscapes-dataset.com/). 

Note: please download leftImg8bit_trainvaltest.zip(11GB) and gtFine_trainvaltest(241MB). 

The Cityscapes dataset scripts for inspection, preparation, and evaluation can download from [here](https://github.com/mcordts/cityscapesScripts).

- You can download camvid dataset from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

The folds of your datasets need satisfy the following structures:

```
├── dataset  					# contains all datasets for the project
|  └── cityscapes 				#  cityscapes dataset
|  |  └── gtCoarse  		
|  |  └── gtFine 			
|  |  └── leftImg8bit 		
|  |  └── cityscapes_test_list.txt
|  |  └── cityscapes_train_list.txt
|  |  └── cityscapes_trainval_list.txt
|  |  └── cityscapes_val_list.txt
|  |  └── cityscapesscripts 	#  cityscapes dataset label convert scripts！
|  └── camvid 					#  camvid dataset 
|  |  └── test
|  |  └── testannot
|  |  └── train
|  |  └── trainannot
|  |  └── val
|  |  └── valannot
|  |  └── camvid_test_list.txt
|  |  └── camvid_train_list.txt
|  |  └── camvid_trainval_list.txt
|  |  └── camvid_val_list.txt
|  └── inform 	
|  |  └── camvid_inform.pkl
|  |  └── cityscapes_inform.pkl
|  └── camvid.py
|  └── cityscapes.py 

```



## Train

```
# cityscapes
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 4.5e-2 --batch_size 4

# camvid
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 1e-3 --batch_size 6
```



## Test

```
# cityscapes
python test.py --dataset cityscapes --checkpoint ./checkpoint/cityscapes/FBSNetbs4gpu1_train/model_1000.pth

# camvid
python test.py --dataset camvid --checkpoint ./checkpoint/camvid/FBSNetbs6gpu1_trainval/model_1000.pth
```

## Predict
only for cityscapes dataset
```
python predict.py --dataset cityscapes 
```

## Results

- Please refer to our article for more details.

| Methods |  Dataset   | Input Size | mIoU(%) |
| :-----: | :--------: | :--------: | :-----: |
| FBSNet  | Cityscapes |  512x1024  |  70.9   |
| FBSNet  |   CamVid   |  360x480   |  68.9   |



## Citation

If you find this project useful for your research, please cite our paper:

```
@article{gao2022fbsnet,
  title={FBSNet: A fast bilateral symmetrical network for real-time semantic segmentation},
  author={Gao, Guangwei and Xu, Guoan and Li, Juncheng and Yu, Yi and Lu, Huimin and Yang, Jian},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```



## Acknowledgements

1. [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)
2. [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
