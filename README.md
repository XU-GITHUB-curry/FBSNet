# FBSNet

This repository is an official PyTorch implementation of our paper"FBSNet: A Fast Bilateral Symmetrical Network for
Real-Time Semantic Segmentation".

# Datasets

We used Cityscapes dataset and CamVid sataset to train our model. Please download them from [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

You can download cityscapes dataset from [here](https://www.cityscapes-dataset.com/).
Note: please download leftImg8bit_trainvaltest.zip(11GB) and gtFine_trainvaltest(241MB).
The Cityscapes dataset scripts for inspection, preparation, and evaluation can download from [here](https://github.com/mcordts/cityscapesScripts).

You can download camvid dataset from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).


# Train
```
python train.py --dataset cityscapes/camvid --train_type train/trainval --max_epochs 1000 --lr 4.5e-2 --batchsize 8
```
