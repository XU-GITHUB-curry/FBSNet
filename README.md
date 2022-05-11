# FBSNet

This repository is an official PyTorch implementation of our paper"FBSNet: A Fast Bilateral Symmetrical Network for
Real-Time Semantic Segmentation".

# Datasets

We used Cityscapes dataset and CamVid sataset to train our model. Please download them from [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

|-- dataset  
|      |-- camvid  
|      |        |-- train  
|  |-- camvid  
|  |  |-- train  
|  |  |-- trainannot  
|  |  |-- val  
|  |  |-- valannot  
|  |  |-- test  
|  |  |-- testannot  
|  |  |-- ...  
|  |-- cityscapes  
|  |  |-- leftImg8bit  
|  |  |  |-- train  
|  |  |  |-- val  
|  |  |  |-- test  
|  |  |-- gtFine  
|  |  |  |-- train  
|  |  |  |-- val  
|  |  |  |-- test  
|  |  |-- ...  

# Train
```
python train.py --dataset cityscapes/camvid --train_type train/trainval --max_epochs 1000 --lr 4.5e-2 --batchsize 8
```
