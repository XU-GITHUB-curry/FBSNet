import os.path as osp #os.path 模块主要用于获取文件的属性
import numpy as np
import random
import cv2  #open cv
from torch.utils import data
import pickle


class CityscapesDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load train set #装载训练集
       Args:
        root: the Cityscapes dataset path, 
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_train_list.txt, include partial path
        mean: bgr_mean (73.15835921, 82.90891754, 72.39239876)   opencv中图片格式：BGR，BGR均值(73.15835921, 82.90891754, 72.39239876)

    """

    def __init__(self, root='', list_path='', max_iters=None,
                 crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size  #self.crop_h = 512，self.crop_w = 1024
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)] #strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
        if not max_iters == None:  #if not (max_iters == None)#训练时根据max_iter数将列表翻倍
            #要计算max_iter要训练多少轮trainset
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))#np.ceil即对于输入 x ，返回最小的整数 i ，使得 i> = x
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])#Python split() 通过指定分隔符对字符串进行切片
            # print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            # print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name        #aachen_000000_000019_leftImg8bit.png
            })

        print("length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)  #数据集长度

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #使用opencv2读取彩色图
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)#读取灰度标签图
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            f_scale = scale[random.randint(0, 5)]  #随机裁剪0.75-2.0
            # f_scale = 0.5 + random.randint(0, 15) / 10.0  # random resize between 0.5 and 2
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)#双线性插值法
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)#最近邻插值法
#cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None)
#scr:原图
#dsize：输出图像尺寸
#fx:沿水平轴的比例因子
#fy:沿垂直轴的比例因子
#interpolation：插值方法
        image = np.asarray(image, np.float32) #asarray可将结构数据转换为ndarray类型

        image -= self.mean    #image = image - self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB   #cv2读取时BGR #列表数组左右翻转
        img_h, img_w = label.shape
        '''pad the inputs if their size is smaller than the crop_size'''
        pad_h = max(self.crop_h - img_h, 0) #若裁剪尺寸>图像尺寸，就pad
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,  #填充边界  #src:源图像 top,bottem,left,right: 分别表示四个方向上边界的长度
                                         pad_w, cv2.BORDER_CONSTANT,   #borderType: 边界的类型 BORDER_CONSTANT　　　　# 常量，增加的变量通通为value色
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        # select a random start-point for croping operation
        h_off = random.randint(0, img_h - self.crop_h)  #随机选取一个点开始裁剪
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        # crop the image and the label 裁剪
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW  N 表示这批图像有几张

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class CityscapesValDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load val set
       Args:
        root: the Cityscapes dataset path, 
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_val_list.txt, include partial path

    """

    def __init__(self, root='',
                 list_path='',
                 f_scale=1, mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.f_scale = f_scale
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            # print(label_file)
            image_name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            # print("image_name:  ",image_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape
        name = datafiles["name"]
        if self.f_scale != 1: #如果不使用随机裁剪
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        # print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name


class CityscapesTestDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load test set
       Args:
        root: the Cityscapes dataset path,
        list_path: cityscapes_test_list.txt, include partial path

    """

    def __init__(self, root='',
                 list_path='', mean=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            image_name = name.strip().split()[0].strip().split('/', 3)[3].split('.')[0]
            # print(image_name)
            self.files.append({
                "img": img_file,
                "name": image_name  #test测试集没有label
            })
        print("lenth of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        size = image.shape

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        return image.copy(), np.array(size), name


class CityscapesTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance. #解决类别不平衡
    """

    def __init__(self, data_dir='', classes=19,
                 train_set_file="", inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)#[0.0,0.0,0.0]
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            # with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img) #去除数组中的重复数字
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, range=(0, 18))
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None
