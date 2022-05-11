import os
import pickle #该pickle模块实现了用于序列化和反序列化Python对象结构的二进制协议
#python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；
#通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象
from torch.utils import data
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(dataset, '_trainval_list.txt')
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl') #两个pkl文件

    # inform_data_file collect the information of mean, std and weight_class，均值/标准差/类别权重
    if not os.path.isfile(inform_data_file): #如果找不到pkl文件，报错，相当于没有找到数据集
        # os.path.isfile用于判断某一对象(需提供绝对路径)是否为文件
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes": #训练信息
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))#从file中读取一个字符串，并将它重构为原来的python对象
#file参数表示的需要打开文件的相对路径(当前工作目录)或者一个绝对路径 rb：以二进制方式读写操作
    if dataset == "cityscapes":
        #训练数据装载
        trainLoader = data.DataLoader(
            CityscapesDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']), #mean=[128,128,128] 灰度
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)
        #验证数据加载
        valLoader = data.DataLoader(
            CityscapesValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),#f_scale=1 表示0.75-2 随机裁剪

            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader

    elif dataset == "camvid":

        trainLoader = data.DataLoader(
            CamVidDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CamVidValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader


def build_dataset_test(dataset, num_workers, none_gt=False):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(dataset, '_trainval_list.txt')
    test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weight_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True #在验证集测试有groundtruth，在测试集测试没有groundtruth
        if none_gt: #test
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),#root='', list_path=''
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else: #val
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    elif dataset == "camvid":

        testLoader = data.DataLoader(
            CamVidValDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader
