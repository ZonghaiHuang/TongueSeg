import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


# 设标签宽W，长H
def fast_hist(a, b, n):
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int)
                       + b[k], minlength=n ** 2).reshape(n,
                                                         n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)



'''
  compute_mIoU函数原始以CityScapes图像分割验证集为例来计算mIoU值的（可以根据自己数据集的不同更改类别数num_classes及类别名称name_classes），本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。主要留意mIoU指标的计算核心代码即可。
'''


def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    with open(r'..\info.json', 'r') as fp:

        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)

    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'pic.txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'pic.txt')  # ground truth和自己的分割结果txt一样

    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]
    # pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#获得验证集图像分割结果路径列表，方便直接读取

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        print(ind)
        print(pred_imgs[ind])
        print(gt_imgs[ind])
        pred = (np.array(Image.open(pred_imgs[ind]).convert("L"))/255).astype(int)  # 读取一张图像分割结果，转化成numpy数组
        label = (np.array(Image.open(gt_imgs[ind]).convert("L")) /255).astype(int) # 读取一张对应的标签，转化成numpy数组


        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
        if  ind % 1 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * per_class_iu(hist)[0]))
            print(per_class_iu(hist))
    acc = np.diag(hist).sum() / hist.sum()
    classAcc = np.diag(hist) / hist.sum(axis=1)
    meanAcc = np.nanmean(classAcc)
    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print("pa:",acc)
    print("cpa:", classAcc)
    print("mpa:", meanAcc)
    return mIoUs


compute_mIoU(r'..\results',
             r'..\datas\datas\open environment\test label',
             r'..\results'
             )
