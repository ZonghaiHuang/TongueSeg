import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int)+ b[k], minlength=n ** 2).reshape(n,n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数

    with open('../info.json', 'r') as fp:

        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)

    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'pic.txt')
    label_path_list = join(devkit_dir, 'pic.txt')

    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        print(ind)
        print(pred_imgs[ind])
        print(gt_imgs[ind])
        pred = (np.array(Image.open(pred_imgs[ind]).convert("L"))/255).astype(int)
        label = (np.array(Image.open(gt_imgs[ind]).convert("L")) /255).astype(int)


        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if  ind % 1 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * per_class_iu(hist)[0]))
            print(per_class_iu(hist))
    acc = np.diag(hist).sum() / hist.sum()
    classAcc = np.diag(hist) / hist.sum(axis=1)
    meanAcc = np.nanmean(classAcc)
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print("pa:",acc)
    print("cpa:", classAcc)
    print("mpa:", meanAcc)
    return mIoUs


compute_mIoU('../results',
             '../datas/datas/open environment/test label',
             '../results'
             )
