# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="../datas/datas/open environment/image",
                 label_path="../datas/datas/open environment/label",
                 test_path="../datas/datas/open environment/test",
                 test_label_path="../datas/datas/open environment/test label",npy_path="./datas", img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.test_label_path = test_label_path

    def create_train_data(self):
        i = 0
        print('Creating training images...')
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        labels = glob.glob(self.label_path+"/*."+self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        print(len(imgs))
        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]
            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        labels = glob.glob(self.test_label_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        testpathlist = []

        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]
            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_test.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1  # 白
        imgs_mask_train[imgs_mask_train <= 0.5] = 0  # 黑
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        imgs_train = np.load(self.npy_path + "/imgs_test.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_test.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1  # 白
        imgs_mask_train[imgs_mask_train <= 0.5] = 0  # 黑
        return imgs_train, imgs_mask_train



if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
