import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *
from data import *
import numpy as np
import os
from keras.preprocessing.image import array_to_img
import pathlib
import cv2
import time
from PIL import Image
import tensorflow.compat.v1 as tf1
# Optimizer / Loss
adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=.9, beta_2=.999, epsilon=1e-08)
bce = keras.losses.BinaryCrossentropy()
import keras.backend as K

def bce_loss(y_true, y_pred):

    y_pred = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_pred)



    return loss0

class myNet(keras.models.Model):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1_1 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv1_2 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2_1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2_2 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop4 = Dropout(0.5)
        self.pool4 = MaxPooling2D(pool_size=(2, 2))
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop5 = Dropout(0.5)
        self.conv6_1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up6 = UpSampling2D(size=(2, 2))
        self.merge6 = Concatenate(axis=3)
        self.conv6_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = UpSampling2D(size=(2, 2))
        self.merge7 = Concatenate(axis=3)
        self.conv7_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = UpSampling2D(size=(2, 2))
        self.merge8 = Concatenate(axis=3)
        self.conv8_2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_3 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = UpSampling2D(size=(2, 2))
        self.merge9 = Concatenate(axis=3)
        self.conv9_2 =  Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(1, 1, activation='sigmoid')
        self.upsample_out_2 = UpSampling2D(size=(1, 1), interpolation='bilinear')
        self.upsample_out_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_out_4 = UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsample_out_5 = UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.upsample_out_6 = UpSampling2D(size=(16, 16), interpolation='bilinear')
        self.side1 = Conv2D(1, (3, 3), padding='same')
        self.side2 = Conv2D(1, (3, 3), padding='same')
        self.side3 = Conv2D(1, (3, 3), padding='same')
        self.side4 = Conv2D(1, (3, 3), padding='same')
        self.side5 = Conv2D(1, (3, 3), padding='same')
        self.side6 = Conv2D(1, (3, 3), padding='same')
        self.upsample_2 = UpSampling2D(size=(1, 1), interpolation='bilinear')
        self.upsample_3 = UpSampling2D(size=(1, 1), interpolation='bilinear')
        self.upsample_4 = UpSampling2D(size=(1, 1), interpolation='bilinear')
        self.upsample_5 = UpSampling2D(size=(1, 1), interpolation='bilinear')
        self.upsample_6 = UpSampling2D(size=(1, 1), interpolation='bilinear')

    def call(self, inputs):
        hx = inputs
        conv1_1 = self.conv1_1(hx)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        drop4 = self.drop4(conv4_2)
        pool4 = self.pool4(drop4)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        drop5 = self.drop5(conv5_2)
        upsample_5 = self.upsample_6(drop5)
        side6 = self.upsample_out_6(self.side6(upsample_5))
        conv6_1 = self.conv6_1(self.up6(drop5))
        merge6 = self.merge6([drop4, conv6_1])
        upsample_6 = self.upsample_5(merge6)
        side5 = self.upsample_out_5(self.side5(upsample_6))
        conv6_2 = self.conv6_2(merge6)
        conv6_3 = self.conv6_3(conv6_2)
        conv7_1 = self.conv7_1(self.up7(conv6_3))
        merge7 = self.merge7([conv3_2, conv7_1])
        upsample_7 = self.upsample_4(merge7)
        side4 = self.upsample_out_4(self.side4(upsample_7))
        conv7_2 = self.conv7_2(merge7)
        conv7_3 = self.conv7_3(conv7_2)
        conv8_1 = self.conv8_1(self.up8(conv7_3))
        merge8 = self.merge8([conv2_2, conv8_1])
        upsample_8 = self.upsample_3(merge8)
        side3 = self.upsample_out_3(self.side3(upsample_8))
        conv8_2 = self.conv8_2(merge8)
        conv8_3 = self.conv8_3(conv8_2)
        conv9_1 = self.conv9_1(self.up9(conv8_3))
        merge9 = self.merge9([conv1_2, conv9_1])
        upsample_9 = self.upsample_2(merge9)
        side2 = self.upsample_out_2(self.side2(upsample_9))
        conv9_2 = self.conv9_2(merge9)
        conv9_3 = self.conv9_3(conv9_2)
        conv9_4 = self.conv9_4(conv9_3)
        side1 = self.side1(conv9_4)

        sig = keras.activations.sigmoid

        return  sig(side1)


import time
def load_data():
    mydata = dataProcess(512,512)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()

    return imgs_train, imgs_mask_train, imgs_test
def train():
    inputs = keras.Input(shape=[512, 512, 3])
    net = myNet()
    out = net(inputs)
    print("out", out)
    model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)
    model.summary()
    print("loading data")
    imgs_train, imgs_mask_train, imgs_test = load_data()
    print(imgs_train.shape)
    print(imgs_test.shape)
    print(imgs_mask_train.shape)
    # print(imgs_train)
    print("loading data done")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.ckpt', save_weights_only=True, verbose=1, save_best_only=True)
    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=12, epochs=1, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
    # model.save('myu2net.h5')
    print('predict test data')
    i = 0
    piclist = []
    for line in open("./results3/pic.txt"):
        line = line.strip()
        picname = line.split('/')[-1]
        piclist.append(picname)
    for img in imgs_test:

        image = img


        input_tensor = np.expand_dims(np.array(image), 0)
        start_time = time.time()
        fused_mask_tensor = model(input_tensor)[0][0]
        print(time.time() - start_time)
        path = "./results3/" + piclist[i]
        # print(fused_mask_tensor)
        output_image = array_to_img(fused_mask_tensor)

        output_image.save(path)
        cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv_pic = cv2.resize(cv_pic, (8000, 8000), interpolation=cv2.INTER_CUBIC)
        binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(path, cv_save)

        i = i + 1

if __name__ == '__main__':
    train()

