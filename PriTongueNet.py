import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from data import *
import numpy as np
from tensorflow.nn import depth_to_space
import os
from keras.preprocessing.image import array_to_img
import pathlib
import cv2
import time
from PIL import Image
import random
os.environ["CUDA_VISIBLE_DEVICE"]="0"
physical_gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
    # 将显存限制提高到6000MB
)
logical_gpus = tf.config.list_logical_devices("GPU")

#Preset tensorflow environment variables to prevent version conflicts
tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
tf.config.run_functions_eagerly(True)

#import losses and optimizer from tensorflow.keras
BCEloss = keras.losses.BinaryCrossentropy()
KLloss = keras.losses.KLDivergence()
adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=.9, beta_2=.999, epsilon=1e-08)
# adam = keras.optimizers.SGD()
#Define hyperparameters
temperature = 3
gama = 0.9
runEpoch = 0
#compute kd
def kd_loss_function(output, target_output):
    output /= temperature
    output_log_softmax = tf.nn.log_softmax(output)
    loss_kd = -tf.reduce_mean(tf.reduce_sum(output_log_softmax*target_output, axis=1))
    return loss_kd

'''
Define total loss  
y_true: ground truth
y_pred: The 512x512x7 feature maps of the output of each stage
shapeloss: Geometric prior loss
sdloss: Self distillation loss
closs: Segmentation loss
'''
def loss(y_true, y_pred):
    # global runEpoch
    # runEpoch = runEpoch+1
    runEpoch=True
    shapeloss, sdloss, closs = 0, 0, 0
    #Align the shape of y_true to y_pred
    y_align_pred = tf.expand_dims(y_pred, axis=-1)
    #compute bce loss
    closs = closs+BCEloss(y_true, y_align_pred[0])
    #compute the loss between attention map and ground truth
    closs = closs+BCEloss(y_true, y_align_pred[-1])
    # distill loss
    temp1 = y_align_pred[0] / temperature
    fused_pred = y_align_pred[0]

    sdloss = sdloss + kd_loss_function(y_align_pred[1], temp1) * (temperature ** 2)
    sdloss = sdloss + kd_loss_function(y_align_pred[2], temp1) * (temperature ** 2)
    sdloss = sdloss + kd_loss_function(y_align_pred[3], temp1) * (temperature ** 2)
    sdloss = sdloss + kd_loss_function(y_align_pred[4], temp1) * (temperature ** 2)


    if runEpoch==True: #if need shapeloss in experiment we set it True after 60 epoch

        #shapeloss will calculate after 60 epoch
        #get the images in a batch
        for i in range(len(fused_pred)):
            #get the binary map
            figure = fused_pred[i].numpy()

            temp_figure = figure
            minf = np.min(temp_figure)
            temp_figure = temp_figure-minf
            maxf = np.max(temp_figure)
            if maxf!=0:
                temp_figure /= maxf
            temp_figure[temp_figure>=0.5] = 1
            temp_figure[temp_figure<0.5] = 0

            #if the result without target the shapeloss will be 0
            if np.count_nonzero(temp_figure[:,:])!=0:
                im_floodfill = temp_figure[:,:].astype(np.uint8).copy()

                h, w, _ = temp_figure[:,:].astype(np.uint8).shape
                mask = np.zeros((h+2, w+2), np.uint8)
                isbreak = False
                for i in range(im_floodfill.shape[0]):
                    for j in range(im_floodfill.shape[1]):
                        if im_floodfill[i][j] == 0:
                            seedPoint = (i , j)
                            isbreak = True
                            break
                        if  isbreak:
                            break
                #get the polygen without hole of result
                cv2.floodFill(im_floodfill, mask, seedPoint, 255)
                #get the inverse
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                #align flood map's shape
                im_floodfill_inv = im_floodfill_inv.reshape([512,512,1])
                #get  the  foreground
                im_out = temp_figure[:,:].astype(np.uint8) | im_floodfill_inv
                #get the imformation of foreground
                numl , labels, stats, center = cv2.connectedComponentsWithStats(im_out)
                stats = stats[1:, :]
                #get the max connection area of foreground
                pupil_candidate = np.argmax(stats[:,4])+1
                #get the all data from the max area
                solutions = np.argwhere(labels==pupil_candidate)
                #draw the area
                area = np.zeros((512,512), np.uint8)
                for s in solutions:
                    area[s[0]][s[1]] = 255

                #get polygen with hole
                temp_figure[:,:][labels!=pupil_candidate] = 0
                #get the edge
                edges = cv2.Canny(temp_figure[:,:].astype(np.uint8), 0.5, 1.0)

                #random choose 8 points to draw ray
                if len(solutions) >8:
                    randnums = random.sample(range(0, len(solutions)),8 )
                else:
                    randnums = []
                l = 0
                point_loss = 0
                for i in range(len(randnums)):
                    rx = solutions[randnums[i]][0]
                    ry = solutions[randnums[i]][1]
                    #exclude the point is on the boundary
                    if edges[rx][ry]==255:
                        continue
                    #draw ray
                    line_map =  np.zeros((512, 512), np.uint8)
                    randy = random.randint(0, 512)
                    cv2.line(line_map, (rx,ry), (512, randy), 255,thickness=1, lineType=cv2.LINE_AA)
                    #intersection
                    intersection = cv2.bitwise_and(line_map, edges)
                    #exclude tangency
                    intersection_points = cv2.findNonZero(intersection)
                    if intersection_points is not None:
                        points_num = len(intersection_points)
                        for n in range(len(intersection_points)):
                            intpx, intpy = intersection_points[n][0][0], intersection_points[n][0][1]
                            #we exclude the point is not on the edge ,so 512-rx will not be 0
                            k = (randy-ry)/(512-rx)
                            y1 = int(round(k+intpy))
                            y2 = int(round(-k+intpy))
                            #if the point is not in the map, let the value to be background
                            if y1<0 or y1>=512 or intpx==511:
                                value1 = -1
                            else:
                                value1 = area[intpx + 1, y1] - 128.5
                            if y2<0 or y2>=512 or intpx==0:
                                value2 = -1
                            else:
                                value2 = area[intpx - 1, y2] - 128.5
                            if value1*value2>0:
                                points_num = points_num-1
                                intersection_points[n][0][0] = -1
                                intersection_points[n][0][1] = -1
                        #calculate shape loss
                        #max dis is 512sqrt(2)<<1000
                        mindis = 1000
                        if points_num!=0 and points_num%2==0:
                            for intp in intersection_points:
                                if intp[0][0]!=-1 and intp[0][1]!=-1:

                                    dis = np.sqrt(pow(intp[0][0] - solutions[i][0], 2) + pow(intp[0][1] - solutions[i][1], 2))
                                    if dis < mindis:
                                        mindis = dis
                                        value_figure = figure[intp[0][0], intp[0][1]]
                            l = l+1
                            point_loss = point_loss + (1-value_figure)*mindis*1e-3
                if  l!=0:
                    shapeloss = shapeloss+point_loss/l

    return shapeloss+(1-gama)*sdloss+gama*closs
#
#SubPixel Convlution
class SubPixelConv(keras.layers.Layer):
    def __init__(self, scale=2, **kwargs):
        super(SubPixelConv, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return depth_to_space(inputs, self.scale)

#build model
class PriTongueNet(keras.models.Model):
    def __init__(self):
        super(PriTongueNet, self).__init__()
        #1st stage encoder
        self.conv1_1 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal',name="conv11")
        self.conv1_2 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal',name="conv12")
        self.pool1 = MaxPooling2D(pool_size=(2, 2),name="pool1")
        #2nd stage encoder
        self.conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name="conv21")
        self.conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',name="conv22")
        self.pool2 = MaxPooling2D(pool_size=(2, 2),name="pool2")
        #3rd stage encoder
        self.conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv31")
        self.conv3_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv32")
        self.pool3 = MaxPooling2D(pool_size=(2, 2),name="pool3")
        #4th stage encoder
        self.conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv41")
        self.conv4_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv42")
        self.pool4 = MaxPooling2D(pool_size=(2, 2),name="pool4")
        #5th stage without endoer and decoder
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv51")
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv52")
        #4th stage decoder/6th option
        self.conv6_1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',name="conv61")
        self.up6 = UpSampling2D(size=(2, 2),name="up6")
        self.merge6 = Concatenate(axis=3)
        self.conv6_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv62")
        self.conv6_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv63")
        #3rd stage decoder/7th option
        self.conv7_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',name="conv71")
        self.up7 = UpSampling2D(size=(2, 2),name="up7")
        self.merge7 = Concatenate(axis=3)
        self.conv7_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv72")
        self.conv7_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv73")
        #2nd stage decoder/8th option
        self.conv8_1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',name="conv81")
        self.up8 = UpSampling2D(size=(2, 2),name="up8")
        self.merge8 = Concatenate(axis=3)
        self.conv8_2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',name="conv82")
        self.conv8_3 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',name="conv83")
        #1st stage decoder/8th option
        self.conv9_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv91")
        self.up9 = UpSampling2D(size=(2, 2),name="up9")
        self.merge9 = Concatenate(axis=3)
        self.conv9_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv92")
        self.conv9_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv93")
        # self.conv9_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal',name="conv94")
        #Obtain the final binary map
        self.conv10 = Conv2D(1, 1, activation='sigmoid',name="conv101")
        #The feature maps of the corresponding stage encoder are obtained for attention guidance map acquisition
        #subpixel conv
        self.subconv1 = SubPixelConv(name="s1")
        self.subconv2 = SubPixelConv(name="s2")
        self.subconv3 = SubPixelConv(name="s3")
        #1x1 conv for expansion channels
        self.econv1 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal',name="econv1")
        self.econv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal',name="econv2")
        self.econv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal',name="econv3")
        #changeable weight
        self.sigma1 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="sigma1")
        self.sigma2 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="sigma2")
        self.sigma3 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="sigma3")
        #genarate attention map conv
        self.gen_conv_final = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal',name="gen_conv")
        #weight of each stage skip connection
        self.beta1 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="beta1")
        self.beta2 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="beta2")
        self.beta3 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="beta3")
        self.beta4 = self.add_weight(shape=[1], initializer='ones', trainable=True,name="beta4")
        #layers and modules for distilling
        self.side1 = Conv2D(1, (3, 3), padding='same',name="side1")
        self.upsample_side_2 = UpSampling2D(size=(2, 2), interpolation='bilinear',name="upside1")
        self.side2 = Conv2D(1, (3, 3), padding='same',name="side2")
        self.upsample_side_3 = UpSampling2D(size=(4, 4), interpolation='bilinear',name="upside2")
        self.side3 = Conv2D(1, (3, 3), padding='same',name="side3")
        self.upsample_side_4 = UpSampling2D(size=(8, 8), interpolation='bilinear',name="upside3")
        self.side4 = Conv2D(1, (3, 3), padding='same',name="side4")



    def call(self, inputs):
        #original UNET encoder structure
        conv1_1 = self.conv1_1(inputs)
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
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        #generate attention map
        conv_sub_3 = self.subconv3(conv4_2)
        conv_e_3 = self.econv3(conv_sub_3)
        gen_map_3 = self.sigma3*conv_e_3+(1-self.sigma3)*conv3_2
        conv_sub_2 = self.subconv2(gen_map_3)
        conv_e_2 = self.econv2(conv_sub_2)
        gen_map_2 = self.sigma2*conv_e_2+(1-self.sigma2)*conv2_2
        conv_sub_1 = self.subconv1(gen_map_2)
        conv_e_1 = self.econv1(conv_sub_1)
        gen_map_1 = self.sigma1 * conv_e_1 + (1 - self.sigma1) * conv1_2

        guidemap = self.gen_conv_final(gen_map_1)

        #skip connection feature map
        #deconv feature map and decoder
        deconv6_1 = self.conv6_1(self.up6(conv5_2))
        att_4 = conv4_2*tf.image.resize(guidemap, [64,64])*self.beta4+(1-self.beta4)*conv4_2
        merge_6 = self.merge6([deconv6_1, att_4])
        conv6_2 = self.conv6_2(merge_6)
        conv6_3 = self.conv6_3(conv6_2)
        deconv7_1 = self.conv7_1(self.up7(conv6_3))
        att_3 = conv3_2*tf.image.resize(guidemap, [128,128])*self.beta3+(1-self.beta3)*conv3_2
        merge_7 = self.merge7([deconv7_1, att_3])
        conv7_2 = self.conv7_2(merge_7)
        conv7_3 = self.conv7_3(conv7_2)
        deconv8_1 = self.conv8_1(self.up8(conv7_3))
        att_2 = conv2_2*tf.image.resize(guidemap, [256, 256])*self.beta2+(1-self.beta2)*conv2_2
        merge_8 = self.merge8([deconv8_1, att_2])
        conv8_2 = self.conv8_2(merge_8)
        conv8_3 = self.conv8_3(conv8_2)
        deconv9_1 = self.conv9_1(self.up9(conv8_3))
        att_1 = conv1_2 *guidemap*self.beta1+(1-self.beta1)*conv1_2
        merge_9 = self.merge9([deconv9_1, att_1])
        conv9_2 = self.conv9_2(merge_9)
        conv9_3 = self.conv9_3(conv9_2)
        #get the result from different stage for distilling study
        #1st
        side1 = self.side1(conv9_3)
        #2nd
        side2 = self.side2(self.upsample_side_2(conv8_3))
        #3rd
        side3 = self.side3(self.upsample_side_3(conv7_3))
        #4th
        side4 = self.side4(self.upsample_side_4(conv6_3))
        #get the final result
        fused_output = self.conv10(tf.concat([side1, side2, side3, side4], axis=3))
        # fused_output = self.conv10(side1)
        sig = keras.activations.sigmoid
        return tf.stack([fused_output, sig(side1), sig(side2), sig(side3), sig(side4), guidemap])

#load data
def load_data():
    mydata = dataProcess(512,512)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test,  imgs_mask_test = mydata.load_test_data()

    return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test

def train():
    inputs = tf.keras.Input(shape=[512, 512, 3])
    imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = load_data()
    net = PriTongueNet()
    out = net(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out, name='PriTongueNet')
    model.compile(optimizer=adam, loss=loss, metrics=None)

    print("loading data")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model.ckpt',verbose=1, save_best_only=True,
                                                          save_weights_only=True)
    print('Fitting model...')
    model.load_weights("naive.h5")
    model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=40, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
    model.save_weights("result.h5")
    print('predict test data')
    i = 0
    ppath = "../datas/datas/open environment/test"
    npath = "./unet_results"
    piclist = os.listdir(ppath)

    for img in imgs_test:
        image = img

        input_tensor = np.expand_dims(np.array(image), 0)
        fused_mask_tensor = model(input_tensor)[0][0]
        path = "./results/" + piclist[i]
        output_image = array_to_img(fused_mask_tensor)

        output_image.save(path)
        cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(path, cv_save)

        i = i + 1


if __name__ == '__main__':
    train()
