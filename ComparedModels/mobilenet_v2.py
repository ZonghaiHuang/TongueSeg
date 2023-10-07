from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.layers import UpSampling2D
from keras import backend as K
from data import *

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    print(x.shape)
    # x = _conv_block(x, tchannel, (3, 3), (1, 1))
    # x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation(relu6)(x)
    print(x.shape)
    x = CoordAtt(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import *
def CoordAtt(inputs):
    input_shape = inputs.shape
    reduction = 4
    inter_channels = input_shape[-1] // reduction
    conv = Conv2D(inter_channels, 1, use_bias=False)
    bn = BatchNormalization()
    conv_h = Conv2D(input_shape[-1], 1)
    conv_w = Conv2D(input_shape[-1], 1)

    h = K.int_shape(inputs)[1]
    x_h = K.mean(inputs, axis=2, keepdims=True)
    x_w = K.mean(inputs, axis=1, keepdims=True)
    x_t = K.permute_dimensions(x_w, (0, 2, 1, 3))

    x = K.concatenate([x_h, x_t], axis=1)
    x = K.relu(bn(conv(x)))

    x_h, x_t = x[:, :h, :, :], x[:, h:, :, :]
    x_w = K.permute_dimensions(x_t, (0, 2, 1, 3))
    x_h = K.sigmoid(conv_h(x_h))
    x_w = K.sigmoid(conv_w(x_w))

    return inputs * x_h * x_w
def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def MobileNetv2(input_shape, k, alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].

    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, last_filters))(x)
    # x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    #
    # x = Activation('softmax', name='softmax')(x)
    # output = Reshape((k,))(x)
    upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsample_6)
    upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsample_5)
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsample_4)
    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsample_3)
    res = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upsample_2)
    res = Conv2D(1, 1, activation='sigmoid')(res)
    model = Model(inputs, res)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
    model.summary()
    return model
def load_data():
    mydata = dataProcess(512,512)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()

    return imgs_train, imgs_mask_train, imgs_test
import keras
if __name__ == '__main__':
    imgs_train, imgs_mask_train, imgs_test = load_data()
    model = MobileNetv2((512, 512, 3), 100, 1.0)
    loss = keras.losses.binary_crossentropy
    opt = keras.optimizers.adam_v2.Adam(1e-4)
    metric = [keras.metrics.binary_accuracy]
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    print(imgs_train.shape)
    # model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=100, verbose=1, shuffle=True)
    # model.save_weights("model.h5")
    model.load_weights("model.h5")
    import numpy as np
    import time
    import cv2
    from keras.preprocessing.image import array_to_img
    i = 0
    piclist = []
    for line in open("./results/pic.txt"):
        line = line.strip()
        picname = line.split('/')[-1]
        piclist.append(picname)
    for img in imgs_test:
        image = img

        input_tensor = np.expand_dims(np.array(image), 0)
        start_time = time.time()
        fused_mask_tensor = model(input_tensor)[0]
        print(time.time() - start_time)
        path = "./results/" + piclist[i].split("image\\")[-1]
        # print(fused_mask_tensor)

        output_image = array_to_img(fused_mask_tensor)

        output_image.save(path)
        cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv_pic = cv2.resize(cv_pic, (8000, 8000), interpolation=cv2.INTER_CUBIC)
        binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(path, cv_save)

        i= i+1
