
from numpy.core.numeric import identity
from numpy.lib.twodim_base import tri
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import numpy as np

def GlobalAvgPool():
    return tf.keras.layers.GlobalAveragePooling2D()

def MaxAvgPool():
    return tf.keras.layers.GlobalMaxPooling2D()


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)

# convolution
def convolution(filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                          dilation_rate=dilation_rate)

# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0001),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0001),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


def max_pool(pool_size=2, stride=2):
    return layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=(stride, stride))

def avg_pool(pool_size=2, stride=2):
    return layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride))

# convolution
def Conv(n_filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                         dilation_rate=dilation_rate)

# Traspose convolution
def Conv_trans(n_filters, kernel_size=2, strides=2, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(n_filters, kernel_size, strides=(strides, strides), padding='same', use_bias=use_bias,
                                  dilation_rate=dilation_rate)

class DepthMaxPool(tf.keras.Model):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super(DepthMaxPool, self).__init__()
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs, type_='MAX'):
        if type_ == 'MAX':
            return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)
        else:
            return tf.nn.avg_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)                    

class channel_atttention(tf.keras.Model):
    def __init__(self, n_filters, ratio=16, outputSize=1):
        super(channel_atttention, self).__init__()
        # self.avg_pool = layers.MaxPool2D(pool_size=(1,64,1,1), strides=1)#GlobalAvgPool()
        # self.avg_pool = DepthMaxPool(64)
        # self.max_pool = DepthMaxPool(64)

        # self.avg_pool =  tf.nn.max_pool(s,
                        # ksize=(1, 1, 1, 3),
                        # strides=(1, 1, 1, 3),
                        # padding="VALID")
        # self.max_pool = layers.MaxPool2D(pool_size=(1,64,1,1), strides=1)#MaxAvgPool()
        self.conv1 = Conv(n_filters//ratio, kernel_size=1)
        self.conv2 = Conv(n_filters, kernel_size=1)
        self.outputSize = outputSize
        self.conv3 = Conv(n_filters//ratio, kernel_size=1)
        self.conv4 = Conv(n_filters, kernel_size=1)

    def call(self, inputs, training=True):
        # conv1 = tf.reshape(self.avg_pool(inputs), (1, 1, 1, -1))
        # _, h, w, ch = K.int_shape(inputs)
         
        stride = 256#np.floor(h/self.outputSize).astype(np.int32)
        kernel = 256#h - (self.outputSize-1) * stride
        conv1 = max_pool(pool_size=kernel, stride=stride)(inputs)
        conv1 = conv1 * inputs
        # conv1 = self.avg_pool(inputs)
        # conv1 = tf.reduce_max(inputs, axis=[3], keepdims=True)
        conv1 = self.conv1(conv1, training=training)
        conv1 = layers.ReLU()(conv1)
        conv2 = self.conv2(conv1, training=training)

        # conv3 = tf.reshape(self.max_pool(inputs), (1, 1, 1, -1))
        conv3 = avg_pool(pool_size=kernel, stride=stride)(inputs)
        conv3 = conv3 * inputs

        # conv3 = self.max_pool(inputs)
        # conv3 = tf.reduce_max(inputs, axis=[3], keepdims=True)
        
        conv3 = self.conv3(conv3, training=training)
        conv3 = layers.ReLU()(conv3)
        conv4 = self.conv4(conv3, training=training)
        # layers.Add
        out = conv2 + conv4
        # out = tf.keras.activations.sigmoid(out)
        out = layers.Activation('sigmoid')(out)

        return out

# def adapPool():


class up_block(tf.keras.Model):
    def __init__(self, n_filters, kernel_size=2, stride=2, dilation_rate=1, trans=True):
        super(up_block, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv_trans = Conv_trans(n_filters, kernel_size, stride)
        self.bn = layers.BatchNormalization()


    def call(self, inputs, activation=True, training=True):
        x = self.conv_trans(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)
        return x

class conv_layer(tf.keras.Model):
    def __init__(self, n_filters, kernel_size, stride=1, dilation_rate=1):
        super(conv_layer, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv = Conv(self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, normalization=True, training=True):
        x = self.conv(inputs, training=training)
        if normalization:
            x = self.bn(inputs, training=training)
        if activation:
            x = layers.ReLU()(x)
        return x


class conv_block(tf.keras.Model):
    def __init__(self, n_filters, kernel_size=3, stride=1, dilation_rate=1):
        super(conv_block, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv1 = Conv(self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.conv2 = Conv(2 * self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.identity = Conv(2 * self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()



    def call(self, inputs, activation=True, normalization=True, training=True):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x1 = layers.ReLU()(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2, training=training)
        x2 = layers.ReLU()(x2)

        identity = self.identity(inputs)
        identity = self.bn3(identity, training=training)
        x2 = x2 + identity
        x2 = self.bn4(x2, training=training)
        x2 = layers.ReLU()(x2)
        return x2


class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x


class Trident_Block(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1):
        super(Trident_Block, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = convolution(filters=filters, kernel_size=3, strides=strides)
        self.bn1 = layers.BatchNormalization()
        # self.conv2 = convolution(filters=filters, kernel_size=3, strides=strides)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = convolution(filters=filters, kernel_size=3, strides=strides)
        self.bn3 = layers.BatchNormalization()
        self.shortcut = convolution(filters=filters, kernel_size=3, strides=strides, dilation_rate=dilation_rate)
        # self.bn4 = layers.BatchNormalization()

    def call(self, inputs, num_branches=3, concat=False, training=True):
        if not isinstance(inputs, list):
            inputs = [inputs]  * num_branches

        out = [self.conv1(x) for i, x in enumerate(inputs)]
        out = [self.bn1(x, training=training) for x in out]
        out = [layers.ReLU()(x) for x in out]
        out = [layers.Conv2D(self.filters, 3, padding='same', use_bias=True,
                          dilation_rate=i+1)(x) for i, x in enumerate(out)]
        out = [self.bn2(x, training=training) for x in out]
        out = [layers.ReLU()(x) for x in out]
        out = [self.conv3(x) for i, x in enumerate(out)]

        shortcut = [self.shortcut(x) for x in inputs]
        out = [out_inputs + x_inputs for out_inputs, x_inputs in zip(out, shortcut)]
        out = [self.bn3(x, training=training) for x in out]
        out = [layers.ReLU()(x) for x in out]
        
        if concat:
            out = layers.concatenate(out, axis=-1)
        return out


class Trident_Block2(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1):
        super(Trident_Block, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        dilation_rates = [1, 2, 3]
        self.conv1 = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rates[0])
        self.bn1 = layers.BatchNormalization()
        self.conv2 = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rates[1])
        self.bn2 = layers.BatchNormalization()
        self.conv3 = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rates[2])
        self.bn3 = layers.BatchNormalization()
        self.shortcut = convolution(filters=filters, kernel_size=1, strides=strides, dilation_rate=dilation_rates[0])
        self.bn4 = layers.BatchNormalization()



    def call(self, inputs, activation=True, training=True):
        x = self.conv1(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x



def Concat():
    return layers.concatenate()

class Trident(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, None, 3), n_filters=64, **kwargs):
        super(Trident, self).__init__(**kwargs)
        print(n_filters, 'Number of filters ')
        self.pool = max_pool()
        # self.concat = Concat()
        # self.conv0_0 = conv_block(n_filters)

        ## Encoder Input 0
        self.conv0_0 = Trident_Block(n_filters)
        self.conv1_0 = Trident_Block(2 * n_filters)
        self.conv2_0 = Trident_Block(4 * n_filters)
        self.conv3_0 = Trident_Block(8 * n_filters)
        self.conv4_0 = Trident_Block(8 * n_filters)


        ## Encoder Input 1
        self.conv0_1 = Trident_Block(n_filters)
        self.conv1_1 = Trident_Block(2 * n_filters)
        self.conv2_1 = Trident_Block(4 * n_filters)
        self.conv3_1 = Trident_Block(8 * n_filters)
        self.conv4_1 = Trident_Block(8 * n_filters)

        
        self.bottleneck = Trident_Block(8 * n_filters)

        ## Upsampling - Decoder 
    

        self.final_up3 = up_block(16 * n_filters)
        self.final_up2 = up_block(8 * n_filters)
        self.final_up1 = up_block(4 * n_filters)
        self.final_up0 = up_block(2 * n_filters)
        self.final_up_final = up_block(n_filters)


        self.up_block_0 = Trident_Block(n_filters)
        self.up_block_1 = Trident_Block(2 * n_filters)
        self.up_block_2 = Trident_Block(4 * n_filters)
        self.up_block_3 = Trident_Block(8 * n_filters)

        self.final_tri = Trident_Block(n_filters)

        self.final_conv = Conv(1, kernel_size=1)

    def call(self, inputs, training=True):
        inputs0, inputs1 = inputs[:, :, :256, :], inputs[:, :, 256:, :]

        x0_0 = [max_pool()(x) for x in (self.conv0_0(inputs0, training=training))]
        x1_0 = [max_pool()(x) for x in (self.conv1_0(x0_0, training=training))]
        x2_0 = [max_pool()(x) for x in (self.conv2_0(x1_0, training=training))]
        x3_0 = [max_pool()(x) for x in (self.conv3_0(x2_0, training=training))]
        x4_0 = [max_pool()(x) for x in (self.conv4_0(x3_0, training=training))]


        x0_1 = [max_pool()(x) for x in (self.conv0_1(inputs1, training=training))]
        x1_1 = [max_pool()(x) for x in (self.conv1_1(x0_1, training=training))]
        x2_1 = [max_pool()(x) for x in(self.conv2_1(x1_1, training=training))]
        x3_1 = [max_pool()(x) for x in(self.conv3_1(x2_1, training=training))]
        x4_1 = [max_pool()(x) for x in (self.conv4_1(x3_1, training=training))]


        concat_x_4 = [layers.concatenate([_x4_0, _x4_1], axis=-1) for _x4_0, _x4_1 in zip(x4_0, x4_1)]
        concat_x_3 = [layers.concatenate([_x3_0, _x3_1], axis=-1) for _x3_0, _x3_1 in zip(x3_0, x3_1)]
        concat_x_2 = [layers.concatenate([_x2_0, _x2_1], axis=-1) for _x2_0, _x2_1 in zip(x2_0, x2_1)]
        concat_x_1 = [layers.concatenate([_x1_0, _x1_1], axis=-1) for _x1_0, _x1_1 in zip(x1_0, x1_1)]
        concat_x_0 = [layers.concatenate([_x0_0, _x0_1], axis=-1) for _x0_0, _x0_1 in zip(x0_0, x0_1)]

   
        bottleneck = self.bottleneck(concat_x_4, training=training)

        up_block3 = [self.final_up3(x, training=training) for x in bottleneck]
        up_block3 = [x0 + x1 for x0, x1 in zip(up_block3, concat_x_3)]
        up_block3 = self.up_block_3(up_block3)

        up_block2 = [self.final_up2(x, training=training) for x in up_block3]
        up_block2 = [x0 + x1 for x0, x1 in zip(up_block2, concat_x_2)]
        up_block2 = self.up_block_2(up_block2)

        up_block1 = [self.final_up1(x, training=training) for x in up_block2]
        up_block1 = [x0 + x1 for x0, x1 in zip(up_block1, concat_x_1)]
        up_block1 = self.up_block_1(up_block1)

        up_block0 = [self.final_up0(x, training=training) for x in up_block1]
        up_block0 = [x0 + x1 for x0, x1 in zip(up_block0, concat_x_0)]
        up_block0 = self.up_block_0(up_block0)

        up_block0 = [self.final_up_final(x, training=training) for x in up_block0]

        out = self.final_tri(up_block0, concat=True)
        out = self.final_conv(out)

        out = layers.Activation('sigmoid')(out)
        return out
