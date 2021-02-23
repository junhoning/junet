import tensorflow as tf
from tensorflow.keras import layers


n_filter = 1

def conv3d_block(inputs, n_filter):
    conv = layers.Conv2D(32*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv2D(64*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    pool = layers.MaxPooling2D(pool_size=2)(conv)
    return conv, pool

def trans_conv3d_block(conv, conv_merge, n_filter):
    up = layers.Conv2DTranspose(512*n_filter, 2, strides=1, padding='same')(conv)
    merge = layers.concatenate([conv_merge, up], axis=-1)
    conv = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(merge)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    return conv


# Build UNet
def get_model(input_shpae, num_classes):
    # Encoder
    inputs = layers.Input(input_shpae)
    conv1 = layers.Conv2D(64*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.Conv2D(64*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = layers.Conv2D(128*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.Conv2D(128*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=2, strides=2)(conv3)

    conv4 = layers.Conv2D(512*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.Conv2D(512*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=2, strides=2)(conv4)

    # Bridge
    conv5 = layers.Conv2D(1024*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU()(conv5)
    conv5 = layers.Conv2D(1024*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU()(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(1024*n_filter, 2, strides=2, padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=-1)
    conv6 = layers.Conv2D(512*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.LeakyReLU()(conv6)
    conv6 = layers.Conv2D(512*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.LeakyReLU()(conv6)

    up7 = layers.Conv2DTranspose(512*n_filter, 2, strides=2, padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=-1)
    conv7 = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.LeakyReLU()(conv7)
    conv7 = layers.Conv2D(256*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.LeakyReLU()(conv7)

    up8 = layers.Conv2DTranspose(256*n_filter, 2, strides=2, padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=-1)
    conv8 = layers.Conv2D(128*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.LeakyReLU()(conv8)
    conv8 = layers.Conv2D(128*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.LeakyReLU()(conv8)

    up9 = layers.Conv2DTranspose(128*n_filter, 2, strides=2, padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=-1)
    conv9 = layers.Conv2D(64*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.LeakyReLU()(conv9)
    conv9 = layers.Conv2D(64*n_filter, 3, strides=1, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.LeakyReLU()(conv9)

    conv9 = layers.Conv2D(num_classes, 1, padding='SAME', kernel_initializer='he_normal', activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv9)
    return model 