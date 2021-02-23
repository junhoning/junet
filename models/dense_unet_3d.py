import tensorflow as tf
from tensorflow.keras import layers


def conv3d_block(layer, n_filter):
    conv = layers.Conv3D(n_filter, 3, padding='SAME')(layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU()(conv)
    return conv


def get_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    conv_blocks = []
    conv_input = inputs
    # Encoder
    for filter_size in [32, 64, 128, 256]:
        conv = conv3d_block(conv_input, filter_size)
        conv = layers.concatenate([conv_input, conv], axis=4)
        conv = conv3d_block(conv, filter_size)
        conv = layers.concatenate([conv_input, conv], axis=4)
        conv_input = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv)
        conv_blocks.append(conv)

    # Bridge
    conv = conv3d_block(conv_input, 512)
    conv = layers.concatenate([conv_input, conv], axis=-1)
    conv = conv3d_block(conv, 512)
    conv = layers.concatenate([conv_input, conv], axis=-1)
    
    # Decoder
    for filter_size in [256, 128, 64, 32]:
        up = layers.concatenate([layers.Conv3DTranspose(filter_size, 2, strides=2, padding='SAME')(conv_input), conv_blocks.pop()], axis=-1)
        conv = conv3d_block(up, filter_size)
        conv = layers.concatenate([up, conv], axis=-1)
        conv = conv3d_block(conv, filter_size)
        conv_input = layers.concatenate([up, conv], axis=-1)

    conv10 = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv_input)

    return tf.keras.models.Model(inputs=[inputs], outputs=[conv10])