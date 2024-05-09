from keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Input, concatenate
from keras.models import Model


def create_model(input_size):

    encoder_input = Input(input_size)

    x0 = encoder_block(encoder_input, 8, 3)
    x1 = encoder_block(x0, 16, 3)
    x2 = encoder_block(x1, 32, 3)
    x3 = encoder_block(x2, 64, 3)
    x4 = encoder_block(x3, 128, 3)

    s3 = skip_conn(x2, 4, 1)
    s4 = skip_conn(x3, 4, 1)

    y4 = decoder_block(x4, 128, 3, skip=s4)
    y3 = decoder_block(y4, 64, 3, skip=s3)
    y2 = decoder_block(y3, 32, 3)
    y1 = decoder_block(y2, 16, 3)
    y0 = decoder_block(y1, 8, 3)

    decoder_output = Conv2D(input_size[-1], 1, padding="same", kernel_initializer='he_normal', activation="sigmoid")(y0)

    model = Model(encoder_input, decoder_output, name="Encoder-Decoder")

    return model


def create_small_model(input_size):

    encoder_input = Input(input_size)

    x0 = encoder_block(encoder_input, 8, 3)
    x1 = encoder_block(x0, 16, 3)
    x2 = encoder_block(x1, 32, 3)

    y2 = decoder_block(x2, 32, 3)
    y1 = decoder_block(y2, 16, 3)
    y0 = decoder_block(y1, 8, 3)

    decoder_output = Conv2D(input_size[-1], 1, padding="same", kernel_initializer='he_normal', activation="sigmoid")(y0)

    model = Model(encoder_input, decoder_output, name="Encoder-Decoder without skip (small)")

    return model


def create_large_model(input_size):

    encoder_input = Input((512,512,32))

    x0 = encoder_block(encoder_input, 128, 3)
    x1 = encoder_block(x0, 128, 3)
    x2 = encoder_block(x1, 128, 3)
    x3 = encoder_block(x2, 128, 3)
    x4 = encoder_block(x3, 128, 3)

    s0 = skip_conn(encoder_input, 4, 1)
    s1 = skip_conn(x0, 4, 1)
    s2 = skip_conn(x1, 4, 1)
    s3 = skip_conn(x2, 4, 1)
    s4 = skip_conn(x3, 4, 1)

    y4 = decoder_block(x4, 128, 3, skip=s4)
    y3 = decoder_block(y4, 128, 3, skip=s3)
    y2 = decoder_block(y3, 128, 3, skip=s2)
    y1 = decoder_block(y2, 128, 3, skip=s1)
    y0 = decoder_block(y1, 128, 3, skip=s0)

    decoder_output = Conv2D(input_size[-1], 1, padding="same", kernel_initializer='he_normal', activation="sigmoid")(y0)

    model = Model(encoder_input, decoder_output, name="Encoder-Decoder (large)")

    return model


def encoder_block(encoder_input, num_fm, kernel_size, alpha=0.2):

    x = Conv2D(num_fm, kernel_size, strides=2, padding="same", kernel_initializer='he_normal')(encoder_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(num_fm, kernel_size, padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)

    return x


def skip_conn(skip_input, num_fm, kernel_size, alpha=0.2):

    s = Conv2D(num_fm, kernel_size, padding="same", kernel_initializer='he_normal')(skip_input)
    s = BatchNormalization()(s)
    s = LeakyReLU(alpha=alpha)(s)

    return s


def decoder_block(decoder_input, num_fm, kernel_size, alpha=0.2, skip=None):

    y = UpSampling2D(size=(2, 2), interpolation='bilinear')(decoder_input)
    y = concatenate([skip, y]) if skip is not None else y
    y = BatchNormalization()(y)

    y = Conv2D(num_fm, kernel_size, padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=alpha)(y)

    y = Conv2D(num_fm, 1, padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=alpha)(y)

    return y
