from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, merge
from keras.models import Model
from keras.optimizers import adam_v2
from keras import backend


LEARNING_RATE = 0.0001
optimizer = adam_v2.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def charbonnier(y_true, y_pred):
    return backend.sqrt(backend.square(y_true - y_pred) + 0.01**2)

def convolve_and_pool(input, filters):
    block = convolve(input, filters)
    block = convolve(block, filters)
    block = MaxPooling2D(pool_size=(2, 2))(block)
    return block

def convolve_and_upsample(input, filters):
    block = convolve(input, filters)
    block = convolve(block, filters)
    block = UpSampling2D(size=(2, 2))(block)
    return block

def convolve(input, filters):
    block = Conv2D(filters, kernel_size=(3,3), activation='relu',padding='same')(input)
    return block

def create_model(input_shape):
    input = Input(input_shape)
    conv_1 = convolve_and_pool(input, 64)

    conv_2 = convolve_and_pool(conv_1, 128)

    conv_3 = convolve_and_pool(conv_2, 256)

    conv_4 = convolve_and_pool(conv_3, 512)

    conv_5 = convolve_and_pool(conv_4, 1024)

    deconv_5 = convolve(conv_5, 1024)
    deconv_5 = convolve(deconv_5, 1024)
    deconv_5 = merge.concatenate([deconv_5, conv_5], axis=3)

    deconv_4 = convolve_and_upsample(deconv_5, 512)
    deconv_4 = merge.concatenate([deconv_4, conv_4], axis=3)

    deconv_3 = convolve_and_upsample(deconv_4, 256)
    deconv_3 = merge.concatenate([deconv_3, conv_3], axis=3)

    deconv_2 = convolve_and_upsample(deconv_3, 128)
    deconv_2 = merge.concatenate([deconv_2, conv_2], axis=3)

    deconv_1 = convolve_and_upsample(deconv_2, 64)
    deconv_1 = merge.concatenate([deconv_1, conv_1], axis=3)

    output = convolve_and_upsample(deconv_1, 32)
    output = merge.concatenate([output, input], axis=3)
    output = Conv2D(1, kernel_size=(1,1),  activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=charbonnier)

    return model

model = create_model((None, None, 2))