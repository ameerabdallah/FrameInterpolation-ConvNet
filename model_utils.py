from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, merge
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.models import Model
from keras import backend

def charbonnier(y_true, y_pred):
    return backend.sqrt(backend.square(y_true - y_pred) + 0.01**2)

def convolve_and_pool(input, filters):
    block = Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same')(input)
    block = Activation('relu')(block)
    block = MaxPooling2D(pool_size=(2, 2))(block)
    return block

def convolve_and_upsample(input, filters):
    block = Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same')(input)
    block = UpSampling2D(size=(2, 2))(block)
    return block

def convolve(input, filters):
    block = Conv2D(filters, kernel_size=(3,3), padding='same')(input)
    block = Activation('relu')(block)
    return block

def create_model(input_shape):
    input = Input(input_shape)
    conv_1 = convolve_and_pool(input, 64)

    conv_2 = convolve_and_pool(conv_1, 128)

    conv_3 = convolve_and_pool(conv_2, 256)

    conv_4 = convolve_and_pool(conv_3, 512)

    conv_5 = convolve_and_pool(conv_4, 1024)

    deconv_5 = convolve(conv_5, 1024)
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
    output = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = create_model((None, None, 2))
model.compile(loss=charbonnier, optimizer="adam")

dot = model_to_dot(model, show_shapes=True, show_layer_names=True).write_png('model.png')