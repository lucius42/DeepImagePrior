import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.losses import mse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error
from Model import *

matplotlib.use('TkAgg')


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = image.astype(np.float32) / 255. if np.max(image) > 1 else image
    image = np.expand_dims(image, axis=-1) if len(image.shape) == 2 else image
    return image


def add_gaussian_noise(image, sigma=25):
    noisy_image_ = image + np.random.normal(0, sigma / 255., image.shape)
    return np.clip(noisy_image_, 0, 1)


def step(model, net_input, noisy_image, loss_function, optimizer):
    with tf.GradientTape() as tape:
        pred = model(net_input)
        pred_flatten = Flatten()(pred)
        loss = loss_function(noisy_image.flatten(), pred_flatten)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return pred, loss[0].numpy()


if __name__ == '__main__':

    #Set parameters
    loss_function = mse
    optimizer = Adam(learning_rate=0.001)
    num_iter = 20000
    num_channels = 3
    noise_type = 'gaussian'  # 'poisson' , 's&p'
    sigma = 25

    #Get Image
    image = preprocess_image("JPG/Tulpe.jpg")

    #Add noise
    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(image, sigma)
    else:
        noisy_image = random_noise(image, noise_type)

    #Show noisy image
    plt.imshow(noisy_image, cmap='gray') if num_channels == 1 else plt.imshow(noisy_image)

    #Create network input
    seed_value = 42
    np.random.seed(seed_value)
    net_input = (np.random.uniform(0, 1, image.shape)).astype(np.float32)
    plt.imshow(net_input, cmap='gray') if num_channels == 1 else plt.imshow(net_input)

    #Create model
    model = create_model(image.shape)
    model.summary()

    #Define variables
    x = np.expand_dims(net_input, axis=0)
    y = noisy_image
    pred_list = []
    loss_list = []
    psnr_list = []
    mse_list = []

    #Start optimization
    for i in range(num_iter + 1):
        pred, loss = step(model, x, y, loss_function, optimizer)
        pred = np.reshape(pred, image.shape)
        psnr = peak_signal_noise_ratio(image, pred)
        mse_ = mean_squared_error(image.flatten(), pred.flatten())
        pred_list.append(pred)
        loss_list.append(loss)
        psnr_list.append(psnr)
        mse_list.append(mse_)
        if i % 500 == 0:
            print('Iteration:', i)
            print('Loss:', np.round(loss, 5))
            print('PSNR:', np.round(psnr, 5))
            print('MSE:', np.round(mse_, 5))
            print('-------------')

    print("fin")
