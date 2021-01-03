from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

import numpy as np;

train_labels = np.loadtxt('labels.txt');
train_images = np.loadtxt('images.txt');# data_in - Данные подающиеся на вход сети - Производные ФПГ Сигнала

test_labels = np.loadtxt('labels_test.txt');
test_images = np.loadtxt('images_test.txt');# data_out - Данные на выходе сети - Положения особых точек ФПГ Сигнала

train_images = train_images/6 # Обезразмеривание производной ФПГ Сигнала
test_images = test_images/6


# Составляем двухуровневую модель с 1000 элементов на входном слое, 300 - на внутреннем, 80 - на внешнем
def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(300,kernel_initializer='glorot_uniform', input_shape=(1000,), activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(80, kernel_initializer = 'lecun_normal', activation='softmax'))
    return model

uniform_model = create_model()

#Функции потерь
#tf.keras.losses.mean_squared_error(y_true, y_pred)
#tf.keras.losses.mean_absolute_error(y_true, y_pred)
#tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
#tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
#tf.keras.losses.squared_hinge(y_true, y_pred)
#tf.keras.losses.hinge(y_true, y_pred)
#tf.keras.losses.categorical_hinge(y_true, y_pred)
#tf.keras.losses.logcosh(y_true, y_pred)
#tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
#tf.keras.losses.binary_crossentropy(y_true, y_pred)
#tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
#tf.keras.losses.poisson(y_true, y_pred)
#tf.keras.losses.cosine_proximity(y_true, y_pred)

#opt = tf.keras.optimizers.SGD(lr=0.05, momentum=0.1, decay=0.0, nesterov=False, clipnorm=1.0, clipvalue=0.5)
#opt= tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=1., clipvalue=0.5)
#opt = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0, clipnorm=1., clipvalue=0.5)
#opt = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0, clipnorm=1., clipvalue=0.5)
#opt =  tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=1., clipvalue=0.5)
#opt = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipnorm=2.0, clipvalue=1.0)
opt = tf.keras.optimizers.Nadam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=1.0, clipvalue=0.5)

# Компилируем модель с выбранным оптимизатором и методом рассчета потерь.
uniform_model.compile(
    loss='categorical_crossentropy',
         #     optimizer='sgd',
                optimizer = opt,
              metrics=['accuracy'])
# Обучаем модель 10 эпох
uniform_model.fit(train_images, train_labels, batch_size=2, epochs=10)
# Рассчитываем потери и точность
test_loss, test_acc = uniform_model.evaluate(test_images, test_labels, verbose=1)
# И выводим их
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(test_loss, test_acc))

#
uniform_model.save_weights('my_model_weights.h5')


print(tf.version.VERSION)
