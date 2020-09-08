#!/usr/bin/env python3
# coding: utf-8

# # CNN Classify Cats & Dogs [Link]
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/


# plot dog photos from the dogs vs cats dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from matplotlib import pyplot
# from matplotlib.image import imread
# from PIL import Image
from collections import OrderedDict
import random
# import shutil
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Ftrl
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

from cnn_config import models_config
from tensorflow.keras import backend as K
import csv


def trim_images2(from_folder, size=50, ratio=1, namesufix=''):
    ''' Read images from directory and convert them into numpy objects '''
    images = []
    labels = []
    no_cats, no_dogs = 0, 0
    images_file = 'data/{}_images_{}_{}.npy'.format(namesufix, size, ratio)
    labels_file = 'data/{}_labels_{}_{}.npy'.format(namesufix, size, ratio)
    if os.path.exists(images_file) and os.path.exists(labels_file):
        images = np.load('data/{}_images_{}_{}.npy'.format(namesufix, size, ratio))
        labels = np.load('data/{}_labels_{}_{}.npy'.format(namesufix, size, ratio))
    else:
        print('getting images from directory')
        random.seed(1)

        for file in os.listdir(from_folder):
            src = from_folder + file
            if random.random() < ratio:
                image = cv2.imread(from_folder + file)
                image = cv2.resize(image, (size, size))
                if file.startswith('cat'):
                    label = [0, 1]
                    no_cats += 1
                elif file.startswith('dog'):
                    label = [1, 0]
                    no_dogs += 1
                labels.append(label)
                images.append(image)
        images = np.array(images)
        labels = np.array(labels)
        np.save(images_file, images)
        np.save(labels_file, labels)

    return no_cats, no_dogs, labels, images


def define_model(model_config):
    ''' Create custom model based on the 
    model_config dictionary that is provided '''
    input_shape = (model_config['target_size'][0],
                   model_config['target_size'][1],
                   3)

    model = Sequential()
    for layer in model_config['layers']:
        model.add(Conv2D(
            layer['filters'],
            layer['kernel_size'],
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            input_shape=input_shape))
        model.add(MaxPooling2D(layer['pooling']))
        if layer['dropout'] is not None:
            model.add(Dropout(layer['dropout']))

    model.add(Flatten())
    model.add(Dense(model_config['base']['units'],
                    activation='relu',
                    kernel_initializer='he_uniform'))
    if model_config['base']['dropout'] is not None:
        model.add(Dropout(model_config['base']['dropout']))
    model.add(Dense(2, activation='softmax'))

    # compile model
    optimizer = eval(model_config['optimizer'])
    if model_config['optimizer'] == 'SGD':
        opt = optimizer(
            lr=model_config['lr'],
            momentum=model_config['momentum'])
    else:
        opt = optimizer(lr=model_config['lr'])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def define_model_tl(model_config):
    ''' Create a Transfer Learning model with some options
    from the model_config model. The requirement to run this
    is to not have any layers. '''
    print('--== Using Transfer Learning model ==--')
    # load model
    model = VGG16(
        include_top=False,
        input_shape=(
            model_config['target_size'][0],
            model_config['target_size'][1],
            3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(model_config['base']['units'],
                   activation='relu',
                   kernel_initializer='he_uniform')(flat1)
    output = Dense(2, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    optimizer = eval(model_config['optimizer'])
    opt = optimizer(
        lr=model_config['lr'],
        momentum=model_config['momentum'])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_eval_harness(model_config):
    ''' Train the images '''
    if len(model_config['layers']) == 0:
        model = define_model_tl(model_config)
        train_datagen = ImageDataGenerator(featurewise_center=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           horizontal_flip=True)
        eval_datagen = ImageDataGenerator(featurewise_center=True,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          horizontal_flip=True)
        # specify imagenet mean values for centering
        train_datagen.mean = [123.68, 116.779, 103.939]
        eval_datagen.mean = [123.68, 116.779, 103.939]
    elif model_config['augmentation']:
        model = define_model(model_config)
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    else:
        model = define_model(model_config)
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # prepare iterators
    train_it = train_datagen.flow(
        tr_images,
        tr_labels,
        batch_size=model_config['batch_size'],
        shuffle=True)
    eval_it = eval_datagen.flow(
        tst_images,
        tst_labels,
        batch_size=model_config['batch_size'],
        shuffle=False)
    # fit model
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=eval_it,
        validation_steps=len(eval_it),
        epochs=model_config['epochs'],
        verbose=1)
    # evaluate model
    _, acc = model.evaluate_generator(eval_it, steps=len(eval_it), verbose=1)
    return model, history, acc


def get_filename(model_config):
    ''' Create a custom filename which specifies the inut data '''
    augm_txt = ''
    if model_config['augmentation']:
        augm_txt = 'augmentation'
    drop_txt = ''
    if model_config['base']['dropout'] is not None:
        drop_txt += 'dropbase{}'.format(model_config['base']['dropout'])
    for layer in model_config['layers']:
        if layer['dropout'] is not None:
            drop_txt += 'droplayer{}'.format(layer['dropout'])

    filename = 'models/base{}_lr{}_momentum{}_VGG{}_img{}_batch{}_{}_epoch{}_{}_{}'.format(
        model_config['base']['units'],
        model_config['lr'],
        model_config['momentum'],
        len(model_config['layers']),
        model_config['target_size'][0],
        model_config['batch_size'],
        model_config['optimizer'],
        model_config['epochs'],
        augm_txt,
        drop_txt).replace('__', '_').strip('_')
    return filename


def get_diagnostics(filename, model_config, acc, f1, prec, rec, cats_metrix, dogs_metrix):
    ''' get diagnostics '''
    droplayers = []
    for layer in model_config['layers']:
        if layer['dropout'] is not None:
            droplayers.append(str(layer['dropout']))
    droplayers_txt = ','.join(droplayers)
    if len(droplayers) == 0:
        droplayers_txt = None
    history = pickle.load(open('{}.hist'.format(filename), 'rb'))

    titles = [
        'hist_loss',
        'hist_val_loss',
        'hist_acc',
        'hist_val_acc',
        'optimiser',
        'epochs',
        'lr',
        'layers',
        'batch',
        'augmentation',
        'transf_learn',
        'img_size',
        'droplayers',
        'base_units',
        'dropbase',
        'test_acc',
        'test_f1',
        'test_precision',
        'test_recall',
        'cat_precision',
        'dog_precision',
        'cat_recall',
        'dog_recall',
        'cat_f1',
        'dog_f1',
        'cat_specificity',
        'dog_specificity',
        'filename',
    ]
    tobesaved = [
        round(history['loss'][-1] * 100, 2),
        round(history['val_loss'][-1] * 100, 2),
        round(history['accuracy'][-1] * 100, 2),
        round(history['val_accuracy'][-1] * 100, 2),
        model_config['optimizer'],
        model_config['epochs'],
        model_config['lr'],
        len(model_config['layers']),
        model_config['batch_size'],
        model_config['augmentation'],
        len(model_config['layers']) == 0,
        model_config['target_size'],
        droplayers_txt,
        model_config['base']['units'],
        model_config['base']['dropout'],
        round(acc * 100, 2),
        round(f1 * 100, 2),
        round(prec * 100, 2),
        round(rec * 100, 2),
        cats_metrix[0],
        dogs_metrix[0],
        cats_metrix[1],
        dogs_metrix[1],
        cats_metrix[2],
        dogs_metrix[2],
        cats_metrix[3],
        dogs_metrix[3],
        '{}.h*'.format(filename),
    ]
    return OrderedDict(zip(titles, tobesaved))


def summarize_diagnostics(filename, model_config, acc=None, f1=None, prec=None, rec=None):
    ''' Plot diagnostic learning curves '''
    droplayers = []
    for layer in model_config['layers']:
        if layer['dropout'] is not None:
            droplayers.append(str(layer['dropout']))
    droplayers_txt = ','.join(droplayers)
    if len(droplayers) == 0:
        droplayers_txt = None

    history = pickle.load(open('{}.hist'.format(filename), 'rb'))
    # plot loss
    pyplot.figure(filename)
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history['loss'], color='blue', label='train')
    pyplot.plot(history['val_loss'], color='orange', label='val')
    pyplot.grid(True)
    # pyplot.ylim(0, 1)
    # pyplot.xlabel('time (s)')
    # pyplot.ylabel('more nans')

    # Add title with info
    pyplot.plot([], [], ' ', label="lr={}".format(model_config['lr']))
    pyplot.plot([], [], ' ', label="Layers={}".format(len(model_config['layers'])))
    pyplot.plot([], [], ' ', label="batch={}".format(model_config['batch_size']))
    pyplot.plot([], [], ' ', label="optimizer={}".format(model_config['optimizer']))
    pyplot.plot([], [], ' ', label="augmentation={}".format(model_config['augmentation']))
    pyplot.plot([], [], ' ', label="transf_learn={}".format(len(model_config['layers'])==0))
    pyplot.plot([], [], ' ', label="img_size={}".format(model_config['target_size']))
    pyplot.plot([], [], ' ', label="droplayers={}".format(droplayers_txt))
    pyplot.plot([], [], ' ', label="dropbase={}".format(model_config['base']['dropout']))
    pyplot.plot([], [], ' ', label="val_acc={}".format(round(history['val_accuracy'][-1]*100, 2)))
    if acc is not None:
        pyplot.plot([], [], ' ', label="acc={}".format(round(acc*100, 2)))
    if f1 is not None:
        pyplot.plot([], [], ' ', label="f1={}".format(round(f1*100, 2)))
    if prec is not None:
        pyplot.plot([], [], ' ', label="prec={}".format(round(prec*100, 2)))
    if rec is not None:
        pyplot.plot([], [], ' ', label="rec={}".format(round(rec*100, 2)))
    pyplot.legend(bbox_to_anchor=(0., 1.3, 1., .102),
                  loc='lower left',
                  ncol=3,
                  borderaxespad=0.)

    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    if 'accuracy' in history:
        pyplot.plot(history['accuracy'], color='blue', label='train')
        pyplot.plot(history['val_accuracy'], color='orange', label='val')
    else:
        pyplot.plot(history['acc'], color='blue', label='train')
        pyplot.plot(history['val_acc'], color='orange', label='val')
    # pyplot.ylim(0.4, 1)
    pyplot.grid(True)

    pyplot.tight_layout()
    pyplot.show(block=True)
    pyplot.close()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def calculate_metrix(TP, FP, TN, FN):
    precision, recall, F1 = None, None, 0
    accuracy = round((TP + TN) / (TP + FN + TN + FP), 3)
    specificity = 1
    if TN + FP > 0:
        specificity = round(TN / (TN + FP), 3)
    if TP + FP > 0:
        precision = round(TP / (TP + FP), 3)
    if TP + FN > 0:
        recall = round(TP / (TP + FN), 3)
    if precision is not None and recall is not None:
        F1 = round((2 * precision * recall) / (precision + recall), 3)
    return precision, recall, F1, accuracy, specificity


def classification_report(model_config, model, eval_folder):
    ''' Get information about the scores based on the test images '''
    eval_cats, eval_dogs, eval_labels, eval_images = trim_images2(
        eval_folder,
        size=model_config['target_size'][0],
        namesufix='eval')

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    eval_it = eval_datagen.flow(
        eval_images,
        eval_labels,
        batch_size=model_config['batch_size'],
        shuffle=False)

    # from keras.preprocessing import image
    # img_width, img_height = 313, 220
    # test_image = image.load_img('mycat.jpg', target_size=(img_width, img_height))
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis=0)
    # test_image = test_image.reshape(img_width, img_height, 3)    # Ambiguity!
    # result = model.predict(test_image)
    # print(result)
    # return 0
    predicted = model.predict(eval_images)
    cat_TP, cat_FP, cat_TN, cat_FN = 0, 0, 0, 0
    dog_TP, dog_FP, dog_TN, dog_FN = 0, 0, 0, 0
    for num, pred in enumerate(predicted):
        pred_identity = 'dog'
        if pred[0] < pred[1]:
            pred_identity = 'cat'
        corr_identity = 'dog'
        if list(eval_labels[num]) == [0, 1]:
            corr_identity = 'cat'

        if corr_identity == 'cat' and pred_identity == 'cat':
            cat_TP += 1
        elif corr_identity == 'cat' and pred_identity != 'cat':
            cat_FN += 1
        elif corr_identity != 'cat' and pred_identity == 'cat':
            cat_FP += 1
        elif corr_identity != 'cat' and pred_identity != 'cat':
            cat_TN += 1

        if corr_identity == 'dog' and pred_identity == 'dog':
            dog_TP += 1
        elif corr_identity == 'dog' and pred_identity != 'dog':
            dog_FN += 1
        elif corr_identity != 'dog' and pred_identity == 'dog':
            dog_FP += 1
        elif corr_identity != 'dog' and pred_identity != 'dog':
            dog_TN += 1

        # print(pred_identity, corr_identity, pred, eval_labels[num])
    cats_metrix = calculate_metrix(cat_TP, cat_FP, cat_TN, cat_FN)
    dogs_metrix = calculate_metrix(dog_TP, dog_FP, dog_TN, dog_FN)

    # compile the model
    model.compile(optimizer=model_config['optimizer'], loss='categorical_crossentropy',
                  metrics=['acc', f1_m, precision_m, recall_m])

    # evaluate the model
    loss, accuracy, f1_score, precision, recall = model.evaluate(
        eval_it, steps=len(eval_it), verbose=1)
    return loss, accuracy, f1_score, precision, recall, cats_metrix, dogs_metrix


# define location of dataset
train_folder = 'machinelearning/train/'
test_folder = 'machinelearning/test/'
eval_folder = 'machinelearning/validation/'


tobesaved = []
for model_config in models_config:
    # filename = None
    filename = get_filename(model_config)
    print('\n----==== {} ====----'.format(filename))

    if not os.path.exists('{}.hist'.format(filename)):
        tr_cats, tr_dogs, tr_labels, tr_images = trim_images2(
            train_folder,
            size=model_config['target_size'][0],
            namesufix='train')
        tst_cats, tst_dogs, tst_labels, tst_images = trim_images2(
            eval_folder,
            size=model_config['target_size'][0],
            namesufix='test')
        print('Train cats: {}, dogs: {}'.format(tr_cats, tr_dogs))
        print('Test cats: {}, dogs: {}'.format(tst_cats, tst_dogs))
        model, history, acc = run_eval_harness(model_config)
        model.save('{}.h5'.format(filename))
        with open('{}.hist'.format(filename), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print('> Accuracy: %.3f' % (acc * 100.0))

    model = load_model('{}.h5'.format(filename))
    print(model.summary())
    loss, accuracy, f1_score, precision, recall, cats_metrix, dogs_metrix = classification_report(
        model_config, model, test_folder)
    tobesaved.append(get_diagnostics(
        filename,
        model_config,
        acc=accuracy,
        f1=f1_score,
        prec=precision,
        rec=recall,
        cats_metrix=cats_metrix,
        dogs_metrix=dogs_metrix))
    # print(tobesaved)
    summarize_diagnostics(
        filename, model_config,
        acc=accuracy,
        f1=f1_score,
        prec=precision,
        rec=recall)
with open('comparison.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=tobesaved[0].keys())
    writer.writeheader()
    for data in tobesaved:
        writer.writerow(data)
