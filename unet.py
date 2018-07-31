import os 
import numpy as np
import tensorflow as tf
import random as rn

# Things to reproduce results, although Keras on multiple GPU is not reproducible.
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as keras
from data import *
import argparse
from keras.utils import plot_model
import keras
from keras import backend as K

K.set_session(sess)

def weighted_binary_crossentropy(y_true, y_pred):

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	binary_crossentropy = K.binary_crossentropy(y_true_f, y_pred_f)
	weighted_vector = y_true_f * 0.21 + (1. - y_true_f) * 0.79
	weighted_binary_crossentropy_loss = weighted_vector * binary_crossentropy

	return K.mean(weighted_binary_crossentropy_loss)

class myUnet(object):

    def __init__(self,args):

        # Copy command line arguments into object's variables
        self.pre_train = args.pre_train
        self.train = args.train
        self.use_pre_train = args.use_pre_train
        self.test = args.test
        self.user = args.u
        self.mydata = dataProcess(256, 256, self.user)
        self.trial = args.trial

        # Version of the keras library is different in the HPC's tensorflow module and our singularity image
        # This is causing an issue with the data augmentation step (ImageDataGenerator)
        self.keras_version = keras.__version__

    def load_data(self):

        # Load training and test data for the network
        imgs_train, imgs_train_labels, imgs_test = self.mydata.load_EC_data(self.keras_version)
        return imgs_train, imgs_train_labels, imgs_test

    def load_pre_train_data(self):

        # Load pre-training train and validation data for the network
        imgs_train, imgs_train_labels, imgs_validation, imgs_validation_labels = self.mydata.load_neuronal_data(self.keras_version)
        return imgs_train, imgs_train_labels, imgs_validation, imgs_validation_labels

    def get_unet(self):

        inputs = Input((256, 256, 1))        

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

        model = Model(inputs = inputs, outputs = conv9)

        model.compile(optimizer = Adam(lr = 1e-4), loss = weighted_binary_crossentropy,  metrics = ['accuracy'])

        model.summary()

        return model

    def train_and_test(self):

        # Pre-training
        if self.pre_train:
            print('-' * 30)
            print('Pre-training on neuronal cell cryo-EM images')
            print('-' * 30)
            
            print('Loading pre-train data (Neuron cells in cryo-EM images)')
            imgs_train, imgs_train_labels, imgs_validation, imgs_validation_labels = self.load_pre_train_data()
            print('Loaded pre-train data (Neuron cells in cryo-EM images) \n')

            # Get the neural network architecture
            print('Loading network architecture')
            model = self.get_unet()
            print('Loaded network architecture \n')

            # Create a checkpoint to save the network weights to a file. Loss on the validation set will be monitored
            model_checkpoint = ModelCheckpoint('unet_pretrain.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

            # Fit the model to the pre-train data
            print('Fitting model to pre-train dataset')
            print('Data generator filenames')
            print(imgs_train.filenames)
            model.fit_generator(zip(imgs_train, imgs_train_labels), validation_data=(zip(imgs_validation, imgs_validation_labels)), steps_per_epoch=20, validation_steps=20, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint])

        # Train the network 
        if self.train:
            print('-' * 30)
            print('Training on EC microscopy images')
            print('-' * 30)

            print('Loading training data (EC cells in microscopy images)')
            imgs_train, imgs_train_labels, imgs_test = self.load_data()
            print('Loaded training data (EC cells in microscopy images) \n')

            # print('Using class weights since it is an unbalanced dataset, more cell pixels than cell border pixels')
            # # Mean over all training labels is used to compute the following values. 21 percent of cell border pixels, 79 percent other.
            # class_weight_dict={0:0.79, 1:0.21}

            # Get the neural network architecture
            print('Loading network architecture')
            model = self.get_unet()
            print('Loaded network architecture \n')

            # Create a checkpoint to save the network weights to a file. Loss on the validation set will be monitored
            model_checkpoint = ModelCheckpoint('unet_train.hdf5', monitor='loss',verbose=1, save_best_only=True)

            if self.use_pre_train:
                print('Loading weights from the pre-trained network')
                model.load_weights('/home/' + self.user + '/endothelial-cell-seg/unet_pretrain.hdf5')
                print('Loaded weights from the pre-trained network \n')

            # Fit the model to the training data
            print('Fitting model to train dataset')
            print('TODO: Fix steps_per_epoch and validation_steps - make sure all training examples are used')
            model.fit_generator(zip(imgs_train, imgs_train_labels), steps_per_epoch=20, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint])

        # Test the network
        if self.test:
            print('-' * 30)
            print('Test on unseen EC microscopy images')
            print('-' * 30)

            # Get the neural network architecture
            print('Loading network architecture')
            model = self.get_unet()
            print('Loaded network architecture \n')

            # Load the network weights
            print('Loading network weights from unet_train.hdf5 file')
            model.load_weights('/home/' + self.user + '/endothelial-cell-seg/unet_train.hdf5')
            print('Loaded weights from the pre-trained network \n')

            # Load training and testing images (only testing is eventually used here)
            print('Loading testing data (EC cells in microscopy images)')
            imgs_train, imgs_train_labels, imgs_test = self.load_data()
            print('Loaded testing data (EC cells in microscopy images) \n')

            # Predict on the test images
            imgs_test_predictions = model.predict(imgs_test, batch_size=1, verbose=1)
            print('Predicted on test EC images')

            # Save predictions to the results folder
            print('Saving predictions on test images to results folder in the current directory')
            self.mydata.save_test_predictions(imgs_test_predictions, self.user, self.trial)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='U-Net for EC cell segmentation')
    parser.add_argument('--pre_train', default=1, type=int,
                        help='Pre-train the network to segment neurons in cryo-EM images (0/1, default 1). Images are usually (512, 512) but have been cropped to (256, 256). Step results in a weights file unet_pretrain.hdf5')
    parser.add_argument('--train', default=1, type=int,
                        help='Train the network to segment EC cells in microscopy images (0/1, default 1). Images are (256, 256). Step results in a weights file unet_train.hdf5')
    parser.add_argument('--use_pre_train', default=1, type=int,
                        help='Use the network weights from pre-training as a starting point (0/1, default 1). Needs a weights file in the current directory called unet_pretrain.hdf5')
    parser.add_argument('--test', default=1, type=int,
                        help='Segment EC cells in a held-out test set of microscopy images (0/1, default 1)')
    parser.add_argument('--u', default='cxk340', type=str,
                        help='Case username (example cxk340). Download files from Github to your folder on the HPC. Data read/write will then be done from the corresponding folders.')
    parser.add_argument('--trial', default=0, type=int,
                        help='Useful if you want to run the software multiple times to check for reproducibility of results')
    args = parser.parse_args()

    print('\nUsing the following command line arguments:')
    print(args)
    print('\n')

    # Make an object of myUnet class
    myunet = myUnet(args)

    # Train and test the U-Net network as needed
    myunet.train_and_test()

    # Draw model architecture to a file (can be used to ensure that the layers are connected properly)
    model = myunet.get_unet()

    if myunet.keras_version is not '2.1.3':
    	plot_model(model, to_file='model.png')