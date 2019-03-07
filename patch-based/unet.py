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
from keras.losses import categorical_crossentropy
from keras import backend as keras
from data import *
import argparse
from keras.utils import plot_model
import keras
from keras import backend as K

K.set_session(sess)

class myUnet(object):

    def __init__(self,args):

        # Copy command line arguments into object's variables
        self.user = args.u
        self.mydata = dataProcess(32, 32, self.user)

    def load_data(self, dataset_select):

        # Load training and test data for the network
        imgs_train, imgs_train_labels, imgs_test = self.mydata.load_EC_data(dataset_select)
        return imgs_train, imgs_train_labels, imgs_test

    def get_unet(self):

        inputs = Input((32, 32, 1))        

        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(inputs)
        drop1 = Dropout(0.1)(conv1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(drop1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool1)
        drop2 = Dropout(0.1)(conv3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(drop2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool2)
        drop3 = Dropout(0.1)(conv5)
        conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(drop3)
        up1 = UpSampling2D(size=(2,2))(conv6)
        
        merge1 = concatenate([conv4, up1], axis=3)

        conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge1)
        drop4 = Dropout(0.1)(conv7)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(drop4)
        up2 = UpSampling2D(size=(2,2))(conv8)

        merge2 = concatenate([conv2, up2], axis = 3)

        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge2)
        drop5 = Dropout(0.1)(conv9)
        conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(drop5)
        conv11 = Conv2D(2, 3, activation = 'softmax', padding = 'same', kernel_initializer = 'glorot_uniform')(conv10)

        model = Model(inputs = inputs, outputs = conv11)

        model.compile(optimizer = Adam(lr = 1e-4), loss = categorical_crossentropy,  metrics = ['accuracy'])

        model.summary()

        return model

    def train_and_test(self):

        print('-' * 30)
        print('Training on EC microscopy images')
        print('-' * 30)

        # 0 indicates train with odd, test with even, 1 for opposite
        dataset_select = 1

        print('Loading training data (EC cells in microscopy images)')
        imgs_train, imgs_train_labels, imgs_test = self.load_data(dataset_select)
        imgs_train = np.expand_dims(imgs_train, axis = -1)
        imgs_test = np.expand_dims(imgs_test, axis = -1)
        print('Loaded training data (EC cells in microscopy images) \n')

        # Get the neural network architecture
        print('Loading network architecture')
        model = self.get_unet()
        print('Loaded network architecture \n')

        # Create a checkpoint to save the network weights to a file. Loss on the training set will be monitored
        if dataset_select == 0:
            model_checkpoint = ModelCheckpoint('unet_train_odd.hdf5', monitor='loss',verbose=1, save_best_only=True)
        else:
            model_checkpoint = ModelCheckpoint('unet_train_even.hdf5', monitor='loss', verbose=1, save_best_only=True)

        # Fit the model to the training data
        print('Fitting model to train dataset')
        model.fit(imgs_train, imgs_train_labels, batch_size=32, epochs=150, verbose=1, callbacks=[model_checkpoint])

        # Predict on the test images
        imgs_test_predictions = model.predict(imgs_test, batch_size=5000, verbose=1)
        print('Predicted on test EC images')

        # Save predictions to the results folder
        print('Saving predictions on test images to results folder in the current directory')
        self.mydata.save_test_predictions(imgs_test_predictions, self.user)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='U-Net for EC cell segmentation')
    parser.add_argument('--u', default='cxk340', type=str,
                        help='Case username (example cxk340).')
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

    plot_model(model, to_file='model.png')