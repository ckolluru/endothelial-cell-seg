import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
import argparse

class myUnet(object):

    def __init__(self,args):

        # Copy command line arguments into object's variables
        self.pre_train = args.pre_train
        self.train = args.train
        self.use_pre_train = args.use_pre_train
        self.test = args.test
        self.user = args.u

    def load_data(self):

        # Load training and test data for the network
        mydata = dataProcess(256, 256, self.user)
        imgs_train, imgs_train_labels, imgs_test = mydata.load_EC_data()
        return imgs_train, imgs_train_labels, imgs_test

    def load_pre_train_data(self):

        # Load pre-training train and validation data for the network
        mydata = dataProcess(256, 256, self.user)
        imgs_train, imgs_train_labels, imgs_validation, imgs_validation_labels = mydata.load_neuronal_data()
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

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

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
            model.fit_generator(zip(imgs_train, imgs_train_labels), validation_data=(zip(imgs_validation, imgs_validation_labels)), steps_per_epoch=20, validation_steps=10, epochs=20, verbose=1, shuffle=True, callbacks=[model_checkpoint])

        # Train the network 
        if self.train:
            print('-' * 30)
            print('Training on EC microscopy images')
            print('-' * 30)

            print('Loading training data (EC cells in microscopy images)')
            imgs_train, imgs_train_labels, imgs_test = self.load_data()
            print('Loaded training data (EC cells in microscopy images) \n')

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
            model.fit_generator(zip(imgs_train, imgs_train_labels), steps_per_epoch=20, epochs=20, validation_steps=10, verbose=1, shuffle=True, callbacks=[model_checkpoint])

        # Test the network
        if self.test:
            print('-' * 30)
            print('Test on unseen EC microscopy images')
            print('-' * 30)

            # Predict on the test images
            imgs_test_predictions = model.predict(imgs_test, batch_size=1, verbose=1)
            print('Predicted on test EC images')

            # Save predictions to the results folder
            print('Saving predictions on test images to results folder in the current directory')
            for i in np.arange(imgs_test_predictions.shape[0]):
                img_test_prediction = array_to_img(imgs_test_predictions[i,:,:])
                img_test_prediction.save('/home/' + self.user + '/endothelial-cell-seg/results/%d.jpg' %(i))

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
    args = parser.parse_args()

    print('\nUsing the following command line arguments:')
    print(args)
    print('\n')

    # Make an object of myUnet class
    myunet = myUnet(args)

    # Train and test the U-Net network as needed
    myunet.train_and_test()