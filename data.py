from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import glob
import re
import math

def natural_keys(text):

    # Assumes that the path is of the form /home/<case ID>/.../1.jpg
    c = re.split('(/\d+)', text)
    print(c)
    return int(c[1].split('/')[1])

class dataProcess(object):

    def __init__(self, img_height, img_width, username):
        self.img_height = img_height
        self.img_width = img_width
        self.username = username

    def load_EC_data(self, keras_version):

        # Augment the training images and labels, because 34 training images is a small number of images for deep learning methods
        datagen_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range= (0.05 * 360. / (2*math.pi)) if keras_version == '2.1.3' else 0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            rescale=1./255)
        
        image_datagen = ImageDataGenerator(**datagen_args)
        labels_datagen = ImageDataGenerator(**datagen_args)

        # Load training images and labels from the respective directories
        image_generator = image_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/EC/train/image/', color_mode='grayscale', class_mode=None, seed=1, batch_size=8)
        labels_generator = labels_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/EC/train/label/', color_mode='grayscale', class_mode=None, seed=1, batch_size=8)

        # Find all test images from the data folder
        self.test_imgs_list = glob.glob('/home/' + self.username + '/endothelial-cell-seg/data/EC/test/*.bmp')

        # Sort so that the list is 1.jpg, 2.jpg etc. and not 1.jpg, 11.jpg etc.
        self.test_imgs_list.sort(key=natural_keys)

        # Read test image files and load them into a numpy array
        imgs_test_stack = np.zeros((len(self.test_imgs_list), 256, 256))
        for i in np.arange(len(self.test_imgs_list)):
            imgs_test_stack[i,:,:] = img_to_array(load_img(self.test_imgs_list[i]))[:,:,1]/255
        
        print('%d EC test images found' %(len(self.test_imgs_list)))
        imgs_test_stack = np.expand_dims(imgs_test_stack, axis=-1)

        return image_generator, labels_generator, imgs_test_stack

    def load_neuronal_data(self, keras_version):

        # The neuron image dataset is (512,512) in size. The images are cropped to (256, 256) and saved in train_crop and test_crop folders.

        # Augment the training images and labels, because 30 training images is a small number of images for deep learning methods.
        datagen_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range= (0.05 * 360. / (2*math.pi)) if keras_version == '2.1.3' else 0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            rescale=1./255)
        
        image_datagen = ImageDataGenerator(**datagen_args)
        labels_datagen = ImageDataGenerator(**datagen_args)

        # Load training images and labels from the respective directories
        image_generator = image_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/neuronal/train_crop/image/', color_mode='grayscale', class_mode=None, seed=1, batch_size=8)
        labels_generator = labels_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/neuronal/train_crop/label/', color_mode='grayscale', class_mode=None, seed=1, batch_size=8)

        # Load testing images and labels from the respective directories
        image_validation_generator = image_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/neuronal/test_crop/image/', color_mode='grayscale', class_mode=None, seed=1, batch_size=32)
        labels_validation_generator = labels_datagen.flow_from_directory('/home/' + self.username + '/endothelial-cell-seg/data/neuronal/test_crop/label/', color_mode='grayscale', class_mode=None, seed=1, batch_size=32)
        
        return image_generator, labels_generator, image_validation_generator, labels_validation_generator

    def save_test_predictions(self, imgs_test_predictions, username, trial):

        for i in np.arange(imgs_test_predictions.shape[0]):
            img_test_prediction = array_to_img(imgs_test_predictions[i,:,:])
            filename_start_index = self.test_imgs_list[i].rfind('/')
            if trial == 0:
                img_test_prediction.save('/home/' + username + '/endothelial-cell-seg/results/%s' %(self.test_imgs_list[i][filename_start_index+1:]))
            else:
                img_test_prediction.save('/home/' + username + '/endothelial-cell-seg/results_' + str(trial) + '/%s' %(self.test_imgs_list[i][filename_start_index+1:]))


if __name__ == '__main__':

    mydata = dataProcess(256,256, 'cxk340')