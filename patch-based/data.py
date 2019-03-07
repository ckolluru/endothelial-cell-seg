from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import glob
import re
import math
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import scipy.io

def natural_keys(text):

    # Assumes that the path is of the form /home/<case ID>/.../1.jpg
    c = re.split('(/\d+)', text)
    return int(c[1].split('/')[1])

class dataProcess(object):

    def __init__(self, img_height, img_width, username):
        self.img_height = img_height
        self.img_width = img_width
        self.username = username

    def load_EC_data(self, dataset_select):

        if dataset_select == 0:
            self.train_imgs_list = glob.glob('/home/' + self.username + '/anna/data/odd/images/*.png')
            self.train_labels_list = glob.glob('/home/' + self.username + '/anna/data/odd/labels/*.png')

            self.test_imgs_list = glob.glob('/home/' + self.username + '/anna/data/even/images/*.png')
            self.test_labels_list = glob.glob('/home/' + self.username + '/anna/data/even/labels/*.png')
        else:
            self.train_imgs_list = glob.glob('/home/' + self.username + '/anna/data/even/images/*.png')
            self.train_labels_list = glob.glob('/home/' + self.username + '/anna/data/even/labels/*.png')

            self.test_imgs_list = glob.glob('/home/' + self.username + '/anna/data/odd/images/*.png')
            self.test_labels_list = glob.glob('/home/' + self.username + '/anna/data/odd/labels/*.png')

        # Sort so that the list is 1.jpg, 2.jpg etc. and not 1.jpg, 11.jpg etc.
        self.train_imgs_list.sort(key=natural_keys)
        self.test_imgs_list.sort(key=natural_keys)
        self.train_labels_list.sort(key=natural_keys)
        self.test_labels_list.sort(key=natural_keys)

        imgs_train_stack = np.zeros((15000, 32, 32))
        labels_train_stack = np.zeros((15000, 32, 32, 2))

        for i in np.arange(len(self.train_imgs_list)):
            start_index = i*1000
            stop_index = start_index + 1000
            current_img = img_to_array(load_img(self.train_imgs_list[i]))[:,:,1]
            current_label = img_to_array(load_img(self.train_labels_list[i]))[:,:,1]/255

            imgs_train_stack[start_index:stop_index,:,:] = image.extract_patches_2d(current_img, patch_size=(32, 32), max_patches= 1000, random_state = 20*i)
            labels_train_stack[start_index:stop_index,:,:, 0] = image.extract_patches_2d(current_label, patch_size=(32, 32), max_patches= 1000, random_state = 20*i)

        labels_train_stack[:,:,:,1] = 1 - labels_train_stack[:,:,:,0]

        # Debugging
        # scipy.io.savemat('labels.mat',{'labels_train_stack':np.squeeze(labels_train_stack[14000:,:,:,1])})
        # scipy.io.savemat('images.mat',{'images_train_stack':np.squeeze(imgs_train_stack[14000:,:,:])})
        
        # 59517 = (152+32-1) * (388+32-1)
        # images are (388, 152) in size
        imgs_test_stack = np.zeros((59517*15, 32, 32))

        for i in np.arange(len(self.test_imgs_list)):
            start_index = i*59517
            stop_index = start_index + 59517
            current_test = img_to_array(load_img(self.test_imgs_list[i]))[:,:,1]
            current_test_np = np.asarray(current_test)
            current_test_padded = np.pad(current_test_np, ((16, 16), (16, 16)), mode='symmetric')

            # make patches with stride 1 pixel, anna had 3 pixels, should get similar results.
            imgs_test_stack[start_index:stop_index,:,:] = image.extract_patches_2d(current_test_padded, patch_size=(32, 32))

        return imgs_train_stack, labels_train_stack, imgs_test_stack

    def save_test_predictions(self, imgs_test_predictions, username):

        for i in np.arange(len(self.test_imgs_list)):
            start_index = i*59517
            stop_index = start_index + 59517            
            predictions_one_image = np.squeeze(imgs_test_predictions[start_index:stop_index, :, :, 0])

            prediction_full_image = image.reconstruct_from_patches_2d(predictions_one_image, (420, 184))
            prediction_full_image = np.expand_dims(prediction_full_image, axis = -1)

            #crop to original size
            prediction_full_image = prediction_full_image[16:388+16, 16:152+16, :]
            img_prediction = array_to_img(prediction_full_image)

            filename_start_index = self.test_imgs_list[i].rfind('/')
            img_prediction.save('/home/' + username + '/anna/results/%s' %(self.test_imgs_list[i][filename_start_index+1:]))

if __name__ == '__main__':

    mydata = dataProcess(32, 32, 'cxk340')