from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
from libtiff import TIFF
import re

def natural_keys(text):
    
    # Assumes that the path is of the form
    # home/cxk340/tensorflow/.../1.tif
    # Any changes in path will need a change here
    c = re.split('(/\d+)', text)
    return int(c[1].split('/')[1])

class myAugmentation(object):
    
    """
    A class used to augmentate image
    Firstly, read train image and label seperately, and then merge them together for the next process
    Secondly, use keras preprocessing to augmentate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, train_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/image",
                 label_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/label", 
                 merge_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/merge",
                 aug_merge_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/aug_merge",
                 aug_train_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/aug_train",
                 aug_label_path="/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/aug_label",
                 img_type="tif"):
        
        """
        Using glob to get all .img_type from path
        """

        self.train_imgs = glob.glob(train_path+"/*."+img_type)
        self.label_imgs = glob.glob(label_path+"/*."+img_type)

        #Sort based on file name which is a number
        self.train_imgs.sort(key=natural_keys)
        self.label_imgs.sort(key=natural_keys)

        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
                                    rotation_range=0.2,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    fill_mode='nearest')


    def Augmentation(self):

        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print ("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train+"/"+str(i+1)+"."+imgtype)
            img_l = load_img(path_label+"/"+str(i+1)+"."+imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:,:,2] = x_l[:,:,0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge+"/"+str(i+1)+"."+imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)
            savedir = path_aug_merge + "/" + str(i+1)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i+1))


    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

        """
        augmentate one image
        """
        datagen = self.datagen
        i = 0

        for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path
        for i in range(self.slices):
            path = path_merge + "/" + str(i+1)
            train_imgs = glob.glob(path+"/*."+self.img_type)
            savedir = path_train + "/" + str(i+1)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i+1)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:,:,2]#cv2 read image rgb->bgr
                img_label = img[:,:,0]

                #cv2.imwrite(path_train+"/"+str(i+1)+"/"+midname+"_train"+"."+self.img_type,img_train)
                #cv2.imwrite(path_label+"/"+str(i+1)+"/"+midname+"_label"+"."+self.img_type,img_label)

                cv2.imwrite(path_train+"/"+midname+"_train"+"."+self.img_type,img_train)
                cv2.imwrite(path_label+"/"+midname+"_label"+"."+self.img_type,img_label)

    def splitTransform(self):

        """
        split perspective transform images
        """
        #path_merge = "transform"
        #path_train = "transform/data/"
        #path_label = "transform/label/"
        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"
        train_imgs = glob.glob(path_merge+"/*."+self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:,:,2]#cv2 read image rgb->bgr
            img_label = img[:,:,0]
            cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
            cv2.imwrite(path_label+midname+"."+self.img_type,img_label)



class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path = "/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/aug_train",
                 label_path = "/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/train/aug_label", 
                 test_path = "/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/test",
                 npy_path = "/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/data/npydata",
                 img_type = "tif"):

        """
        
        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        labs = glob.glob(self.label_path+"/*."+self.img_type)

        print(len(imgs), len(labs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)

        for imgname, labname in zip(imgs, labs):
            midname = imgname[imgname.rindex("/")+1:]
            midlabname = labname[labname.rindex("/")+1:]
            img = load_img(self.data_path + "/" + midname,grayscale = True)
            label = load_img(self.label_path + "/" + midlabname,grayscale = True)
            img = img_to_array(img)
            label = img_to_array(label)
            #img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #img = np.array([img])
            #label = np.array([label])
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        print(self.npy_path + '/imgs_train.npy')
        print(np.size(imgdatas))
        print(np.size(imglabels))
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = glob.glob(self.test_path+"/*."+self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            print(imgname)
            midname = imgname[imgname.rindex("/")+1:]
            img = load_img(self.test_path + "/" + midname,grayscale = True)
            img = img_to_array(img)
            #img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            #img = np.array([img])
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-'*30)
        print('load train images...')
        print('-'*30)
        imgs_train = np.load(self.npy_path+"/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        #mean = imgs_train.mean(axis = 0)
        #imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train,imgs_mask_train

    def load_test_data(self):
        print('-'*30)
        print('load test images...')
        print('-'*30)
        imgs_test = np.load(self.npy_path+"/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        #mean = imgs_test.mean(axis = 0)
        #imgs_test -= mean
        
        for i in range(imgs_test.shape[0]):
            img = imgs_test[i]
            img = array_to_img(img)
            img.save("/home/cxk340/tensorflow/unet_batch_mode_endothelial_cell_segmentation/test_examples/%d.jpg"%(i))
    
        return imgs_test

#srun --x11 -p gpuk40 --gres=gpu:2 -N 1 -c 12 --mem=40g --time=100:00:00 --pty /bin/bash
# module load singularity tensorflow

if __name__ == "__main__":

    aug = myAugmentation()
    aug.Augmentation()
    aug.splitMerge()
    #aug.splitTransform()
    mydata = dataProcess(256,256)
    mydata.create_train_data()
    mydata.create_test_data()
    imgs_train,imgs_mask_train = mydata.load_train_data()
    print (imgs_train.shape,imgs_mask_train.shape)
