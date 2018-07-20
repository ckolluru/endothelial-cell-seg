from unet import *
from data import *

mydata = dataProcess(256,256)

imgs_test = mydata.load_test_data()

myunet = myUnet()

model = myunet.get_unet()

model.load_weights('/home/cxk340/unet_batch_mode_endothelial_cell_segmentation/unet.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size = 1, verbose=1)

np.save('/home/cxk340/unet_batch_mode_endothelial_cell_segmentation/results/imgs_mask_test.npy', imgs_mask_test)

print("array to image")
imgs = np.load('/home/cxk340/unet_batch_mode_endothelial_cell_segmentation/results/imgs_mask_test.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    img = array_to_img(img)
    img.save("/home/cxk340/unet_batch_mode_endothelial_cell_segmentation/results/%d.jpg"%(i))