from unet import *
from data import *
import pydot
import graphviz
from keras.utils import plot_model

myunet = myUnet()

model = myunet.get_unet()

plot_model(model, to_file='model.png')



