# Endothelial Cell Segmentation

Software allows for segmenting cells in the endothelial layer of cornea using deep learning techniques (currently U-Net). 

## Getting Started

Graphical Processing Units (GPU) cards are required for quick neural network training and testing. This code is designed to work well with the GPU nodes (systems) at Case Western Reserve University's High Performance Cluster (HPC). Please create an account for yourself on the cluster under Dr. Wilson's account. Send an email to hpc-support@case.edu for the same and cc Dr. Wilson. All instructions regarding using the cluster are at: https://sites.google.com/a/case.edu/hpc-upgraded-cluster/home. Sign up for their interactive seminars during the semester: https://canvas.case.edu/courses/3014. This step is required if you want to train and test the neural networks and modify the code. If you just want to look at the code for now, you don't have to do this.

## Accessing the High Performance Cluster (HPC) and getting things ready

Files on the HPC can be accessed using WinSCP software. This software allows one to copy files to/from the cluster and edit existing files on the cluster.
![winscp](https://user-images.githubusercontent.com/8373968/43086116-37fa2da4-8eba-11e8-99f3-814d016258a5.PNG)

When connected, you will see a directory listing of your laptop/PC on the left and your folder on the HPC on the right. You can make new folders, delete on each side. Drag and drop from one side to another to copy. Every user gets a folder on the HPC to work in (something like `/home/CaseID/`).

![winscp2](https://user-images.githubusercontent.com/8373968/43086524-2dc8a36e-8ebb-11e8-9834-119aa44a8388.PNG)

Next, to run commands, you can use MobaXTerm, PuTTY or similar software to connect to the HPC (rider.case.edu), with an SSH type connection. MobaXTerm will look like this:

![mobaxterm](https://user-images.githubusercontent.com/8373968/43086282-9c48a754-8eba-11e8-9a04-4fcec3919d08.PNG)

Enter your Case credentials to start a session. After logging in, you will see something like this:

`[cxk340@hpc3 ~]$`

You are now logged into the HEAD node (hpc3). These nodes are general purpose nodes and should not be used for any scientific computing. These nodes can only be used to request a computing node (CPU/GPU) in interactive mode (see below) or submit jobs (see below). Essentially, after you have an SSH connection to the HPC, there are two ways to run your software:

- Interactive mode: `#Interactive mode` You request the resource you need (CPU/GPU, RAM, cores, time etc.), get the resource, and then get a shell to that node to run files etc.
- Job (batch) mode: `#Job (batch) mode`  You submit a job script that contains commands, and the script will be run when the resources you request (CPU/GPUs) are available. The script can contain as many commands as you like. You don't have to wait if the resources you need aren't readily available in this case.

I'll demonstrate the `[interactive mode](#interactive-mode)` now. First, we request a GPU node (essentially just a PC which has some graphic cards on it):
```
srun --x11 -p gpu -C gpup100 -N 1 -c 12 --gres=gpu:2 --mem=186g --time=72:00:00 --pty /bin/bash
```
This requests a NVIDIA P100 GPU with 1 CPU (N), 12 cores (-c), 2 P100 cards (gpu:2), 186 GB of RAM (mem) for 72 hours (time). It opens a command line interface (/bin/bash) to such a system. IMPORTANT: You should ALWAYS free resources back to the cluster when you are done using them. To do this, run the **'exit'** command before closing your session.

There are two ways to run deep learning software:
1. Use the readily available tensorflow module in the HPC. For that, we run the following:
```
export OMP_NUM_THREADS=1
module swap intel gcc
module load tensorflow
```
You now have access to a GPU system with tensorflow and keras libraries loaded. If you want to see the versions:
```
python
import tensorflow as tf
tf.__version__
import keras as K
K.__version__
```
Currently, Tensorflow 1.4.0 and Keras 2.1.3 are installed in the tensorflow module. Modules are maintained by the HPC staff, and might not contain all the python libraries that we might need. Request for adding new libraries to the module can be done by writing to hpc-support@case.edu.

2. We can also use a singularity (a software like Docker) image that we (our lab) can control. New python libraries can be installed on this image which can then be executed on the cluster.
```
module load gcc cuda singularity
singularity shell --nv /home/cxk340/singularity_image/keras_tf.img
```

Both these methods will give you a shell that can run python with the deep learning libraries (keras, tensorflow) available. We can now run our software. Currently, either option 1 or 2 can be used to run this code.

## Downloading the code

The code can be downloaded straight to your folder on the HPC. To do this, navigate to your home folder (`/home/caseID/`) and then type:

```
module load git
git clone git@github.com:ckolluru/endothelial-cell-seg.git
```

You will see a folder named endothelial-cell-seg-master in your (`/home/caseID/`) folder. 

## Folder organization

The repository is organized into the following folders:

```
\data
\results
code files
```

The code allows for pre-training the network to segment neurons in electron microscopy images. This is an optional step, and can be useful since the electron microscopy images look similar to our endothelial cell image dataset. Moreover, the network, U-Net was originally proposed to segment such images. Training and testing images and labels for this dataset are in the `\neuronal` folder. 

Images related to the endothelial cells are present in the `\EC` folder. Both `\EC` and `\neuronal` folders have the following folder organization. Both consist of training and testing images (in `\train` and `\test` folders respectively). `\neuronal` images were cropped to (256, 256) to match the `\EC` image size. The cropped `\neuronal` images are in `\train_crop` and `\test_crop` respectively. 

For `\EC`, `\train` folder consists of `\image` and `\label` folders within it. `\test` only consists of the test images currently. `\results` folder in the parent directory consists the segmentation result on the `\test` images. 

For `\neuronal`, we currently have 15 `\train_crop` images and 15 `\test_crop` images. 

Since the training dataset only consists of 34 training images (in case of `\EC`) and 15 training images (in case of `\neuronal`), we augment the datasets using a combination of rotation steps and translations. Routines to perform the augmentation are called within `data.py` file. The Keras `ImageDataGenerator` function is used. Images that need to be augmented are usually placed inside an additional `\all` folder. Keras only works this way (it needs a path to a folder that has a folder containing the images).

`data.py` consists routines to read and write from the `\data` folder. `unet.py` runs the training and testing of the neural network and saves the network weights to a weight file. 

The pre-trained and trained network weights are stored in unet_pretrain.hdf5 and unet_train.hdf5 respectively. `unet-batch-job.slurm` is a script that runs unet.py in `[job (batch) mode](#job-(batch)-mode)`. In this case, you don't have to wait for the resources (GPU nodes) if they are not readily available. You can run the command `squeue -u <caseID>` to see the status of your jobs.

Note: Please use numbers as filenames for the images in all folders. Software supports images in BMP format only.

### Running unet.py

The following command provides information on how to use the unet.py. This is the only file that needs to be run.
```
python unet.py --help
```

Command line arguments can be sent to `unet.py`. For example,

--pre_train 1 will pre-train the network with the `\neuronal` images. --pre_train 0 will not. (Default 1) <br />
--train 1 will train the network with the `\EC` images. --train 0 will not. (Default 1) <br />
--use_pre_train 1 will load the pre-trained network weights prior to network training on `\EC` images. (Default 1) <br />
--test 1 will run the trained network (looks for the weight file from the trained network) on all `\EC` test images. (Default 1) <br />
--u cxk340 will consider that the code and data exist in the folder /home/cxk340. **Other users**, need to specify their case ID instead or change line 232 in unet.py file. (Default 'cxk340') <br />
--trial 0 is a way to run the code multiple times, results will be stored in a folder named as `results_xy` where `xy` represents the number of trial. For example, you can have a for loop that iterates from 1 to 10 and the results will be written in folders from `results_1` to `results_10`. 0 will simply write to the `results` folder. (Default 0) <br />

Hence, in order to test the software on a completely new image, the image can be cropped (to 256, 256) and copied over to `\data\EC\test`. JPG format is supported. The network can be fully trained (including pre-training) by running `python unet.py`. With no command line arguments, the code will pre-train, train, and predict on all test images in `\data\EC\test` folder. The results will be in the `\results` folder. If you already have a trained network (the corresponding weight file, unet_train.hdf5 in your folder), testing can be done by just running:

```
python unet.py --pre_train 0 --train 0 --test 1 --u cxk340
```

## Built With

* [Tensorflow](http://www.tensorflow.org/) - Neural network training, testing backend
* [Keras](https://keras.io/) - Deep learning front-end API

## Authors

* **Chaitanya Kolluru** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to @zhixuhao (whose U-Net code was used as a starting point)
* Hat tip to @lingchen42 (whose instructions were helpful to create a singularity image for the HPC at CWRU)
* Thanks to CWRU HPC staff (especially Sanjaya Gajurel, Daniel Balague Guardia, Emily Dragowski and Hadrian Djohari)