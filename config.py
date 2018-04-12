# configuration file
# --------------------------------------------------------------------------------
#
# Sergi Valverde 2017
# --------------------------------------------------------------------------------

options = {}

# --------------------------------------------------
# Database options
# --------------------------------------------------

# path to training image folder. In all our experiments, training images were
# inside a folder and image modalities were consistent across training images.
# In the case of leave-one-out experiments, the same folder is used
options['train_folder'] = '/mnt/DATA/w/CNN_NI/test_set2'
options['test_folder'] = '/path/to/testing/data'

# image modalities used (T1, FLAIR, PD, T2, ...)
options['modalities'] = ['T1', 'FLAIR']

# image modalities nifti file names in the same order of options['modalities']
options['x_names'] = ['T1.nii.gz', 'FLAIR.nii.gz']

# lesion annotation nifti file names
options['y_names'] = ['lesion.nii.gz']

# --------------------------------------------------
# Experiment options
# --------------------------------------------------

# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'test_CNN'

# use data generators (slow but necessary on large datasets)
# When used, training data is loaded every consecutively when needed.
# the numbe of training images loaded in paralel can be controlled by
# the parameter load_data_batch.
# NOTE: Also the options['patience'] should be increased if using data
# generators.

options['data_generator'] = True
options['load_data_batch'] = 1

# minimum threshold used to select candidate voxels for training. Note that
# images are normalized to 0 mean 1 standard deviation before thresholding.
# So a value of t > 0.5 on FLAIR is reasonable in most cases to extract all
# WM lesion candidates
options['min_th'] = 0.5

# randomize training features before fitting the model.
options['randomize_train'] = True

# Select between pixel-wise or fully-convolutional training models.
# Although implemented, fully-convolutional
# models have been not tested with this cascaded model
options['fully_convolutional'] = False

# debug mode
options['debug'] = True

# --------------------------------------------------
# model parameters
# --------------------------------------------------

# 3D patch size. So, far only implemented for 3D CNN models.
options['patch_size'] = (11, 11, 11)

# file paths to store the network parameter weights.
# These can be reused for posterior use.
options['weight_paths'] = None
options['load_weights'] = False

# percentage of the training vector that is going to be used to
# validate the model during training
options['train_split'] = 0.25

# maximum number of epochs used to train the model
options['max_epochs'] = 200

# maximum number of epochs without improving validation before
# stopping training
options['patience'] = 75

# Training / Testing batch size
options['batch_size'] = 128

# verbositiy of CNN messaging
# 0 = silent, 1 = progress bar, 2 = one line per epoch.
options['net_verbose'] = 1

# post-processing binary threshold.
options['t_bin'] = 0.5

# post-processing minimum lesion size of soutput candidates
options['l_min'] = 10

# minimum resolution the model can handle in ml.
options['min_error'] = 0.5
