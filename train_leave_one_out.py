# ------------------------------------------------------------------------------------------------------------
#   CNN training using leave-one-out 
#  
#   script used for training and testing a set of images using leave-one-out cross_validation classification.
#   This is an example script which is designed for our own experiments, where each training image is inside
#   a separate folder, and images modality names are consistent across the training set:
#
#   /TRAIN_FOLDER
#    ... /image_1
#         ... T1_name.nii.gz
#         ... FLAIR_name.nii.gz
#         ...
#         ... lesion_annotation.nii.gz
#    .../image_n
#         ... T1_name.nii.gz
#         ... FLAIR_name.nii.gz
#         ...
#         ... lesion_annotation.nii.gz
#
#
#  All model and image options are selected from the config.py file, so this script should work for different
#  databases 
#
#  Sergi Valverde 2017
# ------------------------------------------------------------------------------------------------------------


import os
from collections import OrderedDict
from base import *
from build_model_nolearn import cascade_model
from config import *


list_of_scans = os.listdir(options['train_folder'])
list_of_scans.sort()

modalities = options['modalities']
x_names = options['x_names']
y_names = options['y_names']

for scan in list_of_scans:

    # select training leaving-out the current scan
    train_x_data = {f: {m: os.path.join(options['train_folder'], f, n) for m, n in zip(modalities, x_names)}
                    for f in list_of_scans if f != scan}
    train_y_data = {f: os.path.join(options['train_folder'], f, y_names[0]) for f in list_of_scans if f != scan}

    # organize the experiment: save models and traiining images inside a predifined folder
    # network parameters and weights are stored inside test_folder/experiment/nets/
    # training images are stored inside test_folder/experiment/.train
    # final segmentation images are stored in test_folder/experiment 
    
    exp_folder = os.path.join(options['test_folder'], scan, options['experiment'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
        os.mkdir(os.path.join(exp_folder,'nets'))
        os.mkdir(os.path.join(exp_folder,'.train'))

    options['test_name'] = scan+ '_' + options['experiment'] + '.nii.gz'
    options['test_scan'] = scan 
               
    
    
    # train the model for the current scan 
    print "-------------------------------------------------------"
    print "training net for scan: %s (training size: %d)" %(scan, len(train_x_data.keys()))
    print "-------------------------------------------------------"

    # --------------------------------------------------
    # initialize the CNN
    # --------------------------------------------------
    options['weight_paths'] = os.path.join(options['test_folder'], options['test_scan'])
    model = cascade_model(options)

    # --------------------------------------------------
    # train the cascaded model
    # --------------------------------------------------
    model = train_cascaded_model(model, train_x_data, train_y_data,  options)


    # --------------------------------------------------
    # Testing the cascaded model 
    # --------------------------------------------------
    test_x_data = {scan: {m: os.path.join(options['test_folder'], scan, n) for m, n in zip(modalities, x_names)}}
    out_seg = test_cascaded_model(model, test_x_data, options)
