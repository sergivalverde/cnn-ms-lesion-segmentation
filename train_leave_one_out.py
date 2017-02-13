import os
from collections import OrderedDict
from data_utils import *
from build_model_nolearn import cascade_model

# --------------------------------------------------
# hyper-parameters
# --------------------------------------------------
options = {}
options['train_folder'] = '/mnt/DATA/w/CNN/images/VH_all'
options['test_folder'] = '/mnt/DATA/w/CNN/images/VH_all'  # leave-one-out
options['experiment'] = 'test_CNN'
options['min_th'] = 1
options['patch_size'] = (11,11,11)
options['randomize_train'] = True
options['fully_convolutional'] = False
options['modalities'] = ['T1', 'FLAIR']
options['x_names'] = ['T1.nii.gz', 'FLAIR.nii.gz']
options['y_names'] = ['lesion_bin.nii.gz']

# --------------------------------------------------
# model parameters
# --------------------------------------------------
options['weight_paths'] = None
options['experiment'] = 'VH1_lou'
options['channels'] = 2
options['train_split'] = 0.25
options['max_epochs'] = 200
options['patience'] = 25
options['batch_size'] = 50000
options['t_bin'] = 0.8
options['l_min'] = 20
# --------------------------------------------------
# TRAIN/TEST using leave-one-out training 
# -------------------------------------------------- 

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

    # configure options for testing
    options['test_name'] = scan+ '_' + options['experiment'] + '.nii.gz'
    options['test_scan'] = scan 

    # select test data
    test_x_data = {}
    test_x_data[test_scan] = {}
    test_x_data[test_scan]['FLAIR'] = os.path.join(options['test_folder'], scan, 'FLAIR.nii.gz')
    test_x_data[test_scan]['T1'] = os.path.join(options['test_folder'], scan, 'T1.nii.gz')

    # organize the experiment: save models and traiining images inside a predifined folder
    # network parameters and weights are stored inside test_folder/experiment/nets/
    # training images are stored inside test_folder/experiment/.train
    # final segmentation images are stored in test_folder/experiment 
    
    exp_folder = os.path.join(options['test_folder'], scan, options['experiment'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
        os.mkdir(os.path.join(exp_folder,'nets'))
        os.mkdir(os.path.join(exp_folder,'.train'))
    
    
    # train the model for the current scan 
    print "-------------------------------------------------------"
    print "training net for scan: %s (training size: %d)" %(scan, len(train_x_data.keys()))
    print "-------------------------------------------------------"
    
    # initialize the CNN
    options['weight_paths'] = os.path.join(options['test_folder'], options['test_scan'])
    model = cascade_model(options)

    # --------------------------------------------------
    # first iteration (CNN1):
    # --------------------------------------------------
    print scan, ': cnn1 loading training data....'
    #load training data
    X, Y = load_training_data(train_x_data, train_y_data, options)
    print scan, ': cnn1 train_x ', X.shape

    # fit the model
    model[0].fit(X, Y)

    # --------------------------------------------------
    # second iteration (CNN2):
    # --------------------------------------------------

    # load training data based on CNN1 candidates
    print scan, ': cnn2 loading training data.....'
    X, Y = load_training_data(train_x_data, train_y_data, options,  model = model[0])
    print scan, ': cnn2 train_x ', X.shape
    model[1].fit(X, Y)

    # --------------------------------------------------
    # testing the current scan for CNN1 and CNN2
    # --------------------------------------------------
    print scan, ': testing the model ....'
    options['test_name'] = scan+ '_' + options['experiment'] + '_prob_0.nii.gz'
    t1 = test_scan(model[0], test_x_data, options, save_nifti= True)
    options['test_name'] = scan+ '_' + options['experiment'] + '_prob_1.nii.gz'
    t2 = test_scan(model[1], test_x_data, options, save_nifti= True, candidate_mask = t1>0.5)

    # --------------------------------------------------
    # postprocess the output segmentation
    # --------------------------------------------------
    options['test_name'] = scan+ '_' + options['experiment'] + '_out_CNN.nii.gz'
    out = post_process_segmentation(t2, options)
    
