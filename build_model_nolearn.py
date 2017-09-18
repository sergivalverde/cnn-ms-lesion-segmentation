import os
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, BatchNormLayer
from lasagne.layers import Conv3DLayer, MaxPool3DLayer, Pool3DLayer, batch_norm
from lasagne import nonlinearities, objectives, updates
from lasagne.nonlinearities import softmax, rectify
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
import warnings
warnings.simplefilter("ignore")

class Rotate_batch_Iterator(BatchIterator):
    """
    handle class for on-the-fly data augmentation on batches. 
    Applying 90,180 and 270 degrees rotations and flipping
    """
    def transform(self, Xb, yb):
        Xb, yb = super(Rotate_batch_Iterator, self).transform(Xb, yb)

        # Flip a given percentage of the images at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        x_da = Xb[indices]
    
        # apply rotation to the input batch
        rotate_90 = x_da[:,:,:,::-1,:].transpose(0,1,2,4,3)
        rotate_180 = rotate_90[:,:,:,::-1,:].transpose(0,1,2,4,3)

        # apply flipped versions of rotated patches
        rotate_0_flipped = x_da[:,:,:,:,::-1]
        rotate_180_flipped = rotate_180[:,:,:,:,::-1]

        augmented_x = np.stack([rotate_180,
                                rotate_0_flipped,
                                rotate_180_flipped],
                                axis=1)

        # select random indices from computed transformations
        #r_indices = np.random.randint(0,7,size=augmented_x.shape[0])
        r_indices = np.random.randint(0,3,size=augmented_x.shape[0])

        Xb[indices] = np.stack([augmented_x[i,r_indices[i],:,:,:,:] for i in range(augmented_x.shape[0])])
        
        return Xb, yb


def cascade_model(options):
    """
    3D cascade model using Nolearn and Lasagne
    
    Inputs:
    - model_options:
    - weights_path: path to where weights should be saved

    Output:
    - nets = list of NeuralNets (CNN1, CNN2)
    """

    # model options
    channels = len(options['modalities'])
    train_split_perc = options['train_split']
    num_epochs = options['max_epochs']
    max_epochs_patience = options['patience']

    
    # save model to disk to re-use it. Create an experiment folder
    # organize experiment 
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'])):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment']))
    if not os.path.exists(os.path.join(options['weight_paths'], options['experiment'], 'nets')):
        os.mkdir(os.path.join(options['weight_paths'], options['experiment'], 'nets'))


    # --------------------------------------------------
    # first model
    # --------------------------------------------------
    
    layer1 = InputLayer(name='in1', shape=(None, channels) + options['patch_size'])
    layer1 = batch_norm(Conv3DLayer(layer1, name='conv1_1', num_filters=32, filter_size=3, pad='same'), name = 'BN1')
    layer1 = Pool3DLayer(layer1,  name='avgpool_1', mode='max', pool_size=2, stride=2)
    layer1 = batch_norm(Conv3DLayer(layer1, name='conv2_1', num_filters=64, filter_size=3, pad='same'), name = 'BN2')
    layer1 = Pool3DLayer(layer1,  name='avgpoo2_1', mode='max', pool_size=2, stride=2)
    layer1 = DropoutLayer(layer1, name = 'l2drop', p=0.5)
    layer1 = DenseLayer(layer1, name='d_1', num_units = 256)
    layer1 = DenseLayer(layer1, name = 'out', num_units = 2, nonlinearity=nonlinearities.softmax)

    # save weights 
    net_model = 'model_1'
    net_weights = os.path.join(options['weight_paths'], options['experiment'], 'nets',  net_model + '.pkl' )
    net_history  = os.path.join(options['weight_paths'], options['experiment'], 'nets', net_model + '_history.pkl')
    
    net1 =  NeuralNet(
        layers= layer1,
        objective_loss_function=objectives.categorical_crossentropy,
        batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
        update = updates.adadelta,
        on_epoch_finished=[
            SaveWeights(net_weights, only_best=True, pickle=False),
            SaveTrainingHistory(net_history),
            EarlyStopping(patience=max_epochs_patience)],
        verbose= options['net_verbose'],
        max_epochs= num_epochs,
        train_split=TrainSplit(eval_size= train_split_perc),
    )
    
    # --------------------------------------------------
    # second model
    # --------------------------------------------------
    
    layer2 = InputLayer(name='in2', shape=(None, channels) + options['patch_size'])
    layer2 = batch_norm(Conv3DLayer(layer2, name='conv1_1', num_filters=32, filter_size=3, pad='same'), name = 'BN1')
    layer2 = Pool3DLayer(layer2,  name='avgpool_1', mode='max', pool_size=2, stride=2)
    layer2 = batch_norm(Conv3DLayer(layer2, name='conv2_1', num_filters=64, filter_size=3, pad='same'), name = 'BN2')
    layer2 = Pool3DLayer(layer2,  name='avgpoo2_1', mode='max', pool_size=2, stride=2)
    layer2 = DropoutLayer(layer2, name = 'l2drop', p=0.5)
    layer2 = DenseLayer(layer2, name='d_1', num_units = 256)
    layer2 = DenseLayer(layer2, name = 'out', num_units = 2, nonlinearity=nonlinearities.softmax)

    # save weights 
    net_model = 'model_2'
    net_weights2 = os.path.join(options['weight_paths'], options['experiment'], 'nets',  net_model + '.pkl' )
    net_history2  = os.path.join(options['weight_paths'], options['experiment'], 'nets', net_model + '_history.pkl')
    
    net2 =  NeuralNet(
        layers= layer2,
        objective_loss_function=objectives.categorical_crossentropy,
        batch_iterator_train=Rotate_batch_Iterator(batch_size=128),
        update = updates.adadelta,
        on_epoch_finished=[
            SaveWeights(net_weights2, only_best=True, pickle=False),
            SaveTrainingHistory(net_history2),
            EarlyStopping(patience=max_epochs_patience)],
        verbose= options['net_verbose'],
        max_epochs= num_epochs,
        train_split=TrainSplit(eval_size= train_split_perc),
    )

    # upload weights if set
    if options['load_weights'] == 'True':
        print "    --> CNN, loading weights from", options['experiment'], 'configuration'
        net1.load_params_from(net_weights)
        net2.load_params_from(net_weights2)
    return [net1, net2]

