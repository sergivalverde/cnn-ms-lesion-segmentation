
# Multiple Sclerosis (MS) lesion segmentation of MRI images using a cascade of two 3D convolutional neural networks 

This repository implements the method proposed in: 

```
Valverde, S., Cabezas, M., Roura, E., González-Villà, S., Pareto, D., Vilanova, J.C., Ramió-Torrentà, 
L., Rovira, A., Oliver, A., Lladó, X. (2016). Improving automated multiple sclerosis lesion 
segmentation with a cascaded 3D convolutional neural network approach. In revision (coming soon)
```

# Overview: 

Convolutional neural networks (CNN) are being increasingly used in brain MRI image analysis for segmentation of brain tumors, tissue or pathological lesions. We have developed a novel Multiple Sclerosis (MS) white matter (WM) lesion segmentation method based on a cascade of two 3D patch-wise convolutional neural networks (CNN). The first network is trained to be more sensitive revealing possible candidate lesion voxels while the second network is trained to reduce the number of misclassified voxels coming from the first network. This cascaded CNN architecture tends to learn well from small sets of training data, which can be very interesting in practice, given the difficulty to obtain manual label annotations and the large amount of available unlabeled MRI data. 

The method accepts a variable number of MRI image sequences for training (T1-w, FLAIR, PD-w, T2-w, ...), which are stacked as channels into the model. However, so far, the same number of sequences have to be used for testing. In contrast to other proposed methods, the model is trained using two cascaded networks: for the first network, a balanced training dataset is generated using all positive examples (lesion voxels) and the same number of negative samples (non-lesion voxels), randomly sampled from the entire training voxel distribution. The first network is then used to find the most challenging examples of the entire training distribution, ie. non-lesion voxels which have being classified as lesion with a high probability. From this set of challenging voxels, the second CNN is trained using a new balanced dataset composed by again all positive examples and the same number of randomly sampled voxels from the set of challenging examples. 


![training_pipeline](pipeline_training.png)


The method has been evaluated on different MS lesion segmentation challenges such as the [MICCAI 2008](http://www.ia.unc.edu/MSseg/) and [MICCAI 2016](http://www.ia.unc.edu/MSseg/). On both challenges, the proposed approach yields an excellent performance, outperforming the rest of participating strategies. The method has been also tested on clinical MS data, where our approach exhibits a significant increase in the accuracy segmenting of WM lesions when compared with other two state-of-the-art tissue segmentation methods such as  [SLS](https://github.com/NIC-VICOROB/SLSToolBox) and [LST](http://www.applied-statistics.de/lst.html), highly correlating with the expected lesion volume. 


# Install:

The method relies on some python libraries. All those libraries are available in [pip](https://pypi.python.org/pypi/pip):

```
nibabel==2.1.0
numpy==1.12.0
scipy==0.18.1
lasagne==0.1
nolearn==0.6.0
nolearn_utils==0.2.1
```

For easy installing, please use the `requeriments.txt` file included: 

```python
pip install -r requirements.txt 
```

For speed-up the 3D convolutions, the method has been implemented using [Cuda DNN](http://lasagne.readthedocs.io/en/latest/modules/layers/dnn.html?highlight=cuda%20DNN) layers in `Lasagne`. 

# How to use it: 

The main funtionality of the method can be tested using this [notebook](https://github.com/sergivalverde/cnn-ms-lesion-segmentation/blob/master/example_1.ipynb). For leave-one-out experiments, the custom [script](https://github.com/sergivalverde/cnn-ms-lesion-segmentation/blob/master/train_leave_one_out.py) used in our experiments can be also used. More complex examples and database related noteboks  (MICCAI2008, MICCAI2016) are coming soon. 

