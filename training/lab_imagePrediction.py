#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import ROOT as root
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random

# get stuff from modules
from root_numpy import tree2array
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# user modules
import tools.functions as tools

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
savePDF = False
savePNG = True 

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
h5f = h5py.File("data/NachmanLabJetImages.h5","r")

print "Accessed Jet Images"

# put images in data frames
jetImagesDF = {}
jetImagesDF['QCD'] = h5f['QCD'][()]
jetImagesDF['HH4B'] = h5f['HH4B'][()]
jetImagesDF['HH4W'] = h5f['HH4W'][()]

h5f.close()

print "Made image dataframes"

#==================================================================================
# Load the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
qcdImages = jetImagesDF['QCD'] 
print len(qcdImages)
hh4bImages = numpy.array_split(jetImagesDF['HH4B'], 2)[1]
print len(hh4bImages)
hh4wImages = numpy.array_split(jetImagesDF['HH4W'], 4)[1]
print len(hh4wImages)
jetImages = numpy.concatenate([qcdImages, hh4bImages, hh4wImages ])
jetLabels = numpy.concatenate([numpy.zeros(len(qcdImages) ), numpy.ones(len(hh4bImages) ), numpy.full(len(hh4wImages), 2)] )

print "Stored data and truth information"

# split the training and testing data
trainData, testData, trainTruth, testTruth = train_test_split(jetImages, jetLabels, test_size=0.1, random_state=7)

print "Number of QCD jets in training: ", numpy.sum(trainTruth == 0)
print "Number of H->bb jets in training: ", numpy.sum(trainTruth == 1)
print "Number of H->WW jets in training: ", numpy.sum(trainTruth == 2)

print "Number of QCD jets in testing: ", numpy.sum(testTruth == 0)
print "Number of H->bb jets in testing: ", numpy.sum(testTruth == 1)
print "Number of H->WW jets in testing: ", numpy.sum(testTruth == 2)

# make it so keras results can go in a pkl file
tools.make_keras_picklable()

# get the truth info in the correct form
trainTruth = to_categorical(trainTruth, num_classes=3)
testTruth = to_categorical(testTruth, num_classes=3)

# Define the Neural Network Structure
print "NN input shape: ", trainData.shape[1], trainData.shape[2], trainData.shape[3]
model_BESTNN = Sequential()
model_BESTNN.add( Conv2D(12, (3,3), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(trainData.shape[1], trainData.shape[2], trainData.shape[3]) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( Conv2D(8, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( Conv2D(8, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( Flatten() )
model_BESTNN.add( Dense(72, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dropout(0.20) )
model_BESTNN.add( Dense(3, kernel_initializer="glorot_normal", activation="softmax"))

# Load Weights
model_BESTNN.load_weights("lab_image_model.h5")

# compile the model
model_BESTNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BESTNN.summary() )

print "Make predictions with the neural network!"

# look at images before going through cnn
#plt.figure()
#plt.matshow(t

# visualize cnn layers
layer_outputs = [layer.output for layer in model_BESTNN.layers[:] ]
activation_model = Model(inputs=model_BESTNN.input, outputs=layer_outputs)
activations = activation_model.predict(testData)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#plt.figure()
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in model_BESTNN.layers[:9]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 6
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = numpy.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = numpy.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 2.5 / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig("plots/lab_ConvVis_"+layer_name+".png")
    #plt.tight_layout() #make all the axis labels not get cutoff
    plt.close()

# print model visualization

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
#cm = metrics.confusion_matrix(numpy.argmax(model_BESTNN.predict(testData[:]), axis=1), numpy.argmax(testTruth[:], axis=1) )
#plt.figure()
#targetNames = ['QCD', 'H->bb', 'H->WW']
#tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
#if savePDF == True:
#   plt.savefig('plots/boost_CosTheta_confusion_matrix.pdf')
#if savePNG == True:
#   plt.savefig('plots/boost_CosTheta_confusion_matrix.png')
#plt.close()

# score
#print "Training Score: ", model_BESTNN.evaluate(testData[:], testTruth[:], batch_size=100)

# make file with probability results
#joblib.dump(model_BESTNN, "HHESTIA_keras_CosTheta.pkl")
#joblib.dump(scaler, "HHESTIA_scaler.pkl")

print "Program was a great success!!!"
