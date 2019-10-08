#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
#import ROOT as root
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
#from root_numpy import tree2array
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Conv2D, SeparableConv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# user modules
import tools.functions as tools

# enter batch mode in root (so python can access displays)
#root.gROOT.SetBatch(True)

# set options 
savePDF = False
savePNG = True 

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
# put images and labels in data frames
testFile = h5py.File("images/OpenData_HHESTIAimageOnlyTestData.h5","r")
testImages = testFile['test_images'][()]
testTruth = testFile['test_truth'][()]

print "Accessed HHESTIA test data with Jet Images"

print "Made image dataframes"

#==================================================================================
# Load the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Define the Neural Network Structure
print "NN input shape: ", testImages.shape[1], testImages.shape[2], testImages.shape[3]
model_BESTNN = Sequential()
model_BESTNN.add( SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(testImages.shape[1], testImages.shape[2], testImages.shape[3]) ))
model_BESTNN.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( Flatten() )
model_BESTNN.add( Dropout(0.20) )
model_BESTNN.add( Dense(144, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dense(72, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dense(24, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dropout(0.10) )
model_BESTNN.add( Dense(3, kernel_initializer="glorot_normal", activation="softmax"))

# Load Weights
model_BESTNN.load_weights("OpenData_HHESTIA_imageOnly_model.h5")

# compile the model
model_BESTNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BESTNN.summary() )

print "Make predictions with the neural network!"

# look at images before going through cnn
#plt.figure()
#plt.matshow(t

#==================================================================================
# Visualize cnn layers ////////////////////////////////////////////////////////////
#==================================================================================

layer_outputs = [layer.output for layer in model_BESTNN.layers[:] ]
activation_model = Model(inputs=model_BESTNN.input, outputs=layer_outputs)
activations = activation_model.predict(testImages)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#plt.figure()
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in model_BESTNN.layers[:11]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    # plot layer activations    
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
    plt.savefig("plots/OpenData_boost_CosTheta_ConvVis_"+layer_name+".png")
    #plt.tight_layout() #make all the axis labels not get cutoff
    plt.close()

    # plot filter patterns
    margin = 5
    results = numpy.zeros((4 * size + 3 * margin, 8 * size + 7 * margin, 3) ) # empty image to store results
    for i in range(4): # rows of the results grid
        for j in range(8): # columtns of the result grid
            filter_img = tools.generate_pattern(model_BESTNN, layer_name, i + (j*4), size = size) 
            
            # Put the result in the (i,j) square of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img

    # Display results grid
    plt.figure(figsize=(20,20) )
    plt.imshow(results)
    plt.savefig("plots/OpenData_boost_CosTheta_FilterPattern_"+layer_name+".png")
    plt.close()


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
