#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# combonationROCplotter.py ////////////////////////////////////////////////////////
#==================================================================================
# This program evaluates HHESTIA: HH Event Shape Topology Indentification Algorithm 
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
from scipy import interp
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Conv2D, SeparableConv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import concatenate
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
# Load Test Data //////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
# put images and BES variables in data frames
testFileHHESTIA = h5py.File("images/HHESTIAtestData.h5","r")
testImagesHHESTIA = testFileHHESTIA['test_images'][()]
testBESvarsHHESTIA = testFileHHESTIA['test_BES_vars'][()]
testTruthHHESTIA = testFileHHESTIA['test_truth'][()]

print "Accessed HHESTIA test data with Jet Images and BES variables"

# put images and BES variables in data frames
testFileBES = h5py.File("images/BESonly_HHESTIAtestData.h5","r")
testBESvarsBES = testFileBES['test_BES_vars'][()]
testTruthBES = testFileBES['test_truth'][()]

print "Accessed test data for network with only BES variables"

# put images and BES variables in data frames
testFileBJI = h5py.File("images/HHESTIAimageOnlyTestData.h5","r")
testImagesBJI = testFileBJI['test_images'][()]
testTruthBJI = testFileBJI['test_truth'][()]

print "Accessed HHESTIA test data with Jet Images and BES variables"

#==================================================================================
# Load HHESTIA ////////////////////////////////////////////////////////////////////
#==================================================================================

# Define the Neural Network Structure using functional API
# Create the image portion
imageInputs = Input( shape=(testImagesHHESTIA.shape[1], testImagesHHESTIA.shape[2], testImagesHHESTIA.shape[3]) )

imageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageInputs)
imageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer)
imageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer) 
imageLayer = Flatten()(imageLayer)
imageLayer = Dropout(0.20)(imageLayer)
imageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
imageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
imageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
imageLayer = Dropout(0.10)(imageLayer)

imageModel = Model(inputs = imageInputs, outputs = imageLayer)

# Create the BES variable version
besInputs = Input( shape=(testBESvarsHHESTIA.shape[1], ) )
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)

besModel = Model(inputs = besInputs, outputs = besLayer)

# Add BES variables to the network
combined = concatenate([imageModel.output, besModel.output])

combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combined)
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputHHESTIA = Dense(3, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
model_HHESTIA = Model(inputs = [imageModel.input, besModel.input], outputs = outputHHESTIA)
model_HHESTIA.load_weights("boost_phiCosTheta_image_model.h5")
model_HHESTIA.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print "Loaded the HHESTIA neural network!"

#==================================================================================
# Load BES only network ///////////////////////////////////////////////////////////
#==================================================================================

# Define the Neural Network Structure using functional API
# Create the BES variable version
besInputsBES = Input( shape=(testBESvarsBES.shape[1], ) )
besLayerBES = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputsBES)
besLayerBES = Dropout(0.20)(besLayerBES)
besLayerBES = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayerBES)
besLayerBES = Dropout(0.20)(besLayerBES)
besLayerBES = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayerBES)
besLayerBES = Dropout(0.20)(besLayerBES)
outputBESonly = Dense(3, kernel_initializer="glorot_normal", activation="softmax")(besLayerBES)

# compile the model
model_BESonly = Model(inputs = besInputsBES, outputs = outputBESonly)
model_BESonly.load_weights("HHESTIA_BESonly_model.h5")
model_BESonly.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print "Loaded the BES variable only neural network!"

#==================================================================================
# Load image only network /////////////////////////////////////////////////////////
#==================================================================================

# Define the Neural Network Structure
print "NN input shape: ", testImagesBJI.shape[1], testImagesBJI.shape[2], testImagesBJI.shape[3]
model_imageOnly = Sequential()
model_imageOnly.add( SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(testImagesBJI.shape[1], testImagesBJI.shape[2], testImagesBJI.shape[3]) ))
model_imageOnly.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( BatchNormalization(momentum = 0.6) )
model_imageOnly.add( MaxPool2D(pool_size=(2,2) ) )
model_imageOnly.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( BatchNormalization(momentum = 0.6) )
model_imageOnly.add( MaxPool2D(pool_size=(2,2) ) )
model_imageOnly.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_imageOnly.add( BatchNormalization(momentum = 0.6) )
model_imageOnly.add( MaxPool2D(pool_size=(2,2) ) )
model_imageOnly.add( Flatten() )
model_imageOnly.add( Dropout(0.20) )
model_imageOnly.add( Dense(144, kernel_initializer="glorot_normal", activation="relu" ))
model_imageOnly.add( Dense(72, kernel_initializer="glorot_normal", activation="relu" ))
model_imageOnly.add( Dense(24, kernel_initializer="glorot_normal", activation="relu" ))
model_imageOnly.add( Dropout(0.10) )
model_imageOnly.add( Dense(3, kernel_initializer="glorot_normal", activation="softmax"))

# Load Weights
model_imageOnly.load_weights("HHESTIA_imageOnly_model.h5")

# compile the model
model_imageOnly.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print "Loaded the boosted jet image only neural network!"

#==================================================================================
# Create ROC Curves ///////////////////////////////////////////////////////////////
#==================================================================================

# predict
HHESTIApredict = model_HHESTIA.predict([testImagesHHESTIA[:], testBESvarsHHESTIA[:]])
BESpredict = model_BESonly.predict(testBESvarsBES[:] )
BJIpredict = model_imageOnly.predict(testImagesBJI[:] )

print "Made predictions using the neural networks"

# HHESTIA
# compute ROC curve and area for each class
n_classes = testTruthHHESTIA.shape[1]
fprHHESTIA = dict()
tprHHESTIA = dict()
roc_auc_HHESTIA = dict()
for i in range(n_classes):
    fprHHESTIA[i], tprHHESTIA[i], _ = roc_curve(testTruthHHESTIA[:, i], HHESTIApredict[:, i]) # returns 3 outputs but only care about 2
    roc_auc_HHESTIA[i] = auc(fprHHESTIA[i], tprHHESTIA[i])

# compute micro-average ROC curve and ROC area
fprHHESTIA["micro"], tprHHESTIA["micro"], _ = roc_curve(testTruthHHESTIA.ravel(), HHESTIApredict.ravel() )
roc_auc_HHESTIA["micro"] = auc(fprHHESTIA["micro"], tprHHESTIA["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprHHESTIA = numpy.unique(numpy.concatenate([fprHHESTIA[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprHHESTIA = numpy.zeros_like(all_fprHHESTIA)
for i in range(n_classes):
    mean_tprHHESTIA += interp(all_fprHHESTIA, fprHHESTIA[i], tprHHESTIA[i] )

# average and compute macro AUC
mean_tprHHESTIA /= n_classes

fprHHESTIA["macro"] = all_fprHHESTIA
tprHHESTIA["macro"] = mean_tprHHESTIA
roc_auc_HHESTIA["macro"] = auc(fprHHESTIA["macro"], tprHHESTIA["macro"])

# BES only
# compute ROC curve and area for each class
n_classes = 3
fprBES = dict()
tprBES = dict()
roc_auc_BES = dict()
for i in range(n_classes):
    fprBES[i], tprBES[i], _ = roc_curve(testTruthBES[:, i], BESpredict[:, i])
    roc_auc_BES[i] = auc(fprBES[i], tprBES[i])

# compute micro-average ROC curve and ROC area
fprBES["micro"], tprBES["micro"], _ = roc_curve(testTruthBES.ravel(), BESpredict.ravel() )
roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprBES = numpy.unique(numpy.concatenate([fprBES[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprBES = numpy.zeros_like(all_fprBES)
for i in range(n_classes):
    mean_tprBES += interp(all_fprBES, fprBES[i], tprBES[i] )

# average and compute macro AUC
mean_tprBES /= n_classes

fprBES["macro"] = all_fprBES
tprBES["macro"] = mean_tprBES
roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

# Boosted Jet Images only
# compute ROC curve and area for each class
n_classes = testTruthBJI.shape[1]
fprBJI = dict()
tprBJI = dict()
roc_auc_BJI = dict()
for i in range(n_classes):
    fprBJI[i], tprBJI[i], _ = roc_curve(testTruthBJI[:, i], BJIpredict[:, i]) # returns 3 outputs but only care about 2
    roc_auc_BJI[i] = auc(fprBJI[i], tprBJI[i])

# compute micro-average ROC curve and ROC area
fprBJI["micro"], tprBJI["micro"], _ = roc_curve(testTruthBJI.ravel(), BJIpredict.ravel() )
roc_auc_BJI["micro"] = auc(fprBJI["micro"], tprBJI["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprBJI = numpy.unique(numpy.concatenate([fprBJI[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprBJI = numpy.zeros_like(all_fprBJI)
for i in range(n_classes):
    mean_tprBJI += interp(all_fprBJI, fprBJI[i], tprBJI[i] )

# average and compute macro AUC
mean_tprBJI /= n_classes

fprBJI["macro"] = all_fprBJI
tprBJI["macro"] = mean_tprBJI
roc_auc_BJI["macro"] = auc(fprBJI["macro"], tprBJI["macro"])

print "Created ROC curves"

#==================================================================================
# Plot ROC curves /////////////////////////////////////////////////////////////////
#==================================================================================

# average ROC
plt.figure(1)
plt.plot(fprBES["micro"], tprBES["micro"],
         label='BES micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES["micro"]),
         color='orange', linewidth=2)
plt.plot(fprBJI["micro"], tprBJI["micro"],
         label='BJI micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI["micro"]),
         color='deeppink', linewidth=2)
plt.plot(fprHHESTIA["micro"], tprHHESTIA["micro"],
         label='HHESTIA micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_HHESTIA["micro"]),
         color='blue', linewidth=2)
plt.plot(fprBES["macro"], tprBES["macro"],
         label='BES macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES["macro"]),
         color='orange', linestyle=':', linewidth=4)
plt.plot(fprBJI["macro"], tprBJI["macro"],
         label='BJI macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI["macro"]),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fprHHESTIA["macro"], tprHHESTIA["macro"],
         label='HHESTIA macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_HHESTIA["macro"]),
         color='blue', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_average_ROCplot.png')
plt.close()

# category ROC curves
# QCD
plt.figure(1)
plt.plot(fprBES[0], tprBES[0],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[0]),
         color='orange', linewidth=2)
plt.plot(fprBJI[0], tprBJI[0],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[0]),
         color='deeppink', linewidth=2)
plt.plot(fprHHESTIA[0], tprHHESTIA[0],
         label='HHESTIA ROC curve (area = {0:0.2f})' ''.format(roc_auc_HHESTIA[0]),
         color='blue', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('QCD category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_QCD_ROCplot.png')
plt.close()

# H->bb
plt.figure(1)
plt.plot(fprBES[1], tprBES[1],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[1]),
         color='orange', linewidth=2)
plt.plot(fprBJI[1], tprBJI[1],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[1]),
         color='deeppink', linewidth=2)
plt.plot(fprHHESTIA[1], tprHHESTIA[1],
         label='HHESTIA ROC curve (area = {0:0.2f})' ''.format(roc_auc_HHESTIA[1]),
         color='blue', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('H->bb category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_Hbb_ROCplot.png')
plt.close()

# H->WW
plt.figure(1)
plt.plot(fprBES[2], tprBES[2],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[2]),
         color='orange', linewidth=2)
plt.plot(fprBJI[2], tprBJI[2],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[2]),
         color='deeppink', linewidth=2)
plt.plot(fprHHESTIA[2], tprHHESTIA[2],
         label='HHESTIA ROC curve (area = {0:0.2f})' ''.format(roc_auc_HHESTIA[2]),
         color='blue', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('H->WW category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_HWW_ROCplot.png')
plt.close()

print "Plotted ROC curves"

print "Program was a great success!!!"


