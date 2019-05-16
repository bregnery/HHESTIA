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
testFile = h5py.File("images/BESonly_HHESTIAtestData.h5","r")
testBESvars = testFile['test_BES_vars'][()]
testTruth = testFile['test_truth'][()]

print "Accessed test data with BES variables"

#==================================================================================
# Load the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Define the Neural Network Structure using functional API
# Create the BES variable version
besInputs = Input( shape=(testBESvars.shape[1], ) )
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
outputHHESTIA = Dense(3, kernel_initializer="glorot_normal", activation="softmax")(besLayer)

# compile the model
model_HHESTIA = Model(inputs = besInputs, outputs = outputHHESTIA)

# Load Weights
model_HHESTIA.load_weights("HHESTIA_BESonly_model.h5")

# compile the model
model_HHESTIA.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_HHESTIA.summary() )

print "Make predictions with the neural network!"

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
#cm = metrics.confusion_matrix(numpy.argmax(model_HHESTIA.predict(testData[:]), axis=1), numpy.argmax(testTruth[:], axis=1) )
#plt.figure()
#targetNames = ['QCD', 'H->bb', 'H->WW']
#tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
#if savePDF == True:
#   plt.savefig('plots/boost_CosTheta_confusion_matrix.pdf')
#if savePNG == True:
#   plt.savefig('plots/boost_CosTheta_confusion_matrix.png')
#plt.close()

# score
print "Training Score: ", model_HHESTIA.evaluate(testBESvars[:], testTruth[:], batch_size=100)

print "Program was a great success!!!"
