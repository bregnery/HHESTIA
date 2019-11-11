#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
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
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import GRU, LSTM, ConvLSTM2D, Reshape
from keras.layers import concatenate
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
import tools.functions as tools

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = False
savePNG = True 

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
#h5f = h5py.File("images/phiCosThetaBoostedJetImages.h5","r")

# put images and BES variables in data frames
jetImagesDF = {}
jetBESvarsDF = {}
QCD = h5py.File("images/OpenData_QCDphiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['QCD'] = QCD['QCD_images'][()]
jetBESvarsDF['QCD'] = QCD['QCD_BES_vars'][()]
QCD.close()
HH4B = h5py.File("images/OpenData_HH4BphiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['HH4B'] = HH4B['HH4B_images'][()]
jetBESvarsDF['HH4B'] = HH4B['HH4B_BES_vars'][()]
HH4B.close()
HH4W = h5py.File("images/OpenData_HH4WphiCosThetaBoostedJetImagesX10.h5","r")
jetImagesDF['HH4W'] = HH4W['HH4W_images'][()]
jetBESvarsDF['HH4W'] = HH4W['HH4W_BES_vars'][()]
HH4W.close()

print "Accessed Jet Images and BES variables"

#h5f.close()

print "Made image dataframes"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
qcdImages = jetImagesDF['QCD'] 
qcdBESvars = jetBESvarsDF['QCD']
print "Number of QCD Jet Images: ", len(qcdImages)
hh4bImages = jetImagesDF['HH4B']
hh4bBESvars = jetBESvarsDF['HH4B']
print "Number of H->bb Jet Images: ", len(hh4bImages)
hh4wImages = jetImagesDF['HH4W']
hh4wBESvars = jetBESvarsDF['HH4W']
print "Number of H->WW Jet Images: ", len(hh4wImages)
???LINES MISSING
# Add BES variables to the network
combined = concatenate([imageModel.output, besModel.output])

combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combined)
combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputHHESTIA = Dense(3, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
model_HHESTIA = Model(inputs = [imageModel.input, besModel.input], outputs = outputHHESTIA)
model_HHESTIA.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_HHESTIA.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('OpenData_HHESTIA_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_HHESTIA.fit([trainImages[:], trainBESvars[:] ], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print "Trained the neural network!"

# print model visualization
#plot_model(model_HHESTIA, to_file='plots/boost_CosTheta_NN_Vis.png')

# save the test data
h5f = h5py.File("images/OpenData_HHESTIAtestData.h5","w")
h5f.create_dataset('test_images', data=testImages, compression='lzf')
h5f.create_dataset('test_BES_vars', data=testBESvars, compression='lzf')
h5f.create_dataset('test_truth', data=testTruth, compression='lzf')

print "Saved the testing data!"


#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(numpy.argmax(model_HHESTIA.predict([testImages[:], testBESvars[:] ]), axis=1), numpy.argmax(testTruth[:], axis=1) )
plt.figure()
targetNames = ['QCD', 'H->bb', 'H->WW']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/OpenData_HHESTIA_confusion_matrix.pdf')
if savePNG == True:
   plt.savefig('plots/OpenData_HHESTIA_confusion_matrix.png')
plt.close()

# score
print "Training Score: ", model_HHESTIA.evaluate([testImages[:], testBESvars[:]], testTruth[:], batch_size=100)

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "OpenData_HHESTIA_")
print "plotted HESTIA training Performance"

# make file with probability results
joblib.dump(model_HHESTIA, "OpenData_HHESTIA_keras_CosTheta.pkl")
#joblib.dump(scaler, "HHESTIA_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
