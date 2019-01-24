#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import ROOT as root
import uproot
import numpy
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import copy
import random

# user modules
import tools.functions as tools

# get stuff from modules
from root_numpy import tree2array
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotInputVariables = True
plotProbs = True
savePDF = False
savePNG = True 

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileJJ = root.TFile("preprocess_HHESTIA_QCD.root", "READ")
fileHH4W = root.TFile("preprocess_HHESTIA_HH.root", "READ")
fileHH4B = root.TFile("preprocess_HHESTIA_HH_4B.root", "READ")

# access the trees
treeJJ = fileJJ.Get("run/jetTree")
treeHH4W = fileHH4W.Get("run/jetTree")
treeHH4B = fileHH4B.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = tools.getBranchNames(treeJJ)
treeVars = vars
print "Training with these variables: ", vars

# create selection criteria
#sel = ""
sel = "jetAK8_pt > 50 && jetAK8_mass > 50"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayJJ = tree2array(treeJJ, treeVars, sel)
arrayJJ = tools.appendTreeArray(arrayJJ)

arrayHH4W = tree2array(treeHH4W, treeVars, sel)
arrayHH4W = tools.appendTreeArray(arrayHH4W)

arrayHH4B = tree2array(treeHH4B, treeVars, sel)
arrayHH4B = tools.appendTreeArray(arrayHH4B)

# make an array with all of the datasets
arrayData = [arrayJJ, arrayHH4W, arrayHH4B]

# copy the arrays so that copies of the arrays are not deleted in the randomize function
# without this, deleting arrayData will delete all tree arrays
arrayJJ = copy.copy(arrayJJ)
arrayHH4W = copy.copy(arrayHH4W)
arrayHH4B = copy.copy(arrayHH4B)

print "Made arrays from the datasets"

#==================================================================================
# Plot Input Variables ////////////////////////////////////////////////////////////
#==================================================================================

# store the data in histograms
histsJJ = numpy.array(arrayJJ).T
histsHH4W = numpy.array(arrayHH4W).T
histsHH4B = numpy.array(arrayHH4B).T

# plot with python
if plotInputVariables == True:
   for index, hist in enumerate(histsJJ):
      plt.figure()
      plt.hist(hist, bins=100, color='b', label='QCD', histtype='step', normed=True)
      plt.hist(histsHH4W[index], bins=100, color='m', label='H->WW', histtype='step', normed=True)
      plt.hist(histsHH4B[index], bins=100, color='g', label='H->bb', histtype='step', normed=True)
      plt.xlabel(vars[index])
      plt.legend()
      if savePDF == True:
         plt.savefig("plots/Hist_" + vars[index] + ".pdf")
      if savePNG == True:
         plt.savefig("plots/Hist_" + vars[index] + ".png")
      plt.close()
   print "Plotted each of the variables"

if plotInputVariables == False:
   print "Input Variables will not be plotted. Change options at beginning of the program if this is incorrect."

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# customize the neural network
NDIM = len(VARS)
inputs = Input(shape=(NDIM,), name = 'input')  

# make the hidden layers
hidden = []
dropout = []
hidden.append(Dense(40, name = 'hidden1', kernel_initializer='normal', activation='relu')(inputs) )
for i in range(1, 39):
   dropout.append(Dropout(0.20)(hidden[i-1]) )
   hidden.append(Dense(40, name = 'hidden' + str(i), kernel_initializer='normal', activation='relu')(dropout[i-1]) )
dropout.append(Dropout(0.20)(hidden[39]) )
outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(dropout[39])

# create the model
model = Model(inputs=inputs, outputs=outputs)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print the model summary
model.summary()

# randomize the datasets
trainData, targetData = tools.randomizeData(arrayData)

# standardize the datasets
scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)
arrayJJ = scaler.transform(arrayJJ)
arrayHH4W = scaler.transform(arrayHH4W)
arrayHH4W = scaler.transform(arrayHH4B)

# number of events to train with
numTrain = 60000

# train the neural network
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(40,40,40), verbose=True, activation='relu')
#mlp = tree.DecisionTreeClassifier()
mlp.fit(trainData[:numTrain], targetData[:numTrain])

print "Trained the neural network!"

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(mlp.predict(trainData[10000:]), targetData[10000:])
plt.figure()
targetNames = ['QCD', 'H->WW', 'H->bb']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/confusion_matrix.pdf')
if savePNG == True:
   plt.savefig('plots/confusion_matrix.png')
plt.close()

# score
print "Training Score: ", mlp.score(trainData[10000:], targetData[10000:])

# get the probabilities
probsJJ = mlp.predict_proba(arrayJJ)
probsHH4W = mlp.predict_proba(arrayHH4W)
probsHH4B = mlp.predict_proba(arrayHH4B)

# [ [probArray, label, color], .. ]
probs = [ [probsJJ, 'QCD', 'b'],
          [probsHH4W, 'QCD', 'm'],
          [probsHH4B, 'QCD', 'm'] ]

# plot probability results
if plotProbs == True:
   tools.plotProbabilities(probs)
   print "plotted the HHESTIA probabilities"
if plotProbs == False:
   print "HHESTIA probabilities will not be plotted. This can be changed at the beginning of the program."

# make file with probability results
joblib.dump(mlp, "HHESTIA_mlp.pkl")
joblib.dump(scaler, "HHESTIA_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
