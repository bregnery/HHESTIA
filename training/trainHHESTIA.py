#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import ROOT as root
import numpy
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

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileJJ = root.TFile("out_QCDall.root", "READ")

# access the trees
treeJJ = fileJJ.Get("jetTree")

# get input variable names from branches
vars = tools.getBranchNames(treeJJ)
treeVars = vars

# create selection criteria
sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayJJ = tree2array(treeJJ, treeVars, sel)
arrayJJ = appendTreeArray(arrayJJ)

# make an array with all of the datasets
arrayData = [arrayJJ]

#==================================================================================
# Plot Input Variables ////////////////////////////////////////////////////////////
#==================================================================================

# store the data in histograms
histsJJ = numpy.array(arrayJJ).T

# plot with python
for index, hist in enumerate(histsJJ):
   plt.figure()
   plt.hist(hist, bins=100, color='b', label='QCD', histtype='step', normed=True)
   #plt.hist(histHH4w[index], bins=100, color='m', label='HH->WWWWW', histtype='step', normed=True)
   plt.xlabel(vars[index])
   plt.legend()
   plt.savefig("Hist_" + vars[index] + ".pdf")
   plt.close()

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# randomize the datasets
trainData, targetData = tools.randomizeData(arrayMC)

# standardize the datasets
scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)
arrayJJ = scaler.transform(arrayJJ)

# number of events to train with
numTrain = 500000

# train the neural network
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(40,40,40), verbose=True, activation='relu')
#mlp = tree.DecisionTreeClassifier()
mlp.fit(trainData[:numTrain], targetData[:numTrain])

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm  + metrics.confusion_matrix(mlp.predict(trainData[400000:]), targetData[400000:])
plt.figure()
targetNames = ['j', 'W', 'Z', 'H', 't', 'b']
plot_confusion_matrix(cm.T, targetNames, normalize=True)
plt.savefig('confusion_matrix.pdf')
plt.close()

# score
print "Training Score: " mlp.score(trainData[400000:], targetData[400000:])

# get the probabilities
probsJJ = mlp.predict_proba(arrayJJ)

# [ [probArray, label, color], .. ]
probs = [ [probsJJ, 'QCD', 'b'] ]

# plot probability results
tools.plotProbability(probs)

# make file with probability results
joblib.dump(mlp, "HHESTIA_mlp.pkl")
joblib.dump(scaler, "HHESTIA_scaler.pkl")

