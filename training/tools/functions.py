#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used with HHESTIA //////////////////////////
#==================================================================================

# modules
import ROOT as root
import matplotlib.pyplot as plt
import copy
import random

# functions from modules
from sklearn import svm, metrics, preprocessing

#==================================================================================
# Plot Confusion Matrix ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# cm is the comfusion matrix //////////////////////////////////////////////////////
# classes are the names of the classes that the classifier distributes among //////
#----------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')

#==================================================================================
# Get Branch Names ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# tree is a TTree /////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------

def getBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'nJets' in name:
         continue
      if 'gen' in name:
         continue
      treeVars.append(name)

   return treeVars

#==================================================================================
# Append Arrays from trees ////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is a numpy array made from a TTree ////////////////////////////////////////
#----------------------------------------------------------------------------------

def appendTreeArray(array):

   tmpArray = []
   for entry in array[:] :
      a = list(entry)
      tmpArray.append(a)
   array = copy.copy(tmpArray)
   return array

#==================================================================================
# Randomize Data //////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is an array of TTree arrays ( [ tree1array, tree2array, ...] ) ////////////
#----------------------------------------------------------------------------------

def randomizeData(array):

   trainData = []
   targetData = []
   nEvents = 0
   for iArray in len(array) :
      nEvents = nEvents + len(array[iArray])
   while nEvents > 0:
      rng = random.randomint(0,len(array) )
      if (len(array[rng]) > 0):
         trainData.append(array[rng].pop() )
         targetData.append(rng)
         nEvents = nEvents - 1
   return trainData, targetData

#==================================================================================
# Plot Probabilities //////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# probs is an array of probabilites, labels, and colors ///////////////////////////
#    [ [probArray, label, color], .. ] ////////////////////////////////////////////
#----------------------------------------------------------------------------------

def plotProbabilities(probs):

   for iProb in len(probs) :
      for jProb in len(probs) :
         plt.figure()
         plt.xlabel("Probability for " + probs[iProb][1] + " Classification")
         plt.hist(probs[jProb][0].T[iProb], bins=20, range=(0,1), label=probs[jProb][1], color=probs[jProb][2], histtype='step', 
                  normed=True, log = True)
         plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
         plt.savefig("prob_" + probs[iProb][1] + ".pdf")
         plt.close()
