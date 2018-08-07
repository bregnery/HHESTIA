#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used with HHESTIA //////////////////////////
#==================================================================================

# modules
import ROOT as root
import matplotlib.pyplot as plt
import copy

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

def getBranchNames(tree)

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches()
   name = branch.GetName()
   if 'Njets' in name:
      continue
   if 'target' in name:
      continue
   if 'NNout' in name:
      continue
   if 'sum' in name:
      continue
   if 'gen' in name:
      continue
   if 'flatten' in name:
      continue
   if 'dist' in name:
      continue
   if 'npv' in name:
      continue
   if 'sorting' in name:
      continue
   if name == 'mass':
      continue
   if name == 'minDist':
      continue
   if 'et' in name:
      continue
   treeVars.append(name)

   return treeVars

#==================================================================================
# Append Arrays from trees ////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is a numpy array made from a TTree ////////////////////////////////////////
#----------------------------------------------------------------------------------

def appendTreeArray(array)

   tmpArray = []
   for entry in array[:] :
      a = list(entry)
      tmpArray.append(a)
   array = copy.copy(tmpArray)
   return array 
