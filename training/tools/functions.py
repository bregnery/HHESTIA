#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used with HHESTIA //////////////////////////
#==================================================================================

# modules
import ROOT as root
import numpy
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile
import keras.models

# functions from modules
from sklearn import svm, metrics, preprocessing, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

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
       cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = numpy.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout() #make all the axis labels not get cutoff

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
      if 'SoftDropMass' in name:
         continue
      if 'mass' in name:
         continue
      if 'gen' in name:
         continue
      if 'pt' in name:
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
   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Randomize Data //////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is an array of TTree arrays ( [ tree1array, tree2array, ...] ) ////////////
#----------------------------------------------------------------------------------

def randomizeData(array):

   trainData = []
   targetData = []
   nEvents = 0
   for iArray in range(len(array) ) :
      nEvents = nEvents + len(array[iArray])
   while nEvents > 0:
      rng = random.randint(0,len(array)-1 )
      if (len(array[rng]) > 0):
         trainData.append(array[rng].pop() )
         targetData.append(rng)
         nEvents = nEvents - 1
   return trainData, targetData

#==================================================================================
# Plot Performance ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# loss is an array of loss and loss_val from the training /////////////////////////
# acc is an array of acc and acc_val from the training ////////////////////////////
# train_test is the train data that has not been trained on ///////////////////////
# target_test is the target data that has not been trained on /////////////////////
# target_predict is the models prediction of data that has not been trained on ////
#----------------------------------------------------------------------------------

def plotPerformance(loss, acc): #, train_test, target_test, target_predict):
   
   # plot loss vs epoch
   plt.figure()
   plt.plot(loss[0], label='loss')
   plt.plot(loss[1], label='val_loss')
   plt.legend(loc="upper right")
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.savefig("plots/loss.pdf")
   plt.savefig("plots/loss.png")
   plt.close()

   # plot accuracy vs epoch
   plt.figure()
   plt.plot(acc[0], label='acc')
   plt.plot(acc[1], label='val_acc')
   plt.legend(loc="upper left")
   plt.xlabel('epoch')
   plt.ylabel('acc')
   plt.savefig("plots/acc.pdf")
   plt.savefig("plots/acc.png")
   plt.close()

   # Plot ROC
#   fpr, tpr, thresholds = roc_curve(target_test, target_predict)
#   roc_auc = auc(fpr, tpr)
#   ax = plt.subplot(2, 2, 3)
#   ax.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
#   ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
#   ax.set_xlim([0, 1.0])
#   ax.set_ylim([0, 1.0])
#   ax.set_xlabel('false positive rate')
#   ax.set_ylabel('true positive rate')
#   ax.set_title('receiver operating curve')
#   ax.legend(loc="lower right")

#==================================================================================
# Plot Probabilities //////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# probs is an array of probabilites, labels, and colors ///////////////////////////
#    [ [probArray, label, color], .. ] ////////////////////////////////////////////
#----------------------------------------------------------------------------------

def plotProbabilities(probs):

   for iProb in range(len(probs) ) :
      for jProb in range(len(probs) ) :
         plt.figure()
         plt.xlabel("Probability for " + probs[iProb][1] + " Classification")
         plt.hist(probs[jProb][0].T[iProb], bins=20, range=(0,1), label=probs[jProb][1], color=probs[jProb][2], histtype='step', 
                  normed=True, log = True)
         plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
         plt.savefig("prob_" + probs[iProb][1] + ".pdf")
         plt.close()

#==================================================================================
# Make Keras Picklable  ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# A patch to make Keras give results in pickle format /////////////////////////////
#----------------------------------------------------------------------------------

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

#==================================================================================
# Rotate and Reflect Jet Images ///////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# x is relative eta, i.e. the difference between jet eta and the eta of the ///////
#   each daughter ////////////////////////////////////////////////////////////////
# y is relative phi, i.e. the difference between jet phi and the phi of the ///////
#   each daughter ////////////////////////////////////////////////////////////////
# w is the weight. Typically jet pT ///////////////////////////////////////////////
#----------------------------------------------------------------------------------

def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        dv = numpy.matrix([[ix],[iy]])-numpy.matrix([[x.iloc[0]],[y.iloc[0]]])
        dR = numpy.linalg.norm(dv)
        thisPt = iw
        if dR > 0.35 and thisPt > maxPt:
            maxPt = thisPt
            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            # theta = -numpy.arctan2(iy,ix)-numpy.radians(90)
            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            px = iw * numpy.cos(iy)
            py = iw * numpy.sin(iy)
            pz = iw * numpy.sinh(ix)
            theta = numpy.arctan2(py,pz)+numpy.radians(90)
            
    c, s = numpy.cos(theta), numpy.sin(theta)
    R = numpy.matrix('{} {}; {} {}'.format(c, -s, s, c))
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*numpy.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw * numpy.cos(iy)
        py = iw * numpy.sin(iy)
        pz = iw * numpy.sinh(ix)
        rot = R*numpy.matrix([[py],[pz]])
        px1 = px
        py1 = rot[0,0]
        pz1 = rot[1,0]
        iw1 = numpy.sqrt(px1*px1+py1*py1)
        rix, riy = numpy.arcsinh(pz1/iw1), numpy.arcsin(py1/iw1)
        rot_x.append(rix)
        rot_y.append(riy)
        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y

    return np.array(ref_x), np.array(ref_y)
