#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# test_boost_jetImageCreator.py ///////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
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
import timeit

# get stuff from modules
from root_numpy import tree2array

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras

# user modules
import tools.BESfunctions as tools
import tools.imageOperations as img

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
boostAxis = False
savePDF = False
savePNG = True 

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileTest = root.TFile("../preprocess/preprocess_BEST_TEST.root", "READ")

# access the trees
treeTest = fileTest.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = img.getBoostCandBranchNames(treeTest, "Higgs")
treeVars = vars
print "Variables for jet image creation: ", vars

# create selection criteria
sel = "jetAK8_pt > 500 && jetAK8_mass > 50"

# make arrays from the trees
arrayTest = tree2array(treeTest, treeVars, sel)
arrayTest = tools.appendTreeArray(arrayTest)

print "Number of Jets that will be imaged: ", len(arrayTest)

imgArrayTest = img.makeBoostCandFourVector(arrayTest, treeVars, "Higgs")

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

# get BEST variable names from branches
bestVars = tools.getBESbranchNames(treeTest)
print "Boosted Event Shape Variables: ", bestVars

# make arrays from the trees
bestArrayTest = tree2array(treeTest, bestVars, sel)
bestArrayTest = tools.appendTreeArray(bestArrayTest)

print "Made array with the Boosted Event Shape Variables"

#==================================================================================
# Make Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images"
jetImagesDF['Test_images'] = img.prepareBoostedImages(imgArrayTest, arrayTest, 31, boostAxis)

print "Made jet image data frames"

#==================================================================================
# Store BEST Variables in DataFrame ///////////////////////////////////////////////
#==================================================================================

jetImagesDF['Test_BES_vars'] = bestArrayTest
print "Stored BES variables"

#==================================================================================
# Store Data in h5 file ///////////////////////////////////////////////////////////
#==================================================================================

h5f = h5py.File("images/TestBoostedJetImages.h5","w")
h5f.create_dataset('Test_images', data=jetImagesDF['Test_images'], compression='lzf')
h5f.create_dataset('Test_BES_vars', data=jetImagesDF['Test_BES_vars'], compression='lzf')

print "Saved Test Boosted Jet Images"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF['Test_images'], 'boost_Test', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetImagesDF['Test_images'], 'boost_Test', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['Test'], 'boost_Test', savePNG, savePDF)
print "Program was a great success!!!"

