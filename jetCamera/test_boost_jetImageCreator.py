#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# test_boost_jetImageCreator.py ///////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
#==================================================================================

# modules
import ROOT as root
import uproot
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

# access the TFiles and TTrees
upTree = uproot.open("/uscms_data/d3/bregnery/HHstudies/HHESTIA/CMSSW_9_4_8/src/HHESTIA/preprocess/preprocess_BEST_TEST.root")["run/jetTree"]

# make file to store the images and BES variables
h5f = h5py.File("images/TestBoostedJetImages.h5","w")

# make a data frame to store the images
jetDF = {}

# make boosted jet images
print "Creating boosted Jet Images"
img.boostedJetPhotoshoot(upTree, "Higgs", 31, h5f, jetDF)
print "Finished the jet photoshoot"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

# get BEST variable names from branches
#bestVars = tools.getBESbranchNames(treeTest)
#print "Boosted Event Shape Variables: ", bestVars

# make arrays from the trees
#bestArrayTest = tree2array(treeTest, bestVars, sel)
#jetImagesDF['Test_BES_vars'] = tools.appendTreeArray(bestArrayTest)
#print "Made array with the Boosted Event Shape Variables"

#h5f.create_dataset('Test_BES_vars', data=jetImagesDF['Test_BES_vars'], compression='lzf')
#print "Stored BES variables"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetDF['test_images'], 'boost_Test', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetDF['test_images'], 'boost_Test', savePNG, savePDF)

   img.plotMolleweideBoostedJetImage(jetDF['test_images'], 'boost_Test', 31, savePNG, savePDF)

print "Mischief Managed!!!"

