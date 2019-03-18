#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# boost_jetImageTester.py /////////////////////////////////////////////////////////
#==================================================================================
# Program to test functions that make boosted frame jet images (small scale)  /////
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
import tools.functions as tools
import tools.imageOperations as img

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
boostAxis = True
savePDF = False
savePNG = True 

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileHH4B = root.TFile("preprocess_HHESTIA_HH_4B.root", "READ")

# access the trees
treeHH4B = fileHH4B.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = img.getBoostCandBranchNames(treeHH4B)
treeVars = vars
print "Variables for jet image creation: ", vars

# create selection criteria
#sel = ""
sel = "jetAK8_pt > 500 && jetAK8_mass > 50"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayHH4Blong = tree2array(treeHH4B, treeVars, sel)
arrayHH4B = arrayHH4Blong[:4000]
arrayHH4B = tools.appendTreeArray(arrayHH4B)
imgArrayHH4B = img.makeBoostCandFourVector(arrayHH4B)

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Make Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images for HH->bbbb"
jetImagesDF['HH4B'] = img.prepareBoostedImages(imgArrayHH4B, arrayHH4B, boostAxis)

print "Made jet image data frames"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B_test', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetImagesDF['HH4B'], 'boost_HH4B_test', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
print "Program was a great success!!!"

