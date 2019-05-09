#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# boost_jetImageCreator.py ////////////////////////////////////////////////////////
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
#import functions as tools
#import imageOperations as img
import tools.functions as tools
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
fileHH4W = root.TFile("preprocess_HHESTIA_HH.root", "READ")

# access the trees
treeHH4W = fileHH4W.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = img.getBoostCandBranchNames(treeHH4W)
treeVars = vars
print "Variables for jet image creation: ", vars

# create selection criteria
#sel = ""
sel = "jetAK8_pt > 500 && jetAK8_mass > 50"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
#start, stop, step = 0, 200000, 1
arrayHH4W = tree2array(treeHH4W, treeVars, sel)#, None, start, stop, step )
arrayHH4W = tools.appendTreeArray(arrayHH4W)

print "Number of Jets that will be imaged: ", len(arrayHH4W)

imgArrayHH4W = img.makeBoostCandFourVector(arrayHH4W)

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

# get BEST variable names from branches
bestVars = tools.getBestBranchNames(treeHH4W)
print "Boosted Event Shape Variables: ", bestVars

# make arrays from the trees
start, stop, step = 0, 167262, 1
bestArrayHH4W = tree2array(treeHH4W, bestVars, sel, None, start, stop, step)
bestArrayHH4W = tools.appendTreeArray(bestArrayHH4W)

print "Made array with the Boosted Event Shape Variables"

#==================================================================================
# Make Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images for HH->WWWW"
jetImagesDF['HH4W_images'] = img.prepareBoostedImages(imgArrayHH4W, arrayHH4W, 31, boostAxis)

print "Made jet image data frames"

#==================================================================================
# Store BEST Variables in DataFrame ///////////////////////////////////////////////
#==================================================================================

jetImagesDF['HH4W_BES_vars'] = bestArrayHH4W
print "Stored BES variables"

#==================================================================================
# Store Data in h5 file ///////////////////////////////////////////////////////////
#==================================================================================

h5f = h5py.File("images/HH4WphiCosThetaBoostedJetImagesX10.h5","w")
h5f.create_dataset('HH4W_images', data=jetImagesDF['HH4W_images'], compression='lzf')
h5f.create_dataset('HH4W_BES_vars', data=jetImagesDF['HH4W_BES_vars'], compression='lzf')

print "Saved HH4W Boosted Jet Images"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF['HH4W_images'], 'boost_HH4W', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetImagesDF['HH4W_images'], 'boost_HH4W', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)
print "Program was a great success!!!"

