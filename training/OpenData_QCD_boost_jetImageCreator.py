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
fileQCD = root.TFile("preprocess_HHESTIA_OpenData_QCD.root", "READ")

# access the trees
treeQCD = fileQCD.Get("run/jetTree")

print "Accessed the trees"

# get input variable names from branches
vars = img.getBoostCandBranchNames(treeQCD)
treeVars = vars
print "Variables for jet image creation: ", vars

# create selection criteria
#sel = ""
sel = "jetAK8_pt > 500 && jetAK8_mass > 50"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
start, stop, step = 0, 167262, 1
arrayQCD = tree2array(treeQCD, treeVars, sel, None, start, stop, step )
arrayQCD = tools.appendTreeArray(arrayQCD)

print "Number of Jets that will be imaged: ", len(arrayQCD)

imgArrayQCD = img.makeBoostCandFourVector(arrayQCD)

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

# get BEST variable names from branches
bestVars = tools.getBestBranchNames(treeQCD)
print "Boosted Event Shape Variables: ", bestVars

# make arrays from the trees
start, stop, step = 0, 167262, 1
bestArrayQCD = tree2array(treeQCD, bestVars, sel, None, start, stop, step)
bestArrayQCD = tools.appendTreeArray(bestArrayQCD)

print "Made array with the Boosted Event Shape Variables"

#==================================================================================
# Make Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images for QCD"
jetImagesDF['QCD_images'] = img.prepareBoostedImages(imgArrayQCD, arrayQCD, 31, boostAxis)

print "Made jet image data frames"

#==================================================================================
# Store BEST Variables in DataFrame ///////////////////////////////////////////////
#==================================================================================

jetImagesDF['QCD_BES_vars'] = bestArrayQCD
print "Stored BES variables"

#==================================================================================
# Store Data in h5 file ///////////////////////////////////////////////////////////
#==================================================================================

h5f = h5py.File("images/OpenDataQCDphiCosThetaBoostedJetImagesX10.h5","w")
h5f.create_dataset('QCD_images', data=jetImagesDF['QCD_images'], compression='lzf')
h5f.create_dataset('QCD_BES_vars', data=jetImagesDF['QCD_BES_vars'], compression='lzf')

print "Saved QCD Boosted Jet Images"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF['QCD_images'], 'boost_QCD_OpenData', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetImagesDF['QCD_images'], 'boost_QCD_OpenData', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['QCD'], 'boost_QCD_OpenData', savePNG, savePDF)
print "Program was a great success!!!"

