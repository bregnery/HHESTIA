#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# lab_jetImageCreator.py //////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
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
vars = img.getBoostCandBranchNames(treeJJ)
treeVars = vars
print "Variables for jet image creation: ", vars

# create selection criteria
#sel = ""
sel = "jetAK8_pt > 500 && jetAK8_mass > 50"
#sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayJJ = tree2array(treeJJ, treeVars, sel)
arrayJJ = tools.appendTreeArray(arrayJJ)
imgArrayJJ = img.makeBoostCandFourVector(arrayJJ)

arrayHH4W = tree2array(treeHH4W, treeVars, sel)
arrayHH4W = tools.appendTreeArray(arrayHH4W)
imgArrayHH4W = img.makeBoostCandFourVector(arrayHH4W)

arrayHH4B = tree2array(treeHH4B, treeVars, sel)
arrayHH4B = tools.appendTreeArray(arrayHH4B)
imgArrayHH4B = img.makeBoostCandFourVector(arrayHH4B)

print "Made candidate 4 vector arrays from the datasets"

#==================================================================================
# Make Lab Frame Jet Images ///////////////////////////////////////////////////////
#==================================================================================

jetImagesDF = {}
print "Creating boosted Jet Images for QCD"
jetImagesDF['QCD'] = img.prepareBoostedImages(imgArrayJJ, arrayJJ)
print "Creating boosted Jet Images for HH->bbbb"
jetImagesDF['HH4B'] = img.prepareBoostedImages(imgArrayHH4B, arrayHH4B)
print "Creating boosted Jet Images for HH->WWWW"
jetImagesDF['HH4W'] = img.prepareBoostedImages(imgArrayHH4W, arrayHH4W)

print "Made jet image data frames"

h5f = h5py.File("data/phiCosThetaBoostedJetImages.h5","w")
h5f.create_dataset('QCD', data=jetImagesDF['QCD'], compression='lzf')
h5f.create_dataset('HH4B', data=jetImagesDF['HH4B'], compression='lzf')
h5f.create_dataset('HH4W', data=jetImagesDF['HH4W'], compression='lzf')

print "Saved Boosted Jet Images"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
   img.plotAverageBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
   img.plotAverageBoostedJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)
print "Program was a great success!!!"

