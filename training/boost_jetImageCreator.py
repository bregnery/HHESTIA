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
imgArrayJJ = img.makeBoostCandArray(arrayJJ)

arrayHH4W = tree2array(treeHH4W, treeVars, sel)
arrayHH4W = tools.appendTreeArray(arrayHH4W)
imgArrayHH4W = img.makeBoostCandArray(arrayHH4W)

arrayHH4B = tree2array(treeHH4B, treeVars, sel)
arrayHH4B = tools.appendTreeArray(arrayHH4B)
imgArrayHH4B = img.makeBoostCandArray(arrayHH4B)

print "Made arrays from the datasets"

#==================================================================================
# Make Lab Frame Jet Images ///////////////////////////////////////////////////////
#==================================================================================

candDF = {}
candDF['QCD'] = pd.DataFrame(imgArrayJJ, columns = ['njet', 'jet_pt', 'cand_pt', 'cand_eta', 'cand_phi'])
candDF['HH4B'] = pd.DataFrame(imgArrayHH4B, columns = ['njet', 'jet_pt', 'cand_pt', 'cand_eta', 'cand_phi'])
candDF['HH4W'] = pd.DataFrame(imgArrayHH4W, columns = ['njet', 'jet_pt', 'cand_pt', 'cand_eta', 'cand_phi'])

print "Made particle flow candidate data frames in the Higgs rest frame"

jetImagesDF = {}
print "Creating boosted Jet Images for QCD"
jetImagesDF['QCD'] = img.prepareImages(candDF['QCD'], arrayJJ)
print "Creating boosted Jet Images for HH->bbbb"
jetImagesDF['HH4B'] = img.prepareImages(candDF['HH4B'], arrayHH4B)
print "Creating boosted Jet Images for HH->WWWW"
jetImagesDF['HH4W'] = img.prepareImages(candDF['HH4W'], arrayHH4B)

print "Made jet image data frames"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting jet images"
   img.plotAverageJetImage(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
   img.plotAverageJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
   img.plotAverageJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)

print "Program was a great success!!!"

