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
savePDF = False
savePNG = True 

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
h5f = h5py.File("data/phiCosThetaBoostedJetImages.h5","r")

print "Accessed Jet Images"

# put images in data frames
jetImagesDF = {}
jetImagesDF['QCD'] = h5f['QCD'][()]
jetImagesDF['HH4B'] = h5f['HH4B'][()]
jetImagesDF['HH4W'] = h5f['HH4W'][()]

h5f.close()

print "Made image dataframes"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
print "Plotting Boosted jet images"
img.plotAverageBoostedJetImage(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
img.plotAverageBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
img.plotAverageBoostedJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)

img.plotThreeBoostedJetImages(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
img.plotThreeBoostedJetImages(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
img.plotThreeBoostedJetImages(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)

   #img.plotMolleweideBoostedJetImage(jetImagesDF['QCD'], 'boost_QCD', savePNG, savePDF)
   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4B'], 'boost_HH4B', savePNG, savePDF)
   #img.plotMolleweideBoostedJetImage(jetImagesDF['HH4W'], 'boost_HH4W', savePNG, savePDF)
print "Program was a great success!!!"

