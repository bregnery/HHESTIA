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
print "Put your best candidates forward... it's time for the Jet Photoshoot!"
print "Starting with the Higgs Frame"
img.boostedJetPhotoshoot(upTree, "Higgs", 31, h5f, jetDF)
print "Finished the jet photoshoot"

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

jetDF['test_BES_vars'] = upTree.pandas.df(["jetAK8_phi", "jetAK8_eta", "nSecondaryVertices", "jetAK8_Tau*",
                                       "FoxWolfram*",  "isotropy*", "aplanarity*", "thrust*", "subjet*mass*",
                                       "asymmetry*"])

h5f.create_dataset('test_BES_vars', data=jetDF['test_BES_vars'], compression='lzf')
print "Stored Boosted Event Shape variables"

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

