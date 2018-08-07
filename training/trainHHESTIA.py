#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainHHESTIA.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains HHESTIA: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import ROOT as root
import numpy
import matplotlib.pyplot as plt
import copy

# user modules
import tools.functions as tools

# get stuff from modules
from root_numpy import tree2array
from sklearn import svm, metrics, preprocessing

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
fileJJ = root.TFile("out_QCDall.root", "READ")

# access the trees
treeJJ = fileJJ.Get("jetTree")

# get input variable names from branches
vars = tools.getBranchNames(treeJJ)
treeVars = vars

# create selection criteria
sel = "tau32 < 9999. && et > 500. && et < 2500. && bDisc1 > -0.05 && SDmass < 400"

# make arrays from the trees
arrayJJ = tree2array(treeJJ, treeVars, sel)
arrayJJ = appendTreeArray(arrayJJ)


