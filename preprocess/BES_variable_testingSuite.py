#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BES_variable_testingSuite.py -----------------------------------------------------------
#-----------------------------------------------------------------------------------------
# This program aims to test the Boosted Event Shape Variables created by the edproducer --
#=========================================================================================

# modules
import ROOT   as root
import numpy  as np
import pandas as pd

# get specific functions
from root_numpy import tree2array

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

#==================================================================================
# Append Arrays from trees ////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is a numpy array made from a TTree ////////////////////////////////////////
#----------------------------------------------------------------------------------

def appendTreeArray(array):

    import copy
    tmpArray = []
    for entry in array[:] :
        a = list(entry)
        tmpArray.append(a)
    newArray = copy.copy(tmpArray)
    return newArray

#=========================================================================================
# Load Monte Carlo -----------------------------------------------------------------------
#=========================================================================================

# access the TFiles
fileTest = root.TFile("preprocess_BEST_TEST.root", "READ")

# access the trees
tree = fileTest.Get("run/jetTree")
print "Accessed the trees"

# get branch names
treeVars = []
for branch in tree.GetListOfBranches():
    name = branch.GetName()
    treeVars.append(name)

print "The variables stored in this TTree are: \n", treeVars

# create jet array from tree
jetArray = tree2array(tree, treeVars, "", None)
jetArray = appendTreeArray(jetArray)

print "\n", "This tree has ", len(jetArray), " jets"

#=========================================================================================
# Testing Suite --------------------------------------------------------------------------
#=========================================================================================

# get the jet energy indicies
indHPF = treeVars.index('HiggsFrame_PF_candidate_energy')
indTPF = treeVars.index('TopFrame_PF_candidate_energy')
indWPF = treeVars.index('WFrame_PF_candidate_energy')
indZPF = treeVars.index('ZFrame_PF_candidate_energy')

# get the jet energy indicies
indHsub = treeVars.index('HiggsFrame_subjet_energy')
indTsub = treeVars.index('TopFrame_subjet_energy')
indWsub = treeVars.index('WFrame_subjet_energy')
indZsub = treeVars.index('ZFrame_subjet_energy')

# get foxwolfram indicies
indFoxWolf = [index for index, value in enumerate(treeVars) if "FoxWolf" in value]

# get N Jettiness Variables
indTau = [index for index, value in enumerate(treeVars) if "Tau" in value]

# get mass indicies
indMass = [index for index, value in enumerate(treeVars) if "mass" in value]

# loop over the jets
for ijet in range(0, len(jetArray)):

    #-------------------------------------------------------------------------------------
    # Check For Negative Mass Variables --------------------------------------------------
    #-------------------------------------------------------------------------------------
    
    for iMass in indMass :
        if "SV" not in treeVars[iMass] and jetArray[ijet][iMass] < -0.1 :  # choose negative 0.1 because there seems to be precision errors with smaller values
            print "ERROR: ", treeVars[iMass], " is negative: ", jetArray[ijet][iMass]
 
    #-------------------------------------------------------------------------------------
    # Candidate Tests  -------------------------------------------------------------------
    #-------------------------------------------------------------------------------------
    
    for icand in range(0, len(jetArray[ijet][indHPF])):
        if jetArray[ijet][indHPF][icand] <= 0 or jetArray[ijet][indTPF][icand] <= 0 or jetArray[ijet][indWPF][icand] <= 0 or jetArray[ijet][indZPF][icand] <= 0 :
            
            print "ERROR: NEGATIVE ENERGY in the PF Candidates"

    #-------------------------------------------------------------------------------------
    # Test N-Jettiness -------------------------------------------------------------------
    #-------------------------------------------------------------------------------------
    
    for iTau in indTau :
        if jetArray[ijet][iTau] < 0.0 or jetArray[ijet][iTau] > 1.0 :
            print "ERROR: NJettiness is outside [0.0, 1.0]: ", treeVars[iTau], " = ", jetArray[ijet][iTau]

    #-------------------------------------------------------------------------------------
    # Subjet Tests -----------------------------------------------------------------------
    #-------------------------------------------------------------------------------------

    for isub in range(0, len(jetArray[ijet][indHsub])):
        if jetArray[ijet][indHsub][isub] <= 0 :
            print "ERROR: Negative energy in the Higgs frame subjets"
    for isub in range(0, len(jetArray[ijet][indTsub])):
        if jetArray[ijet][indTsub][isub] <= 0 :
            print "ERROR: Negative energy in the Top frame subjets"
    for isub in range(0, len(jetArray[ijet][indWsub])):
        if jetArray[ijet][indWsub][isub] <= 0 :
            print "ERROR: Negative energy in the W frame subjets"
    for isub in range(0, len(jetArray[ijet][indZsub])):
        if jetArray[ijet][indZsub][isub] <= 0 :
            print "ERROR: Negative energy in the Z frame subjets"

    #-------------------------------------------------------------------------------------
    # Fox Wolfram Moments ----------------------------------------------------------------
    #-------------------------------------------------------------------------------------

    # Test that the Fox Wolfram moments have the correct range
    for iMom in indFoxWolf :
        if abs(jetArray[ijet][iMom]) > 1 :
            print "ERROR: ", treeVars[iMom], " is outside the expected range of [-1, 1], it's value is ", jetArray[ijet][iMom]
    

print "The testing session has concluded"


