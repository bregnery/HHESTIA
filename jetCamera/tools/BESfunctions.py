#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BESfunctions.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used for converting the BES Variables to ///
# the proper python data format ///////////////////////////////////////////////////
#==================================================================================

# modules
import numpy
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile

#==================================================================================
# Get Branch Names ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# tree is a TTree /////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------

def getBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      if 'nJets' in name:
         continue
      if 'SoftDropMass' in name:
         continue
      if 'mass' in name:
         continue
      if 'gen' in name:
         continue
      if 'pt' in name:
         continue
      treeVars.append(name)

   return treeVars

#==================================================================================
# Get BES Branch Names ////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# tree is a TTree /////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------

def getBESbranchNames(tree ):

    # empty array to store names
    treeVars = []
 
    # loop over branches
    for branch in tree.GetListOfBranches():
        name = branch.GetName()
        if 'nJets' in name:
            continue
        if 'SoftDropMass' in name:
            continue
        if 'mass' in name:
            continue
        if 'gen' in name:
            continue
        if 'pt' in name:
            continue
        if 'candidate' in name:
            continue
        if 'SV' in name:
            continue
        if 'PUPPI' in name:
            continue
        if 'subjet' in name:
            continue
        treeVars.append(name)
 
    return treeVars

#==================================================================================
# Append Arrays from trees ////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# array is a numpy array made from a TTree ////////////////////////////////////////
#----------------------------------------------------------------------------------

def appendTreeArray(array):

   tmpArray = []
   for entry in array[:] :
      a = list(entry)
      tmpArray.append(a)
   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Convert Tensor into an Image ////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Utility function to convert a tensor into a valid image -------------------------
#----------------------------------------------------------------------------------

def deprocess_image(x):
   
   # Normalize the tensor
   x -= x.mean()
   x /= (x.std() + 1e-5)
   x *= 0.1

   # clip to [0,1]
   x += 0.5
   #x = numpy.clip(x, 0, 1)

   # convert to RGB array
   x *= 25.5
   x = numpy.clip(x, 0, 255).astype('uint8')

   return x

