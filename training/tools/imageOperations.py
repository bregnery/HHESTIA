#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageOperations.py --------------------------------------------------------------
#==================================================================================
# This module contains functions to make Jet Images -------------------------------
#==================================================================================

# modules
import ROOT as root
import numpy
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile
import timeit

# grab some keras stuff
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import keras.backend as K

#==================================================================================
# Get PF Candidate Branches -------------------------------------------------------
#----------------------------------------------------------------------------------
# tree is TTree -------------------------------------------------------------------
#----------------------------------------------------------------------------------

def getPFcandBranchNames(tree ):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      # Only get PF branches
      if 'PF' in name:
         treeVars.append(name)
      if 'jetAK8_pt' in name:
         treeVars.append(name)

   return treeVars

#==================================================================================
# Make array with PF candidate information ----------------------------------------
#----------------------------------------------------------------------------------
# This function converts the array made from the jetTree to the correct form to ---
#   use with the jet image functions ----------------------------------------------
# array is a numpy array made from a TTree ----------------------------------------
#----------------------------------------------------------------------------------

def makePFcandArray(array):

   tmpArray = []  #use lists not numpy arrays (wayyyyy faster)
   jetCount = 1
   n = 0
   # loop over jets
   while n < len(array) :
      # loop over pf candidates
      for i in range( len(array[n][1][:]) ) :
         jetPt = array[n][0]
         pt = array[n][1][i]
         phi = array[n][2][i]
         eta = array[n][3][i]
         tmpArray.append([jetCount, jetPt, pt, eta, phi])
      jetCount +=1
      n += 1

   newArray = copy.copy(tmpArray)
   return newArray

#==================================================================================
# Make lab frame Jet Images -------------------------------------------------------
#----------------------------------------------------------------------------------
# make jet image histograms using the candidate data frame and the original -------
#    jet array --------------------------------------------------------------------
#----------------------------------------------------------------------------------

def prepareImages(candDF, jetArray):

    nx = 30 # size of image in eta
    ny = 30 # size of image in phi
    xbins = numpy.linspace(-1.4,1.4,nx+1)
    ybins = numpy.linspace(-1.4,1.4,ny+1)

    list_x = []
    list_y = []
    list_w = []
    if K.image_dim_ordering()=='tf':
        # 4D tensor (tensorflow backend)
        # 1st dim is jet index
        # 2nd dim is eta bin
        # 3rd dim is phi bin
        # 4th dim is pt value (or rgb layer, etc.)
        jet_images = numpy.zeros((len(jetArray), nx, ny, 1))
    else:        
        jet_images = numpy.zeros((len(jetArray), 1, nx, ny))
    
    for i in range(0,len(jetArray)):
        if i % 1000 == 0: print "Imaging jet number: ", i+1
        # get the ith jet
        mask = candDF['njet'].values == i + 1 # boolean mask ... the fastest way from stack
        df_unsorted_cand_i = candDF[mask] # new data frame with only ith jet candidates
        df_dict_cand_i = df_unsorted_cand_i.sort_values(by=['cand_pt'], ascending=False) # sort candidates by pt
        # relative eta
        x = df_dict_cand_i['cand_eta']-df_dict_cand_i['cand_eta'].iloc[0]
        # relative phi
        y = df_dict_cand_i['cand_phi']-df_dict_cand_i['cand_phi'].iloc[0]
        weights = df_dict_cand_i['cand_pt'] # pt of candidate is the weight
        x,y = rotate_and_reflect(x,y,weights)
        list_x.append(x)
        list_y.append(y)
        list_w.append(weights)
        hist, xedges, yedges = numpy.histogram2d(x, y, weights=weights, bins=(xbins,ybins))
        for ix in range(0,nx):
            for iy in range(0,ny):
                if K.image_dim_ordering()=='tf':
                    jet_images[i,ix,iy,0] = hist[ix,iy]
                else:
                    jet_images[i,0,ix,iy] = hist[ix,iy]
    return jet_images

#==================================================================================
# Rotate and Reflect Jet Images ---------------------------------------------------
#----------------------------------------------------------------------------------
# x is relative eta, i.e. the difference between jet eta and the eta of each ------
#   daughter ----------------------------------------------------------------------
# y is relative phi, i.e. the difference between jet phi and the phi of each ------
#   daughter ----------------------------------------------------------------------
# w is the weight. Typically jet pT -----------------------------------------------
#----------------------------------------------------------------------------------

def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        dv = numpy.matrix([[ix],[iy]])-numpy.matrix([[x.iloc[0]],[y.iloc[0]]])
        dR = numpy.linalg.norm(dv)
        thisPt = iw
        if dR > 0.35 and thisPt > maxPt:
            maxPt = thisPt
            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            # theta = -numpy.arctan2(iy,ix)-numpy.radians(90)
            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            px = iw * numpy.cos(iy)
            py = iw * numpy.sin(iy)
            pz = iw * numpy.sinh(ix)
            theta = numpy.arctan2(py,pz)+numpy.radians(90)
            
    c, s = numpy.cos(theta), numpy.sin(theta)
    R = numpy.matrix('{} {}; {} {}'.format(c, -s, s, c))
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*numpy.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw * numpy.cos(iy)
        py = iw * numpy.sin(iy)
        pz = iw * numpy.sinh(ix)
        rot = R*numpy.matrix([[py],[pz]])
        px1 = px
        py1 = rot[0,0]
        pz1 = rot[1,0]
        iw1 = numpy.sqrt(px1*px1+py1*py1)
        rix, riy = numpy.arcsinh(pz1/iw1), numpy.arcsin(py1/iw1)
        rot_x.append(rix)
        rot_y.append(riy)
        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y
    
    return numpy.array(ref_x), numpy.array(ref_y)

#==================================================================================
# Plot Averaged Jet Images --------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageJetImage(jetImageDF, title, plotPNG, plotPDF):

   summed = numpy.sum(jetImageDF, axis=0)
   avg = numpy.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none')
   cbar = plt.colorbar()
   cbar.set_label(r'$p_T$ [GeV]')
   if title == 'lab_QCD' :
      plt.title('QCD Lab Jet Image')
   if title == 'lab_HH4W' :
      plt.title(r'$HH\rightarrow WWWW$ Lab Jet Image')
   if title == 'lab_HH4B' :
      plt.title(r'$HH\rightarrow bbbb$ Lab Jet Image')
   plt.xlabel(r'$\eta_i$')
   plt.ylabel(r'$\phi_i$')
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()
