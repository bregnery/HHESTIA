#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageOperations.py --------------------------------------------------------------
#==================================================================================
# This module contains functions to make Jet Images -------------------------------
#==================================================================================

# modules
import ROOT as root
import numpy as np
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
# Get Rest Frame Candidate Branches -----------------------------------------------
#----------------------------------------------------------------------------------
# tree is TTree, frame is a string of the desired rest frame ----------------------
#----------------------------------------------------------------------------------

def getBoostCandBranchNames(tree, frame):

   # empty array to store names
   treeVars = []

   # loop over branches
   for branch in tree.GetListOfBranches():
      name = branch.GetName()
      # Only get PF branches
      if frame + 'Frame_PF' in name:
         treeVars.append(name)
      if 'jetAK8_pt' in name:
         treeVars.append(name)
      if 'jetAK8_phi' in name:
         treeVars.append(name)
      if 'jetAK8_eta' in name:
         treeVars.append(name)
      if 'jetAK8_mass' in name:
         treeVars.append(name)

   return treeVars

#==================================================================================
# Make array with Boosted PF candidate 4 vectors ----------------------------------
#----------------------------------------------------------------------------------
# This function converts the array made from the jetTree to the correct form to ---
#   use with the boosted jet image functions --------------------------------------
# array is a numpy array made from a TTree, treeVars is a list of strings of the --
#   branch names, frame is a string for the rest frame being used -----------------
#----------------------------------------------------------------------------------

def makeBoostCandFourVector(array, treeVars, frame):

    # Get PF candidate indices
    indPx = treeVars.index(frame + 'Frame_PF_candidate_px') 
    indPy = treeVars.index(frame + 'Frame_PF_candidate_py') 
    indPz = treeVars.index(frame + 'Frame_PF_candidate_pz') 
    indE  = treeVars.index(frame + 'Frame_PF_candidate_energy') 

    tmpArray = []  #use lists not np arrays (faster appending)
    jetCount = 1
    entryNum = 0
    n = 0
    # loop over jets
    while n < len(array) :
        if n % 10000 == 0: print "Making 4 vector array for jet number: ", jetCount
        # loop over pf candidates
        for i in range( len(array[n][indE][:]) ) :
            px = array[n][indPx][i]
            py = array[n][indPy][i]
            pz = array[n][indPz][i]
            e  = array[n][indE][i]
            candLV = root.TLorentzVector(px, py, pz, e)
            
            # List the most energetic candidate first
            if i == 0:
                tmpArray.append([jetCount, candLV])
            elif i > 0 and candLV.E() > tmpArray[entryNum - i][1].E() :
                tmpArray.append([jetCount, tmpArray[entryNum - i][1] ])
                tmpArray[entryNum - i] = [jetCount, candLV]
            else:
                tmpArray.append([jetCount, candLV]) 
            entryNum += 1
        jetCount +=1
        n += 1
 
    newArray = copy.copy(tmpArray)
    return newArray

#==================================================================================
# Boosted Candidate Rotations -----------------------------------------------------
#----------------------------------------------------------------------------------
# candArray is an array of four vectors for one jet -------------------------------
#----------------------------------------------------------------------------------

def boostedRotations(candArray):
    phiPrime = []
    thetaPrime = []
 
    # define the rotation angles for first two rotations
    rotPhi = candArray[0].Phi()
    rotTheta = candArray[0].Theta()
    subPsi = 0
 
    # Perform the first two rotations
    subleadE = -1
    leadE = candArray[0].E()
    leadLV = candArray[0]
    for icand in candArray :
 
        # Waring for incorrect energy sorting
        if icand.E() > leadE : print "WARNING: Energy sorting was done incorrectly!"
      
        icand.RotateZ(-rotPhi)
        #set small py values to 0
        if abs(icand.Py() ) < 0.01 : icand.SetPy(0) 
      
        icand.RotateY(np.pi/2 - rotTheta)
      
        # Save leading Candidate LV
        if icand.E() == leadE : 
            # Make sure leading candidate has been fully rotated
            if abs(icand.Pz() ) < 0.01 : 
                icand.SetPz(0)
            leadLV = icand
      
        # Find subleading candidate
        if icand.E() > subleadE and icand.E() < leadE :
            if abs( (icand.Phi() - leadLV.Phi() ) ) > 1.0 :
                subleadE = icand.E()
                # store its y z projection angle to Z axis (psi) for a third rotation
                # arctan2 is very important
                subPsi = np.arctan2(icand.Py(), icand.Pz() )
 
    # Perform the third rotation
    for icand in candArray :
 
        # warning for subleading identification
        if icand.E() > subleadE and icand.E() < leadE : 
            if abs( (icand.Phi() - leadLV.Phi() ) ) > 1.0 : print "WARNING: Subleading candidate was improperly identified!"  
  
        # rotatate about x with psi to get subleading candidate to x-y plane      
        icand.RotateX(subPsi - np.pi/2)
  
        #Make sure that subleading candidate has been fully rotated
        if icand.E() == subleadE and abs(icand.Pz() ) < 0.01 : icand.SetPz(0)
 
        #if icand.M() < -0.1: print "ERROR: Negative Candidate Mass: ", icand.M()
 
        # store image info
        #phiPrime.append(icand.Phi() )
        #thetaPrime.append( icand.CosTheta() )
 
    # Reflect if bottomSum > topSum and/or leftSum > rightSum
    leftSum, rightSum = 0, 0
    topSum, bottomSum = 0, 0
    for icand in candArray :
      
        if icand.CosTheta() > 0 :
            topSum += icand.E()
        if icand.CosTheta() < 0 :
            bottomSum += icand.E()
      
        if icand.Phi() > 0 :
            rightSum += icand.E()
        if icand.Phi() < 0 :
            leftSum += icand.E()
 
    # store image info
    for icand in candArray :
 
        if bottomSum > topSum :
            thetaPrime.append( -icand.CosTheta() )
        if topSum > bottomSum :
            thetaPrime.append( icand.CosTheta() )
       
        if leftSum > rightSum :
            phiPrime.append( -icand.Phi() )
        if leftSum < rightSum :
            phiPrime.append(icand.Phi() )
 
    return np.array(phiPrime), np.array(thetaPrime)

#==================================================================================
# Make boosted frame Jet Images ---------------------------------------------------
#----------------------------------------------------------------------------------
# make jet image histograms using the candidate data frame and the original -------
#    jet array --------------------------------------------------------------------
# refFrame is the reference frame for the images to be created in -----------------
#----------------------------------------------------------------------------------

def prepareBoostedImages(candLV, jetArray, nbins, boostAxis ):

    nx = nbins #30 # number of image bins in phi
    ny = nbins #30 # number of image bins in theta
    # set limits on relative phi and theta for the histogram
    xbins = np.linspace(-np.pi,np.pi,nx+1)
    ybins = np.linspace(-1,1,ny+1)

    if K.image_dim_ordering()=='tf':
        # 4D tensor (tensorflow backend)
        # 1st dim is jet index
        # 2nd dim is eta bin
        # 3rd dim is phi bin
        # 4th dim is pt value (or rgb layer, etc.)
        jet_images = np.zeros((len(jetArray), nx, ny, 1))
    else:        
        jet_images = np.zeros((len(jetArray), 1, nx, ny))

    jetCount = 0    
    candNum = 0
    for i in range(0,len(jetArray)):
        jetNum = i + 1
        if i % 1000 == 0: print "Imaging jet number: ", jetNum

        # make 4 vector of the jet
        jetPt, jetEta, jetPhi, jetMass = jetArray[i][2], jetArray[i][1], jetArray[i][0], jetArray[i][3]
        jetLV = root.TLorentzVector()
        jetLV.SetPtEtaPhiM(jetPt, jetEta, jetPhi, jetMass)

        # get the ith jet candidate 4 vectors
        icandLV = []
        weightList = []
        while jetCount <= jetNum :
            jetCount = candLV[candNum][0]
            if jetCount == jetNum:
               icandLV.append(candLV[candNum][1])
               # use candidate energy as weight
               weightList.append(candLV[candNum][1].E() )
               candNum += 1
            # stop the loop for the last jet
            if candNum == len(candLV):
               break
        
        # perform boosted frame rotations
        if boostAxis == False : #use leading candidate as Z axis in rotations
           phiPrime,thetaPrime = boostedRotations(icandLV)
        if boostAxis == True : # use boost axis as Z axis in rotations
           phiPrime,thetaPrime = boostedRotationsRelBoostAxis(icandLV, jetLV)

        # make the weight list into a np array
        totE = sum(weightList)
        normWeight = [(weight / totE)*10 for weight in weightList] #normalize energy to that of the leading, multiply to be in pixel range (0 to 255)
        weights = np.array(normWeight )
        #weights = np.array(weightList ) #normWeight )

        # make a 2D np hist for the image
        hist, xedges, yedges = np.histogram2d(phiPrime, thetaPrime, weights=weights, bins=(xbins,ybins))
        for ix in range(0,nx):
           for iy in range(0,ny):
              if K.image_dim_ordering()=='tf':
                 jet_images[i,ix,iy,0] = hist[ix,iy]
              else:
                 jet_images[i,0,ix,iy] = hist[ix,iy]

    return jet_images

#==================================================================================
# Plot Averaged Boosted Jet Images ------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageBoostedJetImage(jetImageDF, title, plotPNG, plotPDF):

   # sum and average jet images
   summed = np.sum(jetImageDF, axis=0)
   avg = np.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-np.pi, np.pi, -1, 1], aspect = "auto")
   cbar = plt.colorbar()
   cbar.set_label(r'Energy [GeV]')
   if title == 'boost_QCD' :
      plt.title('QCD Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4W' :
      plt.title(r'$H\rightarrow WW$ Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4B' :
      plt.title(r'$H\rightarrow bb$ Boosted Jet Image', fontsize = 22)
   plt.xlabel(r'$\phi$', fontsize = 18)
   plt.ylabel(r'cos($\theta$)', fontsize = 20)
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()

#==================================================================================
# Plot 3 Boosted Jet Images -------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotThreeBoostedJetImages(jetImageDF, title, plotPNG, plotPDF):

   for i in range (0, 3) :
      # plot the images
      plt.figure('N') 
      plt.imshow(jetImageDF[i,:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-np.pi, np.pi, -1, 1], aspect = "auto")
      cbar = plt.colorbar()
      cbar.set_label(r'Energy [GeV]')
      if title == 'boost_QCD' :
         plt.title('QCD Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4W' :
         plt.title(r'$H\rightarrow WW$ Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4B' :
         plt.title(r'$H\rightarrow bb$ Boosted Jet Image #'+str(i), fontsize = 22)
      plt.xlabel(r'$\phi$', fontsize = 18)
      plt.ylabel(r'cos($\theta$)', fontsize = 20)
      if plotPNG == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.png')
      if plotPDF == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.pdf')
      plt.close()

#==================================================================================
# Plot Molleweide Boosted Jet Images ----------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotMolleweideBoostedJetImage(jetImageDF, title, plotPNG, plotPDF):

   # sum and average jet images
   summed = np.sum(jetImageDF, axis=0)
   avg = np.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   fig = plt.figure()
   ax = fig.add_subplot(111, projection = 'mollweide')

   lon = np.linspace(-np.pi, np.pi, 42) 
   lat = np.linspace(-np.pi/2, np.pi/2, 42) 
   Lon, Lat = np.meshgrid(lon, lat)

   im = ax.pcolormesh(Lon, Lat, avg[:,:,0].T, norm=mpl.colors.LogNorm() )
   cbar = fig.colorbar(im, orientation='horizontal')
   cbar.set_label(r'Energy [GeV]')
   if title == 'boost_QCD' :
      plt.title('QCD Boosted Jet Image')
   if title == 'boost_HH4W' :
      plt.title(r'$H\rightarrow WW$ Boosted Jet Image')
   if title == 'boost_HH4B' :
      plt.title(r'$H\rightarrow bb$ Boosted Jet Image')
   plt.xlabel(r'$\phi_i$')
   plt.ylabel(r'$\theta_i$')
#   plt.xticks(np.arange(-4, 4, step=1.0) )
#   plt.yticks(np.arange(0, 4, step=1.0) )
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()

#==================================================================================
# Plot Averaged Jet Images --------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageJetImage(jetImageDF, title, plotPNG, plotPDF):

   summed = np.sum(jetImageDF, axis=0)
   avg = np.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-1.4,1.4, -1.4, 1.4])
   cbar = plt.colorbar()
   cbar.set_label(r'$p_T$ [GeV]')
   if title == 'lab_QCD' :
      plt.title('QCD Lab Jet Image', fontsize = 22)
   if title == 'lab_HH4W' :
      plt.title(r'$H\rightarrow WW$ Lab Jet Image', fontsize = 22)
   if title == 'lab_HH4B' :
      plt.title(r'$H\rightarrow bb$ Lab Jet Image', fontsize = 22)
   plt.xlabel(r'$\eta_i$', fontsize = 18)
   plt.ylabel(r'$\phi_i$', fontsize = 20)
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()
