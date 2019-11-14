#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageOperations.py --------------------------------------------------------------
#==================================================================================
# This module contains functions to make Jet Images -------------------------------
#==================================================================================

# modules
import ROOT as root
import numpy as np
import pandas as pd
import uproot
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

#====================================================================================
# Make boosted jet images in given rest frame ---------------------------------------
#------------------------------------------------------------------------------------
# 

def boostedJetPhotoshoot(upTree, frame, nbins, h5f, jetDF):

    nx = nbins # number of image bins in phi
    ny = nbins # number of image bins in theta
    print "array length", len(upTree.array(["jetAK8_pt"]) )
    jetDF['test_images'] = np.zeros((len(upTree.array(["jetAK8_pt"]) ), nx, ny, 1) ) # made for tensorFlow

    # Loop over jets using the proper rest frame
    jetCount = 0
    for ijet in upTree.iterate([frame+"Frame_PF_candidate*", "jetAK8_pt", "jetAK8_phi", 
                                "jetAK8_eta", "jetAK8_mass"], entrysteps=1) :
    

        candArray = []
        for i in range( len(ijet[frame+'Frame_PF_candidate_px'][0]) ) :
            px = ijet[frame+'Frame_PF_candidate_px'][0][i]
            py = ijet[frame+'Frame_PF_candidate_py'][0][i]
            pz = ijet[frame+'Frame_PF_candidate_pz'][0][i]
            e  = ijet[frame+'Frame_PF_candidate_energy'][0][i]
            candLV = root.TLorentzVector(px, py, pz, e)
            
            # List the most energetic candidate first
            if i == 0:
                candArray.append(candLV)
            elif i > 0 and candLV.E() > candArray[0].E() :
                candArray.append(candArray[0])
                candArray[0] = candLV
            else:
                candArray.append(candLV) 
       
        # take a picture
        jetPic = boostedJetCamera(candArray, nbins) 
        for ix in range(0,nx):
            for iy in range(0,ny):
                jetDF['test_images'][jetCount,ix,iy,0] = jetPic[ix,iy]
        jetCount += 1  

    # save the jet images to an h5 file
    h5f.create_dataset('test_images', data=jetDF['test_images'], compression='lzf')

#==================================================================================
# Boosted Jet Camera --------------------------------------------------------------
#----------------------------------------------------------------------------------
# make jet image histograms using the candidate data frame and the original -------
#    jet array --------------------------------------------------------------------
# refFrame is the reference frame for the images to be created in -----------------
#----------------------------------------------------------------------------------

def boostedJetCamera(candArray, nbins):

    nx = nbins # number of image bins in phi
    ny = nbins # number of image bins in theta
    # set limits on relative phi and theta for the histogram
    xbins = np.linspace(-np.pi,np.pi,nx+1)
    ybins = np.linspace(-1,1,ny+1)

    # use candidate energy as weight
    weightList = []
    for i in range( len(candArray) ):
        weightList.append(candArray[i].E() )
        
    # perform boosted frame rotations
    phiPrime,thetaPrime = boostedRotations(candArray)

    # make the weight list into a np array
    totE = sum(weightList)
    normWeight = [(weight / totE)*10 for weight in weightList] #normalize energy to that of the leading, multiply to be in pixel range (0 to 255)
    weights = np.array(normWeight )

    # make a 2D np hist for the image
    jet_image_hist, xedges, yedges = np.histogram2d(phiPrime, thetaPrime, weights=weights, bins=(xbins,ybins))

    return jet_image_hist

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
        if icand.E() > leadE : 
            print "ERROR: Energy sorting was done incorrectly!"
            print " 'I stand by what I said ... you would have done well in Slytherin'"
            exit()     

        # rotate so that the leading jet is in the xy plane 
        icand.RotateZ(-rotPhi)

        #make sure leading candidate has been fully rotated
        if icand.E() == leadE : 
            if abs(icand.Py() ) < 0.01 : 
                icand.SetPy(0) 
     
        # rotate so that the leading jet is on the x axis 
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
            if abs( (icand.Phi() - leadLV.Phi() ) ) > 1.0 : 
                print "Error: Subleading candidate was improperly identified!"  
                exit()
  
        # rotatate about x with psi to get subleading candidate to x-y plane      
        icand.RotateX(subPsi - np.pi/2)
  
        #Make sure that subleading candidate has been fully rotated
        if icand.E() == subleadE and abs(icand.Pz() ) < 0.01 : icand.SetPz(0)
 
        if icand.M() < -0.1: 
            print "ERROR: Negative Candidate Mass: ", icand.M()
            print " 'Awful things happen to wizards who meddle with time, Harry'"
            exit()
 
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
 
    for icand in candArray :
 
        if bottomSum > topSum :
            thetaPrime.append( -icand.CosTheta() )
        if topSum > bottomSum :
            thetaPrime.append( icand.CosTheta() )
       
        if leftSum > rightSum :
            phiPrime.append( -icand.Phi() )
        if leftSum < rightSum :
            phiPrime.append(icand.Phi() )
 
    # store image info
    return np.array(phiPrime), np.array(thetaPrime)

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

def plotMolleweideBoostedJetImage(jetImageDF, title, nbins, plotPNG, plotPDF):

   # sum and average jet images
   summed = np.sum(jetImageDF, axis=0)
   avg = np.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   fig = plt.figure()
   ax = fig.add_subplot(111, projection = 'mollweide')

   lon = np.linspace(-1, 1, nbins) 
   lat = np.linspace(-np.pi/2, np.pi/2, nbins) 
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
      plt.savefig('plots/'+title+'_Molleweide_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_Molleweide_jetImage.pdf')
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
