#!/bin/bash

echo "Setting Up Environment"
echo "Starting job on " `date` #Date/time of start of job                                                                                                              
echo "Running on: `uname -a`" #Condor job is running on this node                                                                                                      
echo "System software: `cat /etc/redhat-release`" #Operating System on that node 

echo "Setting Up CMSSW"
source /cvmfs/cms.cern.ch/cmsset_default.sh  
export SCRAM_ARCH=slc6_amd64_gcc630 
eval `scramv1 project CMSSW CMSSW_9_4_8`
cd CMSSW_9_4_8/src/
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo "CMSSW: "$CMSSW_BASE

# Copy over input files
cp ../../HH4B_boost_jetImageCreator.py .
cp ../../preprocess_HHESTIA_HH_4B_all.root .

# Execute program
python HH4B_boost_jetImageCreator.py 
