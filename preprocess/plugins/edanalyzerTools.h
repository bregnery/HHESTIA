//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// edanalyzerTools.h --------------------------------------------------------------
//=================================================================================
// Header file containing functions for use with CMS EDAnalyzer and EDProducer ----
///////////////////////////////////////////////////////////////////////////////////

// make sure the functions are not declared more than once
#ifndef EDANALYZERTOOLS_H 
#define EDANALYZERTOOLS_H

// include files
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TMath.h"

///////////////////////////////////////////////////////////////////////////////////
// Functions ----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

// sort jet collection in terms of pT
std::vector<pat::Jet> * sortJets(std::vector<pat::Jet> *jets);

// find difference in phi
float myDeltaPhi(float phi1, float phi2);

// find the delta R between two objects
float myDeltaR(float eta1, float phi1, float eta2, float phi2);

// delta R match two objects
std::vector<std::vector<bool> > deltaRMatch(std::vector<std::array<float, 2> > etaPhi1, std::vector<std::array<float, 2> > etaPhi2, float limR);

#endif
