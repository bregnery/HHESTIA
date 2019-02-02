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
#include "TLorentzVector.h"

///////////////////////////////////////////////////////////////////////////////////
// Functions ----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

// calculate Legendre Polynomials
float LegendreP(float x, int order);

// calculate Fox Wolfram moments
int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] );

// store the jet variables
void storeJetVariables(std::map<std::string, float> &treeVars, std::vector<pat::Jet>::const_iterator jet); 

#endif
