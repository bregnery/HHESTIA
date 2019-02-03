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
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "TMath.h"
#include "TLorentzVector.h"

///////////////////////////////////////////////////////////////////////////////////
// Functions ----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

// calculate Legendre Polynomials
float LegendreP(float x, int order);

// calculate Fox Wolfram moments
int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] );

// get jet's constituents
void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet);

// store the jet variables
void storeJetVariables(std::map<std::string, float> &treeVars, std::vector<pat::Jet>::const_iterator jet); 

// store the secondary vertex variables
void storeSecVertexVariables(std::map<std::string, float> &treeVars, TLorentzVector jet, 
                             std::vector<reco::VertexCompositePtrCandidate> secVertices);

// store the Higgs frame variables
void storeHiggsFrameVariables(std::map<std::string, float> &treeVars, std::vector<reco::Candidate *> daughtersOfJet,
                              std::vector<pat::Jet>::const_iterator jet); 

#endif
