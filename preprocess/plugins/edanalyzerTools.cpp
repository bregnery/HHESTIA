//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// edanalyzerTools.cpp ------------------------------------------------------------
//=================================================================================
// C++ file containing functions for use with CMS EDAnalyzer and EDProducer -------
///////////////////////////////////////////////////////////////////////////////////

#include "edanalyzerTools.h"

//=================================================================================
// Calculate Legendre Polynomials -------------------------------------------------
//---------------------------------------------------------------------------------
// Simple Legendre polynomial function that can calculate up to order 4 -----------
// Inputs: argument of the polynomial and order desired ---------------------------
//---------------------------------------------------------------------------------

float LegendreP(float x, int order){
   if (order == 0) return 1;
   else if (order == 1) return x;
   else if (order == 2) return 0.5*(3*x*x - 1);
   else if (order == 3) return 0.5*(5*x*x*x - 3*x);
   else if (order == 4) return 0.125*(35*x*x*x*x - 30*x*x + 3);
   else return 0;
}

//=================================================================================
// Calculate Fox Wolfram Moments --------------------------------------------------
//---------------------------------------------------------------------------------
// This function calculates the Fox Wolfram moments for jet constituents ----------
// in various rest frames. --------------------------------------------------------
// Inputs: particles (jet constiuents boosted to rest frame) and empty array that -
//         that will store the FW moments -----------------------------------------
//---------------------------------------------------------------------------------

int FWMoments(std::vector<TLorentzVector> particles, double (&outputs)[5] ){
   
   // get number of particles to loop over
   int numParticles = particles.size();

   // get energy normalization for the FW moments
   float s = 0.0;
   for(int i = 0; i < numParticles; i++){
   	s += particles[i].E();
   }

   float H0 = 0.0;
   float H4 = 0.0;
   float H3 = 0.0;
   float H2 = 0.0;
   float H1 = 0.0;

   for (int i = 0; i < numParticles; i++){

   	for (int j = i; j < numParticles; j++){

                // calculate cos of jet constituent angles
   		float costh = ( particles[i].Px() * particles[j].Px() + particles[i].Py() * particles[j].Py() 
                                   + particles[i].Pz() * particles[j].Pz() ) / ( particles[i].P() * particles[j].P() );
   		float w1 = particles[i].P();
   		float w2 = particles[j].P();

                // calculate legendre polynomials of jet constiteuent angles
   		float fw0 = LegendreP(costh, 0);
   		float fw1 = LegendreP(costh, 1);
   		float fw2 = LegendreP(costh, 2);
   		float fw3 = LegendreP(costh, 3);
   		float fw4 = LegendreP(costh, 4);

                // calculate the Fox Wolfram moments
   		H0 += w1 * w2 * fw0;
   		H1 += w1 * w2 * fw1;
   		H2 += w1 * w2 * fw2;
   		H3 += w1 * w2 * fw3;
   		H4 += w1 * w2 * fw4;

   	}
   }

   // Normalize the Fox Wolfram moments
   if (H0 == 0) H0 += 0.001;      // to prevent dividing by zero
   outputs[0] = (H0);
   outputs[1] = (H1 / H0);
   outputs[2] = (H2 / H0);
   outputs[3] = (H3 / H0);
   outputs[4] = (H4 / H0);

   return 0;
}

//=================================================================================
// Get All Jet Constituents -------------------------------------------------------
//---------------------------------------------------------------------------------
// This gets all the jet constituents (daughters) and stores them as a standard ---
// vector -------------------------------------------------------------------------
//---------------------------------------------------------------------------------

void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet){
   // First get all daughters for the first Soft Drop Subjet
   for (unsigned int i = 0; i < jet->daughter(0)->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(0)->daughter(i) );
   }
   // Get all daughters for the second Soft Drop Subjet
   for (unsigned int i = 0; i < jet->daughter(1)->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(1)->daughter(i));
   }
   // Get all daughters not included in Soft Drop
   for (unsigned int i = 2; i< jet->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(i) );
   }
}

//=================================================================================
// Store Jet Variables ------------------------------------------------------------
//---------------------------------------------------------------------------------
// This takes various jet quantaties and stores them on the map used to fill ------
// the jet tree -------------------------------------------------------------------
//---------------------------------------------------------------------------------  

void storeJetVariables(std::map<std::string, float> &treeVars, std::vector<pat::Jet>::const_iterator jet){ 
                       // pasing a variable with & is pass-by-reference which keeps changes in this func
   // Jet four vector and Soft Drop info
   treeVars["jetAK8_phi"] = jet->phi();
   treeVars["jetAK8_eta"] = jet->eta(); 
   treeVars["jetAK8_pt"] = jet->pt(); 
   treeVars["jetAK8_mass"] = jet->mass(); 
   treeVars["jetAK8_SoftDropMass"] = jet->userFloat("ak8PFJetsCHSSoftDropMass");

   // Store Subjettiness info
   treeVars["jetAK8_Tau4"] = jet->userFloat("NjettinessAK8CHS:tau4");  //important for H->WW jets
   treeVars["jetAK8_Tau3"] = jet->userFloat("NjettinessAK8:tau3");
   treeVars["jetAK8_Tau2"] = jet->userFloat("NjettinessAK8:tau2");
   treeVars["jetAK8_Tau1"] = jet->userFloat("NjettinessAK8:tau1");
}

//=================================================================================
// Store Secondary Vertex Information ---------------------------------------------
//---------------------------------------------------------------------------------
// This takes various secondary vertex quantities and stores them on the map ------
// used to fill the tree ----------------------------------------------------------
//---------------------------------------------------------------------------------

void storeSecVertexVariables(std::map<std::string, float> &treeVars, TLorentzVector jet, 
                             std::vector<reco::VertexCompositePtrCandidate> secVertices){

   int numMatched = 0; // counts number of secondary vertices
   for(std::vector<reco::VertexCompositePtrCandidate>::const_iterator vertBegin = secVertices.begin(), 
              vertEnd = secVertices.end(), ivert = vertBegin; ivert != vertEnd; ivert++){
      TLorentzVector vert(ivert->px(), ivert->py(), ivert->pz(), ivert->energy() );
      // match vertices to jet
      if(jet.DeltaR(vert) < 0.8 ){
         numMatched++;
         // save secondary vertex info for the first three sec vertices
         if(numMatched <= 3){
            std::string i = std::to_string(numMatched);
            treeVars["SV_"+i+"_pt"] = ivert->pt();
            treeVars["SV_"+i+"_eta"] = ivert->eta();
            treeVars["SV_"+i+"_phi"] = ivert->phi();
            treeVars["SV_"+i+"_mass"] = ivert->mass();
            treeVars["SV_"+i+"_nTracks"] = ivert->numberOfDaughters();
            treeVars["SV_"+i+"_chi2"] = ivert->vertexChi2();
            treeVars["SV_"+i+"_Ndof"] = ivert->vertexNdof();
         }
      }
   }
   treeVars["nSecondaryVertices"] = numMatched;
}

//=================================================================================
// Store Higgs Rest Frame Variables -----------------------------------------------
//---------------------------------------------------------------------------------
// This boosts an ak8 jet (and all of its constituents) into the higgs rest frame -
// and then uses it to calculate FoxWolfram moments, Event Shape Variables, -------
// and assymmetry variables -------------------------------------------------------
//---------------------------------------------------------------------------------

void storeHiggsFrameVariables(std::map<std::string, float> &treeVars, std::vector<reco::Candidate *> daughtersOfJet,
                              std::vector<pat::Jet>::const_iterator jet){ 

   using namespace std;

   // get 4 vector for Higgs rest frame
   typedef reco::Candidate::PolarLorentzVector fourv;
   fourv thisJet = jet->polarP4();
   TLorentzVector thisJetLV_H(0.,0.,0.,0.);
   thisJetLV_H.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 125. );

   std::vector<TLorentzVector> particles_H;
   std::vector<math::XYZVector> particles2_H;
   std::vector<reco::LeafCandidate> particles3_H;
   
   double sumPz = 0;
   double sumP = 0;

   // Boost to Higgs rest frame
   for(unsigned int i = 0; i < daughtersOfJet.size(); i++){
      // Do not include low mass subjets
      if (daughtersOfJet[i]->pt() < 0.5) continue;
   
      // Create 4 vector to boost to Higgs frame
      TLorentzVector thisParticleLV_H( daughtersOfJet[i]->px(), daughtersOfJet[i]->py(), daughtersOfJet[i]->pz(), daughtersOfJet[i]->energy() );
   
      // Boost to Higgs rest frame
      thisParticleLV_H.Boost( -thisJetLV_H.BoostVector() );
      particles_H.push_back( thisParticleLV_H );	
      particles2_H.push_back( math::XYZVector( thisParticleLV_H.X(), thisParticleLV_H.Y(), thisParticleLV_H.Z() ));
      particles3_H.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( thisParticleLV_H.X(), thisParticleLV_H.Y(), 
                                                                                      thisParticleLV_H.Z(), thisParticleLV_H.T() ) ));

      // Sum rest frame momenta for asymmetry calculation
      if (daughtersOfJet[i]->pt() < 10) continue;
      sumPz += thisParticleLV_H.Pz();
      sumP += abs( thisParticleLV_H.P() );
   }
   
   // Fox Wolfram Moments
   double fwm_H[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( particles_H, fwm_H);
   treeVars["FoxWolfH1_Higgs"] = fwm_H[1];
   treeVars["FoxWolfH2_Higgs"] = fwm_H[2];
   treeVars["FoxWolfH3_Higgs"] = fwm_H[3];
   treeVars["FoxWolfH4_Higgs"] = fwm_H[4];
   
   // Event Shape Variables
   EventShapeVariables eventShapes_H( particles2_H );
   Thrust thrustCalculator_H( particles3_H.begin(), particles3_H.end() );
   treeVars["isotropy_Higgs"] = eventShapes_H.isotropy();
   treeVars["sphericity_Higgs"] = eventShapes_H.sphericity(2);
   treeVars["aplanarity_Higgs"] = eventShapes_H.aplanarity(2);
   treeVars["thrust_Higgs"] = thrustCalculator_H.thrust();

   // Jet Asymmetry
   double asymmetry = sumPz/sumP;
   treeVars["asymmetry_Higgs"] = asymmetry;
}
