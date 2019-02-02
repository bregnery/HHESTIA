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
