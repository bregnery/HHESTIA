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
// Sort Jets ----------------------------------------------------------------------
//---------------------------------------------------------------------------------
// This function takes in a Jet handle and returns a handle with the top four   ---
// jets sorted by pT                                                            ---
//---------------------------------------------------------------------------------
 
std::vector<pat::Jet> * sortJets(std::vector<pat::Jet> *jets)
{
   std::vector<pat::Jet> *sortedJets = new std::vector<pat::Jet>;
   pat::Jet *jet1 = new pat::Jet;
   pat::Jet *jet2 = new pat::Jet;
   pat::Jet *jet3 = new pat::Jet;
   pat::Jet *jet4 = new pat::Jet;
   
   // Get leading jet
   if(jets->size() > 0){
      for (std::vector<pat::Jet>::const_iterator jetBegin = jets->begin(), jetEnd = jets->end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
          if(!jet1){
             *jet1 = *ijet;
          }
          if(jet1 && ijet->pt() > jet1->pt() ){
             *jet1 = *ijet;
          }
       }
       sortedJets->push_back(*jet1);
       //std::cout << "Jet 1 pT: " << jet1->pt() << std::endl;
   }
   
   // Get subleading jet
   if(jets->size() > 1){
      for (std::vector<pat::Jet>::const_iterator jetBegin = jets->begin(), jetEnd = jets->end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
          if(!jet2 && jet1->pt() > ijet->pt() ){
             *jet2 = *ijet;
          }
          if(jet2 && jet1->pt() > ijet->pt() && ijet->pt() > jet2->pt() ){
             *jet2 = *ijet;
          }
       }
       sortedJets->push_back(*jet2);
       //std::cout << "Jet 2 pT: " << jet2->pt() << std::endl;
   } 

   // Get third leading jet
   if(jets->size() > 2){
      for (std::vector<pat::Jet>::const_iterator jetBegin = jets->begin(), jetEnd = jets->end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
          if(!jet3 && jet2->pt() > ijet->pt() ){
             *jet3 = *ijet;
          }
          if(jet3 && jet2->pt() > ijet->pt() && ijet->pt() > jet3->pt() ){
             *jet3 = *ijet;
          }
       }
       sortedJets->push_back(*jet3);
       //std::cout << "Jet 3 pT: " << jet3->pt() << std::endl;
   } 

   // Get fourth leading jet
   if(jets->size() > 3){
      for (std::vector<pat::Jet>::const_iterator jetBegin = jets->begin(), jetEnd = jets->end(), ijet = jetBegin; ijet != jetEnd; ++ijet){
          if(!jet4 && jet3->pt() > ijet->pt() ){
             *jet4 = *ijet;
          }
          if(jet4 && jet3->pt() > ijet->pt() && ijet->pt() > jet4->pt() ){
             *jet4 = *ijet;
          }
       }
       sortedJets->push_back(*jet4);
       //std::cout << "Jet 4 pT: " << jet4->pt() << std::endl;
   }

   return sortedJets;
}

//=================================================================================
// Calculate delta Phi ////////////////////////////////////////////////////////////
//---------------------------------------------------------------------------------
// phi1 and phi2 are azimuthal angles in the detector -----------------------------
//---------------------------------------------------------------------------------

float myDeltaPhi(float phi1, float phi2)
{
   double pi = 3.14159265358979323846;
   int delPhi = phi1-phi2;
   while(delPhi > pi){
      delPhi -= 2*pi;
   }
   while(delPhi <= -pi){
      delPhi += 2*pi;
   }
   return delPhi;
}

//=================================================================================
// Calculate delta R //////////////////////////////////////////////////////////////
//---------------------------------------------------------------------------------
// phi is the azimuthal angle in the detector -------------------------------------
// eta is psuedorapidity ----------------------------------------------------------
//---------------------------------------------------------------------------------

float myDeltaR(float eta1, float phi1, float eta2, float phi2)
{
   float delEta = eta1 - eta2;
   float delPhi = myDeltaPhi(phi1, phi2);
   float delR = TMath::Sqrt(delEta*delEta + delPhi*delPhi);
   return delR;
}

//=================================================================================
// Match using delta R cone ///////////////////////////////////////////////////////
//---------------------------------------------------------------------------------
// etaPhi is a 2D array of eta phi values ( [ [eta1, phi1], [eta2, phi2], ...] ) --
// limR is the upper bound of delta R for matching --------------------------------
// returns an array where true means matched --------------------------------------
//---------------------------------------------------------------------------------

std::vector<std::vector<bool> > deltaRMatch(std::vector<std::array<float, 2> > etaPhi1, std::vector<std::array<float, 2> > etaPhi2, float limR)
{
   std::vector<std::vector<bool> > matchList;
   bool match = false;

   for(std::size_t i = 0; i < etaPhi1.size(); i++){

      std::vector<bool> nestedMatchList;

      for(std::size_t j = 0; j < etaPhi2.size(); j++){
         float delR = myDeltaR(etaPhi1[i][0], etaPhi1[i][1], etaPhi2[j][0], etaPhi2[j][1]);
         if(delR < limR) match = true;
         if(delR > limR) match = false;
         nestedMatchList.push_back(match);
      }
      matchList.push_back(nestedMatchList);
   }
   return matchList;
}

