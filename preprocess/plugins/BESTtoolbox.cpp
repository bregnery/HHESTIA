//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BESTtoolbox.cpp -------------------------------------------------------------------
//========================================================================================
// C++ file containing functions for use with CMS EDAnalyzer and EDProducer --------------
//////////////////////////////////////////////////////////////////////////////////////////

#include "BESTtoolbox.h"

//========================================================================================
// Calculate Legendre Polynomials --------------------------------------------------------
//----------------------------------------------------------------------------------------
// Simple Legendre polynomial function that can calculate up to order 4 ------------------
// Inputs: argument of the polynomial and order desired ----------------------------------
//----------------------------------------------------------------------------------------

float LegendreP(float x, int order){
   if (order == 0) return 1;
   else if (order == 1) return x;
   else if (order == 2) return 0.5*(3*x*x - 1);
   else if (order == 3) return 0.5*(5*x*x*x - 3*x);
   else if (order == 4) return 0.125*(35*x*x*x*x - 30*x*x + 3);
   else return 0;
}

//========================================================================================
// Calculate Fox Wolfram Moments ---------------------------------------------------------
//----------------------------------------------------------------------------------------
// This function calculates the Fox Wolfram moments for jet constituents -----------------
// in various rest frames. ---------------------------------------------------------------
// Inputs: particles (jet constiuents boosted to rest frame) and empty array that --------
//         that will store the FW moments ------------------------------------------------
//----------------------------------------------------------------------------------------

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

//========================================================================================
// Get All Jet Constituents --------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This gets all the jet constituents (daughters) and stores them as a standard ----------
// vector --------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

void getJetDaughters(std::vector<reco::Candidate * > &daughtersOfJet, std::vector<pat::Jet>::const_iterator jet,
                     std::map<std::string, std::vector<float> > &jetPFcand ){
   // First get all daughters for the first Soft Drop Subjet
   for (unsigned int i = 0; i < jet->daughter(0)->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(0)->daughter(i) );
      jetPFcand["jet_PF_candidate_pt"].push_back(jet->daughter(0)->daughter(i)->pt() );
      jetPFcand["jet_PF_candidate_phi"].push_back(jet->daughter(0)->daughter(i)->phi() );
      jetPFcand["jet_PF_candidate_eta"].push_back(jet->daughter(0)->daughter(i)->eta() );
   }
   // Get all daughters for the second Soft Drop Subjet
   for (unsigned int i = 0; i < jet->daughter(1)->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(1)->daughter(i));
      jetPFcand["jet_PF_candidate_pt"].push_back(jet->daughter(1)->daughter(i)->pt() );
      jetPFcand["jet_PF_candidate_phi"].push_back(jet->daughter(1)->daughter(i)->phi() );
      jetPFcand["jet_PF_candidate_eta"].push_back(jet->daughter(1)->daughter(i)->eta() );
   }
   // Get all daughters not included in Soft Drop
   for (unsigned int i = 2; i< jet->numberOfDaughters(); i++){
      daughtersOfJet.push_back( (reco::Candidate *) jet->daughter(i) );
      jetPFcand["jet_PF_candidate_pt"].push_back(jet->daughter(i)->pt() );
      jetPFcand["jet_PF_candidate_phi"].push_back(jet->daughter(i)->phi() );
      jetPFcand["jet_PF_candidate_eta"].push_back(jet->daughter(i)->eta() );
   }
}

//========================================================================================
// Store Jet Variables -------------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various jet quantaties and stores them on the map used to fill -------------
// the jet tree --------------------------------------------------------------------------
//----------------------------------------------------------------------------------------

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

//========================================================================================
// Store Secondary Vertex Information ----------------------------------------------------
//----------------------------------------------------------------------------------------
// This takes various secondary vertex quantities and stores them on the map -------------
// used to fill the tree -----------------------------------------------------------------
//----------------------------------------------------------------------------------------

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

//========================================================================================
// Store Rest Frame Variables ------------------------------------------------------------
//----------------------------------------------------------------------------------------
// This boosts an ak8 jet (and all of its constituents) into heavy object rest frame -----
// and then uses it to calculate FoxWolfram moments, Event Shape Variables, --------------
// and assymmetry variables --------------------------------------------------------------
//----------------------------------------------------------------------------------------

void storeRestFrameVariables(std::map<std::string, float> &treeVars, std::vector<reco::Candidate *> daughtersOfJet,
                            std::vector<pat::Jet>::const_iterator jet, std::map<std::string, std::vector<float> > &jetPFcand,
                            std::string frame, float mass){

   using namespace std;
   using namespace fastjet;

   // get 4 vector for heavy object rest frame
   typedef reco::Candidate::PolarLorentzVector fourv;
   fourv thisJet = jet->polarP4();
   TLorentzVector thisJetLV(0.,0.,0.,0.);
   thisJetLV.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), mass );

   std::vector<TLorentzVector> particles;
   std::vector<math::XYZVector> particles2;
   std::vector<reco::LeafCandidate> particles3;
   vector<fastjet::PseudoJet> FJparticles;

   double sumPz = 0;
   double sumP = 0;

   // Boost to Higgs rest frame
   for(unsigned int i = 0; i < daughtersOfJet.size(); i++){
      // Do not include low mass subjets
      if (daughtersOfJet[i]->pt() < 0.5) continue;

      // Create 4 vector to boost to Higgs frame
      TLorentzVector thisParticleLV( daughtersOfJet[i]->px(), daughtersOfJet[i]->py(), daughtersOfJet[i]->pz(), daughtersOfJet[i]->energy() );

      // Boost to Higgs rest frame
      thisParticleLV.Boost( -thisJetLV.BoostVector() );
      jetPFcand[frame+"Frame_PF_candidate_px"].push_back(thisParticleLV.Px() );
      jetPFcand[frame+"Frame_PF_candidate_py"].push_back(thisParticleLV.Py() );
      jetPFcand[frame+"Frame_PF_candidate_pz"].push_back(thisParticleLV.Pz() );
      jetPFcand[frame+"Frame_PF_candidate_energy"].push_back(thisParticleLV.E() );

      // Now that PF candidates are stored, make the boost axis the Z-axis
      // Important for BES variables
      //pboost( thisJetLV.Vect(), thisParticleLV.Vect(), thisParticleLV);
      particles.push_back( thisParticleLV );
      particles2.push_back( math::XYZVector( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z() ));
      particles3.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( thisParticleLV.X(), thisParticleLV.Y(),
                                                                                      thisParticleLV.Z(), thisParticleLV.T() ) ));
      FJparticles.push_back( PseudoJet( thisParticleLV.X(), thisParticleLV.Y(), thisParticleLV.Z(), thisParticleLV.T() ) );

      // Sum rest frame momenta for asymmetry calculation
      if (daughtersOfJet[i]->pt() < 10) continue;
      sumPz += thisParticleLV.Pz();
      sumP += abs( thisParticleLV.P() );
   }

   // Fox Wolfram Moments
   double fwm[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( particles, fwm);
   treeVars["FoxWolfH1_"+frame] = fwm[1];
   treeVars["FoxWolfH2_"+frame] = fwm[2];
   treeVars["FoxWolfH3_"+frame] = fwm[3];
   treeVars["FoxWolfH4_"+frame] = fwm[4];

   // Event Shape Variables
   EventShapeVariables eventShapes( particles2 );
   Thrust thrustCalculator( particles3.begin(), particles3.end() );
   treeVars["isotropy_"+frame]   = eventShapes.isotropy();
   treeVars["sphericity_"+frame] = eventShapes.sphericity(2);
   treeVars["aplanarity_"+frame] = eventShapes.aplanarity(2);
   treeVars["thrust_"+frame]     = thrustCalculator.thrust();

   // Jet Asymmetry
   double asymmetry             = sumPz/sumP;
   treeVars["asymmetry_"+frame] = asymmetry;

   // Recluster the jets in the heavy object rest frame
   JetDefinition jet_def(antikt_algorithm, 0.4);
   ClusterSequence cs(FJparticles, jet_def);
   vector<PseudoJet> jetsFJ = sorted_by_pt(cs.inclusive_jets(20.0));

   // Store recluster jet info
   for(unsigned int i = 0; i < jetsFJ.size(); i++){
      jetPFcand[frame+"Frame_subjet_px"].push_back(jetsFJ[i].px());
      jetPFcand[frame+"Frame_subjet_py"].push_back(jetsFJ[i].py());
      jetPFcand[frame+"Frame_subjet_pz"].push_back(jetsFJ[i].pz());
      jetPFcand[frame+"Frame_subjet_energy"].push_back(jetsFJ[i].e());
   }

}

//========================================================================================
// Make boost axis the rest frame z axis -------------------------------------------------
//----------------------------------------------------------------------------------------
// Given jet constituent lab momentum, find momentum relative to beam direction pbeam ----
// plab = Particle 3-vector in Boost Frame -----------------------------------------------
// pbeam = Lab Jet 3-vector --------------------------------------------------------------
//----------------------------------------------------------------------------------------

void pboost( TVector3 pbeam, TVector3 plab, TLorentzVector &pboo ){

    double pl = plab.Dot(pbeam);
    pl *= double(1. / pbeam.Mag());

    // set x axis direction along pbeam x (0,0,1)
    TVector3 pbx;

    pbx.SetX(pbeam.Y());
    pbx.SetY(-pbeam.X());
    pbx.SetZ(0.0);

    pbx *= double(1. / pbx.Mag());

    // set y axis direction along -pbx x pbeam
    TVector3 pby;

    pby = -pbx.Cross(pbeam);
    pby *= double(1. / pby.Mag());

    pboo.SetX((plab.Dot(pbx)));
    pboo.SetY((plab.Dot(pby)));
    pboo.SetZ(pl);

    // Check for errors
    if(pboo.M() <= 0.0){
        std::cout << "ERROR: PF Candidates have negative or zero mass!!!!!" << std::endl;
    }

}
