// -*- C++ -*-
//========================================================================================
// Package:    HHESTIA/preprocess                  ---------------------------------------
// Class:      HHESTIAProducer                     ---------------------------------------
//----------------------------------------------------------------------------------------
/**\class HHESTIAProducer HHESTIAProducer.cc HHESTIA/preprocess/plugins/HHESTIAProducer.cc
------------------------------------------------------------------------------------------
 Description: This class preprocesses MC samples so that they can be used with HHESTIA ---
 -----------------------------------------------------------------------------------------
 Implementation:                                                                       ---
     This EDProducer is meant to be used with CMSSW_9_4_8                              ---
*/
//========================================================================================
// Authors:  Brendan Regnery, Justin Pilot         ---------------------------------------
//         Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////


// system include files
#include <memory>

// FWCore include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Data Formats and tools include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "DataFormats/VertexReco/interface/VertexFwd.h"
//#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"

// Fast Jet Include files
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include "fastjet/tools/Filter.hh"
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ActiveAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>

// ROOT include files
#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TLorentzVector.h"
#include "TCanvas.h"

// user made files
#include "edanalyzerTools.h"

///////////////////////////////////////////////////////////////////////////////////
// Class declaration --------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

class HHESTIAProducer : public edm::stream::EDProducer<> {
   public:
      explicit HHESTIAProducer(const edm::ParameterSet&);
      ~HHESTIAProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      //===========================================================================
      // User functions -----------------------------------------------------------
      //===========================================================================

      //float LegP(float x, int order);
      //int FWMoments( std::vector<TLorentzVector> particles, double (&outputs)[5] );
      //void pboost( TVector3 pbeam, TVector3 plab, TLorentzVector &pboo );	

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      //===========================================================================
      // Member Data --------------------------------------------------------------
      //===========================================================================

      // Input variables
      std::string inputJetColl_;
      bool isSignal_;

      // Tree variables
      TTree *jetTree;
      std::map<std::string, float> treeVars;
      std::vector<std::string> listOfVars;

      // Tokens
      //edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pfCandsToken_;
      edm::EDGetTokenT<std::vector<pat::Jet> > ak8JetsToken_;
      //edm::EDGetTokenT<std::vector<pat::Jet> > ak4JetsToken_;
      edm::EDGetTokenT<std::vector<reco::GenParticle> > genPartToken_;
      edm::EDGetTokenT<std::vector<reco::VertexCompositePtrCandidate> > secVerticesToken_;
      edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;

      //edm::EDGetTokenT<std::vector<pat::Jet> > ak8CHSSoftDropSubjetsToken_;

      //edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
      //edm::EDGetTokenT<bool> BadChCandFilterToken_;
      //edm::EDGetTokenT<bool> BadPFMuonFilterToken_;
};

///////////////////////////////////////////////////////////////////////////////////
// constants, enums and typedefs --------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// static data member definitions -------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// Constructors -------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

HHESTIAProducer::HHESTIAProducer(const edm::ParameterSet& iConfig):
   inputJetColl_ (iConfig.getParameter<std::string>("inputJetColl")),
   isSignal_ (iConfig.getParameter<bool>("isSignal"))
{

   //------------------------------------------------------------------------------
   // Prepare TFile Service -------------------------------------------------------
   //------------------------------------------------------------------------------

   edm::Service<TFileService> fs;
   jetTree = fs->make<TTree>("jetTree","jetTree");

   //------------------------------------------------------------------------------
   // Create tree variables and branches ------------------------------------------
   //------------------------------------------------------------------------------

   // AK8 jet variables
   listOfVars.push_back("nJets");
 
   listOfVars.push_back("jetAK8_phi");
   listOfVars.push_back("jetAK8_eta");
   listOfVars.push_back("jetAK8_pt");
   listOfVars.push_back("jetAK8_mass");
   listOfVars.push_back("jetAK8_SoftDropMass");

   // Vertex Variables
   listOfVars.push_back("nSecondaryVertices");
   listOfVars.push_back("SV_1_pt");
   listOfVars.push_back("SV_1_eta");
   listOfVars.push_back("SV_1_phi");
   listOfVars.push_back("SV_1_mass");
   listOfVars.push_back("SV_1_nTracks");
   listOfVars.push_back("SV_1_chi2");
   listOfVars.push_back("SV_1_Ndof");

   listOfVars.push_back("SV_2_pt");
   listOfVars.push_back("SV_2_eta");
   listOfVars.push_back("SV_2_phi");
   listOfVars.push_back("SV_2_mass");
   listOfVars.push_back("SV_2_nTracks");
   listOfVars.push_back("SV_2_chi2");
   listOfVars.push_back("SV_2_Ndof");

   listOfVars.push_back("SV_3_pt");
   listOfVars.push_back("SV_3_eta");
   listOfVars.push_back("SV_3_phi");
   listOfVars.push_back("SV_3_mass");
   listOfVars.push_back("SV_3_nTracks");
   listOfVars.push_back("SV_3_chi2");
   listOfVars.push_back("SV_3_Ndof");

   // nsubjettiness
   listOfVars.push_back("jetAK8_Tau4");
   listOfVars.push_back("jetAK8_Tau3");
   listOfVars.push_back("jetAK8_Tau2");
   listOfVars.push_back("jetAK8_Tau1");

   // Fox Wolfram Moments
   listOfVars.push_back("FoxWolfH1_Higgs");
   listOfVars.push_back("FoxWolfH2_Higgs");
   listOfVars.push_back("FoxWolfH3_Higgs");
   listOfVars.push_back("FoxWolfH4_Higgs");

   // Event Shape Variables
   listOfVars.push_back("isotropy_Higgs");
   listOfVars.push_back("sphericity_Higgs");
   listOfVars.push_back("aplanarity_Higgs");
   listOfVars.push_back("thrust_Higgs");

   // Jet Asymmetry
   listOfVars.push_back("asymmetry_Higgs");

   // Make Branches for each variable
   for (unsigned i = 0; i < listOfVars.size(); i++){
      treeVars[ listOfVars[i] ] = -999.99;
      jetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
   }

   //------------------------------------------------------------------------------
   // Define input tags -----------------------------------------------------------
   //------------------------------------------------------------------------------

   // AK8 Jets
   edm::InputTag ak8JetsTag_;
   //ak8JetsTag_ = edm::InputTag("slimmedJetsAK8", "", "PAT");
   ak8JetsTag_ = edm::InputTag(inputJetColl_, "", "run");
   ak8JetsToken_ = consumes<std::vector<pat::Jet> >(ak8JetsTag_);

   // Gen Particles
   edm::InputTag genPartTag_;
   genPartTag_ = edm::InputTag("prunedGenParticles", "", "PAT"); 
   genPartToken_ = consumes<std::vector<reco::GenParticle> >(genPartTag_);

   // Primary Vertices
   edm::InputTag verticesTag_;
   verticesTag_ = edm::InputTag("offlineSlimmedPrimaryVertices", "", "PAT");
   verticesToken_ = consumes<std::vector<reco::Vertex> >(verticesTag_);

   // Secondary Vertices
   edm::InputTag secVerticesTag_;
   secVerticesTag_ = edm::InputTag("slimmedSecondaryVertices", "", "PAT");
   secVerticesToken_ = consumes<std::vector<reco::VertexCompositePtrCandidate> >(secVerticesTag_);
}

///////////////////////////////////////////////////////////////////////////////////
// Destructor ---------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

HHESTIAProducer::~HHESTIAProducer()
{

   // do anything that needs to be done at destruction time
   // (eg. close files, deallocate, resources etc.)
 
}

///////////////////////////////////////////////////////////////////////////////////
// Member Functions ---------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

//=================================================================================
// Method called for each event ---------------------------------------------------
//=================================================================================

void
HHESTIAProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace fastjet;
   using namespace std;

   typedef reco::Candidate::PolarLorentzVector fourv;

   //------------------------------------------------------------------------------
   // Create miniAOD object collections -------------------------------------------
   //------------------------------------------------------------------------------
   
   // Find objects corresponding to the token and link to the handle
   Handle< std::vector<pat::Jet> > ak8JetsCollection;
   iEvent.getByToken(ak8JetsToken_, ak8JetsCollection);
   vector<pat::Jet> ak8Jets = *ak8JetsCollection.product();
 
   Handle< std::vector<reco::GenParticle> > genPartCollection;
   iEvent.getByToken(genPartToken_, genPartCollection);
   vector<reco::GenParticle> genPart = *genPartCollection.product();

   Handle< std::vector<reco::Vertex> > vertexCollection;
   iEvent.getByToken(verticesToken_, vertexCollection);
   vector<reco::Vertex> pVertices = *vertexCollection.product();

   Handle< std::vector<reco::VertexCompositePtrCandidate> > secVertexCollection;
   iEvent.getByToken(secVerticesToken_, secVertexCollection);
   vector<reco::VertexCompositePtrCandidate> secVertices = *secVertexCollection.product();

   //------------------------------------------------------------------------------
   // Gen Particles Loop ----------------------------------------------------------
   //------------------------------------------------------------------------------
   // This makes a TLorentz Vector for each generator Higgs to use for jet matching
   //------------------------------------------------------------------------------

   std::vector<TLorentzVector> genHiggs;
   for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
      if(abs(ipart->pdgId() ) == 25){
         genHiggs.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
      }
   }

   //------------------------------------------------------------------------------
   // AK8 Jet Loop ----------------------------------------------------------------
   //------------------------------------------------------------------------------
   // This loop makes a tree entry for each jet of interest -----------------------
   //------------------------------------------------------------------------------

   for (vector<pat::Jet>::const_iterator jetBegin = ak8Jets.begin(), jetEnd = ak8Jets.end(), ijet = jetBegin; ijet != jetEnd; ++ijet){

      //-------------------------------------------------------------------------------
      // AK8 Jets of interest from non-signal samples ---------------------------------
      //-------------------------------------------------------------------------------
      if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && isSignal_ == false){

         // Store Jet Variables
         treeVars["nJets"] = ak8Jets.size();
         storeJetVariables(treeVars, ijet);

         // Secondary Vertex Variables
         TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );
         int numMatched = 0;
         for(vector<reco::VertexCompositePtrCandidate>::const_iterator vertBegin = secVertices.begin(), vertEnd = secVertices.end(), ivert = vertBegin; ivert != vertEnd; ivert++){
            TLorentzVector vert(ivert->px(), ivert->py(), ivert->pz(), ivert->energy() );
            // match vertices to jet
            if(jet.DeltaR(vert) < 0.8 ){
               numMatched++;
               // fill secondary vertex info
               if(numMatched <= 3){
                  string i = to_string(numMatched);
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

         // get 4 vector for Higgs rest frame
         fourv thisJet = ijet->polarP4();
         TLorentzVector thisJetLV_H(0.,0.,0.,0.);
         thisJetLV_H.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 125. );

         std::vector<TLorentzVector> particles_H;
         std::vector<math::XYZVector> particles2_H;
         std::vector<reco::LeafCandidate> particles3_H;

         double sumPz = 0;
         double sumP = 0;

         // Get all of the Jet's daughters
         vector<reco::Candidate * > daughtersOfJet;
            // First get all daughters for the first Soft Drop Subjet
	 for (unsigned int i = 0; i < ijet->daughter(0)->numberOfDaughters(); i++){
            daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(0)->daughter(i) );
	 }
            // Get all daughters for the second Soft Drop Subjet
	 for (unsigned int i = 0; i < ijet->daughter(1)->numberOfDaughters(); i++){
            daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(1)->daughter(i));
	 }
            // Get all daughters not included in Soft Drop
	 for (unsigned int i = 2; i< ijet->numberOfDaughters(); i++){
            daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(i) );
	 }

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
            sumP += abs(thisParticleLV_H.P() );
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

         // Fill the jet entry tree
         jetTree->Fill();
      }

      //-------------------------------------------------------------------------------
      // AK8 Jets of interest from signal samples -------------------------------------
      //-------------------------------------------------------------------------------
      int numJet = 0;
      if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && isSignal_ == true){
         // gen Higgs loop
         for (size_t iHiggs = 0; iHiggs < genHiggs.size(); iHiggs++){
            TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );
           
            numJet++; 
            // match Jet to Higgs
            if(jet.DeltaR(genHiggs[iHiggs]) < 0.1){

               // Store Jet Variables
               treeVars["nJets"] = ak8Jets.size();
               storeJetVariables(treeVars, ijet);

               // Secondary Vertex Variables
               int numMatched = 0;
               for(vector<reco::VertexCompositePtrCandidate>::const_iterator vertBegin = secVertices.begin(), vertEnd = secVertices.end(), ivert = vertBegin; ivert != vertEnd; ivert++){
                  TLorentzVector vert(ivert->px(), ivert->py(), ivert->pz(), ivert->energy() );
                  // match vertices to jet
                  if(jet.DeltaR(vert) < 0.8 ){
                     numMatched++;
                     // fill secondary vertex info
                     if(numMatched <= 3){
                        string i = to_string(numMatched);
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

               // get 4 vector for Higgs rest frame
               fourv thisJet = ijet->polarP4();
               TLorentzVector thisJetLV_H(0.,0.,0.,0.);
               thisJetLV_H.SetPtEtaPhiM(thisJet.Pt(), thisJet.Eta(), thisJet.Phi(), 125. );

               std::vector<TLorentzVector> particles_H;
               std::vector<math::XYZVector> particles2_H;
               std::vector<reco::LeafCandidate> particles3_H;
      
               double sumPz = 0;
               double sumP = 0;

               // Get all of the Jet's daughters
               vector<reco::Candidate * > daughtersOfJet;
                  // First get all daughters for the first Soft Drop Subjet
               for (unsigned int i = 0; i < ijet->daughter(0)->numberOfDaughters(); i++){
                  daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(0)->daughter(i) );
               }
                  // Get all daughters for the second Soft Drop Subjet
               for (unsigned int i = 0; i < ijet->daughter(1)->numberOfDaughters(); i++){
                  daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(1)->daughter(i));
               }
                  // Get all daughters not included in Soft Drop
               for (unsigned int i = 2; i< ijet->numberOfDaughters(); i++){
                  daughtersOfJet.push_back( (reco::Candidate *) ijet->daughter(i) );
               }
      
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

               // Fill the jet entry tree
               jetTree->Fill();
            }
          }
       }
    }
}


//=================================================================================
// Method called once each job just before starting event loop  -------------------
//=================================================================================

void 
HHESTIAProducer::beginStream(edm::StreamID)
{
}

//=================================================================================
// Method called once each job just after ending the event loop  ------------------
//=================================================================================

void 
HHESTIAProducer::endStream() 
{
}

//=================================================================================
// Method fills 'descriptions' with the allowed parameters for the module  --------
//=================================================================================

void
HHESTIAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HHESTIAProducer);
