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
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
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

      //edm::EDGetTokenT<std::vector<pat::Jet> > ak8CHSSoftDropSubjetsToken_;
      //edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;

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

      // AK8 Jets of interest from non-signal samples
      if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && isSignal_ == false){

         // Store Jet Variables
         treeVars["nJets"] = ak8Jets.size();
         treeVars["jetAK8_phi"] = ijet->phi();
         treeVars["jetAK8_eta"] = ijet->eta(); 
         treeVars["jetAK8_pt"] = ijet->pt(); 
         treeVars["jetAK8_mass"] = ijet->mass(); 
         treeVars["jetAK8_SoftDropMass"] = ijet->userFloat("ak8PFJetsCHSSoftDropMass");

         // Fill the jet entry tree
         jetTree->Fill();
      }

      // AK8 Jets of interest from signal samples
      if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && isSignal_ == true){
         // gen Higgs loop
         for (size_t iHiggs = 0; iHiggs < genHiggs.size(); iHiggs++){
            TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

            // match Jet to Higgs
            if(jet.DeltaR(genHiggs[iHiggs]) < 0.1){

               // Store Jet Variables
               treeVars["nJets"] = ak8Jets.size();
               treeVars["jetAK8_phi"] = ijet->phi();
               treeVars["jetAK8_eta"] = ijet->eta(); 
               treeVars["jetAK8_pt"] = ijet->pt(); 
               treeVars["jetAK8_mass"] = ijet->mass(); 
               treeVars["jetAK8_SoftDropMass"] = ijet->userFloat("ak8PFJetsCHSSoftDropMass");
     
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
