// -*- C++ -*-
//========================================================================================
// Package:    BEST/preprocess                  ---------------------------------------
// Class:      BESTProducer                     ---------------------------------------
//----------------------------------------------------------------------------------------
/**\class BESTProducer BESTProducer.cc BEST/preprocess/plugins/BESTProducer.cc
------------------------------------------------------------------------------------------
 Description: This class preprocesses MC samples so that they can be used with BEST ---
 -----------------------------------------------------------------------------------------
 Implementation:                                                                       ---
     This EDProducer is meant to be used with CMSSW_9_4_8                              ---
*/
//========================================================================================
// Authors:  Brendan Regnery, Justin Pilot, Reyer Band, Devin Taylor ---------------------
//         Created:  WED, 8 Aug 2018 21:00:28 GMT  ---------------------------------------
//========================================================================================
//////////////////////////////////////////////////////////////////////////////////////////


// system include files
#include <memory>
#include <thread>
#include <iostream>

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
#include "BESTtoolbox.h"

///////////////////////////////////////////////////////////////////////////////////
// Define a namespace -------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

namespace best {

    // enumerate possible jet types
    enum JetType { H, t, W, Z, Q};

    // create a struct to help with mapping string label to enum value
    struct JetTypeStringToEnum {
        const char label;
        JetType value;
    };

    // Create a mapping from the input jetType to enum
    JetType jetTypeFromString(const std::string& label) {
        static const JetTypeStringToEnum jetTypeStringToEnumMap[] = {
            {'H', H},
            {'t', t},
            {'W', W},
            {'Z', Z},
            {'Q', Q}
        };

        JetType value = (JetType)-1;
        bool found = false;
        for (int i = 0; jetTypeStringToEnumMap[i].label && (!found); ++i){
            if (!strcmp(label.c_str(), &jetTypeStringToEnumMap[i].label) ) {
                found = true;
                value = jetTypeStringToEnumMap[i].value;
            }
        }

        // Throw an error if user inputs an unrecognized type
        if (!found){
            throw cms::Exception("JetTypeError") << label << " is not a recognized JetType";
        }

        return value;
    }
}


///////////////////////////////////////////////////////////////////////////////////
// Class declaration --------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////

class BESTProducer : public edm::stream::EDProducer<> {
   public:
      explicit BESTProducer(const edm::ParameterSet&);
      ~BESTProducer();

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
      best::JetType jetType_;

      // Tree variables
      TTree *jetTree;
      std::map<std::string, float> treeVars;
      std::vector<std::string> listOfVars;
      std::map<std::string, std::vector<float> > jetPFcand;
      std::vector<std::string> listOfJetPFvars;

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

BESTProducer::BESTProducer(const edm::ParameterSet& iConfig):
    inputJetColl_ (iConfig.getParameter<std::string>("inputJetColl")),
    jetType_ (best::jetTypeFromString(iConfig.getParameter<std::string>("jetType")))
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

    listOfVars.push_back("FoxWolfH1_Top");
    listOfVars.push_back("FoxWolfH2_Top");
    listOfVars.push_back("FoxWolfH3_Top");
    listOfVars.push_back("FoxWolfH4_Top");

    listOfVars.push_back("FoxWolfH1_W");
    listOfVars.push_back("FoxWolfH2_W");
    listOfVars.push_back("FoxWolfH3_W");
    listOfVars.push_back("FoxWolfH4_W");

    listOfVars.push_back("FoxWolfH1_Z");
    listOfVars.push_back("FoxWolfH2_Z");
    listOfVars.push_back("FoxWolfH3_Z");
    listOfVars.push_back("FoxWolfH4_Z");

    // Event Shape Variables
    listOfVars.push_back("isotropy_Higgs");
    listOfVars.push_back("sphericity_Higgs");
    listOfVars.push_back("aplanarity_Higgs");
    listOfVars.push_back("thrust_Higgs");

    listOfVars.push_back("isotropy_Top");
    listOfVars.push_back("sphericity_Top");
    listOfVars.push_back("aplanarity_Top");
    listOfVars.push_back("thrust_Top");

    listOfVars.push_back("isotropy_W");
    listOfVars.push_back("sphericity_W");
    listOfVars.push_back("aplanarity_W");
    listOfVars.push_back("thrust_W");

    listOfVars.push_back("isotropy_Z");
    listOfVars.push_back("sphericity_Z");
    listOfVars.push_back("aplanarity_Z");
    listOfVars.push_back("thrust_Z");

    // Jet Asymmetry
    listOfVars.push_back("asymmetry_Higgs");
    listOfVars.push_back("asymmetry_Top");
    listOfVars.push_back("asymmetry_W");
    listOfVars.push_back("asymmetry_Z");

    // Jet PF Candidate Variables
    listOfJetPFvars.push_back("jet_PF_candidate_pt");
    listOfJetPFvars.push_back("jet_PF_candidate_phi");
    listOfJetPFvars.push_back("jet_PF_candidate_eta");

    listOfJetPFvars.push_back("HiggsFrame_PF_candidate_px");
    listOfJetPFvars.push_back("HiggsFrame_PF_candidate_py");
    listOfJetPFvars.push_back("HiggsFrame_PF_candidate_pz");
    listOfJetPFvars.push_back("HiggsFrame_PF_candidate_energy");
    listOfJetPFvars.push_back("HiggsFrame_subjet_px");
    listOfJetPFvars.push_back("HiggsFrame_subjet_py");
    listOfJetPFvars.push_back("HiggsFrame_subjet_pz");
    listOfJetPFvars.push_back("HiggsFrame_subjet_energy");

    listOfJetPFvars.push_back("TopFrame_PF_candidate_px");
    listOfJetPFvars.push_back("TopFrame_PF_candidate_py");
    listOfJetPFvars.push_back("TopFrame_PF_candidate_pz");
    listOfJetPFvars.push_back("TopFrame_PF_candidate_energy");
    listOfJetPFvars.push_back("TopFrame_subjet_px");
    listOfJetPFvars.push_back("TopFrame_subjet_py");
    listOfJetPFvars.push_back("TopFrame_subjet_pz");
    listOfJetPFvars.push_back("TopFrame_subjet_energy");

    listOfJetPFvars.push_back("WFrame_PF_candidate_px");
    listOfJetPFvars.push_back("WFrame_PF_candidate_py");
    listOfJetPFvars.push_back("WFrame_PF_candidate_pz");
    listOfJetPFvars.push_back("WFrame_PF_candidate_energy");
    listOfJetPFvars.push_back("WFrame_subjet_px");
    listOfJetPFvars.push_back("WFrame_subjet_py");
    listOfJetPFvars.push_back("WFrame_subjet_pz");
    listOfJetPFvars.push_back("WFrame_subjet_energy");

    listOfJetPFvars.push_back("ZFrame_PF_candidate_px");
    listOfJetPFvars.push_back("ZFrame_PF_candidate_py");
    listOfJetPFvars.push_back("ZFrame_PF_candidate_pz");
    listOfJetPFvars.push_back("ZFrame_PF_candidate_energy");
    listOfJetPFvars.push_back("ZFrame_subjet_px");
    listOfJetPFvars.push_back("ZFrame_subjet_py");
    listOfJetPFvars.push_back("ZFrame_subjet_pz");
    listOfJetPFvars.push_back("ZFrame_subjet_energy");

    // Make Branches for each variable
    for (unsigned i = 0; i < listOfVars.size(); i++){
        treeVars[ listOfVars[i] ] = -999.99;
        jetTree->Branch( (listOfVars[i]).c_str() , &(treeVars[ listOfVars[i] ]), (listOfVars[i]+"/F").c_str() );
    }

    // Make Branches for each of the jet constituents' variables
    for (unsigned i = 0; i < listOfJetPFvars.size(); i++){
        jetTree->Branch( (listOfJetPFvars[i]).c_str() , &(jetPFcand[ listOfJetPFvars[i] ]) );
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

BESTProducer::~BESTProducer()
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
BESTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
    // This makes a TLorentz Vector for each generator Heavy Object to use for jet
    // matching
    //------------------------------------------------------------------------------
    // Please note that the Jet Type has been enumerated:
    // H -> 0, t -> 1, W -> 2, Z -> 3, Q -> 4
    //------------------------------------------------------------------------------

    // Higgs
    std::vector<TLorentzVector> genHiggs;
    if(jetType_ == 0){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == 25){
                genHiggs.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }

    // Top
    std::vector<TLorentzVector> genTop;
    if(jetType_ == 1){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == 6){
                genTop.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }

    // W
    std::vector<TLorentzVector> genW;
    if(jetType_ == 2){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == 24){
                genW.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }

    // Z
    std::vector<TLorentzVector> genZ;
    if(jetType_ == 3){
        for (vector<reco::GenParticle>::const_iterator genBegin = genPart.begin(), genEnd = genPart.end(), ipart = genBegin; ipart != genEnd; ++ipart){
            if(abs(ipart->pdgId() ) == 23){
                genZ.push_back( TLorentzVector(ipart->px(), ipart->py(), ipart->pz(), ipart->energy() ) );
            }
        }
    }



    //------------------------------------------------------------------------------
    // AK8 Jet Loop ----------------------------------------------------------------
    //------------------------------------------------------------------------------
    // This loop makes a tree entry for each jet of interest -----------------------
    //------------------------------------------------------------------------------

    for (vector<pat::Jet>::const_iterator jetBegin = ak8Jets.begin(), jetEnd = ak8Jets.end(), ijet = jetBegin; ijet != jetEnd; ++ijet){

        //-------------------------------------------------------------------------------
        // AK8 Jets of interest from QCD samples -------=======--------------------------
        //-------------------------------------------------------------------------------
        if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && jetType_ == 4){

            // Store Jet Variables
            treeVars["nJets"] = ak8Jets.size();
            storeJetVariables(treeVars, ijet);

            // Secondary Vertex Variables
            TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );
            storeSecVertexVariables(treeVars, jet, secVertices);

            // Get all of the Jet's daughters
            vector<reco::Candidate * > daughtersOfJet;
            getJetDaughters(daughtersOfJet, ijet, jetPFcand);

            // Higgs Rest Frame Variables
            storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Higgs", 125.);

            // Top Rest Frame Variables
            storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Top", 172.5);

            // W Rest Frame Variables
            storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "W", 80.4);

            // Z Rest Frame Variables
            storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Z", 91.2);

            // Fill the jet entry tree
            jetTree->Fill();
        }

        //-------------------------------------------------------------------------------
        // AK8 Jets of interest from Higgs samples --------------------------------------
        //-------------------------------------------------------------------------------
        if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && jetType_ == 0){
            // gen Higgs loop
            for (size_t iHiggs = 0; iHiggs < genHiggs.size(); iHiggs++){
                TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

                // match Jet to Higgs
                if(jet.DeltaR(genHiggs[iHiggs]) < 0.1){

                    // Store Jet Variables
                    treeVars["nJets"] = ak8Jets.size();
                    storeJetVariables(treeVars, ijet);

                    // Secondary Vertex Variables
                    storeSecVertexVariables(treeVars, jet, secVertices);

                    // Get all of the Jet's daughters
                    vector<reco::Candidate * > daughtersOfJet;
                    getJetDaughters(daughtersOfJet, ijet, jetPFcand);

                    // Higgs Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Higgs", 125.);

                    // Top Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Top", 172.5);

                    // W Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "W", 80.4);

                    // Z Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Z", 91.2);

                    // Fill the jet entry tree
                    jetTree->Fill();
                }
             }
        }

        //-------------------------------------------------------------------------------
        // AK8 Jets of interest from Top samples ----------------------------------------
        //-------------------------------------------------------------------------------
        if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && jetType_ == 1){
            // gen Top loop
            for (size_t iTop = 0; iTop < genTop.size(); iTop++){
                TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

                // match Jet to Top
                if(jet.DeltaR(genTop[iTop]) < 0.1){

                    // Store Jet Variables
                    treeVars["nJets"] = ak8Jets.size();
                    storeJetVariables(treeVars, ijet);

                    // Secondary Vertex Variables
                    storeSecVertexVariables(treeVars, jet, secVertices);

                    // Get all of the Jet's daughters
                    vector<reco::Candidate * > daughtersOfJet;
                    getJetDaughters(daughtersOfJet, ijet, jetPFcand);

                    // Higgs Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Higgs", 125.);

                    // Top Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Top", 172.5);

                    // W Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "W", 80.4);

                    // Z Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Z", 91.2);

                    // Fill the jet entry tree
                    jetTree->Fill();
                }
             }
        }

        //-------------------------------------------------------------------------------
        // AK8 Jets of interest from W samples ------------------------------------------
        //-------------------------------------------------------------------------------
        if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && jetType_ == 2){
            // gen W loop
            for (size_t iW = 0; iW < genW.size(); iW++){
                TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

                // match Jet to W
                if(jet.DeltaR(genW[iW]) < 0.1){

                    // Store Jet Variables
                    treeVars["nJets"] = ak8Jets.size();
                    storeJetVariables(treeVars, ijet);

                    // Secondary Vertex Variables
                    storeSecVertexVariables(treeVars, jet, secVertices);

                    // Get all of the Jet's daughters
                    vector<reco::Candidate * > daughtersOfJet;
                    getJetDaughters(daughtersOfJet, ijet, jetPFcand);

                    // Higgs Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Higgs", 125.);

                    // Top Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Top", 172.5);

                    // W Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "W", 80.4);

                    // Z Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Z", 91.2);

                    // Fill the jet entry tree
                    jetTree->Fill();
                }
             }
         }

        //-------------------------------------------------------------------------------
        // AK8 Jets of interest from Z samples --------------------------------------
        //-------------------------------------------------------------------------------
        if(ijet->numberOfDaughters() >= 2 && ijet->pt() >= 500 && ijet->userFloat("ak8PFJetsCHSSoftDropMass") > 40 && jetType_ == 3){
            // gen Z loop
            for (size_t iZ = 0; iZ < genZ.size(); iZ++){
                TLorentzVector jet(ijet->px(), ijet->py(), ijet->pz(), ijet->energy() );

                // match Jet to Z
                if(jet.DeltaR(genZ[iZ]) < 0.1){

                    // Store Jet Variables
                    treeVars["nJets"] = ak8Jets.size();
                    storeJetVariables(treeVars, ijet);

                    // Secondary Vertex Variables
                    storeSecVertexVariables(treeVars, jet, secVertices);

                    // Get all of the Jet's daughters
                    vector<reco::Candidate * > daughtersOfJet;
                    getJetDaughters(daughtersOfJet, ijet, jetPFcand);

                    // Higgs Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Higgs", 125.);

                    // Top Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Top", 172.5);

                    // W Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "W", 80.4);

                    // Z Rest Frame Variables
                    storeRestFrameVariables(treeVars, daughtersOfJet, ijet, jetPFcand, "Z", 91.2);

                    // Fill the jet entry tree
                    jetTree->Fill();
                }
             }
        }

        //-------------------------------------------------------------------------------
        // Clear and Reset all tree variables -------------------------------------------
        //-------------------------------------------------------------------------------
        for (unsigned i = 0; i < listOfVars.size(); i++){
            treeVars[ listOfVars[i] ] = -999.99;
        }
        for (unsigned i = 0; i < listOfJetPFvars.size(); i++){
            jetPFcand[ listOfJetPFvars[i] ] = vector<float>();
        }
     }
}


//=================================================================================
// Method called once each job just before starting event loop  -------------------
//=================================================================================

void
BESTProducer::beginStream(edm::StreamID)
{
}

//=================================================================================
// Method called once each job just after ending the event loop  ------------------
//=================================================================================

void
BESTProducer::endStream()
{
}

//=================================================================================
// Method fills 'descriptions' with the allowed parameters for the module  --------
//=================================================================================

void
BESTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BESTProducer);
