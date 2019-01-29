import glob
import FWCore.ParameterSet.Config as cms
from JMEAnalysis.JetToolbox.jetToolbox_cff import jetToolbox
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("RecoBTag.Configuration.RecoBTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, '80X_mcRun2_asymptotic_v4')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1),
                                        allowUnscheduled = cms.untracked.bool(True))

# Get file names with unix pattern expander
files = glob.glob("/afs/cern.ch/work/b/bregnery/public/HHwwwwMCgenerator/CMSSW_8_0_21/src/hhMCgenerator/RootFiles/M3500/*.root")
# Add to the beginning of each filename
for ifile in range(len(files)):
    files[ifile] = "file:" + files[ifile]

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        files
	)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#================================================================================
# Remake the Jet Collections ////////////////////////////////////////////////////
#================================================================================

# Select charged hadron subtracted packed PF candidates
#process.pfCHS = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))
#from RecoJets.JetProducers.ak8PFJets_cfi import ak8PFJets

# Define PFJetsCHS
#process.ak8PFJetsCHS = ak8PFJets.clone(src = 'pfCHS')

# Clone the existing TagInfo configurations and adapt them to MiniAOD input
#process.MyImpactParameterTagInfos = process.pfImpactParameterTagInfos.clone(
#    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),
#    candidates = cms.InputTag("packedPFCandidates"),
#    jets = cms.InputTag("ak8PFJetsCHS") # use the above-defined PF jets as input
#)
#process.MySecondaryVertexTagInfos = process.pfSecondaryVertexTagInfos.clone(
#    trackIPTagInfos = cms.InputTag("MyImpactParameterTagInfos") # use the above IP TagInfos as input
#)

# Clone the existing b-tagger configurations and use the above TagInfos as input
#process.MyTrackCountingHighEffBJetTags = process.pfTrackCountingHighEffBJetTags.clone(
#    tagInfos = cms.VInputTag(cms.InputTag("MyImpactParameterTagInfos"))
#)
#process.MySimpleSecondaryVertexHighEffBJetTags = process.pfSimpleSecondaryVertexHighEffBJetTags.clone(
#    tagInfos = cms.VInputTag(cms.InputTag("MySecondaryVertexTagInfos"))
#)

# Adjust the jet collection to include tau4
jetToolbox( process, 'ak8', 'jetsequence', 'out',
    updateCollection = 'slimmedJetsAK8',
    JETCorrPayload= 'AK8PFchs',
    addNsub = True,
    maxTau = 4
)

#================================================================================
# Prepare and run producer //////////////////////////////////////////////////////
#================================================================================

# Apply a preselction
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
    src = cms.InputTag('selectedPatJetsAK8PFCHS'),
    cut = cms.string('pt > 100.0 && abs(eta) < 2.4'),
    filter = cms.bool(True)
    #addTagInfos = cms.bool(True)
)

# Add the tag infos
#getattr(process,'selectedAK8Jets').addTagInfos = cms.bool(True)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(99999),
    src = cms.InputTag("selectedAK8Jets"),
    filter = cms.bool(True)
)

# Run the producer
process.run = cms.EDProducer('HHESTIAProducer',
	inputJetColl = cms.string('selectedAK8Jets'),
        isSignal = cms.bool(True)
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("preprocess_HHESTIA_HH.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               #SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      #'keep *_*AK8*_*_*', #'drop *',
                                                                      'keep *_*run*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

# Organize the running process
process.p = cms.Path(
#    process.pfCHS
#    * process.ak8PFJetsCHS
#    * process.MyImpactParameterTagInfos
#    * process.MyTrackCountingHighEffBJetTags
#    * process.MySecondaryVertexTagInfos
#    * process.MySimpleSecondaryVertexHighEffBJetTags
     process.selectedAK8Jets
    * process.countAK8Jets
    * process.run
)
