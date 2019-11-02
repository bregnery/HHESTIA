import glob
import FWCore.ParameterSet.Config as cms
from JMEAnalysis.JetToolbox.jetToolbox_cff import jetToolbox
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.GlobalTag = GlobalTag(process.GlobalTag, '80X_mcRun2_asymptotic_v4')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1),
                                        allowUnscheduled = cms.untracked.bool(True))

# Get file names with unix pattern expander
#files = glob.glob("/afs/cern.ch/work/b/bregnery/public/HHwwwwMCgenerator/CMSSW_8_0_21/src/hhMCgenerator/RootFiles/M3500/*.root")
# Add to the beginning of each filename
#for ifile in range(len(files)):
#    files[ifile] = "file:" + files[ifile]

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/FA7A80D9-6CEF-E611-9933-FA163E036391.root',
        'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/F66461CC-58EF-E611-AC38-02163E011598.root',
        'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/EEE520BA-5FEF-E611-927D-0025904B2C78.root',
        'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/EAEA38C0-8BEF-E611-ADE3-001E67504F55.root',
        'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/E2B077CF-66EF-E611-85B5-FA163EAE2465.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/68EB3DA7-58EF-E611-9FBF-02163E01616C.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/76B3BF52-5FEF-E611-8082-FA163E4BA1F1.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/7E3ED0CD-D1EF-E611-A5DF-FA163E6AA32D.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/A6FFA4E6-73EF-E611-B469-02163E015C9A.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/B8222974-50EF-E611-8899-FA163E5D03E3.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/BA1255D1-66EF-E611-A3D5-0CC47A04486C.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/BA7CEBA7-58EF-E611-8F1D-02163E013B43.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/C0B11460-5FEF-E611-A2FE-FA163E36D13A.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/C8229465-C0EF-E611-BEA8-02163E00AF5F.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/DC11191B-BFEF-E611-818D-FA163E86100C.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/E0DBB9E5-6CEF-E611-8AEE-FA163E7024AE.root',
	'root://eospublic.cern.ch//eos/opendata/cms/MonteCarlo2016/RunIISummer16MiniAODv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/MINIAODSIM/PUMoriond17_magnetOn_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/E2597BB8-98F5-E611-BF80-001E674DA1AD.root',
	)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# Adjust the jet collection to include tau4
jetToolbox( process, 'ak8', 'jetsequence', 'out',
    updateCollection = 'slimmedJetsAK8',
    JETCorrPayload= 'AK8PFchs',
    addNsub = True,
    maxTau = 4
)

# Apply a jet preselection
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
    src = cms.InputTag('selectedPatJetsAK8PFCHS'),
    cut = cms.string('pt > 100.0 && abs(eta) < 2.4'),
    filter = cms.bool(True)
)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(99999),
    src = cms.InputTag("selectedAK8Jets"),
    filter = cms.bool(True)
)

# Run the producer
process.run = cms.EDProducer('BESTProducer',
	inputJetColl = cms.string('selectedAK8Jets'),
        isSignal = cms.bool(False)
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("preprocess_BEST_OpenData_QCD.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               #SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*run*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

# Organize the running process
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)
