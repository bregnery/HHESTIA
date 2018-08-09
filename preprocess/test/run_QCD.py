import glob
import FWCore.ParameterSet.Config as cms


process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Get file names with unix pattern expander
#files = glob.glob("/afs/cern.ch/work/b/bregnery/public/HHwwwwMCgenerator/CMSSW_8_0_21/src/hhMCgenerator/RootFiles/M3500/*.root")
# Add to the beginning of each filename
#for ifile in range(len(files)):
#    files[ifile] = "file:" + files[ifile]

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
	'root://cms-xrd-global.cern.ch//store/mc/RunIISpring16MiniAODv2/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/70000/847868B8-7E26-E611-9BEA-02163E01413E.root'
	)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.run = cms.EDProducer('HHESTIAProducer')

process.TFileService = cms.Service("TFileService", fileName = cms.string("preprocess_HHESTIA_QCD.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*run*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

process.p = cms.Path(process.run)
