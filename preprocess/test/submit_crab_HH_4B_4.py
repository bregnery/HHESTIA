from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'HH_4B_HHESTIA_input_4'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_HH_4B.py'
config.JobType.outputFiles = ['preprocess_HHESTIA_HH_4B.root']
config.JobType.allowUndistributedCMSSW = True 

config.Data.inputDataset = '/RadionTohhTohbbhbb_narrow_M-1600_13TeV-madgraph/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
