from CRABClient.UserUtilities import config, getUsername
config = config()

config.General.requestName = 'SkimCHANNELNAME_YEAReraERANAME_streamNUMBER_PRODTAG'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'

config.JobType.psetName = 'FILE_TO_SUBMIT_PATH'
config.JobType.pyCfgParams = ['isMC=False']

config.Data.inputDataset = 'DATASET_NAME'
config.Data.allowNonValidInputDataset = True
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 50
config.Data.lumiMask = 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/GOLDEN_JSON_PATH'
#config.Data.publication = True
config.Data.outputDatasetTag = 'SkimCHANNELNAME_YEAReraERANAME_streamNUMBER_PRODTAG'
config.JobType.allowUndistributedCMSSW = True 
config.Site.storageSite = 'T2_IT_Bari'
config.Site.ignoreGlobalBlacklist  = True