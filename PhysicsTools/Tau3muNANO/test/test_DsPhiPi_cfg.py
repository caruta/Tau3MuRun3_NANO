import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing
from PhysicsTools.Tau3muNANO.DsPhiPi_builder_cff import setupDsPhiPi

options = VarParsing.VarParsing('analysis')
options.register('isMC', True, 
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.bool, 
                 "True if MC, False if Data")
options.parseArguments()

isMC = options.isMC
process = cms.Process('NANO', eras.Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("PhysicsTools.NanoAOD.nano_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag
if isMC:
    process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2024_realistic_v26', '')
else:
    process.GlobalTag = GlobalTag(process.GlobalTag, '160X_dataRun3_Prompt_v1', '')

output_name = "dsphipi_output_MC.root" if isMC else "dsphipi_output_Data.root"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

mc_files = [
    'root://xrootd-cms.infn.it//store/mc/Run3Summer23MiniAODv4/DstoPhiPi_Phito2Mu_MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/MINIAODSIM/130X_mcRun3_2023_realistic_v14-v2/2810000/23794a69-978d-4aa5-83c9-60265065cc5b.root',
    #'root://xrootd-cms.infn.it//store/mc/Run3Summer22EEMiniAODv4/BsToJpsiPhi_JMM_PhiMM_MuFilter_SoftQCDnonD_TuneCP5_13p6TeV-pythia8-evtgen/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/50000/43d2b67f-908f-4c7f-952e-22d02e1852a5.root',
]
data_files = [
    'root://xrootd-cms.infn.it//store/data/Run2024C/ParkingDoubleMuonLowMass0/MINIAOD/MINIv6NANOv15-v1/110000/001d80b3-cd40-4fb9-890c-32d1249c6c9e.root',
    #'root://xrootd-cms.infn.it//store/data/Run2022C/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/355/863/00000/389f9ca1-f590-4691-b7f2-41e0146a8a79.root',
]

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(mc_files if isMC else data_files)
)

process.load("PhysicsTools.Tau3muNANO.DsPhiPi_builder_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi") ##if isMC

setupDsPhiPi(process, isMC)

process.out = cms.OutputModule("NanoAODOutputModule",
    fileName = cms.untracked.string(output_name),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p') 
    ),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep nanoaodFlatTable_*_*_*",
    ),
    compressionLevel = cms.untracked.int32(9),
    compressionAlgorithm = cms.untracked.string("ZLIB"),
)

process.p = cms.Path(process.cand2mu1trSequence)
process.endp = cms.EndPath(process.out)

process.MessageLogger.cerr.FwkReport.reportEvery = 100