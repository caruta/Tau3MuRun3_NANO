import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleGenParticleFlatTableProducer_cfi import simpleGenParticleFlatTableProducer
from PhysicsTools.NanoAOD.muons_cff import muonTable


# --- 0. TRIGGER PATHS ---

HLT_path_list = cms.vstring(
        "HLT_DoubleMu3_Trk_Tau3mu", 
        "HLT_DoubleMu3_TkMu_DsTau3Mu_v", 
        "HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v", 
        "HLT_DoubleMu4_3_LowMass_v", 
        "HLT_DoubleMu4_LowMass_Displaced_v"
    )

L1_path_list = cms.vstring(
            "L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9",
            "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4",
            "L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4",
            "L1_DoubleMu4_SQ_OS_dR_Max1p2",
            "L1_DoubleMu4p5_SQ_OS_dR_Max1p2",
            "L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9",
            "L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6",
            "L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5",
            "L1_TripleMu_2SQ_1p5SQ_0OQ_Mass_Max12",
            "L1_TripleMu_3SQ_2p5SQ_0OQ_Mass_Max12"
        )

# --- 1. EVENT FILTERS (HLT & SKIMMING) ---
def setupDsPhiPi(process, isMC):
    # HLT Filter: Select events passing specific trigger paths
    process.hltFilter = cms.EDFilter("HLTHighLevel",
        HLTPaths = cms.vstring([path + "*" for path in HLT_path_list]),
        throw = cms.bool(False)
    )

    # Base Muon Selection: Kinetic cuts and quality requirements
    process.selectedMuons = cms.EDFilter("PATMuonSelector",
        src = cms.InputTag("slimmedMuons"),
        cut = cms.string("pt > 2.0 && abs(eta) < 2.4 && (innerTrack().isNonnull) && (charge!=0) && (innerTrack().hitPattern().numberOfValidPixelHits()>0)"),
        filter = cms.bool(True)
    )

    # Preliminary Filter: Require at least 2 selected muons
    process.TwoMuonsFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag("selectedMuons"),
        minNumber = cms.uint32(2)
    )

    # Track Selection: Select high-quality tracks from PFCandidates
    process.selectedTracks = cms.EDFilter("PATPackedCandidateSelector",
        src = cms.InputTag("packedPFCandidates"),
        cut = cms.string("pt > 2 && abs(eta)<2.4 && (charge!=0) && hasTrackDetails() && trackerLayersWithMeasurement()>5 && pixelLayersWithMeasurement()>=1"),
    )

    # Preliminary Filter: Require at least 1 selected track
    process.oneTrackFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag("selectedTracks"),
        minNumber = cms.uint32(1)
    )

    # Technical Skim: Combiner for Muon pairs (Mass resonance window)
    process.DiMuonCand = cms.EDProducer("CandViewShallowCloneCombiner",
        checkCharge = cms.bool(False),
        cut = cms.string('(abs(charge)==0) && (mass < 1.5) && (mass > 0.5)'),
        decay = cms.string("selectedMuons selectedMuons")
    )

    process.DiMuonCandFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag("DiMuonCand"),
        minNumber = cms.uint32(1)
    )

    # Final Candidate Combiner: 2 Muons + 1 Track (Global mass window)
    process.TwoMuonsOneTrackCand = cms.EDProducer("CandViewShallowCloneCombiner",
        checkCharge = cms.bool(False),
        cut = cms.string('(abs(charge)==1) && ((daughter(0).charge+daughter(1).charge)==0) && (mass < 3.0) && (mass > 0.8)'),
        decay = cms.string("selectedMuons selectedMuons selectedTracks")
    )

    # Main Skim Filter: Drop events without at least one valid triplet candidate
    process.DsCountFilter = cms.EDFilter("CandViewCountFilter",
        src = cms.InputTag("TwoMuonsOneTrackCand"),
        minNumber = cms.uint32(1)
    )

    # --- 2. MONTE CARLO MODULES (Conditional on isMC) ---

    if isMC:
        # Gen Particle Pruning: Keep only interesting particles for signal/background study
        process.myFinalGenParticles = cms.EDProducer("GenParticlePruner",
            src = cms.InputTag("prunedGenParticles"),
            select = cms.vstring(
                "drop *",
                "keep abs(pdgId)==13 | abs(pdgId)==15 | abs(pdgId)==11 | abs(pdgId)==211 | abs(pdgId)==321 | abs(pdgId)==431 | abs(pdgId)==333 | abs(pdgId)==12 | abs(pdgId)==14 | abs(pdgId)==16 | abs(pdgId)==511 | abs(pdgId)==521"
            )
        )


        # Gen Isolation: Calculate isolation for gen-level particles
        process.myGenIso = cms.EDProducer("GenPartIsoProducer",
            genPart = cms.InputTag("myFinalGenParticles"),
            packedGenPart = cms.InputTag("packedGenParticles"),
            additionalPdgId = cms.int32(22),
        )

        # Reco-to-Gen Matching: Match selected muons to gen-level muons
        process.muonGenMatch = cms.EDProducer("MCMatcher",
            src         = cms.InputTag("selectedMuons"),
            matched     = cms.InputTag("myFinalGenParticles"),
            mcPdgId     = cms.vint32(13),
            mcStatus    = cms.vint32(1),
            checkCharge = cms.bool(True),
            maxDPtRel   = cms.double(0.5),
            maxDeltaR   = cms.double(0.3),
            resolveAmbiguities    = cms.bool(True),
            resolveByMatchQuality = cms.bool(True),
        )

        # Embed matching info into muons
        process.muonsWithMatch = cms.EDProducer("MuonMatchEmbedder",
            src      = cms.InputTag("selectedMuons"),
            matching = cms.InputTag("muonGenMatch")
        )

    # --- 3. CANDIDATE BUILDING & TREE CONSTRUCTION ---

    # Define muon source based on MC/Data
    muonSrc = cms.InputTag("muonsWithMatch") if isMC else cms.InputTag("selectedMuons")

    # Ds Candidate Builder: Perform vertex fitting for the 2Mu + 1Trk system
    process.cand2mu1trBuilder = cms.EDProducer("Ds2Mu1TrkBuilder",
        muons    = muonSrc,
        tracks   = cms.InputTag("selectedTracks"),
        vertices = cms.InputTag("offlineSlimmedPrimaryVerticesWithBS"),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        candidates = cms.InputTag("packedPFCandidates"),
    )

    # BPH Muon Selector: Add trigger matching information to muons
    process.muonBPH = cms.EDProducer("MuonTriggerSelector",
        muonCollection = muonSrc,
        bits           = cms.InputTag("TriggerResults", "", "HLT"),
        prescales      = cms.InputTag("patTrigger"),
        objects        = cms.InputTag("slimmedPatTrigger"),
        maxdR_matching = cms.double(0.3),
        muonSelection  = cms.string("pt > 2.0"), 
        HLTPaths       = HLT_path_list,
    )

    # --- 4. FLAT TABLES DEFINITION (NanoAOD Output) ---

    # Cand Table: Stores triplet properties and vertex mass
    process.cand2mu1trTable = cms.EDProducer("MySimpleCompositeCandidateFlatTableProducer",
        src = cms.InputTag("cand2mu1trBuilder"),
        name = cms.string("Cand2MuTrk"),
        variables = cms.PSet(
        pt      = Var("pt", float),
        eta     = Var("eta", float),
        phi     = Var("phi", float),
        charge  = Var("charge", int),

        mu1_idx = Var("userInt('mu1_idx')", int),
        mu2_idx = Var("userInt('mu2_idx')", int),
        tr_idx  = Var("userInt('tr_idx')", int),
        
        # Vertex Info
        mass    = Var("userFloat('sv_mass')", float, doc="Fitted invariant mass"),
        prob    = Var("userFloat('sv_prob')", float, doc="SV fit probability"),
        chi2    = Var("userFloat('sv_chi2')", float),
        ndof    = Var("userFloat('sv_ndof')", float),
        
        # Vertex Positions
        sv_x    = Var("userFloat('sv_x')", float),
        sv_y    = Var("userFloat('sv_y')", float),
        sv_z    = Var("userFloat('sv_z')", float),
        pv_x    = Var("userFloat('pv_x')", float),
        pv_y    = Var("userFloat('pv_y')", float),
        pv_z    = Var("userFloat('pv_z')", float),
        
        # Flight Distance & Displacement
        flightDist    = Var("userFloat('flightDist')", float),
        flightDistErr = Var("userFloat('flightDistErr')", float),
        flightDistSig = Var("userFloat('flightDistSig')", float),
        flightDistXY = Var("userFloat('flightDistXY')", float),
        flightDistXYErr = Var("userFloat('flightDistXYErr')", float),
        flightDistXYSig = Var("userFloat('flightDistXYSig')", float),
        flightDistBS = Var("userFloat('flightDistBS')", float),
        flightDistBSErr = Var("userFloat('flightDistBSErr')", float),
        flightDistBSSig = Var("userFloat('flightDistBSSig')", float),
        
        # Refitted Kinematics (SV fit)
        refit_mu1_pt = Var("userFloat('refit_mu1_pt')", float),
        refit_mu2_pt = Var("userFloat('refit_mu2_pt')", float),
        refit_tr_pt  = Var("userFloat('refit_tr_pt')", float),
        
        # Impact Parameters (PV refit)
        dxy_mu1 = Var("userFloat('dxy_mu1')", float),
        dxy_mu2 = Var("userFloat('dxy_mu2')", float),
        dxy_tr  = Var("userFloat('dxy_tr')", float),
        
        # Isolamento e Angoli
        mindca_iso = Var("userFloat('mindca_iso')", float),
        rel_iso    = Var("userFloat('relative_iso')", float),
        cosPointingAngle = Var("userFloat('cosPointingAngle')", float),
        pointingAngle = Var("userFloat('pointingAngle')", float),

        # High Purity Flags
        mu1_hp = Var("userInt('mu1_innerTrk_hp')", int),
        mu2_hp = Var("userInt('mu2_innerTrk_hp')", int),
        tr_hp  = Var("userInt('tr_innerTrk_hp')", int),

        mu1_match1_dX = Var("userFloat('mu1_match1_dX')", float),
        mu1_match1_pullX = Var("userFloat('mu1_match1_pullX')", float),
        mu2_match1_dX = Var("userFloat('mu2_match1_dX')", float),
        mu2_match1_pullX = Var("userFloat('mu2_match1_pullX')", float),
        
        mu1_match2_dX = Var("userFloat('mu1_match2_dX')", float),
        mu1_match2_pullX = Var("userFloat('mu1_match2_pullX')", float),
        mu2_match2_dX = Var("userFloat('mu2_match2_dX')", float),
        mu2_match2_pullX = Var("userFloat('mu2_match2_pullX')", float),

        mu1_iso03_clean = Var("userFloat('mu1_iso03_clean')", float),
        mu1_iso04_clean = Var("userFloat('mu1_iso04_clean')", float),
        mu2_iso03_clean = Var("userFloat('mu2_iso03_clean')", float),
        mu2_iso04_clean = Var("userFloat('mu2_iso04_clean')", float),
        )
    )

    # Muon Table: BPH specialized Muons
    process.TrgMatchMuonTable = cms.EDProducer("SimplePATMuonFlatTableProducer",
        src  = cms.InputTag("muonBPH", "SelectedMuons"),
        name = cms.string("Muon"),
        variables = cms.PSet(
            # --- Kinematics ---
            p     = Var("p", float, precision=12),
            pt     = Var("pt", float, precision=12),
            eta    = Var("eta", float, precision=12),
            phi    = Var("phi", float, precision=12),
            mass   = Var("mass", float, precision=12),
            energy   = Var("energy", float, precision=12),
            charge = Var("charge", int),

            # --- Impact Parameters (DXY, DZ) ---
            dxy    = Var("dB('PV2D')", float, doc="dxy (with sign) wrt PV[0]", precision=12),
            dxyErr = Var("edB('PV2D')", float, doc="dxy uncertainty", precision=12),
            dz     = Var("dB('PVDZ')", float, doc="dz (with sign) wrt PV[0]", precision=12),
            dzErr  = Var("abs(edB('PVDZ'))", float, doc="dz uncertainty", precision=12),
            ip3d   = Var("abs(dB('PV3D'))", float, doc="3D impact parameter wrt PV[0], in cm"),
            sip3d  = Var("?edB('PV3D')>0?abs(dB('PV3D')/edB('PV3D')):-1", float, doc="3D impact parameter significance wrt PV[0]"),

            # --- Identification & Quality ---
            isPF     = Var("isPFMuon", bool),
            isGlobal = Var("isGlobalMuon", bool),
            isTracker= Var("isTrackerMuon", bool),
            isLoose  = Var("passed('CutBasedIdLoose')", bool),
            isMedium = Var("passed('CutBasedIdMedium')", bool),
            isSoft   = Var("passed('SoftCutBasedId')", bool),
            
            # --- Trigger & Custom ---
            isTriggering = Var("userInt('isTriggering')", bool),
            trgDR        = Var("userFloat('trgDR')", float),
            trgDPT       = Var("userFloat('trgDPT')", float),
            genPartPdgId = Var("userInt('mcMatch')" if isMC else "0", int),
            decayType = Var("userInt('genOrigin')" if isMC else "0", int, doc = "0 if non matching, 1=Ds, 2=B, 3=Other"),
            
            # --- Isolation (Standard) ---
            pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt", float),
            pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt", float),
            
            mvaId = Var("mvaIDValue()", float, doc="MVA Soft ID"),
            softMva = Var("softMvaValue()", float),
            segmentCompatibility = Var("segmentCompatibility()", float),
            caloCompatibility = Var("caloCompatibility()", float),

            sumPt03 = Var("isolationR03().sumPt", float, doc="Tracker isolation sumPt in dR=0.3"),
            sumPt05 = Var("isolationR05().sumPt", float, doc="Tracker isolation sumPt in dR=0.5"),

            nTracks03 = Var("isolationR03().nTracks", int, doc="Number of tracks in dR=0.3 tracker isolation cone"),
            nTracks05 = Var("isolationR05().nTracks", int, doc="Number of tracks in dR=0.5 tracker isolation cone"),
            trkKink = Var("combinedQuality().trkKink", float),
            glbKink = Var("combinedQuality().glbKink", float),
            trkRelChi2 = Var("combinedQuality().trkRelChi2", float),
            staRelChi2 = Var("combinedQuality().staRelChi2", float),
            chi2LocalPosition = Var("combinedQuality().chi2LocalPosition", float),
            chi2LocalMomentum = Var("combinedQuality().chi2LocalMomentum", float),
            localDistance = Var("combinedQuality().localDistance", float),
            globalDeltaEtaPhi = Var("combinedQuality().globalDeltaEtaPhi", float),
            tightMatch = Var("combinedQuality().tightMatch", float),
            glbTrackProbability = Var("combinedQuality().glbTrackProbability", float),

            ## Specific Trigger path matching info
            HLT_DoubleMu3_Trk_Tau3mu = Var("userInt('HLT_DoubleMu3_Trk_Tau3mu')", int, doc="Matched to HLT_DoubleMu3_Trk_Tau3mu"),
            HLT_DoubleMu3_TkMu_DsTau3Mu_v = Var("userInt('HLT_DoubleMu3_TkMu_DsTau3Mu_v')", int, doc="Matched to HLT_DoubleMu3_TkMu_DsTau3Mu_v"),
            HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v = Var("userInt('HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v')", int, doc="Matched to HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v"),
            HLT_DoubleMu4_3_LowMass_v = Var("userInt('HLT_DoubleMu4_3_LowMass_v')", int, doc="Matched to HLT_DoubleMu4_3_LowMass_v"),
            HLT_DoubleMu4_LowMass_Displaced_v = Var("userInt('HLT_DoubleMu4_LowMass_Displaced_v')", int, doc="Matched to HLT_DoubleMu4_LowMass_Displaced_v"),

            # --- Validation Flags ---
            isQValid = Var("isQualityValid()", bool, doc="Muon isQualityValid"),
            isTValid = Var("isTimeValid()", bool, doc="Muon isTimeValid"),
            isIsoValid = Var("isIsolationValid()", bool, doc="Muon isIsolationValid"),

            # --- Global Track Variables ---
            GLnormChi2 = Var("?globalTrack().isNonnull()?globalTrack().normalizedChi2():-99", float, doc="Global track normalized chi2"),
            GL_nValidMuHits = Var("?globalTrack().isNonnull()?globalTrack().hitPattern().numberOfValidMuonHits():-1", int, doc="Number of valid muon hits in global track"),

            # --- Tracker & Hit Pattern ---
            trkLayersWMeas = Var("?innerTrack().isNonnull()?innerTrack().hitPattern().trackerLayersWithMeasurement():-1", int),
            nValidTrackerHits = Var("?innerTrack().isNonnull()?innerTrack().hitPattern().numberOfValidTrackerHits():-1", int),
            nValidPixelHits = Var("?innerTrack().isNonnull()?innerTrack().hitPattern().numberOfValidPixelHits():-1", int),
            validMuonHitComb = Var("?globalTrack().isNonnull()?globalTrack().hitPattern().numberOfValidMuonHits():-1", int), # Spesso mappato così

            # --- Outer Track (Standalone Muon) ---
            outerTrk_P = Var("?outerTrack().isNonnull()?outerTrack().p():-99", float),
            outerTrk_Eta = Var("?outerTrack().isNonnull()?outerTrack().eta():-99", float),
            outerTrk_normChi2 = Var("?outerTrack().isNonnull()?outerTrack().normalizedChi2():-99", float),
            outerTrk_muStValidHits = Var("?outerTrack().isNonnull()?outerTrack().hitPattern().muonStationsWithValidHits():-1", int),

            # --- Inner Track (Tracker Track) ---
            innerTrk_P = Var("?innerTrack().isNonnull()?innerTrack().p():-99", float),
            innerTrk_Eta = Var("?innerTrack().isNonnull()?innerTrack().eta():-99", float),
            innerTrk_normChi2 = Var("?innerTrack().isNonnull()?innerTrack().normalizedChi2():-99", float),
            innerTrk_ValidFraction = Var("?innerTrack().isNonnull()?innerTrack().validFraction():-99", float),

            # --- Combined Quality & Compatibility ---
            QInnerOuter = Var("combinedQuality().updatedSta", bool), # Spesso usato come proxy per QInnerOuter
            cQ_uS = Var("combinedQuality().updatedSta", bool),
            cQ_tK = Var("combinedQuality().trkKink", float),
            cQ_gK = Var("combinedQuality().glbKink", float),
            cQ_tRChi2 = Var("combinedQuality().trkRelChi2", float),
            cQ_sRChi2 = Var("combinedQuality().staRelChi2", float),
            cQ_Chi2LP = Var("combinedQuality().chi2LocalPosition", float),
            cQ_Chi2LM = Var("combinedQuality().chi2LocalMomentum", float),
            cQ_lD = Var("combinedQuality().localDistance", float),
            cQ_gDEP = Var("combinedQuality().globalDeltaEtaPhi", float),
            cQ_tM = Var("combinedQuality().tightMatch", bool),
            cQ_gTP = Var("combinedQuality().glbTrackProbability", float),
            
            segmComp = Var("segmentCompatibility()", float),
            caloComp = Var("caloCompatibility()", float),
        )
    )

    # --- Track Table: Stores comprehensive track parameters for PFCandidates ---
    process.trackTable = cms.EDProducer("MySimplePFCandidateFlatTableProducer",
        src = cms.InputTag("selectedTracks"),
        name = cms.string("Track"),
        variables = cms.PSet(
            # Kinematics
            pt     = Var("pt", float, precision=12),
            eta    = Var("eta", float, precision=12),
            phi    = Var("phi", float, precision=12),
            mass   = Var("mass", float, precision=10),
            charge = Var("charge", int),
            
            # Track Quality and Impact Parameters
            dz      = Var("dz", float, precision=10, doc="longitudinal impact parameter with respect to PV"),
            dxy     = Var("dxy", float, precision=10, doc="transverse impact parameter with respect to PV"),
            dzErr   = Var("dzError", float, precision=10),
            dxyErr  = Var("dxyError", float, precision=10),
            
            # Track Reconstruction Details
            lostInnerHits = Var("lostInnerHits", int, doc="number of lost inner hits"),
            numberOfPixelHits = Var("numberOfPixelHits", int, doc="number of valid pixel hits"),
            numberOfHits = Var("numberOfHits", int, doc="number of valid hits"),
            trackHighPurity = Var("trackHighPurity", bool, doc="track is marked as high purity"),
            
            # Isolation / Environment
            pdgId = Var("pdgId", int, doc="PDG identifier"),
        )
    )

    process.triggerTableDs2Mu1Trk = cms.EDProducer("myTriggerTableProducer",
        l1Src = cms.InputTag("gtStage2Digis"),
        hltSrc = cms.InputTag("TriggerResults", "", "HLT"),
        name = cms.string("Trigger"), # Nome della tabella nel ROOT
        doc = cms.string("L1 and HLT bits for Tau3Mu analysis"),
        extension = cms.bool(False),
        l1Seeds = L1_path_list,
        hltPaths = HLT_path_list
    )

    # BPH Vertex Table
    process.pvTable = cms.EDProducer("PVertexBPHTable",
    pvSrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    goodPvCut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"), 
    pvName = cms.string("PVtx")
    )

    process.puTable = cms.EDProducer("NPUTablesProducer",
            src = cms.InputTag("slimmedAddPileupInfo"),
            pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
            zbins = cms.vdouble( [0.0,1.7,2.6,3.0,3.5,4.2,5.2,6.0,7.5,9.0,12.0] ),
            savePtHatMax = cms.bool(True),
    )

    # Gen Particle Table: Stores MC truth information (Only if isMC)
    if isMC:
        process.myGenParticleTable = simpleGenParticleFlatTableProducer.clone(
            src = cms.InputTag("myFinalGenParticles"),
            name= cms.string("GenPart"),
            externalVariables = cms.PSet(
                iso = ExtVar(cms.InputTag("myGenIso"), float, precision=8, doc="Gen Isolation"),
            ),
            variables = cms.PSet(
                pt               = Var("pt",  float),
                phi              = Var("phi", float),
                eta              = Var("eta",  float),
                mass             = Var("mass",  float),
                pdgId            = Var("pdgId", int),
                status           = Var("status", int),
                genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", "int16"),
            )
        )

    # --- 5. FINAL EXECUTION SEQUENCE ---

    # Filter Sequence
    process.cand2mu1trSequence = cms.Sequence(
        process.hltFilter +
        process.selectedMuons +
        process.TwoMuonsFilter +
        process.selectedTracks +
        process.oneTrackFilter +
        process.DiMuonCand +
        process.DiMuonCandFilter +
        process.TwoMuonsOneTrackCand +
        process.DsCountFilter
    )

    # MC-specific modules
    if isMC:
        process.cand2mu1trSequence += (
            process.myFinalGenParticles +
            process.myGenIso +
            process.muonGenMatch +
            process.muonsWithMatch
        )

    # Candidate construction and table filling
    process.cand2mu1trSequence += (
        process.cand2mu1trBuilder +
        process.cand2mu1trTable +
        process.muonBPH +
        process.TrgMatchMuonTable +
        process.trackTable +
        process.triggerTableDs2Mu1Trk +
        process.pvTable +
        process.puTable
    )

    # Append Gen Table if MC
    if isMC:
        process.cand2mu1trSequence += process.myGenParticleTable