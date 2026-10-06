"""Extra tables for the transformer-training ntuples.

Call addTransformerTables(process, isMC, channel) at the end of setupTau3Mu / setupDsPhiPi.
It extends the candidate and muon tables and adds:
  <prefix>Trk, <prefix>Neu, <prefix>SV : per-candidate context tokens (candIdx = candidate row)
  CtxJet                              : event-level jets (opposite-side information)
  Generator                           : MC generator weight
The schema these tables implement is described in scripts/README_transformer.md.
"""
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var

_CHANNELS = {
    "Tau3Mu": dict(
        candTable="tau3muTable", candName="Tau3Mu", builder="tau3muBuilder", sequence="tau3muSequence",
        legs=["mu1", "mu2", "mu3"], legIdx=["mu1_idx", "mu2_idx", "mu3_idx"],
        legIsTrack=[False, False, False], legTracks="",
    ),
    "DsPhiPi": dict(
        candTable="cand2mu1trTable", candName="Cand2MuTrk", builder="cand2mu1trBuilder", sequence="cand2mu1trSequence",
        legs=["mu1", "mu2", "tr"], legIdx=["mu1_idx", "mu2_idx", "tr_idx"],
        legIsTrack=[False, False, True], legTracks="selectedTracks",
    ),
}

# Extra gen particles to keep, so the leg ancestry flags can see D+, B_s, Lambda_b and light resonances
_EXTRA_GEN = ("keep abs(pdgId)==411 | abs(pdgId)==421 | abs(pdgId)==531 | abs(pdgId)==5122 | "
              "abs(pdgId)==333 | abs(pdgId)==221 | abs(pdgId)==331 | abs(pdgId)==113 | abs(pdgId)==223 | abs(pdgId)==24")


def candidateVariables(legs):
    v = dict(
        channel=Var("userInt('channel')", int, doc="0 = TAU3MU (|Q|=1), 1 = DSPHIPI, 2 = SS3MU (|Q|=3)"),
        pv_idx=Var("userInt('pv_idx')", int, doc="index of the chosen PV in offlineSlimmedPrimaryVerticesWithBS"),
        pv_ndof=Var("userFloat('pv_ndof')", float, doc="ndof of the refitted PV"),
        ctau=Var("userFloat('ctau')", float, doc="L3D * m / p [cm] (mass-correlated)"),
        kin_ok=Var("userInt('kin_ok')", int, doc="kinematic vertex fit succeeded"),
        kin_mass=Var("userFloat('kin_mass')", float, doc="mass from the kinematic vertex fit"),
        kin_massErr=Var("userFloat('kin_massErr')", float, doc="per-candidate mass uncertainty from the kinematic fit"),
        kin_prob=Var("userFloat('kin_prob')", float, doc="kinematic vertex fit probability"),
    )
    for pair in ["12", "13", "23"]:
        v["m" + pair] = Var(f"userFloat('m{pair}')", float, doc=f"refitted mass of legs {pair[0]} and {pair[1]}")
        v["vprob" + pair] = Var(f"userFloat('vprob{pair}')", float, doc=f"2-track vertex probability of legs {pair[0]} and {pair[1]}")
    for vtx in ["sv", "pv"]:
        for el in ["cxx", "cyx", "cyy", "czx", "czy", "czz"]:
            v[f"{vtx}_{el}"] = Var(f"userFloat('{vtx}_{el}')", float, doc=f"{vtx.upper()} position covariance {el[1:]} [cm2]")
    for leg in legs:
        v[f"refit_{leg}_eta"] = Var(f"userFloat('refit_{leg}_eta')", float, precision=12)
        v[f"refit_{leg}_phi"] = Var(f"userFloat('refit_{leg}_phi')", float, precision=12)
        for ref in ["pv", "sv"]:
            v[f"{leg}_ip3d_{ref}"] = Var(f"userFloat('{leg}_ip3d_{ref}')", float, doc=f"3D IP wrt {ref.upper()} [cm]")
            v[f"{leg}_ip3d_{ref}_sig"] = Var(f"userFloat('{leg}_ip3d_{ref}_sig')", float, doc=f"3D IP significance wrt {ref.upper()}")
    return v


def muonVariables(isMC, addSoftMvaRun3):
    v = dict(
        srcIdx=Var("userInt('srcIdx')", int, doc="index in the builder input collection: join key for mu*_idx"),
        isStandalone=Var("isStandAloneMuon", bool),
        nMatchedStations=Var("numberOfMatchedStations()", int),
        inTimeMuon=Var("passed('InTimeMuon')", bool),
        timeAtIpInOut=Var("time().timeAtIpInOut", float, doc="muon time at IP, inside-out [ns]"),
        timeAtIpInOutErr=Var("time().timeAtIpInOutErr", float, doc="uncertainty of timeAtIpInOut [ns]"),
        mvaLowPt=Var("lowptMvaValue()", float, doc="low-pT muon MVA ID"),
    )
    if addSoftMvaRun3:
        # only in releases whose pat::Muon has softMvaRun3Value()
        v["softMvaRun3"] = Var("softMvaRun3Value()", float, doc="soft MVA Run 3 ID")
    if isMC:
        v.update(
            genFlags=Var("userInt('genFlags')", int, doc="bits: 1 tau, 2 Ds, 4 D+, 8 B, 16 W, 32 phi, 64 eta/eta'/rho/omega, 128 decay in flight"),
            genMotherPdgId=Var("userInt('genMotherPdgId')", int, doc="pdgId of the first non-muon ancestor"),
            genTauIdx=Var("userInt('genTauIdx')", int, doc="GenPart index of the tau ancestor (-1 if none)"),
            genHadronPdgId=Var("userInt('genHadronPdgId')", int, doc="pdgId of the gen pi/K matched when no gen muon is (decay in flight)"),
        )
    return v


def addTransformerTables(process, isMC, channel, addSoftMvaRun3=False,
                         maxTrk=64, maxNeu=32, maxSV=8, nDchi2=16, doPairFits=True):
    ch = _CHANNELS[channel]

    candTable = getattr(process, ch["candTable"])
    for name, var in candidateVariables(ch["legs"]).items():
        setattr(candTable.variables, name, var)
    for name, var in muonVariables(isMC, addSoftMvaRun3).items():
        setattr(process.TrgMatchMuonTable.variables, name, var)

    if isMC:
        process.muonsWithMatch.packedGen = cms.InputTag("packedGenParticles")
        process.myFinalGenParticles.select.append(_EXTRA_GEN)

    process.candContext = cms.EDProducer("CandidateContextProducer",
        src=cms.InputTag(ch["builder"]),
        candName=cms.string(ch["candName"]),
        prefix=cms.string(ch["candName"]),
        muons=cms.InputTag("muonsWithMatch" if isMC else "selectedMuons"),
        legTracks=cms.InputTag(ch["legTracks"]),
        legIdx=cms.vstring(ch["legIdx"]),
        legIsTrack=cms.vbool(ch["legIsTrack"]),
        pfCands=cms.InputTag("packedPFCandidates"),
        lostTracks=cms.InputTag("lostTracks"),
        allMuons=cms.InputTag("slimmedMuons"),
        secondaryVertices=cms.InputTag("slimmedSecondaryVertices"),
        trkMinPt=cms.double(0.5),
        trkMaxEta=cms.double(2.5),
        coneDR=cms.double(1.0),     # inCone
        cylDCA=cms.double(0.1),     # inCylinder: DCA to PV->SV line [cm] ...
        cylMaxDR=cms.double(1.5),   # ... within this dR
        neuMinPt=cms.double(0.3),
        neuConeDR=cms.double(0.5),
        maxTrk=cms.uint32(maxTrk),
        maxNeu=cms.uint32(maxNeu),
        maxSV=cms.uint32(maxSV),
        nDchi2=cms.uint32(nDchi2),
        doPairFits=cms.bool(doPairFits),
    )

    process.ctxJetTable = cms.EDProducer("SimplePATJetFlatTableProducer",
        src=cms.InputTag("slimmedJetsPuppi"),
        cut=cms.string("pt > 15 && abs(eta) < 2.5"),
        name=cms.string("CtxJet"),
        doc=cms.string("AK4 Puppi jets for the transformer context (opposite-side b)"),
        singleton=cms.bool(False),
        extension=cms.bool(False),
        variables=cms.PSet(
            pt=Var("pt", float, precision=10),
            eta=Var("eta", float, precision=10),
            phi=Var("phi", float, precision=10),
            mass=Var("mass", float, precision=10),
            nConstituents=Var("numberOfDaughters()", int),
            # -1000 when the tagger is not in the MiniAOD
            btagPNetB=Var("bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:BvsAll')", float, precision=10),
            btagUParTB=Var("bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')", float, precision=10),
        ),
    )

    seq = getattr(process, ch["sequence"])
    seq += process.candContext + process.ctxJetTable

    if isMC:
        process.genWeightTable = cms.EDProducer("SimpleGenEventFlatTableProducer",
            src=cms.InputTag("generator"),
            cut=cms.string(""),
            name=cms.string("Generator"),
            doc=cms.string("Generator information"),
            singleton=cms.bool(True),
            extension=cms.bool(False),
            variables=cms.PSet(weight=Var("weight()", float, doc="MC generator weight")),
        )
        seq += process.genWeightTable
