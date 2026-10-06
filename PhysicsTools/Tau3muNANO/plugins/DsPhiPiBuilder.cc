#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "PhysicsTools/Tau3muNANO/interface/PVRefitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "TVector3.h"
#include "TMath.h"
#include "helper.h"
#include "CandFeatureHelpers.h"

#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"

namespace {
    typedef std::pair<const reco::MuonChamberMatch*, const reco::MuonSegmentMatch*> MatchPair;

    MatchPair getBetterMatch(const MatchPair& match1, const MatchPair& match2) {
        if (match2.first->detector() == MuonSubdetId::DT and
            match1.first->detector() != MuonSubdetId::DT)
            return match2;

        if (std::abs(match1.first->x - match1.second->x) >
            std::abs(match2.first->x - match2.second->x) )
            return match2;
            
        return match1;
    }
}

class Ds2Mu1TrkBuilder : public edm::global::EDProducer<> {
public:
    explicit Ds2Mu1TrkBuilder(const edm::ParameterSet& cfg) :
        muonsToken_(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        tracksToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("tracks"))),
        vtxToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
        pcToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("candidates"))),
        bsToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
        ttbToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))) {
        produces<pat::CompositeCandidateCollection>();
    }

    void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& iSetup) const override {
        edm::Handle<pat::MuonCollection> muons;
        evt.getByToken(muonsToken_, muons);
        edm::Handle<pat::PackedCandidateCollection> tracks;
        evt.getByToken(tracksToken_, tracks);
        edm::Handle<reco::VertexCollection> vertices;
        evt.getByToken(vtxToken_, vertices);
        edm::Handle<pat::PackedCandidateCollection> candidates;
        evt.getByToken(pcToken_, candidates);
        edm::Handle<reco::BeamSpot> beamSpot;
        evt.getByToken(bsToken_, beamSpot);

        const auto& ttb = iSetup.getData(ttbToken_);
        auto ret_val = std::make_unique<pat::CompositeCandidateCollection>();
        PVRefitter pvRefitter;

        for (size_t i = 0; i < muons->size(); ++i) {
            for (size_t j = i + 1; j < muons->size(); ++j) {
                for (size_t k = 0; k < tracks->size(); ++k) {
                    
                    pat::CompositeCandidate cand;
                    const pat::Muon &m1 = muons->at(i), &m2 = muons->at(j);
                    const pat::PackedCandidate &tr = tracks->at(k);
                    auto dimu_p4 = m1.p4() + m2.p4();
                    double dR_13 = reco::deltaR(m1, tr);
                    double dR_23 = reco::deltaR(m2, tr);
                    double dR_12 = reco::deltaR(m1, m2);
                    if(dR_13 < 0.01 || dR_23 < 0.01 || dR_12 < 0.01) { continue; }

                    if (std::abs(m1.eta() - tr.eta()) < 1e-6 ||
                        std::abs(m2.eta() - tr.eta()) < 1e-6 ||
                        std::abs(m1.eta() - m2.eta()) < 1e-6) continue;
                    // 5. Triplet Raw Mass
                    auto p4_raw = dimu_p4 + tr.p4();
                    if (p4_raw.mass() <= 0.8 || p4_raw.mass() >= 3.0) continue;

                    if (m1.innerTrack().isNull() || m2.innerTrack().isNull() || !tr.hasTrackDetails()) continue;
                    if (std::abs(m1.charge() + m2.charge()) != 0) continue;
                    if (std::abs(m1.charge() + m2.charge() + tr.charge()) != 1) continue;
                    auto tt3 = ttb.build(tr.pseudoTrack());
                    if (!tt3.isValid()) { continue; }
                    // Building TTracks
                    std::vector<reco::TransientTrack> muTTracks = {
                        ttb.build(m1.innerTrack()), 
                        ttb.build(m2.innerTrack()), 
                        ttb.build(tr.pseudoTrack())
                    };

                    KalmanVertexFitter svFitter(true);
                    TransientVertex sv = svFitter.vertex(muTTracks);

                    if (!sv.isValid() || !sv.hasRefittedTracks()) {
                        continue;
                    }

                    // --- Refitted Kinematics ---
                    reco::Candidate::LorentzVector p4_ref(0,0,0,0);
                    std::vector<double> refit_pts;
                    std::vector<reco::Track> refit_trks;
                    std::array<reco::Candidate::LorentzVector, 3> leg_p4;
                    int idx_rt = 0;
                    for(const auto& rt : sv.refittedTracks()) {
                        double mass_sq = (idx_rt < 2) ? 0.0111636 : 0.019479; // Muon mass vs Pion mass
                        reco::Candidate::LorentzVector lp4(rt.track().px(), rt.track().py(), rt.track().pz(), std::sqrt(rt.track().p2() + mass_sq));
                        p4_ref += lp4;
                        if (idx_rt < 3) leg_p4[idx_rt] = lp4;
                        refit_pts.push_back(rt.track().pt());
                        refit_trks.push_back(rt.track());
                        idx_rt++;
                    }
                    if (refit_trks.size() != 3) continue;

                    // --- PV Selection ---
                    std::vector<int> validVtxIndices;
                    for (const auto& pfc : *candidates) {
                        if (pfc.charge() == 0 || pfc.vertexRef().isNull() || !pfc.hasTrackDetails()) continue;
                        int fromPV = pfc.fromPV(pfc.vertexRef().key());
                        if (fromPV < 2) continue;
                        if (fromPV == 2 && pfc.pvAssociationQuality() != pat::PackedCandidate::UsedInFitLoose) continue;
                        validVtxIndices.push_back(pfc.vertexRef().key());
                    }
                    std::sort(validVtxIndices.begin(), validVtxIndices.end());
                    validVtxIndices.erase(std::unique(validVtxIndices.begin(), validVtxIndices.end()), validVtxIndices.end());

                    uint b_idx = -1; double maxCos = -1.0;
                    TVector3 svP(sv.position().x(), sv.position().y(), sv.position().z());
                    TVector3 pT(p4_ref.px(), p4_ref.py(), p4_ref.pz());

                    for (size_t v = 0; v < vertices->size(); ++v) {
                        if (std::find(validVtxIndices.begin(), validVtxIndices.end(), (int)v) == validVtxIndices.end()) continue;
                        TVector3 dv(svP.x() - vertices->at(v).x(), svP.y() - vertices->at(v).y(), svP.z() - vertices->at(v).z());
                        double c = dv.Dot(pT) / (dv.Mag() * pT.Mag());
                        if (c > maxCos) { maxCos = c; b_idx = v; }
                    }

                    if (b_idx == (uint)-1) continue; 
                    const reco::Vertex& bestPV = vertices->at(b_idx);

                    std::vector<reco::TransientTrack> pvTTracks;
                    for (const auto& c : *candidates) {
                        if (c.charge() == 0 || !c.hasTrackDetails() || c.vertexRef().isNull() || c.vertexRef().key() != b_idx) continue;
                        int fromPV = c.fromPV(c.vertexRef().key());
                        if (fromPV < 2) continue;
                        if (fromPV == 2 && c.pvAssociationQuality() != pat::PackedCandidate::UsedInFitLoose) continue;
                        pvTTracks.push_back(ttb.build(c.pseudoTrack()));
                    }

                    auto [cleanPV, pvStatus] = pvRefitter.refit(muTTracks, bestPV, pvTTracks);
                    if (!(cleanPV.isValid() && pvStatus == 1)) continue;

                    // --- Displacement & Geometry ---
                    VertexDistanceXY distXY; VertexDistance3D dist3D;
                    auto d2d = distXY.distance(sv, cleanPV);
                    auto d3d = dist3D.distance(sv, cleanPV);
                    
                    reco::Vertex::Point bsP(beamSpot->x0(), beamSpot->y0(), beamSpot->z0());
                    reco::Vertex bsVertex(bsP, beamSpot->covariance3D());
                    auto dBS = distXY.distance(sv, bsVertex);

                    // Pointing Angle
                    GlobalVector flightDir(sv.position().x() - cleanPV.x(), sv.position().y() - cleanPV.y(), sv.position().z() - cleanPV.z());
                    GlobalVector momentum(p4_ref.px(), p4_ref.py(), p4_ref.pz());
                    double cosPointingAngle = 0;
                    if (flightDir.mag2() > 0 && momentum.mag2() > 0) {
                        cosPointingAngle = flightDir.unit().dot(momentum.unit());
                    }

                    cosPointingAngle = std::max(-1.0, std::min(1.0, cosPointingAngle));
                    double pointingAngle = std::acos(cosPointingAngle);
                    // Isolation & DCA
                    float sum_pt_iso = 0; float min_dca = 999.;
                    for (const auto& track : *candidates) {
                        if (track.charge() == 0 || track.pt() < 0.5 || !track.hasTrackDetails()) continue;
                        float dr = reco::deltaR(track, p4_ref);
                        if (reco::deltaR(track, m1) < 0.005 || reco::deltaR(track, m2) < 0.005 || reco::deltaR(track, tr) < 0.005) continue;
                        if (dr < 0.3) sum_pt_iso += track.pt();
                        reco::TransientTrack ttIso = ttb.build(track.pseudoTrack());
                        if (ttIso.isValid()) {
                            auto pca = ttIso.trajectoryStateClosestToPoint(sv.position());
                            if (pca.isValid()) min_dca = std::min(min_dca, (float)(pca.position() - sv.position()).mag());
                        }
                    }

                    // remove contribution of the pion track from the muon PFisolation parameter:
                    float trk_pt = tr.pt();
                    float dr_m1_tr = reco::deltaR(m1, tr);
                    float dr_m2_tr = reco::deltaR(m2, tr);

                    auto get_cleaned_rel_iso = [&](const pat::Muon& mu, float dr_mu_tr, float R) {
                        auto iso = (R == 0.4) ? mu.pfIsolationR04() : mu.pfIsolationR03();
                        float sumCharged = iso.sumChargedHadronPt;
                        
                        // Se la traccia del pione è nel cono, la sottraggo
                        if (dr_mu_tr < R) sumCharged -= trk_pt;
                        
                        float full_iso = std::max(0.f, sumCharged) + 
                                        std::max(0.f, iso.sumNeutralHadronEt + iso.sumPhotonEt - 0.5f * iso.sumPUPt);
                        return full_iso / mu.pt();
                    };

                    float mu1_iso03_clean = get_cleaned_rel_iso(m1, dr_m1_tr, 0.3);
                    float mu1_iso04_clean = get_cleaned_rel_iso(m1, dr_m1_tr, 0.4);
                    float mu2_iso03_clean = get_cleaned_rel_iso(m2, dr_m2_tr, 0.3);
                    float mu2_iso04_clean = get_cleaned_rel_iso(m2, dr_m2_tr, 0.4);

                    // --- Fill Candidate ---
                    cand.setP4(p4_ref);
                    cand.setCharge(m1.charge() + m2.charge() + tr.charge());
                    cand.setVertex(reco::Candidate::Point(sv.position().x(), sv.position().y(), sv.position().z()));

                    cand.addUserInt("mu1_idx", i); cand.addUserInt("mu2_idx", j); cand.addUserInt("tr_idx", k);
                    cand.addUserFloat("mu1_pt",  m1.pt());
                    cand.addUserFloat("mu1_eta", m1.eta());
                    cand.addUserFloat("mu1_phi", m1.phi());
                    cand.addUserFloat("mu1_charge", m1.charge());

                    cand.addUserFloat("mu2_pt",  m2.pt());
                    cand.addUserFloat("mu2_eta", m2.eta());
                    cand.addUserFloat("mu2_phi", m2.phi());
                    cand.addUserFloat("mu2_charge", m2.charge()); 
                    
                    cand.addUserFloat("tr_pt",  tr.pt());
                    cand.addUserFloat("tr_eta", tr.eta());
                    cand.addUserFloat("tr_phi", tr.phi());
                    cand.addUserFloat("tr_charge", tr.charge());

                    cand.addUserFloat("sv_mass", p4_ref.M());
                    cand.addUserFloat("sv_prob", TMath::Prob(sv.totalChiSquared(), sv.degreesOfFreedom()));
                    cand.addUserFloat("sv_chi2", sv.totalChiSquared());
                    cand.addUserFloat("sv_ndof", sv.degreesOfFreedom());
                    
                    cand.addUserFloat("sv_x", sv.position().x());
                    cand.addUserFloat("sv_y", sv.position().y());
                    cand.addUserFloat("sv_z", sv.position().z());
                    // Refit & PV
                    cand.addUserFloat("refit_mu1_pt", refit_pts[0]);
                    cand.addUserFloat("refit_mu2_pt", refit_pts[1]);
                    cand.addUserFloat("refit_tr_pt",  refit_pts[2]);
                    cand.addUserFloat("pv_x", cleanPV.x());
                    cand.addUserFloat("pv_y", cleanPV.y());
                    cand.addUserFloat("pv_z", cleanPV.z());
                    cand.addUserFloat("pv_orig_x", bestPV.x()); 
                    cand.addUserFloat("pv_orig_y", bestPV.y()); 
                    cand.addUserFloat("pv_orig_z", bestPV.z());
                    // Displacement
                    cand.addUserFloat("flightDist", d3d.value());
                    cand.addUserFloat("flightDistErr", d3d.error());
                    cand.addUserFloat("flightDistSig", d3d.value() / d3d.error());
                    cand.addUserFloat("flightDistXY", d2d.value());
                    cand.addUserFloat("flightDistXYErr", d2d.error());
                    cand.addUserFloat("flightDistXYSig", d2d.value() / d2d.error());
                    cand.addUserFloat("flightDistBS", dBS.value());
                    cand.addUserFloat("flightDistBSErr", dBS.error());
                    cand.addUserFloat("flightDistBSSig", dBS.value() / dBS.error());
                    // Impact Parameters
                    cand.addUserFloat("dxy_mu1", m1.innerTrack()->dxy(cleanPV.position()));
                    cand.addUserFloat("dxy_mu2", m2.innerTrack()->dxy(cleanPV.position()));
                    cand.addUserFloat("dxy_tr",  tr.pseudoTrack().dxy(cleanPV.position()));
                    // High Purity & Isolation
                    cand.addUserInt("mu1_innerTrk_hp", m1.innerTrack()->quality(reco::TrackBase::highPurity));
                    cand.addUserInt("mu2_innerTrk_hp", m2.innerTrack()->quality(reco::TrackBase::highPurity));
                    cand.addUserInt("tr_innerTrk_hp",  tr.trackHighPurity());
                    cand.addUserFloat("relative_iso", sum_pt_iso / p4_ref.pt());
                    cand.addUserFloat("mindca_iso", min_dca);
                    cand.addUserFloat("cosPointingAngle", cosPointingAngle);
                    cand.addUserFloat("pointingAngle", pointingAngle);

                    fillMatchInfo(m1, cand, "mu1");
                    fillMatchInfo(m2, cand, "mu2");

                    cand.addUserFloat("mu1_iso03_clean", mu1_iso03_clean);
                    cand.addUserFloat("mu1_iso04_clean", mu1_iso04_clean);
                    cand.addUserFloat("mu2_iso03_clean", mu2_iso03_clean);
                    cand.addUserFloat("mu2_iso04_clean", mu2_iso04_clean);

                    // --- Transformer-ntuple features (see CandFeatureHelpers.h) ---
                    cand.addUserInt("channel", 1);  // 1 = DSPHIPI
                    cand.addUserInt("pv_idx", (int)b_idx);
                    cand.addUserFloat("pv_ndof", cleanPV.ndof());
                    t3m::addVertexCov(cand, "sv", sv.positionError());
                    t3m::addVertexCov(cand, "pv", cleanPV);
                    cand.addUserFloat("ctau", p4_ref.P() > 0 ? d3d.value() * p4_ref.M() / p4_ref.P() : t3m::kMissing);
                    {
                        const reco::Vertex svVtx(sv);
                        const char* legs[3] = {"mu1", "mu2", "tr"};
                        for (int l = 0; l < 3; ++l)
                            t3m::addLegFeatures(cand, legs[l], muTTracks[l], refit_trks[l], cleanPV, svVtx);
                    }
                    t3m::addPairFeatures(cand, leg_p4, muTTracks);
                    t3m::addKinFit(cand, muTTracks,
                                   {bph::MUON_MASS, bph::MUON_MASS, bph::PI_MASS},
                                   {bph::LEP_SIGMA, bph::LEP_SIGMA, bph::PI_SIGMA});

                    ret_val->push_back(cand);
                }
            }
        }
        evt.put(std::move(ret_val));
    }

    void fillMatchInfo(const pat::Muon& muon, pat::CompositeCandidate& cand, std::string prefix) const {
        const int n_stations = 2;
        std::vector<MatchPair> matches(n_stations, {nullptr, nullptr});
        for (const auto& cm : muon.matches()) {
            int s = cm.station() - 1;
            if (s < 0 || s >= n_stations) continue;
            for (const auto& sm : cm.segmentMatches) {
                if (!(sm.isMask(reco::MuonSegmentMatch::BestInStationByDR) && sm.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR))) continue;
                MatchPair curr(&cm, &sm);
                matches[s] = (matches[s].first) ? getBetterMatch(matches[s], curr) : curr;
            }
        }
        for (int s = 0; s < n_stations; ++s) {
            std::string lbl = prefix + "_match" + std::to_string(s + 1);
            if (matches[s].first && matches[s].second) {
                float dX = matches[s].first->x - matches[s].second->x;
                float errX = std::sqrt(std::pow(matches[s].first->xErr, 2) + std::pow(matches[s].second->xErr, 2));
                cand.addUserFloat(lbl + "_dX", dX);
                cand.addUserFloat(lbl + "_pullX", (errX > 0) ? dX / errX : -999.f);
            } else {
                cand.addUserFloat(lbl + "_dX", -999.f); cand.addUserFloat(lbl + "_pullX", -999.f);
            }
        }
    }

private:
    const edm::EDGetTokenT<pat::MuonCollection> muonsToken_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection> tracksToken_;
    const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection> pcToken_;
    const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
    const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
};

DEFINE_FWK_MODULE(Ds2Mu1TrkBuilder);