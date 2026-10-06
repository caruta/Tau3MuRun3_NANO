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
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "TVector3.h"
#include "TMath.h"
#include "helper.h"
#include "diMuonResonances.h"
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

class Tau3MuBuilder : public edm::global::EDProducer<> {
public:
    explicit Tau3MuBuilder(const edm::ParameterSet& cfg) :
        muonsToken_(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("src"))),
        vtxToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
        pcToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("candidates"))),
        bsToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
        // |total charge| values to build: {1} = signal-like triplets, add 3 for the SS3MU control sample
        absCharges_(cfg.getParameter<std::vector<int>>("absCharges")),
        ttbToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))) {
        produces<pat::CompositeCandidateCollection>();
    }

    void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& iSetup) const override {
        edm::Handle<pat::MuonCollection> muons;
        evt.getByToken(muonsToken_, muons);
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
                for (size_t k = j + 1; k < muons->size(); ++k) {
                    // --- Build Candidate ---
                    pat::CompositeCandidate cand;
                    const pat::Muon &m1 = muons->at(i), &m2 = muons->at(j), &m3 = muons->at(k);
                    if (m1.innerTrack().isNull() || m2.innerTrack().isNull() || m3.innerTrack().isNull()) continue;
                    const int absQ = std::abs(m1.charge() + m2.charge() + m3.charge());
                    if (std::find(absCharges_.begin(), absCharges_.end(), absQ) == absCharges_.end()) continue;
                    std::vector<reco::TransientTrack> muTTracks = {ttb.build(m1.innerTrack()), ttb.build(m2.innerTrack()), ttb.build(m3.innerTrack())};
                    KalmanVertexFitter svFitter(true);
                    TransientVertex sv = svFitter.vertex(muTTracks);
                    if (!sv.isValid() || !sv.hasRefittedTracks()){
                        continue;
                    }
                    // --- Refitted Kinematics ---
                    reco::Candidate::LorentzVector p4_ref(0,0,0,0);
                    std::vector<double> refit_pts;
                    std::vector<reco::Track> refit_trks;
                    std::array<reco::Candidate::LorentzVector, 3> leg_p4;
                    for(const auto& rt : sv.refittedTracks()) {
                        double mass_sq = std::sqrt(rt.track().p2() + 0.0111636); //Muon mass
                        reco::Candidate::LorentzVector lp4(rt.track().px(), rt.track().py(), rt.track().pz(), mass_sq);
                        p4_ref += lp4;
                        if (refit_trks.size() < 3) leg_p4[refit_trks.size()] = lp4;
                        refit_pts.push_back(rt.track().pt());
                        refit_trks.push_back(rt.track());
                    }
                    if (refit_trks.size() != 3) continue;

                    // --- PV Selection & Refit Logic ---
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

                    float dr12 = reco::deltaR(m1, m2);
                    float dr23 = reco::deltaR(m2, m3);
                    float dr13 = reco::deltaR(m1, m3);
                    float dr_max = std::max({dr12, dr23, dr13});

                    float dz12 = std::abs(m1.vz() - m2.vz());
                    float dz23 = std::abs(m2.vz() - m3.vz());
                    float dz13 = std::abs(m1.vz() - m3.vz());
                    float dz_max = std::max({dz12, dz23, dz13});

                    float dxy1 = std::abs(m1.innerTrack()->dxy(cleanPV.position()));
                    float dxy2 = std::abs(m2.innerTrack()->dxy(cleanPV.position()));
                    float dxy3 = std::abs(m3.innerTrack()->dxy(cleanPV.position()));

                    float dxy1_sig = dxy1 / m1.innerTrack()->dxyError();
                    float dxy2_sig = dxy2 / m2.innerTrack()->dxyError();
                    float dxy3_sig = dxy3 / m3.innerTrack()->dxyError();

                    float d0 = std::min({dxy1, dxy2, dxy3});
                    float d0_sig = std::min({dxy1_sig, dxy2_sig, dxy3_sig});
                    float d0max_sig = std::max({dxy1_sig, dxy2_sig, dxy3_sig});

                    // Isolation & DCA
                    float sum_pt_iso = 0; float min_dca = 999.;
                    for (const auto& track : *candidates) {
                        if (track.charge() == 0 || track.pt() < 0.5 || !track.hasTrackDetails()) continue;
                        float dr = reco::deltaR(track, p4_ref);
                        if (reco::deltaR(track, m1) < 0.005 || reco::deltaR(track, m2) < 0.005 || reco::deltaR(track, m3) < 0.005) continue;
                        if (dr < 0.3) sum_pt_iso += track.pt();
                        reco::TransientTrack ttIso = ttb.build(track.pseudoTrack());
                        if (ttIso.isValid()) {
                            auto pca = ttIso.trajectoryStateClosestToPoint(sv.position());
                            if (pca.isValid()) min_dca = std::min(min_dca, (float)(pca.position() - sv.position()).mag());
                        }
                    }



                    // --- Fill Candidate ---
                    cand.setP4(p4_ref);
                    cand.setCharge(m1.charge() + m2.charge() + m3.charge());
                    cand.setVertex(reco::Candidate::Point(sv.position().x(), sv.position().y(), sv.position().z()));

                    cand.addUserInt("mu1_idx", i); cand.addUserInt("mu2_idx", j); cand.addUserInt("mu3_idx", k);
                    cand.addUserFloat("mu1_pt",  m1.pt());
                    cand.addUserFloat("mu1_eta", m1.eta());
                    cand.addUserFloat("mu1_phi", m1.phi());
                    cand.addUserFloat("mu1_charge", m1.charge());

                    cand.addUserFloat("mu2_pt",  m2.pt());
                    cand.addUserFloat("mu2_eta", m2.eta());
                    cand.addUserFloat("mu2_phi", m2.phi());
                    cand.addUserFloat("mu2_charge", m2.charge());

                    cand.addUserFloat("mu3_pt",  m3.pt());
                    cand.addUserFloat("mu3_eta", m3.eta());
                    cand.addUserFloat("mu3_phi", m3.phi());
                    cand.addUserFloat("mu3_charge", m3.charge());

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
                    cand.addUserFloat("refit_mu3_pt",  refit_pts[2]);
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
                    cand.addUserFloat("dxy_mu3",  m3.innerTrack()->dxy(cleanPV.position())); // mu3 treated as 'track'
                    // High Purity & Isolation
                    cand.addUserInt("mu1_innerTrk_hp", m1.innerTrack()->quality(reco::TrackBase::highPurity));
                    cand.addUserInt("mu2_innerTrk_hp", m2.innerTrack()->quality(reco::TrackBase::highPurity));
                    cand.addUserInt("mu3_innerTrk_hp", m3.innerTrack()->quality(reco::TrackBase::highPurity));
                    cand.addUserFloat("relative_iso", sum_pt_iso / p4_ref.pt());
                    cand.addUserFloat("mindca_iso", min_dca);
                    cand.addUserFloat("cosPointingAngle", cosPointingAngle);
                    cand.addUserFloat("pointingAngle", pointingAngle);

                    fillMatchInfo(m1, cand, "mu1");
                    fillMatchInfo(m2, cand, "mu2");
                    fillMatchInfo(m3, cand, "mu3");

                    cand.addUserFloat("dr12", dr12); cand.addUserFloat("dr23", dr23); cand.addUserFloat("dr13", dr13);
                    cand.addUserFloat("dr_max", dr_max);
                    cand.addUserFloat("dz12", dz12); cand.addUserFloat("dz23", dz23); cand.addUserFloat("dz13", dz13);
                    cand.addUserFloat("dz_max", dz_max);
                    cand.addUserFloat("d0", d0);
                    cand.addUserFloat("d0_sig", d0_sig);
                    cand.addUserFloat("d0max_sig", d0max_sig);

                    // --- Transformer-ntuple features (see CandFeatureHelpers.h) ---
                    cand.addUserInt("channel", absQ == 1 ? 0 : 2);  // 0 = TAU3MU, 2 = SS3MU
                    cand.addUserInt("pv_idx", (int)b_idx);
                    cand.addUserFloat("pv_ndof", cleanPV.ndof());
                    t3m::addVertexCov(cand, "sv", sv.positionError());
                    t3m::addVertexCov(cand, "pv", cleanPV);
                    cand.addUserFloat("ctau", p4_ref.P() > 0 ? d3d.value() * p4_ref.M() / p4_ref.P() : t3m::kMissing);
                    {
                        const reco::Vertex svVtx(sv);
                        const char* legs[3] = {"mu1", "mu2", "mu3"};
                        for (int l = 0; l < 3; ++l)
                            t3m::addLegFeatures(cand, legs[l], muTTracks[l], refit_trks[l], cleanPV, svVtx);
                    }
                    t3m::addPairFeatures(cand, leg_p4, muTTracks);
                    t3m::addKinFit(cand, muTTracks,
                                   {bph::MUON_MASS, bph::MUON_MASS, bph::MUON_MASS},
                                   {bph::LEP_SIGMA, bph::LEP_SIGMA, bph::LEP_SIGMA});

                    bool isResonance = vetoResonances(evt, ttb, i, j, k, cand);
                    cand.addUserInt("isVetoResonance", isResonance ? 1 : 0);
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

    bool vetoResonances(edm::Event& evt, const TransientTrackBuilder& ttb, 
                        const int idx_mu1, const int idx_mu2, const int idx_mu3,
                        pat::CompositeCandidate& cand) const {
        
        bool isMatchingResonance = false;
        edm::Handle<pat::MuonCollection> all_muons;
        evt.getByToken(muonsToken_, all_muons);

        const int triplet_indices[3] = {idx_mu1, idx_mu2, idx_mu3};
        const float fitProb_min = 0.05;
        float best_prob = -1., best_mass = 0.;

        //std::vector<std::pair<float, float>> resonancesToVeto = {{0.782, 0.02}, {1.019, 0.02}, {3.096, 0.04}};
        //const float SIGMA_TO_EXCLUDE = 3.0;

        for(size_t mu_idx = 0; mu_idx < all_muons->size(); ++mu_idx) {
            if (mu_idx == (size_t)idx_mu1 || mu_idx == (size_t)idx_mu2 || mu_idx == (size_t)idx_mu3) continue;
            
            const pat::Muon& extraMu = all_muons->at(mu_idx);
            if (extraMu.innerTrack().isNull()) continue;
            reco::TransientTrack ttExtra = ttb.build(extraMu.innerTrack());

            for(size_t i = 0; i < 3; ++i) {   
                const pat::Muon& tauMu = all_muons->at(triplet_indices[i]);
                if (tauMu.innerTrack().isNull()) continue;

                if((tauMu.charge() + extraMu.charge()) != 0) continue;

                std::vector<reco::TransientTrack> pairTracks = {ttExtra, ttb.build(tauMu.innerTrack())};
                KalmanVertexFitter fitter(true);
                TransientVertex tv = fitter.vertex(pairTracks);

                if (!tv.isValid()) continue;

                float prob = TMath::Prob(tv.totalChiSquared(), tv.degreesOfFreedom());
                if (prob < fitProb_min) continue;


                reco::Candidate::LorentzVector p4_pair = tauMu.p4() + extraMu.p4();
                float fitMass = p4_pair.M();

                if(prob > best_prob){
                    best_prob = prob;
                    best_mass = fitMass;
                }

                for(const auto& reso : resonancesToVeto){
                    if( std::abs(fitMass - reso.first) < (SIGMA_TO_EXCLUDE * reso.second) ){
                        isMatchingResonance = true;
                    }
                }
            }
        }

        cand.addUserFloat("diMuVtxFit_bestProb", best_prob);
        cand.addUserFloat("diMuVtxFit_bestMass", best_mass);
        return isMatchingResonance;
    }


private:
    const edm::EDGetTokenT<pat::MuonCollection> muonsToken_;
    const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection> pcToken_;
    const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
    const std::vector<int> absCharges_;
    const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
};

DEFINE_FWK_MODULE(Tau3MuBuilder);