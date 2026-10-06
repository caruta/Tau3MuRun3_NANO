// CandidateContextProducer
//
// Writes the per-candidate context tokens for transformer training:
//   <prefix>Trk : charged tracks (packedPFCandidates + lostTracks) around each candidate
//   <prefix>Neu : neutral PF candidates (photons, neutral hadrons) around each candidate
//   <prefix>SV  : IVF secondary vertices of the event, ordered by distance to the candidate SV
// plus an extension of the candidate table (<candName>) with isolation and token counts.
//
// Every token row carries candIdx, the row of the candidate in <candName>. Leg x token
// pair features (leg = the 3 candidate legs) are stored as columns of the token rows,
// e.g. pair_leg1_m_pi, pair_leg2_dca.
//
// Token selection, ordering and truncation follow the ntuple schema:
//   tracks   : inCone (dR < coneDR to candidate momentum) OR inCylinder (DCA to the PV->SV
//              line < cylDCA and dR < cylMaxDR); ordered by 3D IP significance wrt the SV
//   neutrals : inCone (dR < neuConeDR to candidate momentum) OR inFlightCone (dR < neuConeDR
//              to the PV->SV direction); ordered by the smaller of the two dR
//   SVs      : all IVF SVs, ordered by 3D distance to the candidate SV
// The *_total counters in the candidate extension are taken before truncation.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "CandFeatureHelpers.h"

namespace {

  constexpr double kMuMass = 0.10565837;
  constexpr double kPiMass = 0.13957039;
  constexpr double kKMass = 0.493677;
  constexpr float kBig = std::numeric_limits<float>::max();

  struct Leg {
    reco::TransientTrack tt;
    reco::Candidate::LorentzVector p4;  // with the leg mass hypothesis (mu, or pi for the DsPhiPi track)
    float pt, eta, phi;
    int charge;
  };

  struct TrkTok {
    const pat::PackedCandidate* pc;
    reco::TransientTrack tt;
    bool isLost, inCone, inCyl;
    float dR, deta, dphi, dcaFlight;
    float ip3dSV, ip3dSVsig;
    int svTok;
    float muonSoftMva;
    bool isMuon, isLooseMuon;
  };

  struct NeuTok {
    const pat::PackedCandidate* pc;
    bool inCone, inFlightCone;
    float dR, dRflight, deta, dphi;
  };

  struct SVTok {
    int evtIdx;
    float dist, distSig;
  };

  // Distance between the PV->SV line and the straight-line approximation of a track at its
  // point of closest approach to the SV.
  float lineDistance(const GlobalPoint& pv, const GlobalVector& fdir, const GlobalPoint& p, const GlobalVector& d) {
    const GlobalVector w = p - pv;
    const GlobalVector n = fdir.cross(d);
    if (n.mag() > 1e-9)
      return std::abs(w.dot(n)) / n.mag();
    return (w - fdir * w.dot(fdir)).mag();  // parallel lines
  }

  reco::Vertex vertexFromUserFloats(const pat::CompositeCandidate& c, const std::string& pre, double chi2, double ndof) {
    reco::Vertex::Point pos(c.userFloat(pre + "_x"), c.userFloat(pre + "_y"), c.userFloat(pre + "_z"));
    reco::Vertex::Error err;
    err(0, 0) = c.userFloat(pre + "_cxx");
    err(1, 0) = c.userFloat(pre + "_cyx");
    err(1, 1) = c.userFloat(pre + "_cyy");
    err(2, 0) = c.userFloat(pre + "_czx");
    err(2, 1) = c.userFloat(pre + "_czy");
    err(2, 2) = c.userFloat(pre + "_czz");
    return reco::Vertex(pos, err, chi2, ndof, 3);
  }

  template <typename T>
  void addCol(nanoaod::FlatTable& t, const std::string& name, const std::vector<T>& v, const std::string& doc, int bits = -1) {
    t.addColumn<T>(name, v, doc, bits);
  }

}  // namespace

class CandidateContextProducer : public edm::global::EDProducer<> {
public:
  explicit CandidateContextProducer(const edm::ParameterSet& cfg)
      : candToken_(consumes<pat::CompositeCandidateCollection>(cfg.getParameter<edm::InputTag>("src"))),
        muonsToken_(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        pfToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCands"))),
        lostToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("lostTracks"))),
        allMuonsToken_(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("allMuons"))),
        svToken_(consumes<reco::VertexCompositePtrCandidateCollection>(cfg.getParameter<edm::InputTag>("secondaryVertices"))),
        ttbToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
        candName_(cfg.getParameter<std::string>("candName")),
        prefix_(cfg.getParameter<std::string>("prefix")),
        legIdx_(cfg.getParameter<std::vector<std::string>>("legIdx")),
        // vbool is not a valid ParameterSet type: legIsTrack is a vint32 of 0/1
        legIsTrack_(cfg.getParameter<std::vector<int>>("legIsTrack")),
        trkMinPt_(cfg.getParameter<double>("trkMinPt")),
        trkMaxEta_(cfg.getParameter<double>("trkMaxEta")),
        coneDR_(cfg.getParameter<double>("coneDR")),
        cylDCA_(cfg.getParameter<double>("cylDCA")),
        cylMaxDR_(cfg.getParameter<double>("cylMaxDR")),
        neuMinPt_(cfg.getParameter<double>("neuMinPt")),
        neuConeDR_(cfg.getParameter<double>("neuConeDR")),
        maxTrk_(cfg.getParameter<unsigned int>("maxTrk")),
        maxNeu_(cfg.getParameter<unsigned int>("maxNeu")),
        maxSV_(cfg.getParameter<unsigned int>("maxSV")),
        nDchi2_(cfg.getParameter<unsigned int>("nDchi2")),
        doPairFits_(cfg.getParameter<bool>("doPairFits")) {
    if (legIdx_.size() != 3 || legIsTrack_.size() != 3)
      throw cms::Exception("Configuration") << "CandidateContextProducer: legIdx and legIsTrack need 3 entries";
    const auto legTracksTag = cfg.getParameter<edm::InputTag>("legTracks");
    hasLegTracks_ = !legTracksTag.label().empty();
    if (hasLegTracks_)
      legTracksToken_ = consumes<pat::PackedCandidateCollection>(legTracksTag);
    produces<nanoaod::FlatTable>("cand");
    produces<nanoaod::FlatTable>("trk");
    produces<nanoaod::FlatTable>("neu");
    produces<nanoaod::FlatTable>("sv");
  }

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& iSetup) const override;

private:
  bool isLeg(const reco::Candidate& c, const std::vector<Leg>& legs) const {
    for (const auto& l : legs) {
      if (c.charge() != l.charge)
        continue;
      if (reco::deltaR(c.eta(), c.phi(), l.eta, l.phi) < 0.005 && std::abs(c.pt() - l.pt) < 0.05 * l.pt)
        return true;
    }
    return false;
  }

  const edm::EDGetTokenT<pat::CompositeCandidateCollection> candToken_;
  const edm::EDGetTokenT<pat::MuonCollection> muonsToken_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> legTracksToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> lostToken_;
  const edm::EDGetTokenT<pat::MuonCollection> allMuonsToken_;
  const edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> svToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
  const std::string candName_, prefix_;
  const std::vector<std::string> legIdx_;
  const std::vector<int> legIsTrack_;
  const double trkMinPt_, trkMaxEta_, coneDR_, cylDCA_, cylMaxDR_, neuMinPt_, neuConeDR_;
  const unsigned int maxTrk_, maxNeu_, maxSV_, nDchi2_;
  const bool doPairFits_;
  bool hasLegTracks_ = false;
};

void CandidateContextProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& iSetup) const {
  edm::Handle<pat::CompositeCandidateCollection> cands;
  evt.getByToken(candToken_, cands);
  edm::Handle<pat::MuonCollection> muons;
  evt.getByToken(muonsToken_, muons);
  edm::Handle<pat::PackedCandidateCollection> legTracks;
  if (hasLegTracks_)
    evt.getByToken(legTracksToken_, legTracks);
  edm::Handle<pat::PackedCandidateCollection> pfCands;
  evt.getByToken(pfToken_, pfCands);
  edm::Handle<pat::PackedCandidateCollection> lostTracks;
  evt.getByToken(lostToken_, lostTracks);
  edm::Handle<pat::MuonCollection> allMuons;
  evt.getByToken(allMuonsToken_, allMuons);
  edm::Handle<reco::VertexCompositePtrCandidateCollection> ivf;
  evt.getByToken(svToken_, ivf);
  const auto& ttb = iSetup.getData(ttbToken_);

  // IVF SV membership of packed candidates: (product, key) -> event SV index
  std::map<std::pair<edm::ProductID, size_t>, int> trkToIvf;
  for (size_t s = 0; s < ivf->size(); ++s)
    for (size_t d = 0; d < ivf->at(s).numberOfDaughters(); ++d) {
      const auto ptr = ivf->at(s).daughterPtr(d);
      trkToIvf[{ptr.id(), ptr.key()}] = s;
    }

  const size_t nCand = cands->size();

  // candidate extension columns
  std::vector<float> c_iso03(nCand), c_iso05(nCand);
  std::vector<int> c_nVtxCompat(nCand), c_nTrk(nCand), c_nTrkCone(nCand), c_nTrkCyl(nCand), c_nNeu(nCand), c_nSV(nCand);

  // token columns
  std::vector<int> t_cand, t_charge, t_pdgId, t_nPix, t_nHits, t_lostInner, t_fromPV, t_pvQual, t_svIdx;
  std::vector<bool> t_isLost, t_hp, t_hasDetails, t_inCone, t_inCyl, t_isMuon, t_isLooseMuon;
  std::vector<float> t_pt, t_eta, t_phi, t_deta, t_dphi, t_dR, t_dcaFlight, t_dxy, t_dz, t_dxyErr, t_dzErr,
      t_ip3dSV, t_ip3dSVsig, t_normChi2, t_puppi, t_hcalFrac, t_muSoftMva, t_dchi2;
  std::vector<float> t_pair[3][4];  // per leg: m_pi, m_K, dca, vprob

  std::vector<int> n_cand, n_pdgId;
  std::vector<bool> n_inCone, n_inFlightCone;
  std::vector<float> n_pt, n_eta, n_phi, n_deta, n_dphi, n_dR, n_dRflight, n_puppi, n_hcalFrac;
  std::vector<float> n_pair[3];  // per leg: m(leg, neutral)

  std::vector<int> s_cand, s_ntrk, s_shares;
  std::vector<float> s_mass, s_pt, s_eta, s_phi, s_chi2, s_ndof, s_dlen, s_dlenSig, s_dxy, s_dxySig, s_cosPA, s_dist,
      s_distSig;

  VertexDistance3D dist3D;
  VertexDistanceXY distXY;

  for (size_t ic = 0; ic < nCand; ++ic) {
    const auto& cand = cands->at(ic);

    // --- legs ---
    std::vector<Leg> legs;
    bool legsOk = true;
    for (int l = 0; l < 3; ++l) {
      const int idx = cand.userInt(legIdx_[l]);
      Leg leg;
      if (legIsTrack_[l]) {
        if (!hasLegTracks_ || idx < 0 || (size_t)idx >= legTracks->size()) {
          legsOk = false;
          break;
        }
        const auto& pc = legTracks->at(idx);
        const auto& trk = pc.pseudoTrack();
        leg.tt = ttb.build(trk);
        leg.p4 = reco::Candidate::LorentzVector(trk.px(), trk.py(), trk.pz(), std::sqrt(trk.p2() + kPiMass * kPiMass));
        leg.pt = trk.pt(), leg.eta = trk.eta(), leg.phi = trk.phi(), leg.charge = trk.charge();
      } else {
        if (idx < 0 || (size_t)idx >= muons->size() || muons->at(idx).innerTrack().isNull()) {
          legsOk = false;
          break;
        }
        const auto& trk = *muons->at(idx).innerTrack();
        leg.tt = ttb.build(trk);
        leg.p4 = reco::Candidate::LorentzVector(trk.px(), trk.py(), trk.pz(), std::sqrt(trk.p2() + kMuMass * kMuMass));
        leg.pt = trk.pt(), leg.eta = trk.eta(), leg.phi = trk.phi(), leg.charge = trk.charge();
      }
      legs.push_back(leg);
    }

    // --- candidate vertices ---
    const reco::Vertex pvVtx = vertexFromUserFloats(cand, "pv", 0., cand.userFloat("pv_ndof"));
    const reco::Vertex svVtx = vertexFromUserFloats(cand, "sv", cand.userFloat("sv_chi2"), cand.userFloat("sv_ndof"));
    const int pvIdx = cand.hasUserInt("pv_idx") ? cand.userInt("pv_idx") : -1;
    const GlobalPoint pvPos(pvVtx.x(), pvVtx.y(), pvVtx.z());
    const GlobalPoint svPos(svVtx.x(), svVtx.y(), svVtx.z());
    GlobalVector fdir = svPos - pvPos;
    const float fEta = fdir.mag() > 0 ? static_cast<float>(fdir.eta()) : static_cast<float>(cand.eta());
    const float fPhi = fdir.mag() > 0 ? static_cast<float>(fdir.phi()) : static_cast<float>(cand.phi());
    fdir = fdir.mag() > 0 ? fdir.unit() : GlobalVector(cand.px(), cand.py(), cand.pz()).unit();
    const auto cP4 = cand.p4();

    // --- IVF SV tokens ---
    std::vector<SVTok> svToks;
    for (size_t s = 0; s < ivf->size(); ++s) {
      const auto& v = ivf->at(s);
      const reco::Vertex ivfVtx(v.vertex(), v.vertexCovariance(), v.vertexChi2(), v.vertexNdof(), v.numberOfDaughters());
      const auto d = dist3D.distance(ivfVtx, svVtx);
      svToks.push_back({(int)s, (float)d.value(), (float)d.significance()});
    }
    c_nSV[ic] = svToks.size();
    std::sort(svToks.begin(), svToks.end(), [](const SVTok& a, const SVTok& b) { return a.dist < b.dist; });
    if (svToks.size() > maxSV_)
      svToks.resize(maxSV_);
    std::map<int, int> ivfToTok;
    for (size_t t = 0; t < svToks.size(); ++t) {
      const auto& v = ivf->at(svToks[t].evtIdx);
      ivfToTok[svToks[t].evtIdx] = t;
      const reco::Vertex ivfVtx(v.vertex(), v.vertexCovariance(), v.vertexChi2(), v.vertexNdof(), v.numberOfDaughters());
      const auto dl = dist3D.distance(pvVtx, ivfVtx);
      const auto dxy = distXY.distance(pvVtx, ivfVtx);
      const GlobalVector dv(v.vx() - pvVtx.x(), v.vy() - pvVtx.y(), v.vz() - pvVtx.z());
      const GlobalVector pv(v.px(), v.py(), v.pz());
      int shares = 0;
      for (size_t d = 0; d < v.numberOfDaughters(); ++d)
        if (isLeg(*v.daughter(d), legs))
          shares = 1;
      s_cand.push_back(ic);
      s_mass.push_back(v.mass());
      s_pt.push_back(v.pt());
      s_eta.push_back(v.eta());
      s_phi.push_back(v.phi());
      s_ntrk.push_back(v.numberOfDaughters());
      s_chi2.push_back(v.vertexChi2());
      s_ndof.push_back(v.vertexNdof());
      s_dlen.push_back(dl.value());
      s_dlenSig.push_back(dl.significance());
      s_dxy.push_back(dxy.value());
      s_dxySig.push_back(dxy.significance());
      s_cosPA.push_back((dv.mag() > 0 && pv.mag() > 0) ? dv.unit().dot(pv.unit()) : t3m::kMissing);
      s_dist.push_back(svToks[t].dist);
      s_distSig.push_back(svToks[t].distSig);
      s_shares.push_back(shares);
    }

    // --- charged track tokens ---
    std::vector<TrkTok> trks;
    float iso03 = 0, iso05 = 0;
    int nVtxCompat = 0, nCone = 0, nCyl = 0;
    const double maxDR = std::max(coneDR_, cylMaxDR_);
    auto scan = [&](const edm::Handle<pat::PackedCandidateCollection>& coll, bool lost) {
      for (size_t i = 0; i < coll->size(); ++i) {
        const auto& pc = coll->at(i);
        if (pc.charge() == 0 || pc.pt() < trkMinPt_ || std::abs(pc.eta()) > trkMaxEta_)
          continue;
        const float dR = reco::deltaR(pc.eta(), pc.phi(), cP4.eta(), cP4.phi());
        if (dR > maxDR || isLeg(pc, legs))
          continue;
        const int fromPV = (pvIdx >= 0 && pc.vertexRef().isNonnull()) ? pc.fromPV(pvIdx) : -1;
        if (fromPV >= pat::PackedCandidate::PVTight) {
          if (dR < 0.3)
            iso03 += pc.pt();
          if (dR < 0.5)
            iso05 += pc.pt();
        }
        TrkTok t{};
        t.pc = &pc;
        t.isLost = lost;
        t.dR = dR;
        t.deta = pc.eta() - cP4.eta();
        t.dphi = reco::deltaPhi(pc.phi(), cP4.phi());
        t.tt = ttb.build(pc.pseudoTrack());
        t.dcaFlight = kBig;
        t.ip3dSV = t3m::kMissing;
        t.ip3dSVsig = kBig;
        if (t.tt.isValid()) {
          const auto tsc = t.tt.trajectoryStateClosestToPoint(svPos);
          if (tsc.isValid())
            t.dcaFlight = lineDistance(pvPos, fdir, tsc.position(), tsc.momentum().unit());
          const auto ip = IPTools::absoluteImpactParameter3D(t.tt, svVtx);
          if (ip.first) {
            t.ip3dSV = ip.second.value();
            t.ip3dSVsig = ip.second.significance();
          }
        }
        t.inCone = dR < coneDR_;
        t.inCyl = t.dcaFlight < cylDCA_ && dR < cylMaxDR_;
        if (!(t.inCone || t.inCyl))
          continue;
        nCone += t.inCone;
        nCyl += t.inCyl;
        if (t.inCone && t.ip3dSVsig < 2.)
          ++nVtxCompat;
        const auto it = trkToIvf.find({coll.id(), i});
        t.svTok = -1;
        if (it != trkToIvf.end()) {
          const auto jt = ivfToTok.find(it->second);
          if (jt != ivfToTok.end())
            t.svTok = jt->second;
        }
        t.isMuon = false, t.isLooseMuon = false, t.muonSoftMva = t3m::kMissing;
        for (const auto& mu : *allMuons) {
          if (mu.innerTrack().isNull() || mu.charge() != pc.charge())
            continue;
          const auto& it2 = *mu.innerTrack();
          if (reco::deltaR(it2.eta(), it2.phi(), pc.eta(), pc.phi()) < 0.005 && std::abs(it2.pt() - pc.pt()) < 0.05 * pc.pt()) {
            t.isMuon = true;
            t.isLooseMuon = mu.isLooseMuon();
            t.muonSoftMva = mu.softMvaValue();
            break;
          }
        }
        trks.push_back(t);
      }
    };
    scan(pfCands, false);
    scan(lostTracks, true);
    c_nTrk[ic] = trks.size();
    c_nTrkCone[ic] = nCone;
    c_nTrkCyl[ic] = nCyl;
    c_nVtxCompat[ic] = nVtxCompat;
    c_iso03[ic] = iso03;
    c_iso05[ic] = iso05;
    std::sort(trks.begin(), trks.end(), [](const TrkTok& a, const TrkTok& b) { return a.ip3dSVsig < b.ip3dSVsig; });
    if (trks.size() > maxTrk_)
      trks.resize(maxTrk_);

    const float svChi2 = cand.userFloat("sv_chi2");
    for (size_t k = 0; k < trks.size(); ++k) {
      const auto& t = trks[k];
      const auto& pc = *t.pc;
      const auto& trk = pc.pseudoTrack();
      const int fromPV = (pvIdx >= 0 && pc.vertexRef().isNonnull()) ? pc.fromPV(pvIdx) : -1;
      t_cand.push_back(ic);
      t_pt.push_back(pc.pt());
      t_eta.push_back(pc.eta());
      t_phi.push_back(pc.phi());
      t_charge.push_back(pc.charge());
      t_pdgId.push_back(pc.pdgId());
      t_isLost.push_back(t.isLost);
      t_hp.push_back(pc.trackHighPurity());
      t_hasDetails.push_back(pc.hasTrackDetails());
      t_deta.push_back(t.deta);
      t_dphi.push_back(t.dphi);
      t_dR.push_back(t.dR);
      t_inCone.push_back(t.inCone);
      t_inCyl.push_back(t.inCyl);
      t_dcaFlight.push_back(t.dcaFlight < kBig ? t.dcaFlight : t3m::kMissing);
      t_dxy.push_back(trk.dxy(pvVtx.position()));
      t_dz.push_back(trk.dz(pvVtx.position()));
      t_dxyErr.push_back(trk.dxyError());
      t_dzErr.push_back(trk.dzError());
      t_ip3dSV.push_back(t.ip3dSV);
      t_ip3dSVsig.push_back(t.ip3dSVsig < kBig ? t.ip3dSVsig : t3m::kMissing);
      t_nPix.push_back(pc.numberOfPixelHits());
      t_nHits.push_back(pc.numberOfHits());
      t_lostInner.push_back(pc.lostInnerHits());
      t_normChi2.push_back(pc.hasTrackDetails() ? trk.normalizedChi2() : t3m::kMissing);
      t_fromPV.push_back(fromPV);
      t_pvQual.push_back(pc.pvAssociationQuality());
      t_puppi.push_back(pc.puppiWeight());
      t_hcalFrac.push_back(pc.hcalFraction());
      t_isMuon.push_back(t.isMuon);
      t_isLooseMuon.push_back(t.isLooseMuon);
      t_muSoftMva.push_back(t.muonSoftMva);
      t_svIdx.push_back(t.svTok);

      // Delta chi2 of the candidate SV when this track is added (top nDchi2 tracks only)
      float dchi2 = t3m::kMissing;
      if (k < nDchi2_ && legsOk && t.tt.isValid()) {
        try {
          KalmanVertexFitter fitter(false);
          TransientVertex v4 = fitter.vertex(std::vector<reco::TransientTrack>{legs[0].tt, legs[1].tt, legs[2].tt, t.tt});
          if (v4.isValid())
            dchi2 = v4.totalChiSquared() - svChi2;
        } catch (...) {
        }
      }
      t_dchi2.push_back(dchi2);

      // leg x track pair features
      const reco::Candidate::LorentzVector pPi(trk.px(), trk.py(), trk.pz(), std::sqrt(trk.p2() + kPiMass * kPiMass));
      const reco::Candidate::LorentzVector pK(trk.px(), trk.py(), trk.pz(), std::sqrt(trk.p2() + kKMass * kKMass));
      for (int l = 0; l < 3; ++l) {
        float mpi = t3m::kMissing, mk = t3m::kMissing, dca = t3m::kMissing, vprob = t3m::kMissing;
        if (legsOk) {
          mpi = (legs[l].p4 + pPi).M();
          mk = (legs[l].p4 + pK).M();
          if (t.tt.isValid() && legs[l].tt.isValid()) {
            TwoTrackMinimumDistance ttmd;
            if (ttmd.calculate(legs[l].tt.initialFreeState(), t.tt.initialFreeState()))
              dca = ttmd.distance();
            if (doPairFits_)
              vprob = t3m::pairVtxProb(legs[l].tt, t.tt);
          }
        }
        t_pair[l][0].push_back(mpi);
        t_pair[l][1].push_back(mk);
        t_pair[l][2].push_back(dca);
        t_pair[l][3].push_back(vprob);
      }
    }

    // --- neutral tokens ---
    std::vector<NeuTok> neus;
    for (const auto& pc : *pfCands) {
      const int apdg = std::abs(pc.pdgId());
      if (pc.charge() != 0 || (apdg != 22 && apdg != 130) || pc.pt() < neuMinPt_)
        continue;
      NeuTok n{};
      n.pc = &pc;
      n.dR = reco::deltaR(pc.eta(), pc.phi(), cP4.eta(), cP4.phi());
      n.dRflight = reco::deltaR(pc.eta(), pc.phi(), fEta, fPhi);
      n.inCone = n.dR < neuConeDR_;
      n.inFlightCone = n.dRflight < neuConeDR_;
      if (!(n.inCone || n.inFlightCone))
        continue;
      n.deta = pc.eta() - cP4.eta();
      n.dphi = reco::deltaPhi(pc.phi(), cP4.phi());
      neus.push_back(n);
    }
    c_nNeu[ic] = neus.size();
    std::sort(neus.begin(), neus.end(), [](const NeuTok& a, const NeuTok& b) {
      return std::min(a.dR, a.dRflight) < std::min(b.dR, b.dRflight);
    });
    if (neus.size() > maxNeu_)
      neus.resize(maxNeu_);
    for (const auto& n : neus) {
      const auto& pc = *n.pc;
      n_cand.push_back(ic);
      n_pt.push_back(pc.pt());
      n_eta.push_back(pc.eta());
      n_phi.push_back(pc.phi());
      n_pdgId.push_back(pc.pdgId());
      n_deta.push_back(n.deta);
      n_dphi.push_back(n.dphi);
      n_dR.push_back(n.dR);
      n_dRflight.push_back(n.dRflight);
      n_inCone.push_back(n.inCone);
      n_inFlightCone.push_back(n.inFlightCone);
      n_puppi.push_back(pc.puppiWeight());
      n_hcalFrac.push_back(pc.hcalFraction());
      for (int l = 0; l < 3; ++l)
        n_pair[l].push_back(legsOk ? (legs[l].p4 + pc.p4()).M() : t3m::kMissing);
    }
  }

  // --- tables ---
  auto candTab = std::make_unique<nanoaod::FlatTable>(nCand, candName_, false, true);
  addCol(*candTab, "iso_trk_dR03", c_iso03, "sum pT of PV-associated tracks (PVTight or better) in dR<0.3, legs excluded", 10);
  addCol(*candTab, "iso_trk_dR05", c_iso05, "sum pT of PV-associated tracks (PVTight or better) in dR<0.5, legs excluded", 10);
  addCol(*candTab, "n_trk_vtxcompat", c_nVtxCompat, "tracks in cone with 3D IP significance wrt SV < 2");
  addCol(*candTab, "nTrk_total", c_nTrk, "track tokens before truncation (cone OR cylinder)");
  addCol(*candTab, "nTrk_cone", c_nTrkCone, "track tokens inCone before truncation");
  addCol(*candTab, "nTrk_cyl", c_nTrkCyl, "track tokens inCylinder before truncation");
  addCol(*candTab, "nNeu_total", c_nNeu, "neutral tokens before truncation");
  addCol(*candTab, "nSV_total", c_nSV, "IVF SVs in the event before truncation");
  evt.put(std::move(candTab), "cand");

  auto trkTab = std::make_unique<nanoaod::FlatTable>(t_cand.size(), prefix_ + "Trk", false, false);
  addCol(*trkTab, "candIdx", t_cand, "row of the candidate in " + candName_);
  addCol(*trkTab, "pt", t_pt, "pt", 12);
  addCol(*trkTab, "eta", t_eta, "eta", 12);
  addCol(*trkTab, "phi", t_phi, "phi", 12);
  addCol(*trkTab, "charge", t_charge, "charge");
  addCol(*trkTab, "pdgId", t_pdgId, "PF pdgId");
  addCol(*trkTab, "isLostTrack", t_isLost, "from lostTracks");
  addCol(*trkTab, "highPurity", t_hp, "high-purity track");
  addCol(*trkTab, "hasTrackDetails", t_hasDetails, "full track covariance stored (else parametrized)");
  addCol(*trkTab, "deta", t_deta, "eta - candidate eta", 10);
  addCol(*trkTab, "dphi", t_dphi, "phi - candidate phi", 10);
  addCol(*trkTab, "dR", t_dR, "dR to candidate momentum", 10);
  addCol(*trkTab, "inCone", t_inCone, "dR < coneDR to candidate momentum");
  addCol(*trkTab, "inCylinder", t_inCyl, "DCA to PV->SV line < cylDCA and dR < cylMaxDR");
  addCol(*trkTab, "dcaFlight", t_dcaFlight, "distance to the PV->SV line [cm]", 10);
  addCol(*trkTab, "dxy", t_dxy, "dxy wrt candidate refitted PV [cm]", 10);
  addCol(*trkTab, "dz", t_dz, "dz wrt candidate refitted PV [cm]", 10);
  addCol(*trkTab, "dxyErr", t_dxyErr, "dxy uncertainty [cm]", 10);
  addCol(*trkTab, "dzErr", t_dzErr, "dz uncertainty [cm]", 10);
  addCol(*trkTab, "ip3d_sv", t_ip3dSV, "3D IP wrt candidate SV [cm]", 10);
  addCol(*trkTab, "ip3d_sv_sig", t_ip3dSVsig, "3D IP significance wrt candidate SV", 10);
  addCol(*trkTab, "nPixelHits", t_nPix, "valid pixel hits");
  addCol(*trkTab, "nHits", t_nHits, "valid hits");
  addCol(*trkTab, "lostInnerHits", t_lostInner, "lost inner hits");
  addCol(*trkTab, "normChi2", t_normChi2, "track normalized chi2 (-999 without track details)", 10);
  addCol(*trkTab, "fromPV", t_fromPV, "fromPV wrt candidate PV (-1 if unavailable)");
  addCol(*trkTab, "pvAssocQuality", t_pvQual, "pvAssociationQuality");
  addCol(*trkTab, "puppiWeight", t_puppi, "puppi weight", 8);
  addCol(*trkTab, "hcalFraction", t_hcalFrac, "hcal energy fraction", 8);
  addCol(*trkTab, "isMuon", t_isMuon, "matched to a slimmedMuon");
  addCol(*trkTab, "isLooseMuon", t_isLooseMuon, "matched muon passes loose ID");
  addCol(*trkTab, "muon_softMva", t_muSoftMva, "soft MVA of the matched muon (-999 if none)", 10);
  addCol(*trkTab, "svIdx", t_svIdx, "position of the containing IVF SV among this candidate's " + prefix_ + "SV rows (-1 if none)");
  addCol(*trkTab, "sv_dchi2", t_dchi2, "chi2 increase of the candidate SV when adding this track (top nDchi2 only, else -999)", 10);
  const char* pairNames[4] = {"m_pi", "m_K", "dca", "vprob"};
  const char* pairDocs[4] = {"mass(leg, track as pion)", "mass(leg, track as kaon)", "2-track DCA [cm]", "2-track vertex probability (-999 if not computed)"};
  for (int l = 0; l < 3; ++l)
    for (int f = 0; f < 4; ++f)
      addCol(*trkTab, "pair_leg" + std::to_string(l + 1) + "_" + pairNames[f], t_pair[l][f], pairDocs[f], 10);
  evt.put(std::move(trkTab), "trk");

  auto neuTab = std::make_unique<nanoaod::FlatTable>(n_cand.size(), prefix_ + "Neu", false, false);
  addCol(*neuTab, "candIdx", n_cand, "row of the candidate in " + candName_);
  addCol(*neuTab, "pt", n_pt, "pt", 10);
  addCol(*neuTab, "eta", n_eta, "eta", 10);
  addCol(*neuTab, "phi", n_phi, "phi", 10);
  addCol(*neuTab, "pdgId", n_pdgId, "PF pdgId (22 photon, 130 neutral hadron)");
  addCol(*neuTab, "deta", n_deta, "eta - candidate eta", 10);
  addCol(*neuTab, "dphi", n_dphi, "phi - candidate phi", 10);
  addCol(*neuTab, "dR", n_dR, "dR to candidate momentum", 10);
  addCol(*neuTab, "dR_flight", n_dRflight, "dR to the PV->SV direction", 10);
  addCol(*neuTab, "inCone", n_inCone, "dR < neuConeDR to candidate momentum");
  addCol(*neuTab, "inFlightCone", n_inFlightCone, "dR < neuConeDR to the PV->SV direction");
  addCol(*neuTab, "puppiWeight", n_puppi, "puppi weight", 8);
  addCol(*neuTab, "hcalFraction", n_hcalFrac, "hcal energy fraction", 8);
  for (int l = 0; l < 3; ++l)
    addCol(*neuTab, "pair_leg" + std::to_string(l + 1) + "_m", n_pair[l], "mass(leg, neutral)", 10);
  evt.put(std::move(neuTab), "neu");

  auto svTab = std::make_unique<nanoaod::FlatTable>(s_cand.size(), prefix_ + "SV", false, false);
  addCol(*svTab, "candIdx", s_cand, "row of the candidate in " + candName_);
  addCol(*svTab, "mass", s_mass, "mass", 10);
  addCol(*svTab, "pt", s_pt, "pt", 10);
  addCol(*svTab, "eta", s_eta, "eta", 10);
  addCol(*svTab, "phi", s_phi, "phi", 10);
  addCol(*svTab, "ntracks", s_ntrk, "number of tracks");
  addCol(*svTab, "chi2", s_chi2, "vertex chi2", 10);
  addCol(*svTab, "ndof", s_ndof, "vertex ndof", 8);
  addCol(*svTab, "dlen", s_dlen, "3D distance to candidate PV [cm]", 10);
  addCol(*svTab, "dlenSig", s_dlenSig, "3D distance significance to candidate PV", 10);
  addCol(*svTab, "dxy", s_dxy, "2D distance to candidate PV [cm]", 10);
  addCol(*svTab, "dxySig", s_dxySig, "2D distance significance to candidate PV", 10);
  addCol(*svTab, "cosPAngle", s_cosPA, "cosine of the pointing angle wrt candidate PV", 10);
  addCol(*svTab, "distToCandSV", s_dist, "3D distance to the candidate SV [cm]", 10);
  addCol(*svTab, "distToCandSVSig", s_distSig, "3D distance significance to the candidate SV", 10);
  addCol(*svTab, "sharesCandTrack", s_shares, "contains one of the candidate legs");
  evt.put(std::move(svTab), "sv");
}

DEFINE_FWK_MODULE(CandidateContextProducer);
