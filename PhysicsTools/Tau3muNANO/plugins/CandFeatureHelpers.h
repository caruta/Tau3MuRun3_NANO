#ifndef PhysicsTools_Tau3muNANO_CandFeatureHelpers_h
#define PhysicsTools_Tau3muNANO_CandFeatureHelpers_h

// Candidate-level features shared by Tau3MuBuilder and Ds2Mu1TrkBuilder,
// added for the transformer ntuples (see the "Candidate-level features"
// section of the ntuple schema).

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TMath.h"

#include "KinVtxFitter.h"

namespace t3m {

  constexpr float kMissing = -999.f;

  // Six independent elements of a 3x3 symmetric position covariance.
  inline void addVertexCov(pat::CompositeCandidate& cand, const std::string& pre, const GlobalError& e) {
    cand.addUserFloat(pre + "_cxx", e.cxx());
    cand.addUserFloat(pre + "_cyx", e.cyx());
    cand.addUserFloat(pre + "_cyy", e.cyy());
    cand.addUserFloat(pre + "_czx", e.czx());
    cand.addUserFloat(pre + "_czy", e.czy());
    cand.addUserFloat(pre + "_czz", e.czz());
  }

  inline void addVertexCov(pat::CompositeCandidate& cand, const std::string& pre, const reco::Vertex& v) {
    cand.addUserFloat(pre + "_cxx", v.covariance(0, 0));
    cand.addUserFloat(pre + "_cyx", v.covariance(1, 0));
    cand.addUserFloat(pre + "_cyy", v.covariance(1, 1));
    cand.addUserFloat(pre + "_czx", v.covariance(2, 0));
    cand.addUserFloat(pre + "_czy", v.covariance(2, 1));
    cand.addUserFloat(pre + "_czz", v.covariance(2, 2));
  }

  // Probability of a 2-track Kalman vertex fit, -1 if the fit fails.
  inline float pairVtxProb(const reco::TransientTrack& a, const reco::TransientTrack& b) {
    if (!a.isValid() || !b.isValid())
      return -1.f;
    try {
      KalmanVertexFitter fitter(false);
      TransientVertex v = fitter.vertex(std::vector<reco::TransientTrack>{a, b});
      if (!v.isValid() || v.degreesOfFreedom() <= 0)
        return -1.f;
      return TMath::Prob(v.totalChiSquared(), v.degreesOfFreedom());
    } catch (...) {
      return -1.f;
    }
  }

  // Per-leg features: refitted direction and 3D impact parameters wrt the refitted PV and the SV.
  inline void addLegFeatures(pat::CompositeCandidate& cand,
                             const std::string& leg,
                             const reco::TransientTrack& tt,
                             const reco::Track& refitTrack,
                             const reco::Vertex& pv,
                             const reco::Vertex& sv) {
    cand.addUserFloat("refit_" + leg + "_eta", refitTrack.eta());
    cand.addUserFloat("refit_" + leg + "_phi", refitTrack.phi());

    auto ipPV = IPTools::absoluteImpactParameter3D(tt, pv);
    cand.addUserFloat(leg + "_ip3d_pv", ipPV.first ? ipPV.second.value() : kMissing);
    cand.addUserFloat(leg + "_ip3d_pv_sig", ipPV.first ? ipPV.second.significance() : kMissing);

    // Biased (the leg is in the SV fit), but still informative about fit tension.
    auto ipSV = IPTools::absoluteImpactParameter3D(tt, sv);
    cand.addUserFloat(leg + "_ip3d_sv", ipSV.first ? ipSV.second.value() : kMissing);
    cand.addUserFloat(leg + "_ip3d_sv_sig", ipSV.first ? ipSV.second.significance() : kMissing);
  }

  // Pair masses and 2-track vertex probabilities for the three leg pairs (12, 13, 23).
  inline void addPairFeatures(pat::CompositeCandidate& cand,
                              const std::array<reco::Candidate::LorentzVector, 3>& p4,
                              const std::vector<reco::TransientTrack>& tts) {
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    const char* names[3] = {"12", "13", "23"};
    for (int i = 0; i < 3; ++i) {
      const int a = pairs[i][0], b = pairs[i][1];
      cand.addUserFloat(std::string("m") + names[i], (p4[a] + p4[b]).M());
      cand.addUserFloat(std::string("vprob") + names[i], pairVtxProb(tts[a], tts[b]));
    }
  }

  // Kinematic vertex fit of the three legs: fitted mass, its per-candidate uncertainty, fit probability.
  inline void addKinFit(pat::CompositeCandidate& cand,
                        const std::vector<reco::TransientTrack>& tts,
                        const std::vector<double>& masses,
                        const std::vector<float>& sigmas) {
    float mass = kMissing, massErr = kMissing, prob = -1.f;
    int ok = 0;
    try {
      KinVtxFitter fit(tts, masses, sigmas);
      if (fit.success()) {
        mass = fit.fitted_candidate().mass();
        const double var = fit.fitted_candidate().kinematicParametersError().matrix()(6, 6);
        massErr = var > 0 ? std::sqrt(var) : kMissing;
        prob = fit.prob();
        ok = 1;
      }
    } catch (...) {
      ok = 0;
    }
    cand.addUserInt("kin_ok", ok);
    cand.addUserFloat("kin_mass", mass);
    cand.addUserFloat("kin_massErr", massErr);
    cand.addUserFloat("kin_prob", prob);
  }

}  // namespace t3m

#endif
