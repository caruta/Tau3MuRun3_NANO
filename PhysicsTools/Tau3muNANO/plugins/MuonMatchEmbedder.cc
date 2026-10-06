
#include <algorithm>
#include <limits>

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "helper.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace {
  int getGenOrigin(const reco::GenParticleRef& gen) {
    if (gen.isNull()) return 0; // Nessun match a livello Gen

    const reco::Candidate* mother = gen->mother();
    bool foundPhi = false;
    bool foundDs = false;
    bool foundB = false;

    while (mother != nullptr) {
        int pdg = std::abs(mother->pdgId());
        
        // 1. Cerchiamo la Phi (333)
        if (pdg == 333) {
            foundPhi = true;
        }

        // 2. Cerchiamo la Ds (431)
        if (pdg == 431) {
            foundDs = true;
        }

        // 3. Cerchiamo un adrone B (5xx)
        if ((pdg / 100) == 5 || (pdg / 1000) == 5) {
            foundB = true;
            // Se troviamo un B, non serve risalire oltre nella gerarchia
            break; 
        }

        mother = mother->mother();
    }

    // --- Logica di classificazione finale ---
    
    // Se NON è passato per una Phi, è automaticamente "Other" (3)
    if (!foundPhi) return 3;

    // Se è passato per una Phi, controlliamo la provenienza della Phi:
    if (foundB) return 2;  // Non-Prompt (B -> ... -> Phi -> mu)
    if (foundDs) return 1; // Prompt (Ds -> Phi -> mu)

    // Phi prodotta in altri modi (es. frammentazione o decadimenti di adroni leggeri)
    return 3; 
  }
}

namespace {
  // Bits of the genFlags userInt (ancestry of the matched gen muon, or the decay-in-flight hadron match)
  enum GenFlag {
    kFromTau = 1,
    kFromDs = 2,
    kFromDplus = 4,
    kFromB = 8,
    kFromW = 16,
    kFromPhi = 32,
    kFromLightResonance = 64,  // eta, eta', rho, omega
    kDecayInFlight = 128       // no gen muon matched, but a charged pion/kaon is
  };

  bool isBHadron(int apdg) { return (apdg / 100) % 10 == 5 || (apdg / 1000) % 10 == 5; }

  struct GenAncestry {
    int flags = 0;
    int motherPdgId = 0;
    int tauIdx = -1;
  };

  // Walk the mother chain of a matched gen muon. tauIdx is the index of the tau ancestor in the
  // gen collection of the match (the one written as GenPart), so legs from the same tau share it.
  GenAncestry getAncestry(const reco::GenParticleRef& gen) {
    GenAncestry a;
    if (gen.isNull())
      return a;
    // Ref::product() is not public; RefProd gives the collection the Ref points into
    const reco::GenParticleCollection* coll = edm::RefProd<reco::GenParticleCollection>(gen).product();
    const reco::Candidate* mother = gen->mother();
    while (mother != nullptr) {
      const int apdg = std::abs(mother->pdgId());
      if (a.motherPdgId == 0 && apdg != 13)
        a.motherPdgId = mother->pdgId();
      if (apdg == 15) {
        a.flags |= kFromTau;
        if (a.tauIdx < 0 && coll != nullptr && !coll->empty()) {
          const auto* gp = dynamic_cast<const reco::GenParticle*>(mother);
          if (gp != nullptr && gp >= &coll->front() && gp <= &coll->back())
            a.tauIdx = gp - &coll->front();
        }
      }
      if (apdg == 431)
        a.flags |= kFromDs;
      if (apdg == 411)
        a.flags |= kFromDplus;
      if (apdg == 24)
        a.flags |= kFromW;
      if (apdg == 333)
        a.flags |= kFromPhi;
      if (apdg == 221 || apdg == 331 || apdg == 113 || apdg == 223)
        a.flags |= kFromLightResonance;
      if (isBHadron(apdg))
        a.flags |= kFromB;
      mother = mother->mother();
    }
    return a;
  }
}  // namespace

template <typename PATOBJ>
class MatchEmbedder : public edm::global::EDProducer<> {
  // perhaps we need better structure here (begin run etc)

public:
  explicit MatchEmbedder(const edm::ParameterSet &cfg)
      : src_{consumes<PATOBJCollection>(cfg.getParameter<edm::InputTag>("src"))},
        matching_{
            consumes<edm::Association<reco::GenParticleCollection> >(cfg.getParameter<edm::InputTag>("matching"))} {
    // Optional: packed gen particles, to flag decays in flight (reco muon matched to a gen pi/K)
    if (cfg.existsAs<edm::InputTag>("packedGen")) {
      packedGen_ = consumes<std::vector<pat::PackedGenParticle> >(cfg.getParameter<edm::InputTag>("packedGen"));
      hasPackedGen_ = true;
    }
    produces<PATOBJCollection>();
  }

  ~MatchEmbedder() override {}

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  typedef std::vector<PATOBJ> PATOBJCollection;
  const edm::EDGetTokenT<PATOBJCollection> src_;
  const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > matching_;
  edm::EDGetTokenT<std::vector<pat::PackedGenParticle> > packedGen_;
  bool hasPackedGen_ = false;
};

template <typename PATOBJ>
void MatchEmbedder<PATOBJ>::produce(edm::StreamID, edm::Event &evt, edm::EventSetup const &iSetup) const {
  // input
  edm::Handle<PATOBJCollection> src;
  evt.getByToken(src_, src);

  edm::Handle<edm::Association<reco::GenParticleCollection> > matching;
  evt.getByToken(matching_, matching);

  edm::Handle<std::vector<pat::PackedGenParticle> > packedGen;
  if (hasPackedGen_)
    evt.getByToken(packedGen_, packedGen);

  size_t nsrc = src->size();
  // output
  std::unique_ptr<PATOBJCollection> out(new PATOBJCollection());
  out->reserve(nsrc);

  for (unsigned int i = 0; i < nsrc; ++i) {
    edm::Ptr<PATOBJ> ptr(src, i);
    reco::GenParticleRef match = (*matching)[ptr];
    out->emplace_back(src->at(i));
    out->back().addUserInt("mcMatch", match.isNonnull() ? match->pdgId() : 0);
    out->back().addUserInt("genOrigin", getGenOrigin(match));

    GenAncestry anc = getAncestry(match);
    int hadronPdgId = 0;
    if (match.isNull() && hasPackedGen_) {
      // decay in flight / punch-through: closest charged pion or kaon with compatible pT
      const auto& recoObj = src->at(i);
      float bestDR = 0.05;
      for (const auto& g : *packedGen) {
        const int apdg = std::abs(g.pdgId());
        if ((apdg != 211 && apdg != 321) || g.charge() != recoObj.charge())
          continue;
        if (std::abs(g.pt() - recoObj.pt()) > 0.3 * g.pt())
          continue;
        const float dr = reco::deltaR(g.eta(), g.phi(), recoObj.eta(), recoObj.phi());
        if (dr < bestDR) {
          bestDR = dr;
          hadronPdgId = g.pdgId();
        }
      }
      if (hadronPdgId != 0)
        anc.flags |= kDecayInFlight;
    }
    out->back().addUserInt("genFlags", anc.flags);
    out->back().addUserInt("genMotherPdgId", anc.motherPdgId);
    out->back().addUserInt("genTauIdx", anc.tauIdx);
    out->back().addUserInt("genHadronPdgId", hadronPdgId);
  }

  // adding label to be consistent with the muon and track naming
  evt.put(std::move(out));
}

#include "DataFormats/PatCandidates/interface/Muon.h"
typedef MatchEmbedder<pat::Muon> MuonMatchEmbedder;

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef MatchEmbedder<pat::Electron> ElectronMatchEmbedder;

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
typedef MatchEmbedder<pat::CompositeCandidate> CompositeCandidateMatchEmbedder;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonMatchEmbedder);
DEFINE_FWK_MODULE(ElectronMatchEmbedder);
DEFINE_FWK_MODULE(CompositeCandidateMatchEmbedder);