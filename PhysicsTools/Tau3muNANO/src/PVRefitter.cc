#include "PhysicsTools/Tau3muNANO/interface/PVRefitter.h"
#include <algorithm>
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::vector<reco::TransientTrack> PVRefitter::removeTracks(
    const std::vector<reco::TransientTrack>& pvTracks,
    const std::vector<reco::TransientTrack>& svTracks) const {
    
    std::vector<reco::TransientTrack> cleanTracks;
    
    for (const auto& pvTrack : pvTracks) {
        bool isShared = false;
        
        for (const auto& svTrack : svTracks) {
            // Matching criterion: DeltaR < 0.01 and identical charge
            double dr = reco::deltaR(pvTrack.track(), svTrack.track());
            if (dr < 1.e-2 && pvTrack.track().charge() == svTrack.track().charge()) {
                isShared = true;
                
                // Debug message for track removal
                LogDebug("PVRefitter") << "Track removed from PV refit: "
                          << "pt=" << pvTrack.track().pt() 
                          << ", eta=" << pvTrack.track().eta() 
                          << ", dr with SV track=" << dr;
                
                break;
            }
        }
        
        if (!isShared) {
            cleanTracks.push_back(pvTrack);
        }
    }
    return cleanTracks;
}

std::pair<reco::Vertex, int> PVRefitter::refit(const std::vector<reco::TransientTrack>& svTracks, 
                                               const reco::Vertex& primaryVertex,
                                               const std::vector<reco::TransientTrack>& pvTracks) const {
    
    // 1. Remove signal (SV) tracks from the PV track collection
    std::vector<reco::TransientTrack> cleanTracks = removeTracks(pvTracks, svTracks);
    
    // 2. Status -2: Not enough tracks remaining to perform a valid fit (< 2)
    if (cleanTracks.size() < 2) {
        return {reco::Vertex(), -2}; 
    }

    // 3. Attempt the vertex refit using the remaining tracks
    TransientVertex refittedTransientVtx = fitter_.vertex(cleanTracks);
    
    // 4. Status -3: The vertex fitter failed to converge
    if (!refittedTransientVtx.isValid()) {
        return {reco::Vertex(), -3}; 
    }

    // 5. Status 1: Refit successful
    return {reco::Vertex(refittedTransientVtx), 1};
}