#pragma once

// Background thread that owns the kit and renders pads.
//  - markDirty(pad) is lock-free (safe from the audio thread / parameter callbacks);
//    pads re-render after kDebounceMs without changes (immediately when offline).
//  - Finished layer sets are swapped in atomically. Voices keep the set they started
//    with; replaced sets wait in a graveyard and are freed here once no voice holds
//    them, so the audio thread never frees memory.
//  - Offline bounces call waitUntilIdle() so every note plays the current settings.

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include "Layers.h"

#include <atomic>
#include <functional>

class RenderWorker : private juce::Thread
{
public:
    static constexpr int kDebounceMs = 40;

    RenderWorker();
    ~RenderWorker() override;

    // Reads a pad's current settings (called on the worker thread).
    std::function<ldm::PadRecipe (int pad)> getRecipe;
    // Called on the message thread after a kit loads; assignDefaults is false
    // when the host restored a session after the request.
    std::function<void (std::shared_ptr<KitModel>, bool assignDefaults)> onKitLoaded;

    void start();
    void requestKit (const juce::File& kitPt, bool assignDefaults);
    void cancelDefaultAssignment()          { assignDefaultsPending = false; }
    void markDirty (int pad);
    void markAllDirty();
    void setHostRate (double sr);
    void setOffline (bool b)                { offline = b; }

    std::shared_ptr<KitModel> getKit() const;
    ldm::LayerSetPtr getSet (int pad) const;
    bool isBusy() const;
    bool waitUntilIdle (int timeoutMs) const;

    juce::String getStatus() const;
    juce::File getKitFile() const;

private:
    void run() override;
    void loadPending();
    void renderOne (int pad, KitModel& kit);
    void setStatus (const juce::String& s);

    std::shared_ptr<KitModel> kit;                 // std::atomic_load / atomic_store
    std::array<ldm::LayerSetPtr, ldm::kNumPads> sets;
    std::vector<ldm::LayerSetPtr> graveyard;       // worker thread only

    std::atomic<uint32_t> dirty { 0 };
    std::array<std::atomic<juce::uint32>, ldm::kNumPads> changedAt;
    std::atomic<bool> rendering { false }, loadRequested { false }, assignDefaultsPending { false };
    std::atomic<bool> offline { false };
    std::atomic<double> hostRate { 44100.0 };

    mutable juce::CriticalSection lock;            // pendingKit, kitFile, status
    juce::File pendingKit, kitFile;
    juce::String status { "no kit" };
};
