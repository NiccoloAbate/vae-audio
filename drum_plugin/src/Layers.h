#pragma once

// Pad recipes -> render jobs -> velocity-layer sets (shared by the plugin and
// kit_render_test). A pad renders kNumLayers velocity layers x up to kMaxRR
// variations (variation seeds, picked at random or in turn); velocity moves the chosen target from (pad value - depth * range)
// at velocity 0 up to the pad value at full velocity.

#include <juce_audio_basics/juce_audio_basics.h>
#include "KitModel.h"

#include <array>
#include <memory>

namespace ldm
{
constexpr int kNumPads = 8;
constexpr int kMaxControls = 6;
constexpr int kNumLayers = 16;
constexpr int kMaxRR = 4;
constexpr int kMaxSlots = 512;
constexpr int kUserSlotBase = 400;      // slots 400+ are hits encoded from dropped samples
constexpr int kMaxUserHits = kMaxSlots - kUserSlotBase;

// Velocity targets: gain only, morph, the kit's first two controls (Tone, Noise),
// decay, then its other controls (the unlabelled PCs). The editor shows the kit's
// control names. Indices 0-4 are unchanged from the first version (saved sessions).
enum VelTarget { VelGainOnly = 0, VelMorph = 1, VelDecay = 4 };
inline int velTargetForControl (int k) { return k < 2 ? 2 + k : 3 + k; }
inline int controlForVelTarget (int t) { return (t == 2 || t == 3) ? t - 2 : (t >= 5 ? t - 3 : -1); }
inline juce::StringArray velTargetNames()
{
    juce::StringArray s { "Gain only", "Morph A-B", "Control 1", "Control 2", "Decay" };
    for (int k = 2; k < 6; ++k)
        s.add ("Control " + juce::String (k + 1));
    return s;
}

struct PadRecipe
{
    int slotA = 0, slotB = 0, mode = 0;
    UserHitPtr userA, userB;            // resolved user slots (null: bank slot)
    float morph = 0.f, width = 4.f;
    std::array<float, kMaxControls> controls {};
    float variation = 0.f;
    int seed = 0, roundRobin = 1;
    float decay = 0.f, punch = 0.f;
    int velTarget = VelGainOnly;
    float velDepth = 0.f;
    int decodeSeed = 0;
    bool gate = false;
    // Stereo width: > 0 also renders a partner per hit (a fresh decode seed plus
    // its own random offset of size stereoVar); the set then carries side buffers.
    float stereo = 0.f, stereoVar = 0.3f;
};

// Rendered audio for one pad: buffers[layer * numRR + rr], mono, at sampleRate.
// With stereo width, sides[i] = highpass (partner - hit) / 2 for buffers[i], so a voice
// plays L = mid + w * side, R = mid - w * side (L + R = 2 * mid: mono-safe).
struct LayerSet
{
    int numLayers = 1, numRR = 1;
    double sampleRate = 44100.0;
    std::vector<juce::AudioBuffer<float>> buffers, sides;
    const juce::AudioBuffer<float>& get (int layer, int rr) const
    {
        return buffers[(size_t) (layer * numRR + rr)];
    }
    bool hasSides() const { return ! sides.empty(); }
    const juce::AudioBuffer<float>* getSide (int layer, int rr) const
    {
        return sides.empty() ? nullptr : &sides[(size_t) (layer * numRR + rr)];
    }
};
using LayerSetPtr = std::shared_ptr<const LayerSet>;

RenderJob makeJob (const PadRecipe& r, const KitModel& kit, int& numLayers, int& numRR);

// The stereo partners of a job's rows: same recipe, decode seed + 1, and each row's
// variation offset replaced by one of size stereoVar from seeds 32 above the pad's.
RenderJob makePartnerJob (const RenderJob& job, const PadRecipe& r, const KitModel& kit, int numRR);

// Normalise the set to a 0.9 peak (keeping levels between layers), trim trailing
// silence (-60 dB + 50 ms, 10 ms fade), resample from the kit rate to hostRate.
// partners (optional, one per row): side = highpass 150 Hz (partner - row) / 2,
// scaled, trimmed and resampled with its row.
LayerSetPtr buildLayerSet (std::vector<std::vector<float>> rows, int numLayers, int numRR,
                           double kitRate, double hostRate,
                           std::vector<std::vector<float>> partners = {});

LayerSetPtr renderPad (const PadRecipe& r, KitModel& kit, double hostRate);
} // namespace ldm
