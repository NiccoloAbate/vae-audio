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
    float morph = 0.f, width = 4.f;
    std::array<float, kMaxControls> controls {};
    float variation = 0.f;
    int seed = 0, roundRobin = 1;
    float decay = 0.f, punch = 0.f;
    int velTarget = VelGainOnly;
    float velDepth = 0.f;
    int decodeSeed = 0;
    bool gate = false;
};

// Rendered audio for one pad: buffers[layer * numRR + rr], mono, at sampleRate.
struct LayerSet
{
    int numLayers = 1, numRR = 1;
    double sampleRate = 44100.0;
    std::vector<juce::AudioBuffer<float>> buffers;
    const juce::AudioBuffer<float>& get (int layer, int rr) const
    {
        return buffers[(size_t) (layer * numRR + rr)];
    }
};
using LayerSetPtr = std::shared_ptr<const LayerSet>;

RenderJob makeJob (const PadRecipe& r, const KitModel& kit, int& numLayers, int& numRR);

// Normalise the set to a 0.9 peak (keeping levels between layers), trim trailing
// silence (-60 dB + 50 ms, 10 ms fade), resample from the kit rate to hostRate.
LayerSetPtr buildLayerSet (std::vector<std::vector<float>> rows, int numLayers, int numRR,
                           double kitRate, double hostRate);

LayerSetPtr renderPad (const PadRecipe& r, KitModel& kit, double hostRate);
} // namespace ldm
