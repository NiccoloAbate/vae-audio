#pragma once

// Plays rendered layer sets. Each hit is a voice reading two neighbouring velocity
// layers, crossfaded by the exact velocity, until the longer of the two ends.
// Tune is a varispeed read (4-point Hermite); pan and tune can be spread at random
// per hit. With side buffers (stereo width), L = mid + w * side, R = mid - w * side.
// Choke groups fade other pads' voices in the group (5 ms); self-choke fades the
// pad's own ringing hits. Audio thread only.

#include "Layers.h"

struct PadPlay
{
    float gain = 1.f, pan = 0.f, velToGain = 0.7f;
    float panSpread = 0.f;          // pan + uniform(-1, 1) * panSpread per hit
    float tune = 0.f, tuneSpread = 0.f;     // semitones; + uniform(-1, 1) * tuneSpread per hit
    float stereo = 0.f;             // side gain (needs a set with sides)
    int chokeGroup = 0;
    bool selfChoke = false;
    int order = 0;                  // variations: 0 random, 1 cycle, 2 random but never twice in a row
};

class Sampler
{
public:
    static constexpr int kVoices = 48;

    void prepare (double sampleRate);
    void noteOn (int pad, float velocity, const ldm::LayerSetPtr& set, const PadPlay& pp);
    void render (juce::AudioBuffer<float>& out, int start, int num);
    void allOff();
    void fadePad (int pad);                                 // e.g. on mute

private:
    struct Voice
    {
        ldm::LayerSetPtr set;
        const float* lo = nullptr;
        const float* hi = nullptr;
        const float* sLo = nullptr;     // side buffers (null: mono hit)
        const float* sHi = nullptr;
        int lenLo = 0, lenHi = 0, len = 0;
        double pos = 0.0, rate = 1.0;
        float frac = 0.f, gainL = 0.f, gainR = 0.f, side = 0.f;
        int pad = -1, chokeGroup = 0, fade = -1;
        juce::uint32 order = 0;
        bool active = false;
    };

    void startFade (Voice& v);

    std::array<Voice, kVoices> voices;
    std::array<int, ldm::kNumPads> lastVariation {};
    juce::Random rng;
    juce::uint32 counter = 0;
    int fadeLen = 220;
};
