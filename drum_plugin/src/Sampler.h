#pragma once

// Plays rendered layer sets. Each hit is a voice reading two neighbouring velocity
// layers, crossfaded by the exact velocity, until the longer of the two ends.
// Choke groups fade other pads' voices in the group (5 ms); self-choke fades the
// pad's own ringing hits. Audio thread only.

#include "Layers.h"

struct PadPlay
{
    float gain = 1.f, pan = 0.f, velToGain = 0.7f;
    int chokeGroup = 0;
    bool selfChoke = false;
    bool randomOrder = true;        // variations: random (never the same twice in a row) or cycle
};

class Sampler
{
public:
    static constexpr int kVoices = 48;

    void prepare (double sampleRate);
    void noteOn (int pad, float velocity, const ldm::LayerSetPtr& set, const PadPlay& pp);
    void render (juce::AudioBuffer<float>& out, int start, int num);
    void allOff();

private:
    struct Voice
    {
        ldm::LayerSetPtr set;
        const float* lo = nullptr;
        const float* hi = nullptr;
        int lenLo = 0, lenHi = 0, len = 0, pos = 0;
        float frac = 0.f, gainL = 0.f, gainR = 0.f;
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
