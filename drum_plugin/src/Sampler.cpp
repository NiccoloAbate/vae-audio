#include "Sampler.h"

void Sampler::prepare (double sampleRate)
{
    fadeLen = juce::jmax (1, (int) (0.005 * sampleRate));
    allOff();
}

void Sampler::allOff()
{
    for (auto& v : voices)
    {
        v.active = false;
        v.set.reset();
    }
}

void Sampler::fadePad (int pad)
{
    for (auto& v : voices)
        if (v.active && v.pad == pad)
            startFade (v);
}

void Sampler::startFade (Voice& v)
{
    if (v.active && v.fade < 0)
        v.fade = fadeLen;
}

void Sampler::noteOn (int pad, float velocity, const ldm::LayerSetPtr& set, const PadPlay& pp)
{
    if (! set || set->buffers.empty())
        return;

    for (auto& v : voices)
    {
        if (! v.active)
            continue;
        if ((pp.selfChoke && v.pad == pad)
            || (pp.chokeGroup > 0 && v.pad != pad && v.chokeGroup == pp.chokeGroup))
            startFade (v);
    }

    Voice* voice = nullptr;
    for (auto& v : voices)
        if (! v.active) { voice = &v; break; }
    if (voice == nullptr)                                   // steal the oldest
    {
        voice = &voices[0];
        for (auto& v : voices)
            if (v.order < voice->order)
                voice = &v;
    }

    const float vel = juce::jlimit (0.f, 1.f, velocity);
    const int L = set->numLayers;
    const float x = vel * (float) (L - 1);
    const int lo = juce::jlimit (0, L - 1, (int) std::floor (x));
    const int hi = juce::jmin (lo + 1, L - 1);
    const int n = set->numRR;
    int& last = lastVariation[(size_t) pad];
    last = juce::jlimit (0, juce::jmax (0, n - 1), last);   // the set may have fewer variations now
    int rr = 0;
    if (n > 1 && pp.randomOrder)
    {
        rr = rng.nextInt (n - 1);                           // any variation but the last one
        if (rr >= last)
            ++rr;
    }
    else if (n > 1)
        rr = (last + 1) % n;
    last = rr;

    const auto& bLo = set->get (lo, rr);
    const auto& bHi = set->get (hi, rr);
    const float g = pp.gain * (1.f - pp.velToGain * (1.f - vel * vel));
    const float angle = (juce::jlimit (-1.f, 1.f, pp.pan) + 1.f) * juce::MathConstants<float>::pi * 0.25f;

    voice->set = set;                                       // keeps the buffers alive
    voice->lo = bLo.getReadPointer (0);
    voice->hi = bHi.getReadPointer (0);
    voice->lenLo = bLo.getNumSamples();
    voice->lenHi = bHi.getNumSamples();
    voice->len = juce::jmax (voice->lenLo, voice->lenHi);
    voice->frac = x - (float) lo;
    voice->pos = 0;
    voice->gainL = g * std::cos (angle) * juce::MathConstants<float>::sqrt2;
    voice->gainR = g * std::sin (angle) * juce::MathConstants<float>::sqrt2;
    voice->pad = pad;
    voice->chokeGroup = pp.chokeGroup;
    voice->fade = -1;
    voice->order = ++counter;
    voice->active = true;
}

void Sampler::render (juce::AudioBuffer<float>& out, int start, int num)
{
    if (num <= 0)
        return;
    float* L = out.getWritePointer (0, start);
    float* R = out.getNumChannels() > 1 ? out.getWritePointer (1, start) : nullptr;

    for (auto& v : voices)
    {
        if (! v.active)
            continue;
        for (int i = 0; i < num; ++i)
        {
            const float a = v.pos < v.lenLo ? v.lo[v.pos] : 0.f;
            const float b = v.pos < v.lenHi ? v.hi[v.pos] : 0.f;
            float s = a + (b - a) * v.frac;
            if (v.fade >= 0)
            {
                s *= (float) v.fade / (float) fadeLen;
                if (--v.fade < 0) { v.active = false; break; }
            }
            L[i] += s * v.gainL;
            if (R != nullptr)
                R[i] += s * v.gainR;
            if (++v.pos >= v.len) { v.active = false; break; }
        }
        // The set itself is released on the next noteOn that reuses the voice, so
        // a finished voice never drops the last reference here (the worker's
        // graveyard or the pad slot also holds it until the worker frees it).
    }
}
