#include "Sampler.h"

namespace
{
inline float at (const float* x, int n, int i)
{
    return i >= 0 && i < n ? x[i] : 0.f;
}

// 4-point, 3rd-order Hermite at i + t (t in [0, 1)); exact at t = 0.
inline float hermite (const float* x, int n, int i, float t)
{
    const float xm1 = at (x, n, i - 1), x0 = at (x, n, i), x1 = at (x, n, i + 1), x2 = at (x, n, i + 2);
    const float c1 = 0.5f * (x1 - xm1);
    const float c2 = xm1 - 2.5f * x0 + 2.f * x1 - 0.5f * x2;
    const float c3 = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
    return ((c3 * t + c2) * t + c1) * t + x0;
}
} // namespace

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
    if (n > 1 && pp.order == 0)
        rr = rng.nextInt (n);                               // any variation, repeats allowed
    else if (n > 1 && pp.order == 2)
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
    auto spread = [this] (float amount) { return amount > 0.f ? amount * (2.f * rng.nextFloat() - 1.f) : 0.f; };
    const float pan = juce::jlimit (-1.f, 1.f, pp.pan + spread (pp.panSpread));
    const float angle = (pan + 1.f) * juce::MathConstants<float>::pi * 0.25f;
    const float semis = pp.tune + spread (pp.tuneSpread);
    const auto* sLo = set->getSide (lo, rr);
    const auto* sHi = set->getSide (hi, rr);

    voice->set = set;                                       // keeps the buffers alive
    voice->lo = bLo.getReadPointer (0);
    voice->hi = bHi.getReadPointer (0);
    voice->lenLo = bLo.getNumSamples();
    voice->lenHi = bHi.getNumSamples();
    voice->sLo = sLo != nullptr && pp.stereo > 0.f ? sLo->getReadPointer (0) : nullptr;
    voice->sHi = sHi != nullptr && pp.stereo > 0.f ? sHi->getReadPointer (0) : nullptr;
    voice->side = pp.stereo;
    voice->len = juce::jmax (voice->lenLo, voice->lenHi);
    voice->frac = x - (float) lo;
    voice->pos = 0.0;
    voice->rate = std::pow (2.0, (double) semis / 12.0);
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
            const int k = (int) v.pos;
            const float t = (float) (v.pos - (double) k);
            const float a = hermite (v.lo, v.lenLo, k, t);
            const float b = hermite (v.hi, v.lenHi, k, t);
            float m = a + (b - a) * v.frac;
            float s = 0.f;
            if (v.sLo != nullptr)
            {
                const float sa = hermite (v.sLo, v.lenLo, k, t);
                const float sb = hermite (v.sHi, v.lenHi, k, t);
                s = (sa + (sb - sa) * v.frac) * v.side;
            }
            if (v.fade >= 0)
            {
                const float f = (float) v.fade / (float) fadeLen;
                m *= f;
                s *= f;
                if (--v.fade < 0) { v.active = false; break; }
            }
            if (R != nullptr)
            {
                L[i] += (m + s) * v.gainL;
                R[i] += (m - s) * v.gainR;
            }
            else
                L[i] += m * v.gainL;                                // mono out: the mid (L + R = 2 mid)
            v.pos += v.rate;
            if (v.pos >= (double) v.len) { v.active = false; break; }
        }
        // The set itself is released on the next noteOn that reuses the voice, so
        // a finished voice never drops the last reference here (the worker's
        // graveyard or the pad slot also holds it until the worker frees it).
    }
}
