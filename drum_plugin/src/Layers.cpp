#include "Layers.h"

namespace ldm
{
RenderJob makeJob (const PadRecipe& r, const KitModel& kit, int& numLayers, int& numRR)
{
    const int K = juce::jmin (kit.numControls(), kMaxControls);
    const int velControl = controlForVelTarget (r.velTarget);
    const bool layered = r.velTarget != VelGainOnly && r.velDepth > 0.f && velControl < K;
    numLayers = layered ? kNumLayers : 1;
    numRR = r.variation > 0.f ? juce::jlimit (1, kMaxRR, r.roundRobin) : 1;

    RenderJob job;
    job.slotA = r.slotA;
    job.slotB = r.slotB;
    job.userA = r.userA;
    job.userB = r.userB;
    job.mode = juce::jlimit (0, juce::jmax (0, kit.modeNames().size() - 1), r.mode);
    job.width = r.width;
    job.decodeSeed = r.decodeSeed;
    job.gateDb = r.gate ? -50.f : -300.f;

    for (int l = 0; l < numLayers; ++l)
    {
        const float v = numLayers == 1 ? 1.f : (float) l / (float) (numLayers - 1);
        const float shift = r.velDepth * (1.f - v);           // 0 at full velocity
        for (int rr = 0; rr < numRR; ++rr)
        {
            RenderRow row;
            row.morph = r.morph;
            row.decay = r.decay;
            row.punch = r.punch;
            row.varAmount = r.variation;
            row.varSeed = (r.seed + rr) % juce::jmax (1, kit.numSeeds());
            row.controls.assign ((size_t) kit.numControls(), 0.f);
            for (int k = 0; k < K; ++k)
                row.controls[(size_t) k] = r.controls[(size_t) k];
            if (layered)
            {
                if (r.velTarget == VelMorph)
                    row.morph = r.morph * (1.f - shift);                     // soft -> towards A
                else if (velControl >= 0)
                    row.controls[(size_t) velControl] -= 3.f * shift;        // soft -> lower
                else if (r.velTarget == VelDecay)
                    row.decay -= 2.f * shift;                                // soft -> shorter
            }
            job.rows.push_back (std::move (row));
        }
    }
    return job;
}

LayerSetPtr buildLayerSet (std::vector<std::vector<float>> rows, int numLayers, int numRR,
                           double kitRate, double hostRate)
{
    auto set = std::make_shared<LayerSet>();
    set->numLayers = numLayers;
    set->numRR = numRR;
    set->sampleRate = hostRate;

    float peak = 0.f;
    for (auto& r : rows)
        for (float x : r)
            peak = juce::jmax (peak, std::abs (x));
    const float norm = peak > 1e-9f ? 0.9f / peak : 1.f;
    const float floorAbs = peak * norm * 1e-3f;               // -60 dB below the set's peak

    for (auto& r : rows)
    {
        for (auto& x : r)
            x *= norm;
        int last = 0;
        for (int i = (int) r.size() - 1; i >= 0; --i)
            if (std::abs (r[(size_t) i]) > floorAbs) { last = i; break; }
        const int len = juce::jmin ((int) r.size(), last + (int) (0.05 * kitRate));
        const int fade = juce::jmin (len, (int) (0.01 * kitRate));
        for (int i = 0; i < fade; ++i)
            r[(size_t) (len - 1 - i)] *= (float) i / (float) fade;

        juce::AudioBuffer<float> buf;
        if (std::abs (hostRate - kitRate) < 1e-6)
        {
            buf.setSize (1, juce::jmax (1, len));
            buf.copyFrom (0, 0, r.data(), len);
        }
        else
        {
            const double ratio = kitRate / hostRate;           // input samples per output sample
            const int outLen = juce::jmax (1, (int) std::ceil (len / ratio));
            std::vector<float> in ((size_t) len + 16, 0.f);
            std::copy (r.begin(), r.begin() + len, in.begin());
            buf.setSize (1, outLen);
            juce::LagrangeInterpolator interp;
            interp.process (ratio, in.data(), buf.getWritePointer (0), outLen);
        }
        set->buffers.push_back (std::move (buf));
    }
    return set;
}

LayerSetPtr renderPad (const PadRecipe& r, KitModel& kit, double hostRate)
{
    int L = 1, R = 1;
    auto job = makeJob (r, kit, L, R);
    auto rows = kit.render (job);
    return buildLayerSet (std::move (rows), L, R, (double) kit.sampleRate(), hostRate);
}
} // namespace ldm
