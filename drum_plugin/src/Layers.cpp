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

RenderJob makePartnerJob (const RenderJob& job, const PadRecipe& r, const KitModel& kit, int numRR)
{
    const int S = juce::jmax (1, kit.numSeeds());
    RenderJob p = job;
    p.decodeSeed = (job.decodeSeed + 1) % S;
    numRR = juce::jmax (1, numRR);
    for (size_t i = 0; i < p.rows.size(); ++i)
    {
        p.rows[i].varAmount = r.stereoVar;
        p.rows[i].varSeed = (r.seed + (int) i % numRR + 32) % S;     // rows: layer * numRR + rr
    }
    return p;
}

// Zero-phase 2nd-order Butterworth high-pass (RBJ biquad, forward then backward).
static void highpassZeroPhase (std::vector<float>& x, double fc, double fs)
{
    const double w = juce::MathConstants<double>::twoPi * fc / fs, cw = std::cos (w);
    const double alpha = std::sin (w) / juce::MathConstants<double>::sqrt2;   // sin(w) / (2Q), Q = 1/sqrt2
    const double a0 = 1.0 + alpha;
    const double b0 = (1.0 + cw) / 2.0 / a0, b1 = -(1.0 + cw) / a0, b2 = b0;
    const double a1 = -2.0 * cw / a0, a2 = (1.0 - alpha) / a0;
    auto pass = [&] (auto begin, auto end)
    {
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (auto it = begin; it != end; ++it)
        {
            const double x0 = *it;
            const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1; x1 = x0; y2 = y1; y1 = y0;
            *it = (float) y0;
        }
    };
    pass (x.begin(), x.end());
    pass (x.rbegin(), x.rend());
}

// side -= g * mid per 50%-overlap periodic-Hann frame (the windows sum to 1), g the
// frame's least-squares gain of side on mid. (partner - hit) carries minus the hit's own
// texture, so it anti-correlates with the hit and R = mid - side came out louder (up to
// ~9 dB on noisy hits at width 2). With the correlated part removed, L and R have equal
// energy in every frame, and L + R is still 2 * mid.
static void orthogonalise (std::vector<float>& side, const std::vector<float>& mid, int n = 1024)
{
    const int hop = n / 2, N = (int) side.size();
    std::vector<float> w ((size_t) n), out ((size_t) N, 0.f);
    for (int i = 0; i < n; ++i)
        w[(size_t) i] = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) n);
    for (int start = -hop; start < N; start += hop)
    {
        double sm = 0.0, mm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const int j = start + i;
            if (j < 0 || j >= N)
                continue;
            const double a = side[(size_t) j] * w[(size_t) i], b = mid[(size_t) j] * w[(size_t) i];
            sm += a * b;
            mm += b * b;
        }
        const float g = mm > 1e-12 ? (float) (sm / mm) : 0.f;
        for (int i = 0; i < n; ++i)
        {
            const int j = start + i;
            if (j >= 0 && j < N)
                out[(size_t) j] += w[(size_t) i] * (side[(size_t) j] - g * mid[(size_t) j]);
        }
    }
    side.swap (out);
}

LayerSetPtr buildLayerSet (std::vector<std::vector<float>> rows, int numLayers, int numRR,
                           double kitRate, double hostRate, std::vector<std::vector<float>> partners)
{
    const bool stereo = partners.size() == rows.size() && ! rows.empty();
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

    auto toHost = [&] (const std::vector<float>& r, int len)
    {
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
        return buf;
    };

    for (size_t i = 0; i < rows.size(); ++i)
    {
        auto& r = rows[i];
        std::vector<float> side;
        if (stereo)
        {
            const auto& b = partners[i];
            side.assign (r.size(), 0.f);
            for (size_t j = 0; j < r.size() && j < b.size(); ++j)
                side[j] = 0.5f * (b[j] - r[j]);
            highpassZeroPhase (side, 150.0, kitRate);
            auto midHp = r;
            highpassZeroPhase (midHp, 150.0, kitRate);           // keeps the side high-passed
            orthogonalise (side, midHp);
        }
        for (auto& x : r)
            x *= norm;
        int last = 0;
        for (int j = (int) r.size() - 1; j >= 0; --j)
            if (std::abs (r[(size_t) j]) > floorAbs) { last = j; break; }
        const int len = juce::jmin ((int) r.size(), last + (int) (0.05 * kitRate));
        const int fade = juce::jmin (len, (int) (0.01 * kitRate));
        for (int j = 0; j < fade; ++j)
            r[(size_t) (len - 1 - j)] *= (float) j / (float) fade;
        set->buffers.push_back (toHost (r, len));
        if (stereo)
        {
            for (auto& x : side)
                x *= norm;
            for (int j = 0; j < fade; ++j)
                side[(size_t) (len - 1 - j)] *= (float) j / (float) fade;
            set->sides.push_back (toHost (side, len));
        }
    }
    return set;
}

LayerSetPtr renderPad (const PadRecipe& r, KitModel& kit, double hostRate)
{
    int L = 1, R = 1;
    auto job = makeJob (r, kit, L, R);
    auto rows = kit.render (job);
    std::vector<std::vector<float>> partners;
    if (r.stereo > 0.f)
        partners = kit.render (makePartnerJob (job, r, kit, R));
    return buildLayerSet (std::move (rows), L, R, (double) kit.sampleRate(), hostRate, std::move (partners));
}
} // namespace ldm
