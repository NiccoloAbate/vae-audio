#include "HitLoader.h"

namespace ldm
{
bool isAudioFile (const juce::File& f)
{
    return f.hasFileExtension ("wav;aif;aiff;flac;mp3;ogg;m4a;caf");
}

std::vector<float> prepareHit (const juce::File& f, const KitModel& kit, juce::String& error)
{
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (f));
    if (reader == nullptr)
    {
        error = "can't read " + f.getFileName();
        return {};
    }
    const double fileRate = reader->sampleRate;
    const int n = (int) juce::jmin<juce::int64> (reader->lengthInSamples, (juce::int64) (20.0 * fileRate));
    if (n <= 0)
    {
        error = f.getFileName() + " is empty";
        return {};
    }
    juce::AudioBuffer<float> in ((int) reader->numChannels, n);
    reader->read (&in, 0, n, 0, true, true);

    // mono (channel mean)
    std::vector<float> mono ((size_t) n, 0.f);
    for (int c = 0; c < in.getNumChannels(); ++c)
        for (int i = 0; i < n; ++i)
            mono[(size_t) i] += in.getSample (c, i) / (float) in.getNumChannels();

    // resample to the kit rate
    const double sr = (double) kit.sampleRate();
    std::vector<float> y;
    if (std::abs (fileRate - sr) < 1e-6)
        y = std::move (mono);
    else
    {
        // the sinc interpolator delays its output by getBaseLatency() input samples:
        // render that much extra and drop it, so the timing matches the file
        const double ratio = fileRate / sr;
        const int outLen = juce::jmax (1, (int) std::floor (n / ratio));
        const int skip = juce::roundToInt (juce::WindowedSincInterpolator::getBaseLatency() / ratio);
        mono.resize ((size_t) n + 512, 0.f);
        y.assign ((size_t) (outLen + skip), 0.f);
        juce::WindowedSincInterpolator interp;
        interp.process (ratio, mono.data(), y.data(), outLen + skip);
        y.erase (y.begin(), y.begin() + skip);
    }

    // onset alignment
    float peak = 0.f;
    for (float v : y)
        peak = juce::jmax (peak, std::abs (v));
    size_t start = 0;
    if (kit.onsetThresh > 0.f && peak > 0.f)
    {
        size_t onset = 0;
        while (onset < y.size() && std::abs (y[onset]) <= kit.onsetThresh * peak)
            ++onset;
        start = onset > (size_t) kit.prePad ? onset - (size_t) kit.prePad : 0;
    }

    std::vector<float> out ((size_t) kit.numSamples(), 0.f);
    for (size_t i = 0; i < out.size() && start + i < y.size(); ++i)
        out[i] = y[start + i];
    if (kit.peakNorm)
    {
        float m = 0.f;
        for (float v : out)
            m = juce::jmax (m, std::abs (v));
        if (m > 1e-6f)
            for (auto& v : out)
                v = v / m * 0.95f;
    }
    return out;
}

std::shared_ptr<UserHit> encodeFile (const juce::File& f, KitModel& kit)
{
    juce::String err;
    auto x = prepareHit (f, kit, err);
    if (x.empty())
        throw std::runtime_error (err.toStdString());
    auto h = kit.encode (x, f.getFileNameWithoutExtension());
    h->file = f;
    return h;
}
} // namespace ldm
