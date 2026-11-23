// kit_render_test <kit.pt> <out.wav> [slotA | sample.wav] [velTarget]
// Renders one pad the way the plugin does (16 velocity layers x 2 variations,
// velocity -> Control 1 by default), reports timing and writes the layers one
// after another (soft to loud) to a WAV file.

#include <juce_audio_formats/juce_audio_formats.h>
#include "Layers.h"
#include "HitLoader.h"

int main (int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: kit_render_test <kit.pt> <out.wav> [slotA] [velTarget 0-4]\n";
        return 1;
    }
    juce::String err;
    auto t0 = juce::Time::getMillisecondCounterHiRes();
    auto kit = KitModel::load (juce::File (argv[1]), err);
    if (! kit)
    {
        std::cerr << "load failed: " << err << "\n";
        return 1;
    }
    std::cout << "loaded " << kit->getName() << ": " << kit->entries().size() << " hits, "
              << kit->numControls() << " controls, " << kit->getDeviceName() << " ("
              << juce::roundToInt (juce::Time::getMillisecondCounterHiRes() - t0) << " ms)\n";

    ldm::PadRecipe r;
    r.slotA = argc > 3 ? juce::String (argv[3]).getIntValue() : 0;
    if (argc > 3 && ! juce::String (argv[3]).containsOnly ("0123456789"))     // a sample file: encode it
    {
        try
        {
            t0 = juce::Time::getMillisecondCounterHiRes();
            const auto in = juce::File::getCurrentWorkingDirectory().getChildFile (argv[3]);
            r.userA = ldm::encodeFile (in, *kit);
            // debugging aids: the prepared input and the latent mean, raw float32
            juce::String err;
            auto prep = ldm::prepareHit (in, *kit, err);
            juce::File (juce::String (argv[2]) + ".prep.f32").replaceWithData (prep.data(), prep.size() * sizeof (float));
            juce::File (juce::String (argv[2]) + ".mu.f32").replaceWithData (r.userA->mu.data(), r.userA->mu.size() * sizeof (float));
            std::cout << "encoded " << r.userA->name << " (peak frame " << r.userA->peak << ", end " << r.userA->end
                      << ") in " << juce::roundToInt (juce::Time::getMillisecondCounterHiRes() - t0) << " ms\n";
            r.slotA = ldm::kUserSlotBase;
        }
        catch (const std::exception& e)
        {
            std::cerr << "encode failed: " << e.what() << "\n";
            return 1;
        }
    }
    r.slotB = (r.slotA + 1) % (int) kit->entries().size();
    r.variation = 0.1f;
    r.roundRobin = 2;
    r.velTarget = argc > 4 ? juce::String (argv[4]).getIntValue() : ldm::velTargetForControl (0);
    r.velDepth = 0.5f;
    r.morph = r.velTarget == ldm::VelMorph ? 1.f : 0.f;

    for (int pass = 0; pass < 2; ++pass)           // the first call includes warm-up
    {
        t0 = juce::Time::getMillisecondCounterHiRes();
        auto set = ldm::renderPad (r, *kit, 48000.0);
        const double ms = juce::Time::getMillisecondCounterHiRes() - t0;
        std::cout << "pass " << pass << ": " << set->numLayers << " layers x " << set->numRR << " rr in "
                  << juce::roundToInt (ms) << " ms\n";
        if (pass == 1)
        {
            int total = 0;
            for (auto& b : set->buffers)
                total += b.getNumSamples() + 4800;
            juce::AudioBuffer<float> all (1, total);
            all.clear();
            int pos = 0;
            for (int l = 0; l < set->numLayers; ++l)
            {
                auto& b = set->get (l, 0);
                all.copyFrom (0, pos, b, 0, 0, b.getNumSamples());
                pos += b.getNumSamples() + 4800;
                std::cout << "  layer " << l << ": " << b.getNumSamples() << " samples, peak "
                          << juce::String (b.getMagnitude (0, b.getNumSamples()), 3) << "\n";
            }
            juce::File out (argv[2]);
            out.deleteFile();
            juce::WavAudioFormat wav;
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (
                new juce::FileOutputStream (out), 48000.0, 1, 24, {}, 0));
            if (w)
                w->writeFromAudioSampleBuffer (all, 0, pos);
            std::cout << "wrote " << out.getFullPathName() << "\n";
        }
    }

    // Stereo width: partners rendered, side level vs mid on the loudest layer, and
    // that layer as L = mid + side, R = mid - side (width 1) to <out>.stereo.wav
    r.stereo = 1.f;
    t0 = juce::Time::getMillisecondCounterHiRes();
    auto st = ldm::renderPad (r, *kit, 48000.0);
    std::cout << "stereo: " << st->numLayers << " layers x " << st->numRR << " rr + partners in "
              << juce::roundToInt (juce::Time::getMillisecondCounterHiRes() - t0) << " ms\n";
    if (! st->hasSides())
    {
        std::cerr << "stereo render has no side buffers\n";
        return 1;
    }
    const int top = st->numLayers - 1;
    for (int k = 0; k < st->numRR; ++k)
    {
        const auto& m = st->get (top, k);
        const auto& s = *st->getSide (top, k);
        const double rm = m.getRMSLevel (0, 0, m.getNumSamples()), rs = s.getRMSLevel (0, 0, s.getNumSamples());
        std::cout << "  rr " << k << ": side " << juce::String (juce::Decibels::gainToDecibels (rs / rm), 1)
                  << " dB vs mid, " << m.getNumSamples() << " / " << s.getNumSamples() << " samples\n";
    }
    {
        const auto& m = st->get (top, 0);
        const auto& s = *st->getSide (top, 0);
        juce::AudioBuffer<float> lr (2, m.getNumSamples());
        for (int i = 0; i < m.getNumSamples(); ++i)
        {
            lr.setSample (0, i, m.getSample (0, i) + s.getSample (0, i));
            lr.setSample (1, i, m.getSample (0, i) - s.getSample (0, i));
        }
        juce::File out (juce::String (argv[2]) + ".stereo.wav");
        out.deleteFile();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (
            new juce::FileOutputStream (out), 48000.0, 2, 24, {}, 0));
        if (w)
            w->writeFromAudioSampleBuffer (lr, 0, lr.getNumSamples());
        std::cout << "wrote " << out.getFullPathName() << "\n";
    }
    return 0;
}
