// kit_render_test <kit.pt> <out.wav> [slotA] [velTarget]
// Renders one pad the way the plugin does (16 velocity layers x 2 variations,
// velocity -> Control 1 by default), reports timing and writes the layers one
// after another (soft to loud) to a WAV file.

#include <juce_audio_formats/juce_audio_formats.h>
#include "Layers.h"

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
    return 0;
}
