#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "RenderWorker.h"
#include "Sampler.h"

namespace ldm
{
struct PadDefault
{
    const char* name;
    const char* type;
    int note, choke;
    juce::StringArray prefer;               // name keywords for the default bank slot
};
const std::array<PadDefault, kNumPads>& padDefaults();
juce::String pid (int pad, const char* param);
} // namespace ldm

class DrumMachineProcessor : public juce::AudioProcessor,
                             private juce::AudioProcessorValueTreeState::Listener
{
public:
    DrumMachineProcessor();
    ~DrumMachineProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using juce::AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Latent Drum Machine"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 1.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    // UI
    void loadKit (const juce::File& kitPt) { worker.requestKit (kitPt, true); }
    void audition (int pad, float velocity);
    float getFlash (int pad) const { return flash[(size_t) pad].load(); }
    void toggle (int pad, const char* param);               // mute / solo buttons
    void decayFlash();
    void setSlot (int pad, bool b, int index);

    juce::AudioProcessorValueTreeState apvts;
    RenderWorker worker;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void parameterChanged (const juce::String& id, float) override;
    ldm::PadRecipe readRecipe (int pad) const;
    PadPlay readPlay (int pad) const;
    void trigger (int pad, float velocity);
    void assignDefaultSlots (const KitModel& kit);
    float raw (int pad, const char* p) const;

    Sampler sampler;
    std::array<std::atomic<float>, ldm::kNumPads> flash;

    // UI clicks -> audio thread
    juce::AbstractFifo uiFifo { 64 };
    std::array<std::pair<int, float>, 64> uiNotes;

    // Playback parameters, cached so the audio thread does no lookups
    struct PlayParams { std::atomic<float> *note, *gain, *pan, *velGain, *choke, *selfChoke, *varOrder, *mute, *solo; };
    std::array<PlayParams, ldm::kNumPads> play {};
    std::array<bool, ldm::kNumPads> audible;                // mute / solo, audio thread

    std::atomic<float>* masterGain = nullptr;
    std::atomic<float>* decodeSeed = nullptr;
    std::atomic<float>* gate = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumMachineProcessor)
};
