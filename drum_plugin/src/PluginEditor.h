#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

// One pad: click to select and play (higher on the pad = louder).
class PadButton : public juce::Component
{
public:
    std::function<void (int pad, float velocity)> onPress;
    int pad = 0;
    juce::String title, sub, note;
    juce::Colour colour;
    float flash = 0.f;
    bool selected = false;

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;
};

// The selected pad's rendered hit: loudest layer, softest layer faint behind it.
class WaveView : public juce::Component
{
public:
    void setSet (ldm::LayerSetPtr s) { set = std::move (s); repaint(); }
    const ldm::LayerSet* current() const { return set.get(); }
    void paint (juce::Graphics&) override;

private:
    ldm::LayerSetPtr set;
};

class DrumMachineEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit DrumMachineEditor (DrumMachineProcessor&);
    ~DrumMachineEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void kitChanged();                       // called on the message thread

private:
    using SA = juce::AudioProcessorValueTreeState::SliderAttachment;
    using CA = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    using BA = juce::AudioProcessorValueTreeState::ButtonAttachment;

    void timerCallback() override;
    void selectPad (int pad);
    void fillSlotCombos();
    void syncSlotCombos();
    juce::String slotName (int index) const;
    juce::Label& labelFor (juce::Component& c, const juce::String& text);

    juce::LookAndFeel_V4 laf;                // declared first: outlives every child
    DrumMachineProcessor& proc;
    int selected = 0;
    std::shared_ptr<KitModel> kit;

    juce::Label title, kitLabel, statusLabel, padTitle;
    juce::TextButton loadButton { "Load kit..." };
    std::unique_ptr<juce::FileChooser> chooser;
    std::array<PadButton, ldm::kNumPads> pads;
    WaveView wave;

    juce::ComboBox slotA, slotB, mode, velTarget, choke, varOrder;
    juce::Slider morph, width, decay, punch, variation, seed, rr, velDepth, velGain, gain, pan, note;
    std::array<juce::Slider, ldm::kMaxControls> controls;
    juce::ToggleButton selfChoke { "Self-choke" };
    juce::Slider master, decodeSeed;
    juce::ToggleButton gate { "Noise gate (-50 dB)" };

    std::vector<std::unique_ptr<juce::Label>> labels;
    std::vector<std::unique_ptr<SA>> sliderAtt;
    std::vector<std::unique_ptr<CA>> comboAtt;
    std::vector<std::unique_ptr<BA>> buttonAtt;
    std::vector<std::unique_ptr<SA>> globalSliderAtt;
    std::unique_ptr<BA> gateAtt;
    std::array<juce::Label*, ldm::kMaxControls> controlLabels {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumMachineEditor)
};
