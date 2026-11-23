#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

// One pad: click to select and play (higher on the pad = louder).
class PadButton : public juce::Component
{
public:
    std::function<void (int pad, float velocity)> onPress;
    std::function<void (int pad, bool solo)> onToggle;      // the M / S buttons
    bool muted = false, soloed = false, dimmed = false;
    int pad = 0;
    juce::String title, sub, note;
    juce::Colour colour;
    float flash = 0.f;
    bool selected = false;

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;

private:
    juce::Rectangle<float> toggleBox (bool solo) const;
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

class DrumMachineEditor : public juce::AudioProcessorEditor,
                          public juce::FileDragAndDropTarget,
                          private juce::Timer
{
public:
    explicit DrumMachineEditor (DrumMachineProcessor&);
    ~DrumMachineEditor() override;

    void paint (juce::Graphics&) override;
    void paintOverChildren (juce::Graphics&) override;

    // Drop samples: on a pad -> its slot A (two files: A and B); on Slot A / Slot B
    // -> that slot of the selected pad; elsewhere on the right panel -> slot A.
    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void fileDragMove (const juce::StringArray& files, int x, int y) override;
    void fileDragExit (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int x, int y) override;
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
    struct DropTarget { int pad = -1; bool slotB = false; juce::Rectangle<int> area; };
    DropTarget dropTargetAt (int x, int y) const;
    juce::Rectangle<int> dropHighlight;

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
    juce::Slider panSpread, tune, tuneSpread, stereo, stereoVar;
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
