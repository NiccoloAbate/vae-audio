#pragma once

// A drum kit exported by export_kit.py: kit.pt (TorchScript KitRenderer) + kit.json.
// render() runs on the render worker's thread only.

#include <juce_core/juce_core.h>
#include <torch/script.h>

#include <memory>
#include <mutex>
#include <vector>

struct KitEntry
{
    juce::String name, type;
};

// One rendered hit: the per-row inputs of KitRenderer.render().
struct RenderRow
{
    float morph = 0.f, varAmount = 0.f, decay = 0.f, punch = 0.f;
    int varSeed = 0;
    std::vector<float> controls;    // one per kit control
};

struct RenderJob
{
    int slotA = 0, slotB = 0, mode = 0;
    float width = 4.f;
    int decodeSeed = 0;
    float gateDb = -300.f;          // <= -200: off
    std::vector<RenderRow> rows;
};

class KitModel
{
public:
    static std::shared_ptr<KitModel> load (const juce::File& kitPt, juce::String& error);

    // One mono buffer per row, at sampleRate(), numSamples() long. Throws on failure.
    std::vector<std::vector<float>> render (const RenderJob& job);

    int sampleRate() const              { return sr; }
    int numSamples() const              { return samples; }
    int numSeeds() const                { return nSeeds; }
    int numControls() const             { return controls.size(); }
    const std::vector<KitEntry>& entries() const { return bank; }
    const juce::StringArray& controlNames() const { return controls; }
    const juce::StringArray& modeNames() const    { return modes; }
    const juce::String& getName() const       { return name; }
    const juce::String& getDeviceName() const { return deviceName; }
    const juce::File& getFile() const         { return file; }

private:
    KitModel() = default;

    torch::jit::script::Module module;
    torch::Device device { torch::kCPU };
    int sr = 44100, samples = 65536, nSeeds = 64;
    int chunk = 1;                  // rows per call: batching is slow on CPU, fast on MPS
    std::vector<KitEntry> bank;
    juce::StringArray controls, modes;
    juce::String name, deviceName;
    juce::File file;
    std::mutex mutex;
};
