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

// A hit encoded from a dropped sample (KitModel::encode). Only valid for the model
// that encoded it (modelId = the kit's checkpoint).
struct UserHit
{
    juce::String name, modelId;
    juce::File file;
    int D = 0, T = 0;
    std::vector<float> mu, sigma;       // D * T each, row-major
    float peak = 0.f, end = 1.f;        // timing frames, for the re-timed morphs
};
using UserHitPtr = std::shared_ptr<const UserHit>;

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
    UserHitPtr userA, userB;            // set: use these instead of bank slots
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

    // Prepared hit (numSamples(), see preprocessing) -> UserHit. Throws on failure.
    std::shared_ptr<UserHit> encode (const std::vector<float>& prepared, const juce::String& name);
    bool canEncode() const              { return hasEncoder; }
    bool accepts (const UserHit& h) const { return h.D == latentDim && h.T == tLat; }
    const juce::String& getModelId() const { return modelId; }
    float onsetThresh = 0.02f;
    int prePad = 512;
    bool peakNorm = true;

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
    int latentDim = 32, tLat = 128;
    bool hasEncoder = false;        // kits exported before encode() / render_z() existed: bank only
    juce::String modelId;
    std::vector<KitEntry> bank;
    juce::StringArray controls, modes;
    juce::String name, deviceName;
    juce::File file;
    std::mutex mutex;
};
