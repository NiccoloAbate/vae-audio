#pragma once

// Reads a dropped sample and prepares it exactly as the training data was
// (dataset/datasets.py OneShotAudioDataset): mono, resampled to the kit rate, the
// onset (first sample above onsetThresh x peak) moved to prePad samples, cut or
// zero-padded to the model window, peak-normalised to 0.95.

#include <juce_audio_formats/juce_audio_formats.h>
#include "KitModel.h"

namespace ldm
{
bool isAudioFile (const juce::File& f);
// Empty on failure (error says why).
std::vector<float> prepareHit (const juce::File& f, const KitModel& kit, juce::String& error);
// Load + prepare + encode. Throws on failure.
std::shared_ptr<UserHit> encodeFile (const juce::File& f, KitModel& kit);
} // namespace ldm
