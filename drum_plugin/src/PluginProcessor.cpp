#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "BuildConfig.h"

#include <map>

namespace ldm
{
const std::array<PadDefault, kNumPads>& padDefaults()
{
    static const std::array<PadDefault, kNumPads> d { {
        { "Kick",       "kick",   36, 0, {} },
        { "Snare",      "snare",  38, 0, {} },
        { "Clap",       "clap",   39, 0, {} },
        { "Closed Hat", "hat",    42, 1, { "closed", "chh", "close", "tight" } },
        { "Open Hat",   "hat",    46, 1, { "open", "ohh" } },
        { "Low Tom",    "tom",    45, 0, { "low", "floor" } },
        { "High Tom",   "tom",    48, 0, { "high", "hi", "rack" } },
        { "Crash",      "cymbal", 49, 0, { "crash" } },
    } };
    return d;
}

juce::String pid (int pad, const char* param)
{
    return "p" + juce::String (pad) + "_" + param;
}
} // namespace ldm

using namespace ldm;

// Parameters that change the rendered audio (anything else is applied at playback)
static const juce::StringArray& renderParams()
{
    static const juce::StringArray p { "slot_a", "slot_b", "morph", "mode", "width",
                                       "c0", "c1", "c2", "c3", "c4", "c5",
                                       "var", "seed", "rr", "decay", "punch",
                                       "vel_target", "vel_depth" };
    return p;
}

DrumMachineProcessor::DrumMachineProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "LDM", createLayout())
{
    for (auto& f : flash)
        f = 0.f;
    for (int p = 0; p < kNumPads; ++p)
        for (auto& name : renderParams())
            apvts.addParameterListener (pid (p, name.toRawUTF8()), this);
    apvts.addParameterListener ("decode_seed", this);
    apvts.addParameterListener ("gate", this);
    for (int p = 0; p < kNumPads; ++p)
        play[(size_t) p] = { apvts.getRawParameterValue (pid (p, "note")), apvts.getRawParameterValue (pid (p, "gain")),
                             apvts.getRawParameterValue (pid (p, "pan")), apvts.getRawParameterValue (pid (p, "vel_gain")),
                             apvts.getRawParameterValue (pid (p, "choke")), apvts.getRawParameterValue (pid (p, "self_choke")),
                             apvts.getRawParameterValue (pid (p, "var_order")),
                             apvts.getRawParameterValue (pid (p, "mute")), apvts.getRawParameterValue (pid (p, "solo")) };
    audible.fill (true);
    masterGain = apvts.getRawParameterValue ("master");
    decodeSeed = apvts.getRawParameterValue ("decode_seed");
    gate = apvts.getRawParameterValue ("gate");

    worker.getRecipe = [this] (int pad) { return readRecipe (pad); };
    worker.onHitEncoded = [this] (std::shared_ptr<UserHit> hit, RenderWorker::EncodeRequest r) { hitEncoded (hit, r); };
    worker.onKitLoaded = [this] (std::shared_ptr<KitModel> kit, bool assign)
    {
        if (assign && kit)
            assignDefaultSlots (*kit);
        if (kit)
            reencodeForKit (*kit);
        if (auto* ed = dynamic_cast<DrumMachineEditor*> (getActiveEditor()))
            ed->kitChanged();
    };
    worker.start();

    juce::File def (LDM_DEFAULT_KIT);
    if (def.existsAsFile())
        worker.requestKit (def, true);
}

DrumMachineProcessor::~DrumMachineProcessor()
{
    for (int p = 0; p < kNumPads; ++p)
        for (auto& name : renderParams())
            apvts.removeParameterListener (pid (p, name.toRawUTF8()), this);
    apvts.removeParameterListener ("decode_seed", this);
    apvts.removeParameterListener ("gate", this);
}

juce::AudioProcessorValueTreeState::ParameterLayout DrumMachineProcessor::createLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;
    auto noteName = [] (int v, int) { return MidiMessage::getMidiNoteName (v, true, true, 3); };
    const auto twoDp = AudioParameterFloatAttributes().withStringFromValueFunction ([] (float v, int) { return String (v, 2); });
    const StringArray modes { "Blend", "Blend (slerp)", "Shape-aligned", "Tail only", "Crossover in time" };
    const StringArray chokes { "None", "Group 1", "Group 2", "Group 3", "Group 4" };

    for (int p = 0; p < kNumPads; ++p)
    {
        const auto& d = padDefaults()[(size_t) p];
        const String n = String (d.name) + " ";
        auto addF = [&] (const char* id, const String& name, float lo, float hi, float def, float step = 0.f)
        {
            layout.add (std::make_unique<AudioParameterFloat> (ParameterID { pid (p, id), 1 }, n + name,
                                                               NormalisableRange<float> (lo, hi, step), def, twoDp));
        };
        auto addI = [&] (const char* id, const String& name, int lo, int hi, int def)
        {
            layout.add (std::make_unique<AudioParameterInt> (ParameterID { pid (p, id), 1 }, n + name, lo, hi, def));
        };
        auto addC = [&] (const char* id, const String& name, const StringArray& items, int def)
        {
            layout.add (std::make_unique<AudioParameterChoice> (ParameterID { pid (p, id), 1 }, n + name, items, def));
        };
        layout.add (std::make_unique<AudioParameterInt> (ParameterID { pid (p, "note"), 1 }, n + "Note", 0, 127, d.note,
                                                         AudioParameterIntAttributes().withStringFromValueFunction (noteName)));
        addI ("slot_a", "Slot A", 0, kMaxSlots - 1, 0);
        addI ("slot_b", "Slot B", 0, kMaxSlots - 1, 1);
        addF ("morph", "Morph", 0.f, 1.f, 0.f);
        addC ("mode", "Morph mode", modes, 0);
        addF ("width", "Crossfade width", 1.f, 16.f, 4.f, 1.f);
        for (int k = 0; k < kMaxControls; ++k)
            addF (("c" + std::to_string (k)).c_str(), "Control " + String (k + 1), -3.f, 3.f, 0.f);
        addF ("var", "Variation", 0.f, 2.f, 0.1f);
        addI ("seed", "Variation seed", 0, 63, (p * 7) % 64);
        addI ("rr", "Variations", 1, kMaxRR, 3);
        addC ("var_order", "Variation order", { "Random", "Cycle", "Random, no repeat" }, 0);
        addF ("decay", "Decay", -2.f, 2.f, 0.f);
        addF ("punch", "Punch", -0.75f, 4.f, 0.f, 0.25f);
        addC ("vel_target", "Velocity target", velTargetNames(), velTargetForControl (0));
        addF ("vel_depth", "Velocity depth", 0.f, 1.f, 0.3f);
        addF ("vel_gain", "Velocity to gain", 0.f, 1.f, 0.7f);
        addF ("gain", "Gain (dB)", -48.f, 12.f, 0.f);
        addF ("pan", "Pan", -1.f, 1.f, 0.f);
        addC ("choke", "Choke group", chokes, d.choke);
        layout.add (std::make_unique<AudioParameterBool> (ParameterID { pid (p, "self_choke"), 1 }, n + "Self-choke", false));
        layout.add (std::make_unique<AudioParameterBool> (ParameterID { pid (p, "mute"), 1 }, n + "Mute", false));
        layout.add (std::make_unique<AudioParameterBool> (ParameterID { pid (p, "solo"), 1 }, n + "Solo", false));
    }
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "master", 1 }, "Master (dB)",
                                                       NormalisableRange<float> (-48.f, 12.f), 0.f, twoDp));
    layout.add (std::make_unique<AudioParameterInt> (ParameterID { "decode_seed", 1 }, "Decode seed", 0, 63, 0));
    layout.add (std::make_unique<AudioParameterBool> (ParameterID { "gate", 1 }, "Noise gate", false));
    return layout;
}

float DrumMachineProcessor::raw (int pad, const char* p) const
{
    return apvts.getRawParameterValue (pid (pad, p))->load();
}

PadRecipe DrumMachineProcessor::readRecipe (int pad) const
{
    PadRecipe r;
    r.slotA = (int) raw (pad, "slot_a");
    r.slotB = (int) raw (pad, "slot_b");
    if (r.slotA >= kUserSlotBase)
        r.userA = getUserHit (r.slotA - kUserSlotBase);
    if (r.slotB >= kUserSlotBase)
        r.userB = getUserHit (r.slotB - kUserSlotBase);
    r.mode = (int) raw (pad, "mode");
    r.morph = raw (pad, "morph");
    r.width = raw (pad, "width");
    for (int k = 0; k < kMaxControls; ++k)
        r.controls[(size_t) k] = raw (pad, ("c" + std::to_string (k)).c_str());
    r.variation = raw (pad, "var");
    r.seed = (int) raw (pad, "seed");
    r.roundRobin = (int) raw (pad, "rr");
    r.decay = raw (pad, "decay");
    r.punch = raw (pad, "punch");
    r.velTarget = (int) raw (pad, "vel_target");
    r.velDepth = raw (pad, "vel_depth");
    r.decodeSeed = (int) decodeSeed->load();
    r.gate = gate->load() > 0.5f;
    return r;
}

PadPlay DrumMachineProcessor::readPlay (int pad) const
{
    const auto& q = play[(size_t) pad];
    PadPlay pp;
    pp.gain = juce::Decibels::decibelsToGain (q.gain->load(), -48.f);
    pp.pan = q.pan->load();
    pp.velToGain = q.velGain->load();
    pp.chokeGroup = (int) q.choke->load();
    pp.selfChoke = q.selfChoke->load() > 0.5f;
    pp.order = (int) q.varOrder->load();
    return pp;
}

void DrumMachineProcessor::parameterChanged (const juce::String& id, float)
{
    if (id == "decode_seed" || id == "gate")
    {
        worker.markAllDirty();
        return;
    }
    if (id.startsWithChar ('p'))
        worker.markDirty (id.substring (1).getIntValue());
}

bool DrumMachineProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::stereo() || out == juce::AudioChannelSet::mono();
}

void DrumMachineProcessor::prepareToPlay (double sampleRate, int)
{
    sampler.prepare (sampleRate);
    worker.setHostRate (sampleRate);
}

void DrumMachineProcessor::trigger (int pad, float velocity)
{
    auto set = worker.getSet (pad);
    if (! set)
        return;
    sampler.noteOn (pad, velocity, set, readPlay (pad));
    flash[(size_t) pad] = 1.f;
}

void DrumMachineProcessor::audition (int pad, float velocity)
{
    const auto scope = uiFifo.write (1);
    if (scope.blockSize1 > 0)
        uiNotes[(size_t) scope.startIndex1] = { pad, velocity };
    else if (scope.blockSize2 > 0)
        uiNotes[(size_t) scope.startIndex2] = { pad, velocity };
}

void DrumMachineProcessor::decayFlash()
{
    for (auto& f : flash)
        f = f.load() * 0.8f;
}

void DrumMachineProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();

    // Offline bounce: wait for pending renders so every note uses current settings
    worker.setOffline (isNonRealtime());
    if (isNonRealtime() && worker.isBusy())
        worker.waitUntilIdle (30000);

    // Mute / solo: muted pads (or pads left out of a solo) ignore MIDI, and their
    // ringing hits fade out when they become inaudible. Pad clicks always play.
    bool anySolo = false;
    for (auto& q : play)
        anySolo = anySolo || q.solo->load() > 0.5f;
    for (int p = 0; p < kNumPads; ++p)
    {
        const auto& q = play[(size_t) p];
        const bool a = q.mute->load() < 0.5f && (! anySolo || q.solo->load() > 0.5f);
        if (audible[(size_t) p] && ! a)
            sampler.fadePad (p);
        audible[(size_t) p] = a;
    }

    {
        const auto scope = uiFifo.read (uiFifo.getNumReady());
        for (int i = 0; i < scope.blockSize1; ++i)
            trigger (uiNotes[(size_t) (scope.startIndex1 + i)].first, uiNotes[(size_t) (scope.startIndex1 + i)].second);
        for (int i = 0; i < scope.blockSize2; ++i)
            trigger (uiNotes[(size_t) (scope.startIndex2 + i)].first, uiNotes[(size_t) (scope.startIndex2 + i)].second);
    }

    int pos = 0;
    for (const auto meta : midi)
    {
        const auto msg = meta.getMessage();
        const int t = juce::jlimit (0, buffer.getNumSamples(), meta.samplePosition);
        if (msg.isNoteOn())
        {
            sampler.render (buffer, pos, t - pos);
            pos = t;
            for (int p = 0; p < kNumPads; ++p)
                if (audible[(size_t) p] && (int) play[(size_t) p].note->load() == msg.getNoteNumber())
                    trigger (p, msg.getFloatVelocity());
        }
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            sampler.render (buffer, pos, t - pos);
            pos = t;
            sampler.allOff();
        }
    }
    sampler.render (buffer, pos, buffer.getNumSamples() - pos);
    buffer.applyGain (juce::Decibels::decibelsToGain (masterGain->load(), -48.f));
}

void DrumMachineProcessor::toggle (int pad, const char* param)
{
    if (auto* p = apvts.getParameter (pid (pad, param)))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->getValue() > 0.5f ? 0.f : 1.f);
        p->endChangeGesture();
    }
}

void DrumMachineProcessor::setSlot (int pad, bool b, int index)
{
    if (auto* p = apvts.getParameter (pid (pad, b ? "slot_b" : "slot_a")))
        p->setValueNotifyingHost (p->convertTo0to1 ((float) index));
}

void DrumMachineProcessor::assignDefaultSlots (const KitModel& kit)
{
    const auto& entries = kit.entries();
    std::map<juce::String, int> used;                  // spread pads of one type over the bank
    for (int p = 0; p < kNumPads; ++p)
    {
        const auto& d = padDefaults()[(size_t) p];
        std::vector<int> cand;
        for (int i = 0; i < (int) entries.size(); ++i)
            if (entries[(size_t) i].type == d.type)
                cand.push_back (i);
        if (cand.empty())
            for (int i = 0; i < (int) entries.size(); ++i)
                cand.push_back (i);
        int a = -1;
        for (int i : cand)
        {
            const auto nm = entries[(size_t) i].name.toLowerCase();
            for (auto& kw : d.prefer)
                if (nm.contains (kw)) { a = i; break; }
            if (a >= 0)
                break;
        }
        int& u = used[d.type];
        if (a < 0)
            a = cand[(size_t) (u % (int) cand.size())];
        ++u;
        int b = a;
        for (size_t j = 0; j < cand.size(); ++j)
            if (cand[j] == a) { b = cand[(j + 1) % cand.size()]; break; }
        setSlot (p, false, a);
        setSlot (p, true, b);
    }
}

// ── User hits (dropped samples) ──────────────────────────────────────────────

UserHitPtr DrumMachineProcessor::getUserHit (int index) const
{
    const juce::ScopedLock sl (userLock);
    return index >= 0 && index < (int) userHits.size() ? userHits[(size_t) index] : nullptr;
}

int DrumMachineProcessor::numUserHits() const
{
    const juce::ScopedLock sl (userLock);
    return (int) userHits.size();
}

void DrumMachineProcessor::importSamples (const juce::Array<juce::File>& files, int pad, bool slotB)
{
    // one file: that slot; two or more on slot A: the first two go to A and B
    for (int i = 0; i < files.size() && i < 2; ++i)
    {
        if (i == 1 && slotB)
            break;
        worker.requestEncode ({ files[i], pad, slotB || i == 1, -1 });
    }
}

void DrumMachineProcessor::hitEncoded (std::shared_ptr<UserHit> hit, const RenderWorker::EncodeRequest& r)
{
    int index = r.replaceIndex;
    {
        const juce::ScopedLock sl (userLock);
        if (index >= 0 && index < (int) userHits.size())
            userHits[(size_t) index] = hit;
        else
        {
            index = -1;                                     // the same file again: reuse its slot
            for (size_t i = 0; i < userHits.size(); ++i)
                if (userHits[i]->file == hit->file && userHits[i]->modelId == hit->modelId)
                    index = (int) i;
            if (index < 0)
            {
                if ((int) userHits.size() >= kMaxUserHits)
                    return;
                userHits.push_back (hit);
                index = (int) userHits.size() - 1;
            }
            else
                userHits[(size_t) index] = hit;
        }
    }
    if (r.pad >= 0)
        setSlot (r.pad, r.slotB, kUserSlotBase + index);
    worker.markAllDirty();                                  // any pad may use this slot
    if (onUserHitsChanged)
        onUserHitsChanged (r.pad);
}

void DrumMachineProcessor::reencodeForKit (const KitModel& kit)
{
    const juce::ScopedLock sl (userLock);
    for (size_t i = 0; i < userHits.size(); ++i)
        if (userHits[i]->modelId != kit.getModelId() && userHits[i]->file.existsAsFile())
            worker.requestEncode ({ userHits[i]->file, -1, false, (int) i });
}

static juce::String floatsToBase64 (const std::vector<float>& v)
{
    return juce::MemoryBlock (v.data(), v.size() * sizeof (float)).toBase64Encoding();
}

static std::vector<float> floatsFromBase64 (const juce::String& s)
{
    juce::MemoryBlock mb;
    mb.fromBase64Encoding (s);
    std::vector<float> v (mb.getSize() / sizeof (float));
    std::memcpy (v.data(), mb.getData(), v.size() * sizeof (float));
    return v;
}

void DrumMachineProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    auto state = apvts.copyState();
    state.setProperty ("kitPath", worker.getKitFile().getFullPathName(), nullptr);
    juce::ValueTree hits ("UserHits");
    {
        const juce::ScopedLock sl (userLock);
        for (auto& h : userHits)
        {
            juce::ValueTree t ("Hit");
            t.setProperty ("name", h->name, nullptr);
            t.setProperty ("file", h->file.getFullPathName(), nullptr);
            t.setProperty ("model", h->modelId, nullptr);
            t.setProperty ("D", h->D, nullptr);
            t.setProperty ("T", h->T, nullptr);
            t.setProperty ("peak", h->peak, nullptr);
            t.setProperty ("end", h->end, nullptr);
            t.setProperty ("mu", floatsToBase64 (h->mu), nullptr);
            t.setProperty ("sigma", floatsToBase64 (h->sigma), nullptr);
            hits.appendChild (t, nullptr);
        }
    }
    state.removeChild (state.getChildWithName ("UserHits"), nullptr);
    state.appendChild (hits, nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary (*xml, dest);
}

void DrumMachineProcessor::setStateInformation (const void* data, int size)
{
    auto xml = getXmlFromBinary (data, size);
    if (! xml || ! xml->hasTagName (apvts.state.getType()))
        return;
    auto state = juce::ValueTree::fromXml (*xml);
    worker.cancelDefaultAssignment();                 // the session's slots win
    {
        std::vector<UserHitPtr> loaded;
        auto hits = state.getChildWithName ("UserHits");
        for (int i = 0; i < hits.getNumChildren(); ++i)
        {
            auto t = hits.getChild (i);
            auto h = std::make_shared<UserHit>();
            h->name = t.getProperty ("name").toString();
            h->file = juce::File (t.getProperty ("file").toString());
            h->modelId = t.getProperty ("model").toString();
            h->D = (int) t.getProperty ("D");
            h->T = (int) t.getProperty ("T");
            h->peak = (float) t.getProperty ("peak");
            h->end = (float) t.getProperty ("end");
            h->mu = floatsFromBase64 (t.getProperty ("mu").toString());
            h->sigma = floatsFromBase64 (t.getProperty ("sigma").toString());
            if ((int) h->mu.size() == h->D * h->T && h->mu.size() == h->sigma.size())
                loaded.push_back (h);
        }
        const juce::ScopedLock sl (userLock);
        userHits = std::move (loaded);
    }
    state.removeChild (state.getChildWithName ("UserHits"), nullptr);
    apvts.replaceState (state);
    const juce::File kitFile (state.getProperty ("kitPath").toString());
    if (kitFile.existsAsFile() && kitFile != worker.getKitFile())
        worker.requestKit (kitFile, false);
    else if (auto k = worker.getKit())
        reencodeForKit (*k);
    if (onUserHitsChanged)
        juce::MessageManager::callAsync ([this] { if (onUserHitsChanged) onUserHitsChanged (-1); });
    worker.markAllDirty();
}

juce::AudioProcessorEditor* DrumMachineProcessor::createEditor()
{
    return new DrumMachineEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DrumMachineProcessor();
}
