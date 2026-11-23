#include "PluginEditor.h"

using namespace ldm;

namespace
{
const juce::Colour bg (0xff15171c), panel (0xff1d2027), text (0xffd8dce6), dim (0xff8a90a0), accent (0xff6fb3ff);

juce::Colour typeColour (const juce::String& t)
{
    if (t == "kick")   return juce::Colour (0xffe8894a);
    if (t == "snare")  return juce::Colour (0xff5d9cec);
    if (t == "clap")   return juce::Colour (0xffd770ad);
    if (t == "hat")    return juce::Colour (0xffe6c84f);
    if (t == "tom")    return juce::Colour (0xff6cc788);
    if (t == "cymbal") return juce::Colour (0xff4fc6c6);
    return juce::Colour (0xffa0a0a0);
}

void styleSlider (juce::Slider& s)
{
    s.setSliderStyle (juce::Slider::LinearHorizontal);
    s.setTextBoxStyle (juce::Slider::TextBoxRight, false, 58, 20);
    s.setColour (juce::Slider::trackColourId, accent.withAlpha (0.7f));
    s.setColour (juce::Slider::thumbColourId, accent);
    s.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
}
} // namespace

// ── PadButton ────────────────────────────────────────────────────────────────

juce::Rectangle<float> PadButton::toggleBox (bool solo) const
{
    return { 9.f + (solo ? 22.f : 0.f), (float) getHeight() - 25.f, 19.f, 16.f };
}

void PadButton::paint (juce::Graphics& g)
{
    if (dimmed)
        g.beginTransparencyLayer (0.45f);
    auto r = getLocalBounds().toFloat().reduced (2.f);
    g.setColour (panel.brighter (0.05f));
    g.fillRoundedRectangle (r, 8.f);
    g.setColour (colour.withAlpha (0.18f + 0.6f * flash));
    g.fillRoundedRectangle (r, 8.f);
    g.setColour (selected ? colour : colour.withAlpha (0.35f));
    g.drawRoundedRectangle (r, 8.f, selected ? 2.5f : 1.f);

    auto t = r.reduced (9.f, 7.f);
    g.setColour (text);
    g.setFont (juce::FontOptions (15.f, juce::Font::bold));
    g.drawText (title, t.removeFromTop (20.f), juce::Justification::topLeft);
    g.setColour (dim);
    g.setFont (juce::FontOptions (11.f));
    g.drawFittedText (sub, t.removeFromTop (30.f).toNearestInt(), juce::Justification::topLeft, 2);
    g.drawText (note, t.removeFromBottom (14.f), juce::Justification::bottomRight);
    if (dimmed)
        g.endTransparencyLayer();

    // M / S toggles, bottom left: filled when on
    for (bool solo : { false, true })
    {
        const bool on = solo ? soloed : muted;
        const auto box = toggleBox (solo);
        const auto c = solo ? juce::Colour (0xffe6c84f) : juce::Colour (0xffe0604f);
        if (on)
        {
            g.setColour (c);
            g.fillRoundedRectangle (box, 3.f);
        }
        g.setColour (on ? c : dim.withAlpha (0.6f));
        g.drawRoundedRectangle (box, 3.f, 1.f);
        g.setColour (on ? bg : dim);
        g.setFont (juce::FontOptions (11.f, juce::Font::bold));
        g.drawText (solo ? "S" : "M", box, juce::Justification::centred);
    }
}

void PadButton::mouseDown (const juce::MouseEvent& e)
{
    for (bool solo : { false, true })
        if (toggleBox (solo).expanded (2.f).contains (e.position))
        {
            if (onToggle)
                onToggle (pad, solo);
            return;
        }
    if (onPress)
        onPress (pad, juce::jlimit (0.05f, 1.f, 1.f - (float) e.y / (float) juce::jmax (1, getHeight())));
}

// ── WaveView ─────────────────────────────────────────────────────────────────

void WaveView::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    g.setColour (panel);
    g.fillRoundedRectangle (r, 6.f);
    if (! set || set->buffers.empty())
    {
        g.setColour (dim);
        g.drawText ("rendering...", r, juce::Justification::centred);
        return;
    }
    const double seconds = 1.5;
    auto draw = [&] (const juce::AudioBuffer<float>& b, juce::Colour c)
    {
        const float* d = b.getReadPointer (0);
        const int n = b.getNumSamples();
        const float w = r.getWidth(), mid = r.getCentreY(), h = r.getHeight() * 0.46f;
        const double perPx = seconds * set->sampleRate / (double) w;
        juce::Path p;
        for (int x = 0; x < (int) w; ++x)
        {
            const int s0 = (int) (x * perPx), s1 = juce::jmin (n, (int) ((x + 1) * perPx));
            float mx = 0.f;
            for (int s = s0; s < s1; ++s)
                mx = juce::jmax (mx, std::abs (d[s]));
            p.addRectangle (r.getX() + (float) x, mid - mx * h, 1.f, juce::jmax (1.f, 2.f * mx * h));
        }
        g.setColour (c);
        g.fillPath (p);
    };
    if (set->numLayers > 1)
        draw (set->get (0, 0), accent.withAlpha (0.3f));
    draw (set->get (set->numLayers - 1, 0), accent.withAlpha (0.85f));
    g.setColour (dim);
    g.setFont (juce::FontOptions (11.f));
    g.drawText (juce::String (set->numLayers) + " velocity layers x " + juce::String (set->numRR) + " variations"
                    + (set->hasSides() ? " + stereo partners" : "") + (set->numLayers > 1 ? "  (faint: softest)" : ""),
                r.reduced (8.f, 4.f), juce::Justification::topRight);
    g.drawText ("1.5 s", r.reduced (8.f, 4.f), juce::Justification::bottomRight);
}

// ── Editor ───────────────────────────────────────────────────────────────────

juce::Label& DrumMachineEditor::labelFor (juce::Component& c, const juce::String& t)
{
    labels.push_back (std::make_unique<juce::Label> ("", t));
    auto& l = *labels.back();
    l.setColour (juce::Label::textColourId, dim);
    l.setFont (juce::FontOptions (13.f));
    l.setJustificationType (juce::Justification::centredRight);
    l.attachToComponent (&c, true);
    addAndMakeVisible (l);
    return l;
}

DrumMachineEditor::DrumMachineEditor (DrumMachineProcessor& p)
    : AudioProcessorEditor (&p), proc (p)
{
    laf.setColour (juce::ResizableWindow::backgroundColourId, bg);
    laf.setColour (juce::ComboBox::backgroundColourId, panel.brighter (0.08f));
    laf.setColour (juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
    laf.setColour (juce::Slider::textBoxTextColourId, text);
    setLookAndFeel (&laf);

    title.setText ("Latent Drum Machine", juce::dontSendNotification);
    title.setFont (juce::FontOptions (22.f, juce::Font::bold));
    title.setColour (juce::Label::textColourId, text);
    for (auto* l : { &kitLabel, &statusLabel })
    {
        l->setColour (juce::Label::textColourId, dim);
        l->setFont (juce::FontOptions (12.f));
    }
    statusLabel.setJustificationType (juce::Justification::centredRight);
    padTitle.setFont (juce::FontOptions (17.f, juce::Font::bold));
    padTitle.setColour (juce::Label::textColourId, text);
    for (auto* c : std::initializer_list<juce::Component*> { &title, &kitLabel, &statusLabel, &padTitle, &loadButton, &wave })
        addAndMakeVisible (c);

    loadButton.onClick = [this]
    {
        chooser = std::make_unique<juce::FileChooser> ("Load a kit (kit.pt from export_kit.py)",
                                                       proc.worker.getKitFile().getParentDirectory(), "*.pt");
        chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                              [this] (const juce::FileChooser& fc)
                              {
                                  if (fc.getResult().existsAsFile())
                                      proc.loadKit (fc.getResult());
                              });
    };

    for (int i = 0; i < kNumPads; ++i)
    {
        auto& pb = pads[(size_t) i];
        pb.pad = i;
        pb.title = padDefaults()[(size_t) i].name;
        pb.colour = typeColour (padDefaults()[(size_t) i].type);
        pb.onPress = [this] (int pad, float vel)
        {
            selectPad (pad);
            proc.audition (pad, vel);
        };
        pb.onToggle = [this] (int pad, bool solo) { proc.toggle (pad, solo ? "solo" : "mute"); };
        addAndMakeVisible (pb);
    }

    for (auto* s : { &morph, &width, &decay, &punch, &variation, &seed, &rr, &velDepth, &velGain, &gain, &pan, &note,
                     &panSpread, &tune, &tuneSpread, &stereo, &stereoVar, &master, &decodeSeed })
    {
        styleSlider (*s);
        addAndMakeVisible (*s);
    }
    for (auto& s : controls)
    {
        styleSlider (s);
        addAndMakeVisible (s);
    }
    for (auto* c : { &slotA, &slotB, &mode, &velTarget, &choke, &varOrder })
        addAndMakeVisible (*c);
    addAndMakeVisible (selfChoke);
    addAndMakeVisible (gate);

    labelFor (slotA, "Slot A");
    labelFor (slotB, "Slot B");
    labelFor (morph, "Morph A-B");
    labelFor (mode, "Morph mode");
    labelFor (width, "Xfade width");
    labelFor (decay, "Decay");
    labelFor (punch, "Punch");
    labelFor (variation, "Variation");
    labelFor (seed, "Var. seed");
    labelFor (rr, "Variations");
    labelFor (varOrder, "Var. order");
    labelFor (note, "MIDI note");
    for (int k = 0; k < kMaxControls; ++k)
        controlLabels[(size_t) k] = &labelFor (controls[(size_t) k], "Control " + juce::String (k + 1));
    labelFor (velTarget, "Velocity ->");
    labelFor (velDepth, "Vel. depth");
    labelFor (velGain, "Vel. -> gain");
    labelFor (gain, "Gain (dB)");
    labelFor (pan, "Pan");
    labelFor (panSpread, "Pan spread");
    labelFor (tune, "Tune (st)");
    labelFor (tuneSpread, "Tune spread");
    labelFor (stereo, "Stereo width");
    labelFor (stereoVar, "Width var.");
    labelFor (choke, "Choke");
    labelFor (master, "Master (dB)");
    labelFor (decodeSeed, "Decode seed");

    globalSliderAtt.push_back (std::make_unique<SA> (proc.apvts, "master", master));
    globalSliderAtt.push_back (std::make_unique<SA> (proc.apvts, "decode_seed", decodeSeed));
    gateAtt = std::make_unique<BA> (proc.apvts, "gate", gate);

    auto onSlot = [this] (juce::ComboBox& cb, bool b)
    {
        cb.onChange = [this, &cb, b]
        {
            if (cb.getSelectedId() > 0)
                proc.setSlot (selected, b, cb.getSelectedId() - 1);
        };
    };
    onSlot (slotA, false);
    onSlot (slotB, true);

    proc.onUserHitsChanged = [this] (int pad)
    {
        fillSlotCombos();
        if (pad >= 0)
            selectPad (pad);
    };

    setSize (1120, 640);
    kitChanged();
    startTimerHz (30);
}

DrumMachineEditor::~DrumMachineEditor()
{
    proc.onUserHitsChanged = nullptr;
    stopTimer();
    setLookAndFeel (nullptr);
}

void DrumMachineEditor::kitChanged()
{
    kit = proc.worker.getKit();
    if (kit)
        kitLabel.setText ("kit: " + kit->getName() + "  |  " + juce::String ((int) kit->entries().size()) + " hits  |  "
                              + kit->getDeviceName() + "  |  " + kit->getFile().getParentDirectory().getFullPathName(),
                          juce::dontSendNotification);
    else
        kitLabel.setText ("no kit loaded: Load kit... (kit.pt from export_kit.py)", juce::dontSendNotification);
    const int K = kit ? juce::jmin (kit->numControls(), kMaxControls) : 0;
    for (int k = 0; k < kMaxControls; ++k)
    {
        const bool on = k < K;
        controls[(size_t) k].setVisible (on);
        controlLabels[(size_t) k]->setVisible (on);
        if (on)
            controlLabels[(size_t) k]->setText (kit->controlNames()[k], juce::dontSendNotification);
    }
    fillSlotCombos();
    selectPad (selected);
}

juce::String DrumMachineEditor::slotName (int index) const
{
    if (index >= kUserSlotBase)
    {
        auto h = proc.getUserHit (index - kUserSlotBase);
        return h ? "* " + h->name : "-";
    }
    if (! kit || index < 0 || index >= (int) kit->entries().size())
        return "-";
    return kit->entries()[(size_t) index].name;
}

void DrumMachineEditor::fillSlotCombos()
{
    for (auto* cb : { &slotA, &slotB })
    {
        cb->clear (juce::dontSendNotification);
        if (! kit)
            continue;
        // your dropped hits first, then the bank grouped by drum type, the pad's type first
        if (const int nUser = proc.numUserHits(); nUser > 0)
        {
            cb->addSectionHeading ("your hits (drop samples here)");
            for (int i = 0; i < nUser; ++i)
                if (auto h = proc.getUserHit (i))
                    cb->addItem ("* " + h->name, kUserSlotBase + i + 1);
        }
        juce::StringArray types;
        types.add (padDefaults()[(size_t) selected].type);
        for (auto& e : kit->entries())
            types.addIfNotAlreadyThere (e.type);
        for (auto& t : types)
        {
            bool heading = false;
            for (int i = 0; i < (int) kit->entries().size(); ++i)
            {
                const auto& e = kit->entries()[(size_t) i];
                if (e.type != t)
                    continue;
                if (! heading)
                {
                    cb->addSectionHeading (t);
                    heading = true;
                }
                cb->addItem (juce::String (i) + "  " + e.name, i + 1);
            }
        }
    }
    syncSlotCombos();
}

void DrumMachineEditor::syncSlotCombos()
{
    const int a = (int) proc.apvts.getRawParameterValue (pid (selected, "slot_a"))->load();
    const int b = (int) proc.apvts.getRawParameterValue (pid (selected, "slot_b"))->load();
    if (slotA.getSelectedId() != a + 1)
        slotA.setSelectedId (a + 1, juce::dontSendNotification);
    if (slotB.getSelectedId() != b + 1)
        slotB.setSelectedId (b + 1, juce::dontSendNotification);
}

void DrumMachineEditor::selectPad (int pad)
{
    const bool padChanged = pad != selected;
    selected = pad;
    for (auto& pb : pads)
        pb.selected = pb.pad == pad;
    padTitle.setText ("Pad " + juce::String (pad + 1) + ": " + padDefaults()[(size_t) pad].name, juce::dontSendNotification);

    sliderAtt.clear();
    comboAtt.clear();
    buttonAtt.clear();

    mode.clear (juce::dontSendNotification);
    mode.addItemList ({ "Blend", "Blend (slerp)", "Shape-aligned", "Tail only", "Crossover in time" }, 1);
    velTarget.clear (juce::dontSendNotification);
    auto vt = velTargetNames();
    for (int k = 0; k < kMaxControls; ++k)
        vt.set (velTargetForControl (k), kit && k < kit->numControls() ? kit->controlNames()[k]
                                                               : "(no control " + juce::String (k + 1) + ")");
    velTarget.addItemList (vt, 1);
    choke.clear (juce::dontSendNotification);
    choke.addItemList ({ "None", "Group 1", "Group 2", "Group 3", "Group 4" }, 1);
    varOrder.clear (juce::dontSendNotification);
    varOrder.addItemList ({ "Random", "Cycle", "Random, no repeat" }, 1);

    auto& ap = proc.apvts;
    auto sa = [&] (const char* id, juce::Slider& s) { sliderAtt.push_back (std::make_unique<SA> (ap, pid (pad, id), s)); };
    sa ("morph", morph);
    sa ("width", width);
    sa ("decay", decay);
    sa ("punch", punch);
    sa ("var", variation);
    sa ("seed", seed);
    sa ("rr", rr);
    sa ("vel_depth", velDepth);
    sa ("vel_gain", velGain);
    sa ("gain", gain);
    sa ("pan", pan);
    sa ("note", note);
    sa ("pan_rand", panSpread);
    sa ("tune", tune);
    sa ("tune_rand", tuneSpread);
    sa ("stereo", stereo);
    sa ("stereo_var", stereoVar);
    for (int k = 0; k < kMaxControls; ++k)
        sa (("c" + std::to_string (k)).c_str(), controls[(size_t) k]);
    comboAtt.push_back (std::make_unique<CA> (ap, pid (pad, "mode"), mode));
    comboAtt.push_back (std::make_unique<CA> (ap, pid (pad, "vel_target"), velTarget));
    comboAtt.push_back (std::make_unique<CA> (ap, pid (pad, "choke"), choke));
    comboAtt.push_back (std::make_unique<CA> (ap, pid (pad, "var_order"), varOrder));
    buttonAtt.push_back (std::make_unique<BA> (ap, pid (pad, "self_choke"), selfChoke));

    if (padChanged)
        fillSlotCombos();
    else
        syncSlotCombos();
    wave.setSet (proc.worker.getSet (pad));
    repaint();
}

void DrumMachineEditor::timerCallback()
{
    proc.decayFlash();
    bool anySolo = false;
    for (int p = 0; p < kNumPads; ++p)
        anySolo = anySolo || proc.apvts.getRawParameterValue (pid (p, "solo"))->load() > 0.5f;
    for (auto& pb : pads)
    {
        const bool m = proc.apvts.getRawParameterValue (pid (pb.pad, "mute"))->load() > 0.5f;
        const bool s = proc.apvts.getRawParameterValue (pid (pb.pad, "solo"))->load() > 0.5f;
        const bool d = m || (anySolo && ! s);
        if (m != pb.muted || s != pb.soloed || d != pb.dimmed)
        {
            pb.muted = m;
            pb.soloed = s;
            pb.dimmed = d;
            pb.repaint();
        }
        const float f = proc.getFlash (pb.pad);
        const int a = (int) proc.apvts.getRawParameterValue (pid (pb.pad, "slot_a"))->load();
        const int n = (int) proc.apvts.getRawParameterValue (pid (pb.pad, "note"))->load();
        const auto sub = slotName (a);
        const auto nn = juce::MidiMessage::getMidiNoteName (n, true, true, 3) + " (" + juce::String (n) + ")";
        if (std::abs (f - pb.flash) > 0.01f || sub != pb.sub || nn != pb.note)
        {
            pb.flash = f;
            pb.sub = sub;
            pb.note = nn;
            pb.repaint();
        }
    }
    statusLabel.setText (proc.worker.getStatus() + (proc.worker.isBusy() ? "  ..." : ""), juce::dontSendNotification);
    auto set = proc.worker.getSet (selected);
    if (set.get() != wave.current())
        wave.setSet (set);
    if (proc.worker.getKit() != kit)
        kitChanged();
    syncSlotCombos();
}

DrumMachineEditor::DropTarget DrumMachineEditor::dropTargetAt (int x, int y) const
{
    const juce::Point<int> pt (x, y);
    for (auto& pb : pads)
        if (pb.getBounds().contains (pt))
            return { pb.pad, false, pb.getBounds() };
    if (slotA.getBounds().expanded (4).contains (pt))
        return { selected, false, slotA.getBounds() };
    if (slotB.getBounds().expanded (4).contains (pt))
        return { selected, true, slotB.getBounds() };
    if (x > 494 && y > 60)
        return { selected, false, slotA.getBounds() };
    return {};
}

bool DrumMachineEditor::isInterestedInFileDrag (const juce::StringArray& files)
{
    for (auto& f : files)
        if (isAudioFile (juce::File (f)))
            return true;
    return false;
}

void DrumMachineEditor::fileDragMove (const juce::StringArray&, int x, int y)
{
    const auto t = dropTargetAt (x, y);
    if (t.area != dropHighlight)
    {
        dropHighlight = t.area;
        repaint();
    }
}

void DrumMachineEditor::fileDragExit (const juce::StringArray&)
{
    dropHighlight = {};
    repaint();
}

void DrumMachineEditor::filesDropped (const juce::StringArray& files, int x, int y)
{
    dropHighlight = {};
    repaint();
    const auto t = dropTargetAt (x, y);
    if (t.pad < 0)
        return;
    if (kit && ! kit->canEncode())
    {
        statusLabel.setText ("this kit has no encoder: re-export it with export_kit.py", juce::dontSendNotification);
        return;
    }
    juce::Array<juce::File> audio;
    for (auto& f : files)
        if (isAudioFile (juce::File (f)))
            audio.add (juce::File (f));
    if (audio.isEmpty())
        return;
    selectPad (t.pad);
    proc.importSamples (audio, t.pad, t.slotB);
}

void DrumMachineEditor::paintOverChildren (juce::Graphics& g)
{
    if (! dropHighlight.isEmpty())
    {
        g.setColour (accent);
        g.drawRoundedRectangle (dropHighlight.toFloat().expanded (3.f), 6.f, 2.5f);
    }
}

void DrumMachineEditor::paint (juce::Graphics& g)
{
    g.fillAll (bg);
    g.setColour (panel);
    g.fillRoundedRectangle (getLocalBounds().toFloat().withTrimmedLeft (494.f).withTrimmedTop (60.f).reduced (6.f), 8.f);
}

void DrumMachineEditor::resized()
{
    title.setBounds (16, 10, 260, 30);
    loadButton.setBounds (280, 14, 100, 24);
    kitLabel.setBounds (392, 8, 720, 18);
    statusLabel.setBounds (392, 28, 712, 18);

    const int pw = 108, ph = 104, gap = 8, x0 = 16, y0 = 64;
    for (int i = 0; i < kNumPads; ++i)
        pads[(size_t) i].setBounds (x0 + (i % 4) * (pw + gap), y0 + (i / 4) * (ph + gap), pw, ph);
    wave.setBounds (x0, y0 + 2 * (ph + gap) + 6, 4 * pw + 3 * gap, 190);

    const int gy = wave.getBottom() + 18, lw = 92;
    master.setBounds (x0 + lw, gy, 4 * pw + 3 * gap - lw, 24);
    decodeSeed.setBounds (x0 + lw, gy + 30, 4 * pw + 3 * gap - lw, 24);
    gate.setBounds (x0 + lw, gy + 60, 220, 24);

    // right panel: two columns of label + control rows
    padTitle.setBounds (516, 70, 400, 26);
    const int cx1 = 500 + 14 + lw, cx2 = 812 + lw, cw = 196, rh = 28, ry = 104;
    auto col = [&] (int x, std::initializer_list<juce::Component*> comps)
    {
        int y = ry;
        for (auto* c : comps)
        {
            if (c != nullptr)
                c->setBounds (x, y, cw, 24);
            y += rh;
        }
    };
    col (cx1, { &slotA, &slotB, &morph, &mode, &width, nullptr, &decay, &punch, nullptr, &variation, &seed, &rr,
                &varOrder, nullptr, &stereo, &stereoVar, nullptr, &note });
    col (cx2, { &controls[0], &controls[1], &controls[2], &controls[3], &controls[4], &controls[5], nullptr,
                &velTarget, &velDepth, &velGain, nullptr, &gain, &pan, &panSpread, &tune, &tuneSpread, &choke, &selfChoke });
}
