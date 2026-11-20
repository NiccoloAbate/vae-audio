#include "RenderWorker.h"

RenderWorker::RenderWorker() : juce::Thread ("LDM render")
{
    for (auto& c : changedAt)
        c = 0;
}

RenderWorker::~RenderWorker()
{
    stopThread (5000);
}

void RenderWorker::start()
{
    startThread (juce::Thread::Priority::normal);
}

void RenderWorker::requestKit (const juce::File& kitPt, bool assignDefaults)
{
    {
        const juce::ScopedLock sl (lock);
        pendingKit = kitPt;
    }
    assignDefaultsPending = assignDefaults;
    loadRequested = true;
    notify();
}

void RenderWorker::markDirty (int pad)
{
    if (pad < 0 || pad >= ldm::kNumPads)
        return;
    changedAt[(size_t) pad] = juce::Time::getMillisecondCounter();
    dirty.fetch_or (1u << pad);
    notify();
}

void RenderWorker::markAllDirty()
{
    for (int p = 0; p < ldm::kNumPads; ++p)
        markDirty (p);
}

void RenderWorker::setHostRate (double sr)
{
    if (std::abs (hostRate.exchange (sr) - sr) > 1e-6)
        markAllDirty();
}

std::shared_ptr<KitModel> RenderWorker::getKit() const
{
    return std::atomic_load (&kit);
}

ldm::LayerSetPtr RenderWorker::getSet (int pad) const
{
    return std::atomic_load (&sets[(size_t) pad]);
}

bool RenderWorker::isBusy() const
{
    return loadRequested || rendering || encodeRequested || (dirty.load() != 0 && getKit() != nullptr);
}

void RenderWorker::requestEncode (const EncodeRequest& r)
{
    {
        const juce::ScopedLock sl (lock);
        encodeQueue.push_back (r);
    }
    encodeRequested = true;
    notify();
}

void RenderWorker::encodePending (KitModel& k)
{
    std::vector<EncodeRequest> jobs;
    {
        const juce::ScopedLock sl (lock);
        jobs.swap (encodeQueue);
        encodeRequested = false;
    }
    for (auto& r : jobs)
    {
        try
        {
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            auto hit = ldm::encodeFile (r.file, k);
            setStatus ("encoded " + r.file.getFileName() + " in "
                       + juce::String (juce::roundToInt (juce::Time::getMillisecondCounterHiRes() - t0)) + " ms");
            if (onHitEncoded)
            {
                auto cb = onHitEncoded;
                juce::MessageManager::callAsync ([cb, hit, r] { cb (hit, r); });
            }
        }
        catch (const std::exception& e)
        {
            setStatus ("could not encode " + r.file.getFileName() + ": " + e.what());
        }
    }
}

bool RenderWorker::waitUntilIdle (int timeoutMs) const
{
    const auto end = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
    while (isBusy())
    {
        if (juce::Time::getMillisecondCounter() > end)
            return false;
        juce::Thread::sleep (2);
    }
    return true;
}

juce::String RenderWorker::getStatus() const
{
    const juce::ScopedLock sl (lock);
    return status;
}

juce::File RenderWorker::getKitFile() const
{
    const juce::ScopedLock sl (lock);
    return kitFile;
}

void RenderWorker::setStatus (const juce::String& s)
{
    const juce::ScopedLock sl (lock);
    status = s;
}

void RenderWorker::loadPending()
{
    juce::File f;
    {
        const juce::ScopedLock sl (lock);
        f = pendingKit;
    }
    setStatus ("loading " + f.getParentDirectory().getFileName() + "...");
    juce::String err;
    auto k = KitModel::load (f, err);
    if (! k)
    {
        setStatus ("kit error: " + err);
        return;
    }
    std::atomic_store (&kit, k);
    {
        const juce::ScopedLock sl (lock);
        kitFile = f;
    }
    setStatus (k->getName() + " loaded on " + k->getDeviceName());
    markAllDirty();
    const bool assign = assignDefaultsPending.exchange (false);
    if (onKitLoaded)
    {
        auto cb = onKitLoaded;
        juce::MessageManager::callAsync ([cb, k, assign] { cb (k, assign); });
    }
}

void RenderWorker::renderOne (int pad, KitModel& k)
{
    if (! getRecipe)
        return;
    const auto recipe = getRecipe (pad);
    const auto t0 = juce::Time::getMillisecondCounterHiRes();
    try
    {
        auto set = ldm::renderPad (recipe, k, hostRate.load());
        auto old = std::atomic_exchange (&sets[(size_t) pad], ldm::LayerSetPtr (set));
        if (old)
            graveyard.push_back (std::move (old));
        setStatus ("pad " + juce::String (pad + 1) + ": " + juce::String (set->numLayers) + " layers x "
                   + juce::String (set->numRR) + " in " + juce::String (juce::roundToInt (juce::Time::getMillisecondCounterHiRes() - t0))
                   + " ms (" + k.getDeviceName() + ")");
    }
    catch (const std::exception& e)
    {
        setStatus ("render error, pad " + juce::String (pad + 1) + ": " + e.what());
    }
}

void RenderWorker::run()
{
    while (! threadShouldExit())
    {
        if (loadRequested)
        {
            loadRequested = false;
            rendering = true;
            loadPending();
            rendering = false;
        }

        bool waiting = false;
        if (auto k = getKit())
        {
            if (encodeRequested)
            {
                rendering = true;
                encodePending (*k);
                rendering = false;
            }
            for (int p = 0; p < ldm::kNumPads && ! threadShouldExit(); ++p)
            {
                const uint32_t bit = 1u << p;
                if ((dirty.load() & bit) == 0)
                    continue;
                const auto age = juce::Time::getMillisecondCounter() - changedAt[(size_t) p].load();
                if (! offline && age < (juce::uint32) kDebounceMs)
                {
                    waiting = true;
                    continue;
                }
                rendering = true;
                dirty.fetch_and (~bit);
                renderOne (p, *k);
                rendering = false;
                if (loadRequested)
                    break;
            }
        }

        graveyard.erase (std::remove_if (graveyard.begin(), graveyard.end(),
                                         [] (const ldm::LayerSetPtr& s) { return s.use_count() == 1; }),
                         graveyard.end());

        if (! loadRequested && ! (encodeRequested && getKit() != nullptr) && (dirty.load() == 0 || getKit() == nullptr))
            wait (graveyard.empty() ? 1000 : 100);
        else if (waiting)
            wait (5);
    }
}
