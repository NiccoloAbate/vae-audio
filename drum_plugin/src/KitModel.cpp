#include "KitModel.h"
#include <ATen/Parallel.h>

#if __has_include(<torch/mps.h>)
 #include <torch/mps.h>
 #define LDM_HAS_MPS 1
#else
 #define LDM_HAS_MPS 0
#endif

std::shared_ptr<KitModel> KitModel::load (const juce::File& kitPt, juce::String& error)
{
    if (! kitPt.existsAsFile())
    {
        error = "kit not found: " + kitPt.getFullPathName();
        return nullptr;
    }
    auto json = kitPt.getSiblingFile ("kit.json");
    auto meta = juce::JSON::parse (json);
    if (! meta.isObject())
    {
        error = "missing or invalid kit.json next to " + kitPt.getFileName();
        return nullptr;
    }

    std::shared_ptr<KitModel> k (new KitModel());
    k->file    = kitPt;
    k->name    = meta.getProperty ("name", kitPt.getParentDirectory().getFileName()).toString();
    k->sr      = (int) meta.getProperty ("sr", 44100);
    k->samples = (int) meta.getProperty ("samples", 65536);
    k->nSeeds  = juce::jmax (1, (int) meta.getProperty ("n_seeds", 64));
    if (auto* arr = meta.getProperty ("entries", {}).getArray())
        for (auto& e : *arr)
            k->bank.push_back ({ e.getProperty ("name", "").toString(), e.getProperty ("type", "").toString() });
    if (auto* arr = meta.getProperty ("controls", {}).getArray())
        for (auto& c : *arr)
            k->controls.add (c.getProperty ("name", "").toString());
    if (auto* arr = meta.getProperty ("modes", {}).getArray())
        for (auto& m : *arr)
            k->modes.add (m.toString());
    if (k->bank.empty())
    {
        error = "kit.json lists no bank entries";
        return nullptr;
    }

    try
    {
        at::set_num_threads (4);
        k->deviceName = "CPU";
       #if LDM_HAS_MPS
        if (torch::mps::is_available())
        {
            k->device = torch::Device (torch::kMPS);
            k->deviceName = "GPU (MPS)";
            k->chunk = 16;
        }
       #endif
        k->module = torch::jit::load (kitPt.getFullPathName().toStdString(), k->device);
        k->module.eval();
    }
    catch (const std::exception& e)
    {
        error = juce::String ("could not load kit.pt: ") + e.what();
        return nullptr;
    }
    return k;
}

std::vector<std::vector<float>> KitModel::render (const RenderJob& job)
{
    std::lock_guard<std::mutex> lock (mutex);
    torch::NoGradGuard noGrad;

    const int L = (int) job.rows.size();
    const int K = numControls();
    const int nBank = (int) bank.size();
    std::vector<std::vector<float>> out;
    out.reserve ((size_t) L);

    for (int start = 0; start < L; start += chunk)
    {
        const int n = juce::jmin (chunk, L - start);
        auto morph = torch::empty ({ n });
        auto var   = torch::empty ({ n });
        auto decay = torch::empty ({ n });
        auto punch = torch::empty ({ n });
        auto seeds = torch::empty ({ n }, torch::kLong);
        auto ctl   = torch::zeros ({ n, K });
        auto m = morph.accessor<float, 1>();
        auto v = var.accessor<float, 1>();
        auto d = decay.accessor<float, 1>();
        auto p = punch.accessor<float, 1>();
        auto s = seeds.accessor<int64_t, 1>();
        auto c = ctl.accessor<float, 2>();
        for (int i = 0; i < n; ++i)
        {
            const auto& r = job.rows[(size_t) (start + i)];
            m[i] = r.morph;
            v[i] = r.varAmount;
            d[i] = r.decay;
            p[i] = r.punch;
            s[i] = juce::jlimit (0, nSeeds - 1, r.varSeed);
            for (int k = 0; k < K && k < (int) r.controls.size(); ++k)
                c[i][k] = r.controls[(size_t) k];
        }

        std::vector<torch::jit::IValue> args {
            (int64_t) juce::jlimit (0, nBank - 1, job.slotA),
            (int64_t) juce::jlimit (0, nBank - 1, job.slotB),
            (int64_t) job.mode, (double) job.width,
            morph, ctl, var, seeds, decay, punch,
            (int64_t) juce::jlimit (0, nSeeds - 1, job.decodeSeed), (double) job.gateDb };

        auto y = module.get_method ("render") (args).toTensor().to (torch::kCPU).contiguous();
        const int len = (int) y.size (1);
        const float* data = y.data_ptr<float>();
        for (int i = 0; i < n; ++i)
            out.emplace_back (data + (size_t) i * (size_t) len, data + (size_t) (i + 1) * (size_t) len);
    }
    return out;
}
