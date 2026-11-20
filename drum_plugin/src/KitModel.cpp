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
    k->latentDim = (int) meta.getProperty ("latent_dim", 32);
    k->tLat = (int) meta.getProperty ("t_lat", 128);
    k->modelId = meta.getProperty ("checkpoint", "").toString();
    auto pre = meta.getProperty ("preprocess", {});
    k->onsetThresh = (float) (double) pre.getProperty ("onset_thresh", 0.02);
    k->prePad = (int) pre.getProperty ("pre_pad", 512);
    k->peakNorm = (bool) pre.getProperty ("peak_norm", true);
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
        k->hasEncoder = k->module.find_method ("encode").has_value() && k->module.find_method ("render_z").has_value();
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

        torch::Tensor y;
        const auto tail = std::vector<torch::jit::IValue> { (int64_t) job.mode, (double) job.width,
                                                            morph, ctl, var, seeds, decay, punch,
                                                            (int64_t) juce::jlimit (0, nSeeds - 1, job.decodeSeed),
                                                            (double) job.gateDb };
        if (hasEncoder)
        {
            // latents for A and B: a user hit, or the bank entry
            auto source = [&] (const UserHitPtr& u, int slot) -> std::pair<torch::Tensor, torch::Tensor>
            {
                if (u && accepts (*u))
                {
                    auto mu = torch::from_blob (const_cast<float*> (u->mu.data()), { u->D, u->T }).clone();
                    auto sg = torch::from_blob (const_cast<float*> (u->sigma.data()), { u->D, u->T }).clone();
                    return { torch::cat ({ mu, sg }, 0), torch::tensor ({ u->peak, u->end }) };
                }
                auto e = module.get_method ("entry") ({ (int64_t) juce::jlimit (0, nBank - 1, slot) }).toTuple();
                return { e->elements()[0].toTensor(), e->elements()[1].toTensor() };
            };
            auto a = source (job.userA, job.slotA);
            auto b = source (job.userB, job.slotB);
            std::vector<torch::jit::IValue> args { a.first, b.first, a.second, b.second };
            args.insert (args.end(), tail.begin(), tail.end());
            y = module.get_method ("render_z") (args).toTensor().to (torch::kCPU).contiguous();
        }
        else
        {
            std::vector<torch::jit::IValue> args { (int64_t) juce::jlimit (0, nBank - 1, job.slotA),
                                                   (int64_t) juce::jlimit (0, nBank - 1, job.slotB) };
            args.insert (args.end(), tail.begin(), tail.end());
            y = module.get_method ("render") (args).toTensor().to (torch::kCPU).contiguous();
        }
        const int len = (int) y.size (1);
        const float* data = y.data_ptr<float>();
        for (int i = 0; i < n; ++i)
            out.emplace_back (data + (size_t) i * (size_t) len, data + (size_t) (i + 1) * (size_t) len);
    }
    return out;
}

std::shared_ptr<UserHit> KitModel::encode (const std::vector<float>& prepared, const juce::String& hitName)
{
    if (! hasEncoder)
        throw std::runtime_error ("this kit has no encoder: re-export it with export_kit.py");
    std::lock_guard<std::mutex> lock (mutex);
    torch::NoGradGuard noGrad;
    auto x = torch::from_blob (const_cast<float*> (prepared.data()), { (int64_t) prepared.size() }).clone();
    auto out = module.get_method ("encode") ({ x }).toTuple();
    auto mu = out->elements()[0].toTensor().to (torch::kCPU).contiguous();
    auto sg = out->elements()[1].toTensor().to (torch::kCPU).contiguous();
    auto tm = out->elements()[2].toTensor().to (torch::kCPU).contiguous();

    auto h = std::make_shared<UserHit>();
    h->name = hitName;
    h->modelId = modelId;
    h->D = (int) mu.size (0);
    h->T = (int) mu.size (1);
    h->mu.assign (mu.data_ptr<float>(), mu.data_ptr<float>() + mu.numel());
    h->sigma.assign (sg.data_ptr<float>(), sg.data_ptr<float>() + sg.numel());
    h->peak = tm.data_ptr<float>()[0];
    h->end = tm.data_ptr<float>()[1];
    return h;
}
