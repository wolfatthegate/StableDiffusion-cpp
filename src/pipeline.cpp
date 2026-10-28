// ============================================================================
// pipeline.cpp - Model loading and the generation loops. See pipeline.h.
// ============================================================================
#include "pipeline.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "scheduler.h"

namespace sd {

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Parses <path> as JSON; returns a Null value if the file is absent.
Json read_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) return Json();
    std::stringstream ss;
    ss << in.rdbuf();
    return parse_json(ss.str());
}

double json_num(const Json& j, const char* key, double fallback) {
    const Json* v = j.type == Json::Object ? j.find(key) : nullptr;
    return v && v->type == Json::Number ? v->num : fallback;
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void validate(const GenerateParams& p) {
    if (p.prompt.empty()) throw std::runtime_error("prompt is empty");
    if (p.steps < 1 || p.steps > 500) throw std::runtime_error("steps must be between 1 and 500");
    if (p.width < 64 || p.height < 64 || p.width % 64 || p.height % 64 || p.width > 2048 || p.height > 2048)
        throw std::runtime_error("width and height must be multiples of 64 between 64 and 2048");
    if (!(p.cfg >= 0.0f && p.cfg <= 50.0f)) throw std::runtime_error("cfg must be between 0 and 50");
}

void fill_noise(Tensor& x, int64_t seed, float sigma) {
    std::mt19937_64 rng((uint64_t)seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& v : x.data) v = normal(rng) * sigma;
}

void to_rgb8(const Tensor& img, GenerateResult& r) {
    r.width = (int)img.dim(3);
    r.height = (int)img.dim(2);
    const int HW = r.width * r.height;
    r.rgb.resize((size_t)HW * 3);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < HW; ++i) {
            float v = (img.ptr()[c * HW + i] * 0.5f + 0.5f) * 255.0f;
            r.rgb[(size_t)i * 3 + c] = (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, v)));
        }
}

}  // namespace

// ---------------------------------------------------------------------------
Pipeline::Pipeline(const std::string& model_dir, Device device) : dir_(model_dir), device_(device) {
    Json index = read_json(dir_ + "/model_index.json");
    std::string cls;
    if (const Json* c = index.type == Json::Object ? index.find("_class_name") : nullptr) cls = c->str;
    if (cls.find("ZImage") != std::string::npos || file_exists(dir_ + "/transformer/config.json"))
        family_ = Family::ZImage;

    Json vae_cfg = read_json(dir_ + "/vae/config.json");
    const float vae_scale = (float)json_num(vae_cfg, "scaling_factor", VAEDecoder::kSDScalingFactor);
    const float vae_shift = (float)json_num(vae_cfg, "shift_factor", 0.0);
    vae_w_.load_model(dir_ + "/vae", "diffusion_pytorch_model");
    vae_w_.to_device(device_);
    vae_.reset(new VAEDecoder(vae_w_, vae_scale, vae_shift));

    if (family_ == Family::SD1) {
        tokenizer_.reset(new CLIPTokenizer(dir_ + "/tokenizer/vocab.json", dir_ + "/tokenizer/merges.txt"));
        clip_w_.load_model(dir_ + "/text_encoder", "model");
        clip_.reset(new CLIPTextEncoder(clip_w_));
        unet_w_.load_model(dir_ + "/unet", "diffusion_pytorch_model");
        unet_w_.to_device(device_);
        unet_.reset(new UNet(unet_w_));
    } else {
        qwen_tok_.reset(new QwenTokenizer(dir_ + "/tokenizer"));
        Json te_cfg = read_json(dir_ + "/text_encoder/config.json");
        qwen_w_.load_model(dir_ + "/text_encoder", "model", /*keep_bf16=*/true);
        qwen_.reset(new Qwen3Encoder(qwen_w_, (float)json_num(te_cfg, "rope_theta", 1e6),
                                     (float)json_num(te_cfg, "rms_norm_eps", 1e-6)));
        Json tr_cfg = read_json(dir_ + "/transformer/config.json");
        ZImageTransformer::Config cfg;
        cfg.heads = (int)json_num(tr_cfg, "n_heads", cfg.heads);
        cfg.norm_eps = (float)json_num(tr_cfg, "norm_eps", cfg.norm_eps);
        cfg.rope_theta = (float)json_num(tr_cfg, "rope_theta", cfg.rope_theta);
        cfg.t_scale = (float)json_num(tr_cfg, "t_scale", cfg.t_scale);
        if (const Json* ad = tr_cfg.type == Json::Object ? tr_cfg.find("axes_dims") : nullptr)
            for (size_t i = 0; i < 3 && i < ad->arr.size(); ++i) cfg.axes_dims[i] = (int)ad->arr[i].num;
        if (const Json* ps = tr_cfg.type == Json::Object ? tr_cfg.find("all_patch_size") : nullptr)
            if (!ps->arr.empty()) cfg.patch = (int)ps->arr[0].num;
        dit_w_.load_model(dir_ + "/transformer", "diffusion_pytorch_model", /*keep_bf16=*/true);
        dit_w_.to_device(device_);
        dit_.reset(new ZImageTransformer(dit_w_, cfg));
        Json sch = read_json(dir_ + "/scheduler/scheduler_config.json");
        flow_shift_ = (float)json_num(sch, "shift", 3.0);
    }
}

GenerateParams Pipeline::defaults() const {
    GenerateParams p;
    if (family_ == Family::ZImage) {
        p.sampler = "flow";
        p.steps = 9;
        p.cfg = 0.0f;
        p.width = p.height = 1024;
    }
    return p;
}

std::vector<std::string> Pipeline::samplers() const {
    if (family_ == Family::ZImage) return {"flow"};
    return {"euler", "ddim"};
}

GenerateResult Pipeline::generate(const GenerateParams& p, const ProgressFn& progress) const {
    validate(p);
    return family_ == Family::ZImage ? generate_zimage(p, progress) : generate_sd1(p, progress);
}

// ---------------------------------------------------------------------------
GenerateResult Pipeline::generate_sd1(const GenerateParams& p, const ProgressFn& progress) const {
    const double t_start = now();
    GenerateResult result;
    result.seed = p.seed >= 0 ? p.seed : (int64_t)(std::random_device{}() & 0x7fffffff);
    auto report = [&](Progress pr) { return !progress || progress(pr); };

    Progress pr;
    pr.phase = Progress::TextEncoding;
    pr.total = p.steps;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    const bool use_cfg = p.cfg != 1.0f && p.cfg != 0.0f;
    Tensor context = clip_->forward(tokenizer_->encode(p.prompt));
    if (use_cfg) context = concat_batch(clip_->forward(tokenizer_->encode(p.negative)), context);
    context = context.to(device_);

    std::unique_ptr<Scheduler> sched = make_scheduler(p.sampler == "flow" ? "euler" : p.sampler);
    sched->set_timesteps(p.steps);
    Tensor x({1, 4, p.height / 8, p.width / 8});
    fill_noise(x, result.seed, sched->init_noise_sigma());

    const auto& ts = sched->timesteps();
    for (int i = 0; i < p.steps; ++i) {
        const double t0 = now();
        Tensor xin = x;
        sched->scale_model_input(xin, i);
        Tensor eps;
        if (use_cfg) {
            Tensor both = unet_->forward(repeat_batch(xin.to(device_), 2), ts[i], context).to(Device::CPU);
            eps = slice_batch(both, 0);
            const Tensor cond = slice_batch(both, 1);
            for (int64_t j = 0; j < eps.numel(); ++j) eps.ptr()[j] += p.cfg * (cond.ptr()[j] - eps.ptr()[j]);
        } else {
            eps = unet_->forward(xin.to(device_), ts[i], context).to(Device::CPU);
        }
        Tensor x0 = sched->predict_x0(x, eps, i);
        sched->step(eps, i, x);
        pr.phase = Progress::Denoising;
        pr.step = i + 1;
        pr.timestep = ts[i];
        pr.step_seconds = now() - t0;
        pr.preview = &x0;
        if (!report(pr)) {
            result.cancelled = true;
            result.seconds = now() - t_start;
            return result;
        }
    }

    pr.phase = Progress::Decoding;
    pr.preview = nullptr;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    to_rgb8(vae_->decode(x.to(device_)).to(Device::CPU), result);
    result.seconds = now() - t_start;
    return result;
}

// ---------------------------------------------------------------------------
// Z-Image: text through Qwen3 (penultimate layer, no final norm), flow
// matching from pure noise, model called with t = 1 - sigma.
GenerateResult Pipeline::generate_zimage(const GenerateParams& p, const ProgressFn& progress) const {
    const double t_start = now();
    GenerateResult result;
    result.seed = p.seed >= 0 ? p.seed : (int64_t)(std::random_device{}() & 0x7fffffff);
    auto report = [&](Progress pr) { return !progress || progress(pr); };

    Progress pr;
    pr.phase = Progress::TextEncoding;
    pr.total = p.steps;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    const int layers = qwen_->num_layers() - 1;  // hidden_states[-2]
    Tensor cap = qwen_->forward(qwen_tok_->encode_prompt(p.prompt, max_text_tokens_), layers, false).to(device_);
    const bool use_cfg = p.cfg > 0.0f;
    Tensor cap_neg;
    if (use_cfg) cap_neg = qwen_->forward(qwen_tok_->encode_prompt(p.negative, max_text_tokens_), layers, false).to(device_);

    FlowMatchScheduler sched(flow_shift_);
    sched.set_timesteps(p.steps);
    Tensor x({1, dit_->in_channels(), p.height / 8, p.width / 8});
    fill_noise(x, result.seed, 1.0f);

    for (int i = 0; i < p.steps; ++i) {
        const double t0 = now();
        const Tensor x_dev = x.to(device_);
        Tensor out = dit_->forward(x_dev, sched.model_time(i), cap).to(Device::CPU);
        if (use_cfg) {
            Tensor neg = dit_->forward(x_dev, sched.model_time(i), cap_neg).to(Device::CPU);
            for (int64_t j = 0; j < out.numel(); ++j) out.ptr()[j] += p.cfg * (out.ptr()[j] - neg.ptr()[j]);
        }
        Tensor x0 = sched.predict_x0(x, out, i);
        sched.step(out, i, x);
        pr.phase = Progress::Denoising;
        pr.step = i + 1;
        pr.timestep = sched.timesteps()[i];
        pr.step_seconds = now() - t0;
        pr.preview = &x0;
        if (!report(pr)) {
            result.cancelled = true;
            result.seconds = now() - t_start;
            return result;
        }
    }

    pr.phase = Progress::Decoding;
    pr.preview = nullptr;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    to_rgb8(vae_->decode(x.to(device_)).to(Device::CPU), result);
    result.seconds = now() - t_start;
    return result;
}

// ---------------------------------------------------------------------------
// Per-channel RGB weights widely used for cheap previews: SD 1.x (4 channels)
// and the FLUX VAE latent space (16 channels, with a bias).
void Pipeline::latent_to_rgb(const Tensor& latent, int& width, int& height, std::vector<uint8_t>& rgb) {
    static const float sd_factors[4][3] = {
        {0.298f, 0.207f, 0.208f}, {0.187f, 0.286f, 0.173f}, {-0.158f, 0.189f, 0.264f}, {-0.184f, -0.271f, -0.473f}};
    static const float flux_factors[16][3] = {
        {-0.0346f, 0.0244f, 0.0681f}, {0.0034f, 0.0210f, 0.0687f},   {0.0275f, -0.0668f, -0.0433f},
        {-0.0174f, 0.0160f, 0.0617f}, {0.0859f, 0.0721f, 0.0329f},   {0.0004f, 0.0383f, 0.0115f},
        {0.0405f, 0.0861f, 0.0915f},  {-0.0236f, -0.0185f, -0.0259f}, {-0.0245f, 0.0250f, 0.1180f},
        {0.1008f, 0.0755f, -0.0421f}, {-0.0515f, 0.0201f, 0.0011f},  {0.0428f, -0.0012f, -0.0036f},
        {0.0817f, 0.0765f, 0.0749f},  {-0.1264f, -0.0522f, -0.1103f}, {-0.0280f, -0.0881f, -0.0499f},
        {-0.1262f, -0.0982f, -0.0778f}};
    static const float flux_bias[3] = {-0.0329f, -0.0718f, -0.0851f};
    const int64_t C = latent.dim(1);
    height = (int)latent.dim(2);
    width = (int)latent.dim(3);
    const int HW = width * height;
    rgb.assign((size_t)HW * 3, 0);
    for (int i = 0; i < HW; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = C == 16 ? flux_bias[c] : 0.0f;
            for (int64_t l = 0; l < C && l < 16; ++l)
                v += latent.ptr()[l * HW + i] * (C == 16 ? flux_factors[l][c] : sd_factors[l][c]);
            v = (v + 1.0f) * 0.5f * 255.0f;
            rgb[(size_t)i * 3 + c] = (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, v)));
        }
}

}  // namespace sd
