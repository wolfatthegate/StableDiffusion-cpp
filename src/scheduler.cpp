// ============================================================================
// scheduler.cpp - Beta schedule, Euler and DDIM samplers. See scheduler.h for
// the maths and the calling convention.
// ============================================================================
#include "scheduler.h"

#include <cmath>
#include <stdexcept>

namespace sd {

// "scaled_linear": betas are linear in sqrt-space, then squared. The
// cumulative product is accumulated in double to avoid drift over 1000 terms.
Scheduler::Scheduler() {
    const int T = 1000;
    const float b0 = std::sqrt(0.00085f), b1 = std::sqrt(0.012f);
    alphas_cumprod_.resize(T);
    double prod = 1.0;
    for (int i = 0; i < T; ++i) {
        float beta = b0 + (b1 - b0) * (float)i / (float)(T - 1);
        beta *= beta;
        prod *= 1.0 - beta;
        alphas_cumprod_[i] = (float)prod;
    }
}

// ---------------------------------------------------------------------------
void EulerScheduler::set_timesteps(int steps) {
    if (steps < 1) throw std::runtime_error("steps must be >= 1");
    const int T = (int)alphas_cumprod_.size();
    // sigma_t for every training timestep; increases from ~0.03 to ~14.6.
    std::vector<float> all_sigmas(T);
    for (int i = 0; i < T; ++i) all_sigmas[i] = std::sqrt((1.0f - alphas_cumprod_[i]) / alphas_cumprod_[i]);

    timesteps_.resize(steps);
    sigmas_.resize(steps + 1);
    // "linspace": t_i evenly spaced from T-1 down to 0. Fractional timesteps
    // get a sigma linearly interpolated between neighbouring integers; the
    // UNet receives the fractional t as is.
    for (int i = 0; i < steps; ++i) {
        const float t = steps == 1 ? (float)(T - 1) : (float)(T - 1) * (float)(steps - 1 - i) / (float)(steps - 1);
        timesteps_[i] = t;
        const int lo = (int)std::floor(t);
        const int hi = std::min(lo + 1, T - 1);
        const float frac = t - (float)lo;
        sigmas_[i] = all_sigmas[lo] + (all_sigmas[hi] - all_sigmas[lo]) * frac;
    }
    sigmas_[steps] = 0.0f;  // final target: a clean sample
}

// x has standard deviation sqrt(1 + sigma^2); dividing by it recovers the
// x_t = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps the UNet was trained on.
void EulerScheduler::scale_model_input(Tensor& x, int step) const {
    const float s = sigmas_[step];
    scale_(x, 1.0f / std::sqrt(s * s + 1.0f));
}

void EulerScheduler::step(const Tensor& eps, int step, Tensor& x) const {
    const float sigma = sigmas_[step], dt = sigmas_[step + 1] - sigma;
    // x0 = x - sigma * eps ; d = (x - x0) / sigma = eps ; x += d * dt
    float* p = x.ptr();
    const float* e = eps.ptr();
    for (int64_t i = 0; i < x.numel(); ++i) p[i] += e[i] * dt;
}

// x = x0 + sigma * eps  =>  x0 = x - sigma * eps
Tensor EulerScheduler::predict_x0(const Tensor& x, const Tensor& eps, int step) const {
    const float sigma = sigmas_[step];
    Tensor x0 = x;
    float* p = x0.ptr();
    const float* e = eps.ptr();
    for (int64_t i = 0; i < x0.numel(); ++i) p[i] -= sigma * e[i];
    return x0;
}

// ---------------------------------------------------------------------------
void DDIMScheduler::set_timesteps(int steps) {
    if (steps < 1) throw std::runtime_error("steps must be >= 1");
    const int T = (int)alphas_cumprod_.size();
    step_ratio_ = T / steps;
    // "leading" spacing with steps_offset = 1: 1, 1 + r, 1 + 2r, ... reversed.
    timesteps_.resize(steps);
    for (int i = 0; i < steps; ++i) timesteps_[i] = (float)((steps - 1 - i) * step_ratio_ + 1);
}

void DDIMScheduler::step(const Tensor& eps, int step, Tensor& x) const {
    const int t = (int)timesteps_[step];
    const int prev_t = t - step_ratio_;
    const float a_t = alphas_cumprod_[t];
    // After the last step prev_t is negative; diffusers' default
    // (set_alpha_to_one = False) then uses alpha_bar_0.
    const float a_prev = prev_t >= 0 ? alphas_cumprod_[prev_t] : alphas_cumprod_[0];
    const float sqrt_at = std::sqrt(a_t), sqrt_1m_at = std::sqrt(1.0f - a_t);
    const float sqrt_ap = std::sqrt(a_prev), sqrt_1m_ap = std::sqrt(1.0f - a_prev);
    float* p = x.ptr();
    const float* e = eps.ptr();
    // Predict x0 from the noise estimate, then re-noise it to level a_prev
    // using the same eps (deterministic, eta = 0).
    for (int64_t i = 0; i < x.numel(); ++i) {
        const float x0 = (p[i] - sqrt_1m_at * e[i]) / sqrt_at;
        p[i] = sqrt_ap * x0 + sqrt_1m_ap * e[i];
    }
}

Tensor DDIMScheduler::predict_x0(const Tensor& x, const Tensor& eps, int step) const {
    const int t = (int)timesteps_[step];
    const float a_t = alphas_cumprod_[t];
    const float sqrt_at = std::sqrt(a_t), sqrt_1m_at = std::sqrt(1.0f - a_t);
    Tensor x0 = x;
    float* p = x0.ptr();
    const float* e = eps.ptr();
    for (int64_t i = 0; i < x0.numel(); ++i) p[i] = (p[i] - sqrt_1m_at * e[i]) / sqrt_at;
    return x0;
}

// ---------------------------------------------------------------------------
void FlowMatchScheduler::set_timesteps(int steps) {
    if (steps < 1) throw std::runtime_error("steps must be >= 1");
    timesteps_.resize(steps);
    sigmas_.resize(steps + 1);
    for (int i = 0; i < steps; ++i) {
        const float s = steps == 1 ? 1.0f : 1.0f - (float)i * (1.0f - 1.0f / (float)steps) / (float)(steps - 1);
        const float shifted = shift_ * s / (1.0f + (shift_ - 1.0f) * s);
        sigmas_[i] = shifted;
        timesteps_[i] = shifted * 1000.0f;
    }
    sigmas_[steps] = 0.0f;
}

void FlowMatchScheduler::step(const Tensor& out, int step, Tensor& x) const {
    const float dt = sigmas_[step] - sigmas_[step + 1];
    float* p = x.ptr();
    const float* o = out.ptr();
    for (int64_t i = 0; i < x.numel(); ++i) p[i] += dt * o[i];
}

Tensor FlowMatchScheduler::predict_x0(const Tensor& x, const Tensor& out, int step) const {
    const float s = sigmas_[step];
    Tensor x0 = x;
    for (int64_t i = 0; i < x0.numel(); ++i) x0.ptr()[i] += s * out.ptr()[i];
    return x0;
}

std::unique_ptr<Scheduler> make_scheduler(const std::string& name) {
    if (name == "euler") return std::unique_ptr<Scheduler>(new EulerScheduler());
    if (name == "ddim") return std::unique_ptr<Scheduler>(new DDIMScheduler());
    if (name == "flow") return std::unique_ptr<Scheduler>(new FlowMatchScheduler());
    throw std::runtime_error("unknown sampler '" + name + "' (use euler, ddim or flow)");
}

}  // namespace sd
