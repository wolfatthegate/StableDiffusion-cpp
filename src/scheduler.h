// ============================================================================
// scheduler.h - Noise schedules and samplers.
//
// SD 1.x was trained with T = 1000 diffusion steps and a "scaled_linear" beta
// schedule:
//     beta_i = (sqrt(0.00085) + i/(T-1) * (sqrt(0.012) - sqrt(0.00085)))^2
//     alpha_bar_t = prod_{i<=t} (1 - beta_i)
// A noisy latent at step t is x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps.
//
// A sampler picks a short sequence of timesteps and, given the UNet's noise
// prediction at each one, moves x towards x_0. Usage:
//
//     sched->set_timesteps(steps);
//     x = randn() * sched->init_noise_sigma();
//     for i in 0..steps-1:
//         xin = x; sched->scale_model_input(xin, i);
//         eps = unet(xin, sched->timesteps()[i], context);
//         sched->step(eps, i, x);
// ============================================================================
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

namespace sd {

// Base class: owns the training schedule and the chosen inference timesteps.
class Scheduler {
public:
    virtual ~Scheduler() = default;
    // Chooses `steps` timesteps from the 1000-step training schedule.
    // Must be called before any other method. Throws if steps < 1.
    virtual void set_timesteps(int steps) = 0;
    // Timestep value passed to the UNet for each inference step, from noisy
    // (high) to clean (low).
    const std::vector<float>& timesteps() const { return timesteps_; }
    // Standard deviation to multiply the initial N(0, 1) noise by.
    virtual float init_noise_sigma() const { return 1.0f; }
    // Rescales the latent before it is fed to the UNet at step `step`.
    // The identity for DDIM; see EulerScheduler for why it is needed there.
    virtual void scale_model_input(Tensor& x, int step) const { (void)x; (void)step; }
    // Advances x in place from timestep `step` to the next one, given the
    // UNet's (guided) noise prediction eps.
    virtual void step(const Tensor& eps, int step, Tensor& x) const = 0;
    // The clean latent x_0 implied by the current latent x and the noise
    // prediction eps at step `step`. Used for live previews; x is not
    // modified.
    virtual Tensor predict_x0(const Tensor& x, const Tensor& eps, int step) const = 0;

protected:
    // Builds alphas_cumprod_ for the scaled-linear schedule.
    Scheduler();
    std::vector<float> alphas_cumprod_;  // alpha_bar_t for t = 0..999
    std::vector<float> timesteps_;
};

// Euler method on the probability-flow ODE in the k-diffusion parameterisation
// (diffusers EulerDiscreteScheduler, "linspace" spacing).
//
// It works with sigma = sqrt((1 - alpha_bar) / alpha_bar) and the unscaled
// latent x = x_t / sqrt(alpha_bar), so x = x_0 + sigma * eps. Then
//     dx/dsigma = eps     and one step is   x += eps * (sigma_next - sigma).
// Because x has variance 1 + sigma^2 but the UNet was trained on x_t, the
// input is divided by sqrt(sigma^2 + 1) before each UNet call.
class EulerScheduler : public Scheduler {
public:
    // Timesteps evenly spaced from 999 to 0; sigmas linearly interpolated
    // from the training schedule at those (possibly fractional) timesteps.
    void set_timesteps(int steps) override;
    // The initial latent is pure noise at the largest sigma (about 14.6).
    float init_noise_sigma() const override { return sigmas_.front(); }
    void scale_model_input(Tensor& x, int step) const override;
    void step(const Tensor& eps, int step, Tensor& x) const override;
    Tensor predict_x0(const Tensor& x, const Tensor& eps, int step) const override;

private:
    std::vector<float> sigmas_;  // steps + 1 entries; the last one is 0
};

// Deterministic DDIM (eta = 0) with "leading" spacing and steps_offset = 1,
// matching the SD 1.x diffusers defaults. For each step:
//     x0_pred = (x_t - sqrt(1 - a_t) eps) / sqrt(a_t)
//     x_prev  = sqrt(a_prev) x0_pred + sqrt(1 - a_prev) eps
class DDIMScheduler : public Scheduler {
public:
    // Timesteps 1 + k * (1000 / steps) for k = steps-1 down to 0.
    void set_timesteps(int steps) override;
    void step(const Tensor& eps, int step, Tensor& x) const override;
    Tensor predict_x0(const Tensor& x, const Tensor& eps, int step) const override;

private:
    int step_ratio_ = 50;  // 1000 / steps (integer division)
};

// Factory: "euler" or "ddim". Throws std::runtime_error for anything else.
std::unique_ptr<Scheduler> make_scheduler(const std::string& name);

}  // namespace sd
