// Noise schedulers for the SD 1.x "scaled_linear" beta schedule (1000 training steps).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

namespace sd {

class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual void set_timesteps(int steps) = 0;
    // Timestep values (as passed to the UNet) for each inference step.
    const std::vector<float>& timesteps() const { return timesteps_; }
    // Standard deviation of the initial latent noise.
    virtual float init_noise_sigma() const { return 1.0f; }
    // Input scaling applied before the UNet (identity for DDIM).
    virtual void scale_model_input(Tensor& x, int step) const { (void)x; (void)step; }
    // x <- x_{t_next} given epsilon prediction at step `step`.
    virtual void step(const Tensor& eps, int step, Tensor& x) const = 0;

protected:
    Scheduler();
    std::vector<float> alphas_cumprod_;  // 1000 entries
    std::vector<float> timesteps_;
};

// Euler (k-diffusion style), "linspace" timestep spacing.
class EulerScheduler : public Scheduler {
public:
    void set_timesteps(int steps) override;
    float init_noise_sigma() const override { return sigmas_.front(); }
    void scale_model_input(Tensor& x, int step) const override;
    void step(const Tensor& eps, int step, Tensor& x) const override;

private:
    std::vector<float> sigmas_;  // steps + 1 entries, trailing 0
};

// DDIM with eta = 0, "leading" spacing and steps_offset = 1 (SD 1.x defaults).
class DDIMScheduler : public Scheduler {
public:
    void set_timesteps(int steps) override;
    void step(const Tensor& eps, int step, Tensor& x) const override;

private:
    int step_ratio_ = 50;
};

std::unique_ptr<Scheduler> make_scheduler(const std::string& name);

}  // namespace sd
