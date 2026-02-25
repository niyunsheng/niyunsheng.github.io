import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def shifted_logit_normal(t, shift=1.0):
    """
    Simulates the probability density of sampling timestep t.
    Convention: t=0 (Pure Noise), t=1 (Clean Data).
    """
    # Avoid div-by-zero at boundaries
    t = np.clip(t, 1e-5, 1 - 1e-5)
    
    # 1. Standard Logit Transform
    logit_t = np.log(t / (1 - t))
    
    # 2. Shift the distribution
    # A shift > 1 pushes density towards t=0 (Noise / Macro-structure)
    shifted_logit_t = logit_t + np.log(shift)
    
    # 3. Calculate Density (Jacobian included for the logit transform)
    pdf = norm.pdf(shifted_logit_t) / (t * (1 - t))
    return pdf

# Plotting the density
t = np.linspace(0, 1, 500)
plt.figure(figsize=(10, 5))

# Shift = 1.0 (Standard bell curve, centered)
plt.plot(t, shifted_logit_normal(t, shift=1.0), label="Shift=1.0 (Base/Low Res)", color='gray')

# Shift = 3.0 (SD3 High Res - Pushed towards noise)
plt.fill_between(t, shifted_logit_normal(t, shift=3.0), alpha=0.4, color='blue', label="Shift=3.0 (SD3 Image)")

# Shift = 7.0 (HunyuanVideo - Aggressively pushed towards noise)
plt.fill_between(t, shifted_logit_normal(t, shift=7.0), alpha=0.4, color='red', label="Shift=7.0 (Hunyuan Video)")

plt.title("Timestep Sampling Density (The 'Shift' Trick)")
plt.xlabel("Timestep (t=0: Pure Noise / Macro-structure  -->  t=1: Clean Data / Micro-details)")
plt.ylabel("Sampling Probability Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("timestep_sampling_density.png")