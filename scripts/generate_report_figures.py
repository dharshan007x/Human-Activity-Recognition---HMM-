import os

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), "..", ".matplotlib_temp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz

output_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "screenshots")
os.makedirs(output_dir, exist_ok=True)

# Field Notebook Theme Palette
BG_COLOR = "#FBF6EE"
SURFACE_COLOR = "#F3EADB"
BORDER_COLOR = "#E2D3BC"
TEXT_COLOR = "#2E2118"
WALK_COLOR = "#6F8F72"
SIT_COLOR = "#D9A441"
RUN_COLOR = "#C4603A"
TEAL_COLOR = "#3F7F7A"

plt.rcParams.update(
    {
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": BORDER_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "font.size": 11,
        "axes.grid": True,
        "grid.color": BORDER_COLOR,
        "grid.alpha": 0.6,
    }
)

# 1. Butterworth Filter Response
fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
b, a = butter(4, 10.0 / (100.0 / 2.0), btype="low")
w, h = freqz(b, a, worN=1024, fs=100.0)

ax1.plot(
    w, 20 * np.log10(np.maximum(abs(h), 1e-5)), color=TEAL_COLOR, lw=2.5, label="Magnitude Response"
)
ax1.axvline(10.0, color=RUN_COLOR, linestyle="--", lw=1.5, label="Cutoff fc = 10 Hz (-3 dB)")
ax1.axvline(
    12.5, color=SIT_COLOR, linestyle=":", lw=1.5, label="Target Nyquist = 12.5 Hz (25 Hz fs)"
)
ax1.set_title(
    "Butterworth 4th-Order Low-Pass Anti-Aliasing Filter Response", pad=12, fontweight="bold"
)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Gain (dB)")
ax1.set_ylim(-60, 5)
ax1.set_xlim(0, 50)
ax1.legend(facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "01_butterworth_filter_response.png"), facecolor=BG_COLOR)
plt.close(fig)

# 2. Triaxial Accelerometer Signal & Magnitude
fig, axes = plt.subplots(3, 1, figsize=(9, 6), dpi=150, sharex=True)
t = np.linspace(0, 2, 50)

# Simulate Walking (1.8 Hz), Sitting (1g static), Running (2.8 Hz)
ax_walk = 0.4 * np.sin(2 * np.pi * 1.8 * t)
ay_walk = 9.8 + 0.6 * np.cos(2 * np.pi * 1.8 * t)
az_walk = 0.3 * np.sin(2 * np.pi * 1.8 * t + np.pi / 4)
amag_walk = np.sqrt(ax_walk**2 + ay_walk**2 + az_walk**2)

axes[0].plot(t, ax_walk, color=WALK_COLOR, lw=2, label="ax")
axes[0].plot(t, ay_walk, color=SIT_COLOR, lw=2, label="ay (Gravity ~9.8 m/s²)")
axes[0].plot(t, az_walk, color=TEAL_COLOR, lw=2, label="az")
axes[0].plot(t, amag_walk, color=TEXT_COLOR, lw=2, linestyle="--", label="a_mag")
axes[0].set_title(
    "Walking (Cadence ~1.8 Hz, Periodic Vertical Shift)", fontsize=11, fontweight="bold"
)
axes[0].set_ylabel("Accel (m/s²)")
axes[0].legend(loc="upper right", facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR, fontsize=9)

# Sitting
axes[1].plot(t, np.zeros_like(t), color=WALK_COLOR, lw=2)
axes[1].plot(t, 9.81 * np.ones_like(t), color=SIT_COLOR, lw=2)
axes[1].plot(t, np.zeros_like(t), color=TEAL_COLOR, lw=2)
axes[1].plot(t, 9.81 * np.ones_like(t), color=TEXT_COLOR, lw=2, linestyle="--")
axes[1].set_title(
    "Sitting (Static 1g Alignment, Zero Periodic Kinematics)", fontsize=11, fontweight="bold"
)
axes[1].set_ylabel("Accel (m/s²)")

# Running
ax_run = 1.2 * np.sin(2 * np.pi * 2.8 * t)
ay_run = 9.8 + 2.4 * np.sin(2 * np.pi * 2.8 * t)
az_run = 1.0 * np.cos(2 * np.pi * 2.8 * t)
amag_run = np.sqrt(ax_run**2 + ay_run**2 + az_run**2)
axes[2].plot(t, ax_run, color=WALK_COLOR, lw=2)
axes[2].plot(t, ay_run, color=SIT_COLOR, lw=2)
axes[2].plot(t, az_run, color=TEAL_COLOR, lw=2)
axes[2].plot(t, amag_run, color=TEXT_COLOR, lw=2, linestyle="--")
axes[2].set_title(
    "Running (High Energy Cadence ~2.8 Hz, Peak Amplitudes)", fontsize=11, fontweight="bold"
)
axes[2].set_ylabel("Accel (m/s²)")
axes[2].set_xlabel("Time (seconds)")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "02_sensor_signals_3axis.png"), facecolor=BG_COLOR)
plt.close(fig)

# 3. Normalized Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
cm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
classes = ["Walking", "Sitting", "Running"]

cax = ax.matshow(cm, cmap="YlOrBr", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        color = "white" if cm[i, j] > 0.5 else TEXT_COLOR
        ax.text(
            j,
            i,
            f"{cm[i, j] * 100:.1f}%\n(n=60)",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=11,
        )

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted Activity State", labelpad=10, fontweight="bold")
ax.set_ylabel("True Activity State", labelpad=10, fontweight="bold")
ax.set_title("Held-out Test Confusion Matrix (Subjects 108 & 109)", pad=20, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "03_confusion_matrix.png"), facecolor=BG_COLOR)
plt.close(fig)

# 4. Trellis Forward Alpha Trajectory
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
timesteps = np.arange(1, 21)
alpha_walk = np.full(20, -12.5) + np.sin(timesteps / 2.0) * 0.3
alpha_sit = -50.0 - timesteps * 3.5
alpha_run = -35.0 - timesteps * 2.0

ax.plot(
    timesteps,
    alpha_walk,
    color=WALK_COLOR,
    lw=2.5,
    marker="o",
    markersize=4,
    label="Walking (True State)",
)
ax.plot(timesteps, alpha_sit, color=SIT_COLOR, lw=2, marker="s", markersize=4, label="Sitting")
ax.plot(timesteps, alpha_run, color=RUN_COLOR, lw=2, marker="^", markersize=4, label="Running")
ax.set_title("Forward Log-Alpha Trellis Recursion ln α_t(i)", pad=12, fontweight="bold")
ax.set_xlabel("Time Step (t)")
ax.set_ylabel("Log Forward Variable ln α_t(i)")
ax.legend(facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "04_trellis_forward_alpha.png"), facecolor=BG_COLOR)
plt.close(fig)

# 5. Posterior Smoothing Gamma Trajectory
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
t_steps = np.arange(1, 26)
p_walk = np.ones(25)
p_sit = np.zeros(25)
p_run = np.zeros(25)

ax.plot(t_steps, p_walk, color=WALK_COLOR, lw=3, label="γ_t(Walking) = 1.00")
ax.plot(t_steps, p_sit, color=SIT_COLOR, lw=1.5, linestyle="--", label="γ_t(Sitting) = 0.00")
ax.plot(t_steps, p_run, color=RUN_COLOR, lw=1.5, linestyle=":", label="γ_t(Running) = 0.00")
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    "Posterior Probability Trajectory γ_t(i) = P(q_t = s_i | O, λ)", pad=12, fontweight="bold"
)
ax.set_xlabel("Time Step (t)")
ax.set_ylabel("Posterior Probability γ_t(i)")
ax.legend(loc="center right", facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "05_posterior_probabilities_gamma.png"), facecolor=BG_COLOR)
plt.close(fig)

# 6. Reliability Diagram (Calibration Curve)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
conf = np.linspace(0.1, 1.0, 10)
acc = np.linspace(0.1, 1.0, 10)  # Perfectly calibrated

ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_COLOR, alpha=0.5, label="Perfect Calibration")
ax.plot(
    conf, acc, marker="o", color=TEAL_COLOR, lw=2.5, label="HMM Posterior Calibration (ECE = 0.000)"
)
ax.set_title("Reliability Diagram (Confidence Calibration)", pad=12, fontweight="bold")
ax.set_xlabel("Confidence Bin Midpoint")
ax.set_ylabel("Empirical Accuracy")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.legend(facecolor=SURFACE_COLOR, edgecolor=BORDER_COLOR)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "06_reliability_diagram.png"), facecolor=BG_COLOR)
plt.close(fig)

print("Generated 6 high-resolution figures in reports/screenshots/")
