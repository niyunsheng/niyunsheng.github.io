"""Figures + reference numbers for the RoPE / Timestep-Embedding post.

Everything here assumes the classic setup: dim = 128, base = 10000,
so we get m = 64 two-dimensional groups with omega_i = base ** (-i / m).
"""

import math

import matplotlib.pyplot as plt
import numpy as np

DIM = 128
BASE = 10000.0
M = DIM // 2

i = np.arange(M)
omega = BASE ** (-i / M)
period = 2 * np.pi / omega

# Timestep embeddings are usually 256-dim (HunyuanVideo, Qwen-Image, Sana, ...),
# while RoPE runs on a 128-dim head. Same ladder, sampled twice as densely.
TS_DIM = 256
TS_M = TS_DIM // 2
ts_omega = BASE ** (-np.arange(TS_M) / TS_M)


def print_frequency_table():
    """The representative rows quoted in the post."""
    print(f"{'i':>3} {'omega (rad/step)':>18} {'deg/step':>12} {'period T_i':>14}")
    for idx in [0, 1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 56, 63]:
        w = omega[idx]
        print(f"{idx:>3} {w:>18.9f} {math.degrees(w):>12.6f} {2 * np.pi / w:>14.3f}")


def print_step_delta():
    """Section 5 numbers: the 256-dim timestep embedding."""
    print(f"\n[timestep embedding, dim={TS_DIM}, m={TS_M}]")
    print(f"{'i':>4} {'omega':>14} {'|v_i(1)-v_i(0)|':>18} {'phase over 999 steps':>22}")
    for idx in [0, 32, 64, 96, 127]:
        w = ts_omega[idx]
        print(f"{idx:>4} {w:>14.9f} {2 * abs(math.sin(w / 2)):>18.9f} {999 * w:>19.3f} rad")

    e = _embed(np.array([0.0, 1.0]))
    e0, e1 = e[0], e[1]
    n = np.linalg.norm(e0)
    print(f"\n||e(t)||        = {n:.6f}  (= sqrt({TS_M}))")
    print(f"||e(1) - e(0)|| = {np.linalg.norm(e1 - e0):.6f}")
    print(f"cos_sim(e0, e1) = {e0 @ e1 / n ** 2:.6f}")


def plot_frequency_ladder(path="freq_ladder.png"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(i, omega, marker="o", ms=3, color="tab:blue", label=r"$\omega_i$ (rad/step)")
    ax.set_xlabel("frequency group index $i$")
    ax.set_ylabel(r"angular frequency $\omega_i$ [rad/step]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.semilogy(i, period, marker="s", ms=3, color="tab:red", label=r"$T_i$ (steps)")
    ax2.set_ylabel(r"period $T_i = 2\pi/\omega_i$ [steps]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    for idx, note in [(0, "fastest group\n6.3 steps/turn"), (63, "slowest group\n54410 steps/turn")]:
        ax.annotate(
            note,
            xy=(idx, omega[idx]),
            xytext=(idx + (6 if idx == 0 else -20), omega[idx] * (0.05 if idx == 0 else 20)),
            arrowprops=dict(arrowstyle="->", alpha=0.6),
            fontsize=9,
        )

    ax.set_title(f"The geometric frequency ladder (dim={DIM}, base={int(BASE)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase_circles(path="phase_circles.png"):
    """Same coordinate x, different groups: who has moved, who has not."""
    xs = [0, 1, 10, 100, 1000]
    groups = [0, 16, 32, 63]
    fig, axes = plt.subplots(1, len(groups), figsize=(13, 3.4), subplot_kw={"aspect": "equal"})
    circle = np.linspace(0, 2 * np.pi, 200)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(xs)))

    for ax, g in zip(axes, groups):
        ax.plot(np.cos(circle), np.sin(circle), color="0.85", lw=1)
        for x, c in zip(xs, colors):
            th = x * omega[g]
            ax.plot([0, np.cos(th)], [0, np.sin(th)], color=c, lw=2, label=f"x={x}")
        ax.set_title(rf"group $i$={g}" "\n" rf"$\omega$={omega[g]:.2e}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=9)
    fig.suptitle("One coordinate, many groups: phase $x\\omega_i$ on the unit circle", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _embed(ts, om=ts_omega):
    return np.concatenate(
        [np.cos(ts[:, None] * om[None, :]), np.sin(ts[:, None] * om[None, :])], axis=-1
    )


def _sim_to_zero(ts, om=ts_omega):
    """cosine similarity between e(t) and e(0)."""
    emb = _embed(ts, om)
    e0 = _embed(np.zeros(1), om)[0]
    return emb.dot(e0) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(e0))


def plot_timestep_heatmap(path="timestep_heatmap.png"):
    """The full schedule as a heatmap, with a few rows drawn as raw waveforms below."""
    t = np.arange(1000)
    rows = [48, 64, 96]          # 在 1000 步内分别走 5.0 / 1.6 / 0.16 个周期

    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.28)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    im = ax0.imshow(_embed(t).T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest", extent=[0, 999, TS_DIM, 0])
    ax0.set_xlim(0, 1090)
    ax0.axhline(TS_M, color="k", lw=1.2)
    ax0.set_xticks([0, 200, 400, 600, 800, 999])
    ax0.set_ylabel("embedding channel\n(each half: top = fast, bottom = slow)")
    ax0.set_title(f"Each column is one {TS_DIM}-dim $e(t)$; each row is a sine of fixed period"
                  "   (max_period=10000)")
    ax0.text(1015, TS_M / 2, "$\\cos$\nhalf", va="center", fontsize=9.5)
    ax0.text(1015, TS_M + TS_M / 2, "$\\sin$\nhalf", va="center", fontsize=9.5)

    lbl = dict(fontsize=7.5, va="center", ha="center",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for k, r in enumerate(rows):
        ax0.plot([0, 999], [r, r], color="k", lw=0.9, ls=":")
        ax0.text(880, r, f"$i$={r}", **lbl)

    for k, r in enumerate(rows):
        ax1.plot(t, np.cos(t * ts_omega[r]), lw=1.5, color=f"C{k}",
                 label=f"$i$={r},  $T$={2 * np.pi / ts_omega[r]:.0f} steps")
    ax1.set_xlim(0, 1090)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_xticks([0, 200, 400, 600, 800, 999])
    ax1.set_xlabel("timestep $t$")
    ax1.set_ylabel("$\\cos(t\\omega_i)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=3, loc="lower center", framealpha=0.9)

    fig.colorbar(im, ax=[ax0, ax1], pad=0.015, fraction=0.03)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timestep_similarity(path="timestep_similarity.png"):
    """Same frequencies, two schedules: discrete 0..999 vs continuous [0, 1]."""
    t_disc = np.arange(1000)                 # 1000 integer samples, step = 1
    t_cont = np.linspace(0.0, 1.0, 2001)     # 2001 float samples, step = 5e-4
    s_disc, s_cont = _sim_to_zero(t_disc), _sim_to_zero(t_cont)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))

    axL.plot(t_disc, s_disc, color="tab:purple", lw=1)
    axL.set_title(f"Discrete schedule  $t = 0 \\ldots 999$\n{len(t_disc)} samples, step 1")
    axL.set_xlabel("timestep $t$")
    axL.set_ylabel(r"cosine similarity with $e(0)$")
    axL.set_ylim(0, 1.05)
    axL.grid(True, alpha=0.3)
    floor = s_disc[400:].mean()
    axL.axhline(floor, color="0.5", ls="--", lw=1)
    axL.annotate(f"$t$=1: {s_disc[1]:.4f}", xy=(1, s_disc[1]), xytext=(160, 0.95),
                 arrowprops=dict(arrowstyle="->", alpha=0.6), fontsize=9)
    axL.annotate(f"floor $\\approx$ {floor:.2f}: the frozen high-$i$ channels\n"
                 "never stop matching $e(0)$; the jitter above it\nis the fast channels wrapping",
                 xy=(700, floor), xytext=(330, 0.66), fontsize=8.5, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.45"),
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))

    axR.plot(t_cont, s_cont, color="tab:green", lw=1.6)
    axR.set_title(f"Continuous schedule  $t \\in [0, 1]$\n{len(t_cont)} samples, step 5e-4")
    axR.set_xlabel("timestep $t$")
    axR.set_ylabel(r"cosine similarity with $e(0)$")
    axR.grid(True, alpha=0.3)
    axR.annotate(f"$t$=1 lands at {s_cont[-1]:.4f}\n(the whole schedule moves it by {1 - s_cont[-1]:.3f})",
                 xy=(1.0, s_cont[-1]), xytext=(0.12, 0.9775),
                 arrowprops=dict(arrowstyle="->", alpha=0.7), fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))

    fig.suptitle("Same frequency table, two conventions for $t$ — note the different $y$ scales", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_base_effect(path="base_effect.png"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for b, c in [(1000.0, "tab:green"), (10000.0, "tab:blue"), (100000.0, "tab:orange")]:
        ax.semilogy(i, b ** (-i / M), color=c, lw=2, label=f"base = {int(b)}")
    ax.set_xlabel("frequency group index $i$")
    ax.set_ylabel(r"$\omega_i$ [rad/step]")
    ax.set_title(r"Changing base pivots the ladder around $\omega_0=1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print_frequency_table()
    print_step_delta()
    plot_frequency_ladder()
    plot_phase_circles()
    plot_timestep_heatmap()
    plot_timestep_similarity()
    plot_base_effect()
