#!/usr/bin/env python3
"""P2 figures (WO-TBM-P2, diagnosis only).

Fig A: episode paths on the grid (wall, door, start, goal) for four
episodes — two slow cases, two controls.
Fig B: greedy ticks vs start phase, slow vs control pairs.
Fig C: median phase decomposition (approach / staging / escape),
stacked bars for all 12 traced pairs.
Reads nn/traces_p2.jsonl and nn/p2_analysis.json. Additive only.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_HERE, "figures_p2")
os.makedirs(FIGDIR, exist_ok=True)

eps = {}
for line in open(os.path.join(_HERE, "traces_p2.jsonl")):
    e = json.loads(line)
    eps[(e["arm"], e["seed"], e["period"], e["start_phase"])] = e
ana = json.load(open(os.path.join(_HERE, "p2_analysis.json")))["pairs"]


def draw_path(ax, arm, seed, period, phase, title):
    e = eps[(arm, seed, period, phase)]
    steps = e["steps"]
    xs = [s["x"] for s in steps] + [9]
    ys = [s["y"] for s in steps] + [9]
    # Wall column x=5 except door (5,5).
    for y in range(10):
        if y != 5:
            ax.add_patch(plt.Rectangle((4.5, y - 0.5), 1, 1,
                                       facecolor="#444444"))
    ax.plot(xs, ys, "-o", ms=3, lw=1.2, color="#1f77b4")
    ax.plot([0], [0], "gs", ms=10, label="start")
    ax.plot([5], [5], "w*", ms=16, markeredgecolor="black",
            label="door")
    ax.plot([9], [9], "r*", ms=16, label="goal")
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(9.7, -0.7)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(True, lw=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
draw_path(axes[0, 0], "oracle_ind", 101, 30, 8,
          "oracle_ind/101 ph8: 58 ticks\nspawn freeze + escape cycle")
draw_path(axes[0, 1], "oracle_ind", 102, 30, 0,
          "CONTROL oracle_ind/102 ph0: 18 ticks")
draw_path(axes[1, 0], "wrapped_ind", 108, 20, 14,
          "wrapped/108 ph14: 39 ticks\n10-tick stay at (1,5)")
draw_path(axes[1, 1], "wrapped_ind", 109, 20, 0,
          "wrapped/109 ph0: 41 ticks\nstaging + escape bounce")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figA_paths.png"), dpi=120)
plt.close(fig)


def phase_profile(ax, pairs, period, title):
    for arm, seed, style in pairs:
        rows = [(ph, eps[(arm, seed, period, ph)]["ticks"])
                for ph in range(period)]
        ax.plot([r[0] for r in rows], [r[1] for r in rows], style,
                label="%s/%d" % (arm, seed))
    ax.set_xlabel("start phase")
    ax.set_ylabel("greedy ticks to goal")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
phase_profile(axes[0], [("oracle_ind", 101, "r-"),
                        ("oracle_new", 101, "m-"),
                        ("phase_only_new", 101, "y-"),
                        ("wrapped_ind", 101, "g-"),
                        ("oracle_ind", 102, "b-")], 30, "30/14")
phase_profile(axes[1], [("wrapped_ind", 108, "r-"),
                        ("wrapped_ind", 109, "m-"),
                        ("wrapped_ind", 106, "y-"),
                        ("wrapped_ind", 101, "c-"),
                        ("wrapped_ind", 102, "g-"),
                        ("wrapped_ind", 104, "b-")], 20, "20/10")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figB_phase_profiles.png"), dpi=120)
plt.close(fig)


labels = []
app, stg, esc = [], [], []
for key in sorted(ana):
    labels.append(key)
    app.append(ana[key]["approach"]["median"])
    stg.append(ana[key]["staging"]["median"])
    esc.append(ana[key]["escape"]["median"])
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(labels))
ax.bar(x, app, label="approach")
ax.bar(x, stg, bottom=app, label="staging wait")
ax.bar(x, esc, bottom=np.array(app) + np.array(stg), label="escape")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("median ticks")
ax.set_title("Median phase decomposition (approach / staging / escape)",
             fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figC_decomposition.png"), dpi=120)
plt.close(fig)
print("figures in %s" % FIGDIR)
