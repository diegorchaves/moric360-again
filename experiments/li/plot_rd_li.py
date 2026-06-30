#!/usr/bin/env python3
"""
RD-curve comparison plot:
  - ours     : experiments/01/results.csv  (mask_type=erp, wsmse_tag=1, swhdc_tag=1)
  - lic360   : experiments/li/codecs_renamed.csv  (codec=lic360_777965)
  - lic_plus : experiments/li/codecs_renamed.csv  (codec=lic_plus_777974)

X-axis: mean bpp (eval_total_rate_bpp)
Y-axis: mean PSNR (eval_psnr)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── colour palette ──────────────────────────────────────────────────────────
CLR_OURS = "#2563EB"  # blue
CLR_LIC360 = "#DC2626"  # red
CLR_LICPLUS = "#16A34A"  # green
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"
PANEL_COLOR = "#F9FAFB"

# ── load data ────────────────────────────────────────────────────────────────
ours_raw = pd.read_csv(
    "/home/diego/Desktop/moric360-again/experiments/vcip_consolidated/results.csv"
)
ours = ours_raw[
    (ours_raw["mask_type"] == "erp")
    & (ours_raw["wsmse_tag"] == 1)
    & (ours_raw["swhdc_tag"] == 1)
].copy()

ref = pd.read_csv("experiments/li/codecs_renamed.csv")
lic360 = ref[ref["codec"] == "lic360_777965"].copy()
lic_plus = ref[ref["codec"] == "lic_plus_777974"].copy()

# ── aggregate: mean over images per rate point ───────────────────────────────
ours_rd = (
    ours.groupby("lambda_rate")[["eval_total_rate_bpp", "eval_psnr"]]
    .mean()
    .sort_values("eval_total_rate_bpp")
    .reset_index()
)

lic360_rd = (
    lic360.groupby("model_idx")[["bpp", "eval_psnr"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)
lic_plus_rd = (
    lic_plus.groupby("model_idx")[["bpp", "eval_psnr"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)

# ── plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#9CA3AF",
    }
)

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
ax.set_facecolor(PANEL_COLOR)

ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

MARKER_KW = dict(markersize=7, markeredgewidth=1.2, linewidth=2.0, zorder=5)

ax.plot(
    lic360_rd["bpp"],
    lic360_rd["eval_psnr"],
    color=CLR_LIC360,
    marker="s",
    markeredgecolor=CLR_LIC360,
    label="LIC360",
    **MARKER_KW,
)

ax.plot(
    lic_plus_rd["bpp"],
    lic_plus_rd["eval_psnr"],
    color=CLR_LICPLUS,
    marker="^",
    markeredgecolor=CLR_LICPLUS,
    label="LIC+",
    **MARKER_KW,
)

ax.plot(
    ours_rd["eval_total_rate_bpp"],
    ours_rd["eval_psnr"],
    color=CLR_OURS,
    marker="o",
    markeredgecolor=CLR_OURS,
    label="Ours",
    **MARKER_KW,
)

ax.set_xlabel("Rate (bpp)", fontsize=13, labelpad=8)
ax.set_ylabel("WS-PSNR (dB)", fontsize=13, labelpad=8)
ax.set_title(
    "Rate-Distortion", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=14
)

ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.tick_params(axis="both", which="both", direction="in", length=4)
ax.tick_params(which="minor", length=2)

legend = ax.legend(
    frameon=True,
    framealpha=0.95,
    facecolor="#FFFFFF",
    edgecolor="#9CA3AF",
    fontsize=11,
    loc="lower right",
    handlelength=2.2,
)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

for spine in ax.spines.values():
    spine.set_edgecolor("#9CA3AF")

plt.tight_layout(pad=1.5)
out_path = "experiments/01/rd_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved -> {out_path}")
plt.show()
