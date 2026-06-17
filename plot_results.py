import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    default="/home/diego/Desktop/moric360-again/experiments_wspsnr_5_img_swhdc_100k/results.csv",
)
parser.add_argument(
    "--out",
    default="./home/diego/Desktop/moric360-again/experiments_wspsnr_5_img_swhdc_100k/rd_curve.png",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df = pd.read_csv(args.csv)

scenarios = [
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 1},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 1},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 1},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 1},
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 0},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 0},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 0},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 0},
]

colors = [
    "#4C8EF7",
    "#F76C6C",
    "#4BCB8A",
    "#F7A84C",  # swhdc=1
    "#1A4FA0",
    "#A01A1A",
    "#1A7A4A",
    "#A06010",  # swhdc=0 (darker)
]
markers = ["o", "s", "^", "D", "o", "s", "^", "D"]
linestyles = ["-", "-", "-", "-", "--", "--", "--", "--"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

legend_handles = []

for scenario, color, marker, ls in zip(scenarios, colors, markers, linestyles):
    mask = scenario["mask_type"]
    wsmse = scenario["wsmse_tag"]
    swhdc = scenario["swhdc_tag"]

    subset = df[
        (df["mask_type"] == mask)
        & (df["wsmse_tag"] == wsmse)
        & (df["swhdc_tag"] == swhdc)
    ].copy()

    if subset.empty:
        print(f"[aviso] Nenhum dado para mask={mask}, wsmse={wsmse}, swhdc={swhdc}")
        continue

    subset = (
        subset.groupby("lambda_rate")[["eval_psnr", "eval_total_rate_bpp"]]
        .mean()
        .reset_index()
        .sort_values("eval_total_rate_bpp")
    )

    label = f"mask={mask}, wsmse={wsmse}, swhdc={swhdc}"
    ax.plot(
        subset["eval_total_rate_bpp"],
        subset["eval_psnr"],
        color=color,
        marker=marker,
        linestyle=ls,
        markersize=7,
        linewidth=2,
        label=label,
    )

    # Custom handle: shows the actual linestyle + marker in the legend
    handle = mlines.Line2D(
        [],
        [],
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=2,
        markersize=7,
        label=label,
    )
    legend_handles.append(handle)

# ---------------------------------------------------------------------------
# Estética
# ---------------------------------------------------------------------------
ax.set_xlabel("Total Rate (bpp)", fontsize=12)
ax.set_ylabel("WS-PSNR (dB)", fontsize=12)
ax.set_title("Rate-Distortion Curves", fontsize=14, fontweight="bold")
ax.legend(
    handles=legend_handles,
    fontsize=9,
    framealpha=0.9,
    loc="lower right",
    ncol=2,
    handlelength=2.5,  # wider line sample so dashes are clearly visible
)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
plt.tight_layout()

# ---------------------------------------------------------------------------
# Salvar
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f"Gráfico salvo em: {args.out}")
plt.show()
