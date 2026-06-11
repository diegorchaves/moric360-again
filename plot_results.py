import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    default="./experiments/results.csv",
    help="Caminho para o CSV com os resultados",
)
parser.add_argument(
    "--out",
    default="./experiments/rd_curve.png",
    help="Caminho para salvar o gráfico gerado",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df = pd.read_csv(args.csv)

scenarios = [
    {"mask_type": "full", "wsmse_tag": 0},
    {"mask_type": "full", "wsmse_tag": 1},
    {"mask_type": "erp", "wsmse_tag": 0},
    {"mask_type": "erp", "wsmse_tag": 1},
]

colors = ["#4C8EF7", "#F76C6C", "#4BCB8A", "#F7A84C"]
markers = ["o", "s", "^", "D"]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

for scenario, color, marker in zip(scenarios, colors, markers):
    mask = scenario["mask_type"]
    wsmse = scenario["wsmse_tag"]

    subset = df[(df["mask_type"] == mask) & (df["wsmse_tag"] == wsmse)].copy()
    # Agrega por lambda_rate (média entre imagens caso haja mais de uma)
    subset = (
        subset.groupby("lambda_rate")[["eval_psnr", "eval_total_rate_bpp"]]
        .mean()
        .reset_index()
        .sort_values("eval_total_rate_bpp")
    )

    label = f"mask={mask}, wsmse={wsmse}"
    ax.plot(
        subset["eval_total_rate_bpp"],
        subset["eval_psnr"],
        color=color,
        marker=marker,
        markersize=7,
        linewidth=2,
        label=label,
    )

# ---------------------------------------------------------------------------
# Estética
# ---------------------------------------------------------------------------
ax.set_xlabel("Total Rate (bpp)", fontsize=12)
ax.set_ylabel("WS-PSNR (dB)", fontsize=12)
ax.set_title("Rate-Distortion Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
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
