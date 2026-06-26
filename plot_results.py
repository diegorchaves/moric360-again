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
    default="/home/diego/Desktop/moric360-again/experiments/erp_padding_test/results_combined.csv",
)
parser.add_argument(
    "--out",
    default="/home/diego/Desktop/moric360-again/experiments/erp_padding_test/rd_curve.png",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df = pd.read_csv(args.csv)

# Todos os 16 cenários expandidos
scenarios = [
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 1, "erp_padding": 0},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 1, "erp_padding": 0},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 1, "erp_padding": 0},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 1, "erp_padding": 0},
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 0, "erp_padding": 0},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 0, "erp_padding": 0},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 0, "erp_padding": 0},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 0, "erp_padding": 0},
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 1, "erp_padding": 1},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 1, "erp_padding": 1},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 1, "erp_padding": 1},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 1, "erp_padding": 1},
    {"mask_type": "full", "wsmse_tag": 0, "swhdc_tag": 0, "erp_padding": 1},
    {"mask_type": "full", "wsmse_tag": 1, "swhdc_tag": 0, "erp_padding": 1},
    {"mask_type": "erp", "wsmse_tag": 0, "swhdc_tag": 0, "erp_padding": 1},
    {"mask_type": "erp", "wsmse_tag": 1, "swhdc_tag": 0, "erp_padding": 1},
]

# Base de 8 cores (Claras para swhdc=1, Escuras para swhdc=0)
base_colors = [
    "#4C8EF7",
    "#F76C6C",
    "#4BCB8A",
    "#F7A84C",  # swhdc=1
    "#1A4FA0",
    "#A01A1A",
    "#1A7A4A",
    "#A06010",  # swhdc=0
]
# Duplicamos as cores e marcadores para os 16 cenários
colors = base_colors * 2
markers = ["o", "s", "^", "D", "o", "s", "^", "D"] * 2

# Linha contínua para os primeiros 8 (pad=0), tracejada para os últimos 8 (pad=1)
linestyles = ["-"] * 8 + ["--"] * 8

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
# Aumentamos um pouco a largura do gráfico para a legenda não esmagar as curvas
fig, ax = plt.subplots(figsize=(12, 7))

legend_handles = []

for scenario, color, marker, ls in zip(scenarios, colors, markers, linestyles):
    mask = scenario["mask_type"]
    wsmse = scenario["wsmse_tag"]
    swhdc = scenario["swhdc_tag"]
    erp_pad = scenario["erp_padding"]

    # Filtro completo com as 4 tags requisitadas
    subset = df[
        (df["mask_type"] == mask)
        & (df["wsmse_tag"] == wsmse)
        & (df["swhdc_tag"] == swhdc)
        & (df["erp_padding"] == erp_pad)
    ].copy()

    if subset.empty:
        print(
            f"[aviso] Sem dados para mask={mask}, wsmse={wsmse}, swhdc={swhdc}, pad={erp_pad}"
        )
        continue

    subset = (
        subset.groupby("lambda_rate")[["eval_psnr", "eval_total_rate_bpp"]]
        .mean()
        .reset_index()
        .sort_values("eval_total_rate_bpp")
    )

    label = f"mask={mask}, wsmse={wsmse}, swhdc={swhdc}, pad={erp_pad}"

    ax.plot(
        subset["eval_total_rate_bpp"],
        subset["eval_psnr"],
        color=color,
        marker=marker,
        linestyle=ls,
        markersize=6,  # Reduzido levemente para evitar poluição visual com 16 curvas
        linewidth=1.8,
        label=label,
    )

    # Custom handle para a legenda refletir o estilo de linha correto
    handle = mlines.Line2D(
        [],
        [],
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=1.8,
        markersize=6,
        label=label,
    )
    legend_handles.append(handle)

# ---------------------------------------------------------------------------
# Estética
# ---------------------------------------------------------------------------
ax.set_xlabel("Total Rate (bpp)", fontsize=12)
ax.set_ylabel("WS-PSNR (dB)", fontsize=12)
ax.set_title("Rate-Distortion Curves", fontsize=14, fontweight="bold")

# Ajustes na legenda para comportar 16 itens sem sumir com o gráfico
ax.legend(
    handles=legend_handles,
    fontsize=8,
    framealpha=0.9,
    loc="lower right",
    ncol=2,  # Mantido em 2 colunas (ficará com 8 linhas)
    handlelength=3.5,  # Aumentado para o tracejado das linhas ficar bem nítido
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
