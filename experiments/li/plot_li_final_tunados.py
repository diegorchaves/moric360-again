#!/usr/bin/env python3
"""
RD-curve comparison plot (PSNR vs SSIM):
  - ours (PSNR) : Tuned for WS-PSNR
  - ours (SSIM) : Tuned for WS-SSIM
  - lic360      : experiments/li/codecs_renamed.csv  (codec=lic360_777965)
  - lic_plus    : experiments/li/codecs_renamed.csv  (codec=lic_plus_777974)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── CONFIGURAÇÃO DE CAMINHOS ─────────────────────────────────────────────────
PATH_OURS_PSNR = (
    "/home/diego/Desktop/moric360-again/experiments/vcip_consolidated/results.csv"
)
PATH_OURS_SSIM = "/home/diego/Desktop/moric360-again/experiments/vcip_wssim_tunado_diegoPC/results.csv"
PATH_REF = "experiments/li/codecs_renamed.csv"

# ── PALETA DE CORES E ESTILOS ────────────────────────────────────────────────
CLR_OURS = "#2563EB"  # Azul
CLR_LIC360 = "#DC2626"  # Vermelho
CLR_LICPLUS = "#16A34A"  # Verde
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"
PANEL_COLOR = "#F9FAFB"

MARKER_KW = dict(markersize=7, markeredgewidth=1.2, linewidth=2.0, zorder=5)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#9CA3AF",
        "xtick.labelsize": 20,  # Tamanho dos números no eixo X
        "ytick.labelsize": 20,  # Tamanho dos números no eixo Y
    }
)

# ── CARREGAMENTO E FILTRAGEM DOS DADOS ───────────────────────────────────────
# 1. Referências (LIC)
ref = pd.read_csv(PATH_REF)
lic360 = ref[ref["codec"] == "lic360_777965"].copy()
lic_plus = ref[ref["codec"] == "lic_plus_777974"].copy()


# 2. Ours - Tunado para PSNR
ours_psnr_raw = pd.read_csv(PATH_OURS_PSNR)
ours_psnr = ours_psnr_raw[
    (ours_psnr_raw["mask_type"] == "erp") & (ours_psnr_raw["swhdc_tag"] == 1)
].copy()

# 3. Ours - Tunado para SSIM
ours_ssim_raw = pd.read_csv(PATH_OURS_SSIM)
ours_ssim = ours_ssim_raw[
    (ours_ssim_raw["mask_type"] == "erp") & (ours_ssim_raw["swhdc_tag"] == 1)
].copy()

# ── AGREGAÇÃO (MÉDIA POR RATE POINT) ─────────────────────────────────────────
# Agrupamento para PSNR (Ambos usam 'eval_psnr')
ours_psnr_rd = (
    ours_psnr.groupby("lambda_rate")[["eval_total_rate_bpp", "eval_psnr"]]
    .mean()
    .sort_values("eval_total_rate_bpp")
    .reset_index()
)
lic360_psnr_rd = (
    lic360.groupby("model_idx")[["bpp", "eval_psnr"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)
lic_plus_psnr_rd = (
    lic_plus.groupby("model_idx")[["bpp", "eval_psnr"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)

lic_plus_psnr_rd = lic_plus_psnr_rd[lic_plus_psnr_rd["bpp"] > 0.1]
ours_psnr_rd = ours_psnr_rd[ours_psnr_rd["eval_total_rate_bpp"] < 0.7]

# Agrupamento para SSIM (Ours usa 'eval_wsssim' e LIC usa 'ws_ssim')
ours_ssim_rd = (
    ours_ssim.groupby("lambda_rate")[["eval_total_rate_bpp", "eval_wsssim"]]
    .mean()
    .sort_values("eval_total_rate_bpp")
    .reset_index()
)
lic360_ssim_rd = (
    lic360.groupby("model_idx")[["bpp", "ws_ssim"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)
lic_plus_ssim_rd = (
    lic_plus.groupby("model_idx")[["bpp", "ws_ssim"]]
    .mean()
    .sort_values("bpp")
    .reset_index()
)

lic_plus_ssim_rd = lic_plus_ssim_rd[lic_plus_ssim_rd["bpp"] > 0.1]
ours_ssim_rd = ours_ssim_rd[ours_ssim_rd["eval_total_rate_bpp"] < 0.7]


# ── FUNÇÃO AUXILIAR PARA PLOTAGEM DO PAINEL ──────────────────────────────────
def make_panel(
    ax,
    ours_rd,
    lic360_rd,
    lic_plus_rd,
    x_ours_col,
    x_ref_col,
    y_ours_col,
    y_ref_col,
    y_label,
    title,
):
    ax.set_facecolor(PANEL_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Baselines (LIC) usando y_ref_col
    ax.plot(
        lic360_rd[x_ref_col],
        lic360_rd[y_ref_col],
        color=CLR_LIC360,
        marker="s",
        markeredgecolor=CLR_LIC360,
        label="Li et al. [8]",
        **MARKER_KW,
    )
    ax.plot(
        lic_plus_rd[x_ref_col],
        lic_plus_rd[y_ref_col],
        color=CLR_LICPLUS,
        marker="^",
        markeredgecolor=CLR_LICPLUS,
        label="Li et al. [9]",
        **MARKER_KW,
    )

    # Nosso modelo usando y_ours_col
    ax.plot(
        ours_rd[x_ours_col],
        ours_rd[y_ours_col],
        color=CLR_OURS,
        marker="o",
        markeredgecolor=CLR_OURS,
        label="ZOC-360",
        **MARKER_KW,
    )

    # Estilização de eixos e títulos
    ax.set_xlabel("Rate (bpp)", fontsize=30, labelpad=8)
    ax.set_ylabel(y_label, fontsize=30, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=14)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="both", which="both", direction="in", length=4)
    ax.tick_params(which="minor", length=2)

    legend = ax.legend(
        frameon=True,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#9CA3AF",
        fontsize=30,
        loc="lower right",
        handlelength=2.2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_edgecolor("#9CA3AF")


# ── RENDERIZAÇÃO E SALVAMENTO DOS GRAFICOS ───────────────────────────────────

# 1. Gráfico Separado: WS-PSNR
fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
make_panel(
    ax1,
    ours_psnr_rd,
    lic360_psnr_rd,
    lic_plus_psnr_rd,
    "eval_total_rate_bpp",
    "bpp",
    "eval_psnr",
    "eval_psnr",
    "WS-PSNR (dB)",
    "",
)
plt.tight_layout(pad=1.5)
out_psnr = "experiments/01/rd_comparison_psnr.pdf"
plt.savefig(out_psnr, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"Saved -> {out_psnr}")

# 2. Gráfico Separado: WS-SSIM
fig2, ax2 = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
# Aqui passamos 'eval_wsssim' para o nosso e 'ws_ssim' para as referências
make_panel(
    ax2,
    ours_ssim_rd,
    lic360_ssim_rd,
    lic_plus_ssim_rd,
    "eval_total_rate_bpp",
    "bpp",
    "eval_wsssim",
    "ws_ssim",
    "WS-SSIM",
    "",
)
plt.tight_layout(pad=1.5)
out_ssim = "experiments/01/rd_comparison_ssim.pdf"
plt.savefig(out_ssim, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"Saved -> {out_ssim}")

# 3. Gráfico Combinado (Lado a Lado)
fig_comb, (ax_comb1, ax_comb2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR)
make_panel(
    ax_comb1,
    ours_psnr_rd,
    lic360_psnr_rd,
    lic_plus_psnr_rd,
    "eval_total_rate_bpp",
    "bpp",
    "eval_psnr",
    "eval_psnr",
    "WS-PSNR (dB)",
    "",
)
make_panel(
    ax_comb2,
    ours_ssim_rd,
    lic360_ssim_rd,
    lic_plus_ssim_rd,
    "eval_total_rate_bpp",
    "bpp",
    "eval_wsssim",
    "ws_ssim",
    "WS-SSIM",
    "",
)
plt.tight_layout(pad=1.5)
out_combined = "experiments/01/rd_comparison_combined.pdf"
plt.savefig(out_combined, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved -> {out_combined}")

# Mostra o resultado combinado em tela no final
plt.show()
