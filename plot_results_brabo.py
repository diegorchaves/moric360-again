import argparse
import colorsys
import itertools
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BASE_DIR = (
    "/home/diego/Desktop/moric360-again/experiments/vcip_combined_loss_busca_lambdas/"
)

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=f"{BASE_DIR}/results.csv")
parser.add_argument(
    "--out-psnr",
    default=f"{BASE_DIR}/rd_curve_psnr.png",
    help="Caminho de saida do grafico de WS-PSNR",
)
parser.add_argument(
    "--out-wsssim",
    default=f"{BASE_DIR}/rd_curve_wsssim.png",
    help="Caminho de saida do grafico de WS-SSIM",
)
parser.add_argument(
    "--out-combined",
    default=f"{BASE_DIR}/rd_curve_combined.png",
    help="Caminho de saida do grafico combinado (PSNR + WS-SSIM lado a lado)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df = pd.read_csv(args.csv)

# Descobrimos dinamicamente os valores presentes no CSV, ao invés de fixar
# uma lista de cenarios. Isso permite qualquer numero de loss_type.
mask_types = sorted(df["mask_type"].dropna().unique().tolist())
loss_types = sorted(df["loss_type"].dropna().unique().tolist())
swhdc_tags = sorted(df["swhdc_tag"].dropna().unique().tolist())
erp_paddings = sorted(df["erp_padding"].dropna().unique().tolist())
lambda_dist = sorted(df["lambda_dist"].dropna().unique().tolist())

scenarios = [
    {"mask_type": m, "loss_type": lt, "swhdc_tag": s, "erp_padding": p}
    for m, lt, s, p in itertools.product(
        mask_types, loss_types, swhdc_tags, erp_paddings
    )
]

# ---------------------------------------------------------------------------
# Estilo visual
#   - cor      -> loss_type (uma cor base por tipo de loss)
#   - tom      -> swhdc_tag (mais claro = swhdc=1 / valor maior; mais escuro = swhdc=0)
#   - marcador -> mask_type
#   - linha    -> erp_padding (solida = 0 / valor menor; tracejada = 1 / valor maior)
# ---------------------------------------------------------------------------
marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
mask_markers = {
    m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(mask_types)
}

linestyle_cycle = ["-", "--", ":", "-."]
pad_linestyles = {
    p: linestyle_cycle[i % len(linestyle_cycle)] for i, p in enumerate(erp_paddings)
}

cmap_name = "tab10" if len(loss_types) <= 10 else "tab20"
try:
    # matplotlib >= 3.9
    cmap = matplotlib.colormaps[cmap_name]
except AttributeError:
    # matplotlib < 3.9
    cmap = cm.get_cmap(cmap_name)
n_loss = max(len(loss_types), 1)
loss_base_colors = {
    lt: cmap((i % cmap.N) / max(n_loss, 1)) for i, lt in enumerate(loss_types)
}


def shade_color(rgba, factor):
    """Clareia (factor > 1) ou escurece (factor < 1) uma cor RGBA."""
    r, g, b = rgba[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


# Fatores de tonalidade: o maior swhdc_tag fica mais claro, o menor mais escuro.
n_swhdc = max(len(swhdc_tags), 1)
if n_swhdc == 1:
    shade_factors = [1.0]
else:
    shade_factors = [1.35 - i * (0.75 / (n_swhdc - 1)) for i in range(n_swhdc)]
# swhdc_tags esta em ordem crescente; queremos o maior valor mais claro,
# entao invertemos a lista de fatores.
swhdc_shade = dict(zip(swhdc_tags, reversed(shade_factors)))

# Estilo (cor/marker/linestyle/label) fixo por cenario, compartilhado entre
# os graficos de PSNR e WS-SSIM para que as curvas sejam comparaveis.
scenario_styles = []
for scenario in scenarios:
    mask = scenario["mask_type"]
    loss = scenario["loss_type"]
    swhdc = scenario["swhdc_tag"]
    erp_pad = scenario["erp_padding"]

    scenario_styles.append(
        {
            **scenario,
            "color": shade_color(loss_base_colors[loss], swhdc_shade[swhdc]),
            "marker": mask_markers[mask],
            "linestyle": pad_linestyles[erp_pad],
            "label": f"mask={mask}, loss={loss}, swhdc={swhdc}, pad={erp_pad}",
        }
    )

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Funcao auxiliar: desenha as curvas de um eixo Y (eval_psnr ou eval_wsssim)
# em um dado Axes
# ---------------------------------------------------------------------------
def draw_rd_curves(ax, y_col, y_label, y_fmt="%.1f"):
    legend_handles = []

    for style in scenario_styles:
        mask = style["mask_type"]
        loss = style["loss_type"]
        swhdc = style["swhdc_tag"]
        erp_pad = style["erp_padding"]

        subset = df[
            (df["mask_type"] == mask)
            & (df["loss_type"] == loss)
            & (df["swhdc_tag"] == swhdc)
            & (df["erp_padding"] == erp_pad)
        ].copy()

        if subset.empty:
            print(
                f"[aviso] Sem dados para mask={mask}, loss={loss}, swhdc={swhdc}, pad={erp_pad}"
            )
            continue

        subset = (
            subset.groupby("lambda_rate")[[y_col, "eval_total_rate_bpp"]]
            .mean()
            .reset_index()
            .sort_values("eval_total_rate_bpp")
        )

        ax.plot(
            subset["eval_total_rate_bpp"],
            subset[y_col],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=6,
            linewidth=1.8,
            label=style["label"],
        )

        handle = mlines.Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            label=style["label"],
        )
        legend_handles.append(handle)

    ax.set_xlabel("Total Rate (bpp)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

    return legend_handles


def add_legend(ax, legend_handles, ncol=None):
    n_curves = len(legend_handles)
    if ncol is None:
        ncol = 2 if n_curves <= 16 else 3
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        framealpha=0.9,
        loc="lower right",
        ncol=ncol,
        handlelength=3.5,
    )


def save_fig(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Grafico salvo em: {out_path}")


# ---------------------------------------------------------------------------
# Grafico 1: WS-PSNR
# ---------------------------------------------------------------------------
fig_psnr, ax_psnr = plt.subplots(figsize=(12, 7))
handles_psnr = draw_rd_curves(ax_psnr, "eval_psnr", "WS-PSNR (dB)", y_fmt="%.1f")
ax_psnr.set_title("Rate-Distortion Curves (WS-PSNR)", fontsize=14, fontweight="bold")
add_legend(ax_psnr, handles_psnr)
fig_psnr.tight_layout()
save_fig(fig_psnr, args.out_psnr)

# ---------------------------------------------------------------------------
# Grafico 2: WS-SSIM
# ---------------------------------------------------------------------------
fig_ssim, ax_ssim = plt.subplots(figsize=(12, 7))
handles_ssim = draw_rd_curves(ax_ssim, "eval_wsssim", "WS-SSIM", y_fmt="%.3f")
ax_ssim.set_title("Rate-Distortion Curves (WS-SSIM)", fontsize=14, fontweight="bold")
add_legend(ax_ssim, handles_ssim)
fig_ssim.tight_layout()
save_fig(fig_ssim, args.out_wsssim)

# ---------------------------------------------------------------------------
# Grafico 3: combinado, lado a lado, com legenda unica compartilhada
# ---------------------------------------------------------------------------
n_curves = max(len(handles_psnr), len(handles_ssim))
ncol_combined = min(n_curves, 4) if n_curves > 0 else 1

fig_combined, (ax_c_psnr, ax_c_ssim) = plt.subplots(1, 2, figsize=(20, 8))

handles_c_psnr = draw_rd_curves(ax_c_psnr, "eval_psnr", "WS-PSNR (dB)", y_fmt="%.1f")
ax_c_psnr.set_title("WS-PSNR", fontsize=13, fontweight="bold")

handles_c_ssim = draw_rd_curves(ax_c_ssim, "eval_wsssim", "WS-SSIM", y_fmt="%.3f")
ax_c_ssim.set_title("WS-SSIM", fontsize=13, fontweight="bold")

fig_combined.suptitle("Rate-Distortion Curves", fontsize=15, fontweight="bold")

# Legenda unica, compartilhada pelas duas curvas, abaixo dos dois graficos
legend_source = handles_c_psnr if handles_c_psnr else handles_c_ssim
fig_combined.legend(
    handles=legend_source,
    fontsize=8,
    framealpha=0.9,
    loc="lower center",
    ncol=ncol_combined,
    handlelength=3.5,
    bbox_to_anchor=(0.5, -0.02),
)

fig_combined.tight_layout(rect=[0, 0.08, 1, 0.96])
save_fig(fig_combined, args.out_combined)

plt.show()
