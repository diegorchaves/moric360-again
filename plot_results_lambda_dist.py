import argparse
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BASE_DIR = "/home/diego/Desktop/moric360-again/experiments/vcip_combined_loss_busca_lambdas_wsssimMax"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=f"{BASE_DIR}/results.csv")
parser.add_argument(
    "--out-psnr",
    default=f"{BASE_DIR}/rd_curve_lambda_dist_psnr.png",
    help="Caminho de saida do grafico de WS-PSNR",
)
parser.add_argument(
    "--out-wsssim",
    default=f"{BASE_DIR}/rd_curve_lambda_dist_wsssim.png",
    help="Caminho de saida do grafico de WS-SSIM",
)
parser.add_argument(
    "--out-combined",
    default=f"{BASE_DIR}/rd_curve_lambda_dist_combined.png",
    help="Caminho de saida do grafico combinado (PSNR + WS-SSIM lado a lado)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
df = pd.read_csv(args.csv)

# lambda_dist eh float, entao arredondamos para agrupar com seguranca
# possiveis ruidos de ponto flutuante ao ler o CSV.
df["lambda_dist_rounded"] = df["lambda_dist"].round(6)
lambda_dist_values = sorted(df["lambda_dist_rounded"].dropna().unique().tolist())

if len(lambda_dist_values) == 0:
    raise ValueError("Nenhum valor de lambda_dist encontrado no CSV.")
if len(lambda_dist_values) > 2:
    print(
        f"[aviso] Esperados 2 valores de lambda_dist, mas foram encontrados "
        f"{len(lambda_dist_values)}: {lambda_dist_values}. Todos serao plotados."
    )

# ---------------------------------------------------------------------------
# Estilo visual: uma cor + marcador distintos por valor de lambda_dist
# ---------------------------------------------------------------------------
color_cycle = ["#4C8EF7", "#F76C6C", "#4BCB8A", "#F7A84C", "#8E5CD9", "#20C0C0"]
marker_cycle = ["o", "s", "^", "D", "v", "P"]

lambda_dist_styles = {
    val: {
        "color": color_cycle[i % len(color_cycle)],
        "marker": marker_cycle[i % len(marker_cycle)],
    }
    for i, val in enumerate(lambda_dist_values)
}

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Funcao auxiliar: desenha as curvas (uma por valor de lambda_dist) em um
# dado Axes, para a coluna y_col escolhida
# ---------------------------------------------------------------------------
def draw_rd_curves(ax, y_col, y_label, y_fmt="%.1f"):
    legend_handles = []

    for lambda_dist in lambda_dist_values:
        style = lambda_dist_styles[lambda_dist]

        subset = df[df["lambda_dist_rounded"] == lambda_dist].copy()

        if subset.empty:
            print(f"[aviso] Sem dados para lambda_dist={lambda_dist}")
            continue

        subset = (
            subset.groupby("lambda_rate")[[y_col, "eval_total_rate_bpp"]]
            .mean()
            .reset_index()
            .sort_values("eval_total_rate_bpp")
        )

        label = f"lambda_dist={lambda_dist}"

        ax.plot(
            subset["eval_total_rate_bpp"],
            subset[y_col],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            markersize=7,
            linewidth=2.0,
            label=label,
        )

        handle = mlines.Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=label,
        )
        legend_handles.append(handle)

    ax.set_xlabel("Total Rate (bpp)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

    return legend_handles


def add_legend(ax, legend_handles):
    ax.legend(
        handles=legend_handles,
        fontsize=10,
        framealpha=0.9,
        loc="lower right",
        handlelength=3.0,
    )


def save_fig(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Grafico salvo em: {out_path}")


# ---------------------------------------------------------------------------
# Grafico 1: WS-PSNR
# ---------------------------------------------------------------------------
fig_psnr, ax_psnr = plt.subplots(figsize=(10, 6.5))
handles_psnr = draw_rd_curves(ax_psnr, "eval_psnr", "WS-PSNR (dB)", y_fmt="%.1f")
ax_psnr.set_title("Rate-Distortion Curves (WS-PSNR)", fontsize=14, fontweight="bold")
add_legend(ax_psnr, handles_psnr)
fig_psnr.tight_layout()
save_fig(fig_psnr, args.out_psnr)

# ---------------------------------------------------------------------------
# Grafico 2: WS-SSIM
# ---------------------------------------------------------------------------
fig_ssim, ax_ssim = plt.subplots(figsize=(10, 6.5))
handles_ssim = draw_rd_curves(ax_ssim, "eval_wsssim", "WS-SSIM", y_fmt="%.3f")
ax_ssim.set_title("Rate-Distortion Curves (WS-SSIM)", fontsize=14, fontweight="bold")
add_legend(ax_ssim, handles_ssim)
fig_ssim.tight_layout()
save_fig(fig_ssim, args.out_wsssim)

# ---------------------------------------------------------------------------
# Grafico 3: combinado, lado a lado, com legenda unica compartilhada
# ---------------------------------------------------------------------------
fig_combined, (ax_c_psnr, ax_c_ssim) = plt.subplots(1, 2, figsize=(14, 6.5))

handles_c_psnr = draw_rd_curves(ax_c_psnr, "eval_psnr", "WS-PSNR (dB)", y_fmt="%.1f")
ax_c_psnr.set_title("WS-PSNR", fontsize=13, fontweight="bold")

handles_c_ssim = draw_rd_curves(ax_c_ssim, "eval_wsssim", "WS-SSIM", y_fmt="%.3f")
ax_c_ssim.set_title("WS-SSIM", fontsize=13, fontweight="bold")

fig_combined.suptitle("Rate-Distortion Curves", fontsize=15, fontweight="bold")

legend_source = handles_c_psnr if handles_c_psnr else handles_c_ssim
fig_combined.legend(
    handles=legend_source,
    fontsize=10,
    framealpha=0.9,
    loc="lower center",
    ncol=len(legend_source) if legend_source else 1,
    handlelength=3.0,
    bbox_to_anchor=(0.5, -0.02),
)

fig_combined.tight_layout(rect=[0, 0.06, 1, 0.95])
save_fig(fig_combined, args.out_combined)

plt.show()
