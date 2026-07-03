#!/usr/bin/env python3
"""
Bjontegaard ablation study — compara configs internas do results.csv.

Âncora (baseline interno):  mask_type=full, wsmse_tag=0, swhdc_tag=0
Test:                        todas as demais combinações presentes no CSV

Para cada config "test" calcula BD-Rate e BD-PSNR/BD-WSSSIM em relação
ao âncora, usando a lib bjontegaard (pip install bjontegaard).

Reads:
  experiments/vcip_consolidated/results.csv

Outputs in:
  experiments/bjontegaard_ablation/
"""

import argparse
import warnings
from pathlib import Path

import bjontegaard as bd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
RESULTS_CSV = "/home/diego/Desktop/moric360-again/experiments/01/results.csv"
OUTPUT_DIR  = SCRIPT_DIR / "bjontegaard_ablation_menor_1000x1000"

BD_METHOD = "pchip"

# Âncora (baseline interno)
ANCHOR_MASK  = "full"
ANCHOR_WSMSE = 0
ANCHOR_SWHDC = 0
# erp_padding não restringe o âncora — pegamos todas as linhas que batem
# nos três campos acima e usamos a média

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # cols = [
    #     "image_name", "mask_type", "wsmse_tag", "swhdc_tag",
    #     "erp_padding", "lambda_rate",
    #     "eval_psnr", "eval_wsssim", "eval_total_rate_bpp",
    # ]
    cols = [
        "image_name", "mask_type", "wsmse_tag", "swhdc_tag", "lambda_rate",
        "eval_psnr", "eval_total_rate_bpp",
    ]
    df = df[cols].copy()
    # for c in ("eval_psnr", "eval_wsssim", "eval_total_rate_bpp"):
    for c in ("eval_psnr", "eval_total_rate_bpp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# RD aggregation — média sobre imagens, agrupada por lambda_rate
# ---------------------------------------------------------------------------

def aggregate_rd(df: pd.DataFrame, metric_col: str):
    agg = (
        df.groupby("lambda_rate")[["eval_total_rate_bpp", metric_col]]
        .mean()
        .reset_index()
        .sort_values("eval_total_rate_bpp")
    )
    agg = agg[agg["eval_total_rate_bpp"] > 0.5]
    return agg["eval_total_rate_bpp"].values, agg[metric_col].values

# ---------------------------------------------------------------------------
# BD wrappers
# ---------------------------------------------------------------------------

def safe_bd_rate(r_anchor, m_anchor, r_test, m_test) -> float:
    try:
        return bd.bd_rate(r_anchor, m_anchor, r_test, m_test,
                          method=BD_METHOD, require_matching_points=False)
    except Exception as e:
        warnings.warn(f"bd_rate: {e}")
        return float("nan")


def safe_bd_metric(r_anchor, m_anchor, r_test, m_test) -> float:
    try:
        return bd.bd_psnr(r_anchor, m_anchor, r_test, m_test,
                          method=BD_METHOD, require_matching_points=False)
    except Exception as e:
        warnings.warn(f"bd_psnr: {e}")
        return float("nan")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STYLES = [
    dict(linestyle="-",  marker="o"),
    dict(linestyle="--", marker="s"),
    dict(linestyle="-.", marker="^"),
    dict(linestyle=":",  marker="D"),
    dict(linestyle="-",  marker="v"),
    dict(linestyle="--", marker="P"),
    dict(linestyle="-.", marker="X"),
    dict(linestyle=":",  marker="*"),
]

ANCHOR_COLOR = "#e63946"   # vermelho — baseline interno

# Cores distintas para os modelos test (evitam vermelho)
TEST_COLORS = [
    "#2dc653",  # verde
    "#0077b6",  # azul
    "#ff6b35",  # laranja
    "#9b5de5",  # roxo
    "#f9c74f",  # amarelo
    "#ff006e",  # pink
    "#43aa8b",  # teal
    "#f77f00",  # laranja escuro
]


def test_color(i: int) -> str:
    return TEST_COLORS[i % len(TEST_COLORS)]


def plot_rd(anchor_label, anchor_df, test_configs, metric_col,
            metric_label, output_path, title=""):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Âncora
    r, m = aggregate_rd(anchor_df, metric_col)
    ax.plot(r, m, color=ANCHOR_COLOR, lw=3, marker="o", ms=8,
            label=f"[ÂNCORA] {anchor_label}", zorder=5)

    # Test configs
    for i, (label, df) in enumerate(test_configs):
        r, m = aggregate_rd(df, metric_col)
        if len(r) < 2:
            continue
        ax.plot(r, m, color=test_color(i), lw=2.5,
                label=label, zorder=4,
                **STYLES[i % len(STYLES)], ms=7, alpha=1.0)

    ax.set_xlabel("Bit-rate (bpp)", fontsize=13)
    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(title or f"RD — {metric_label}", fontsize=14)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.88)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# def plot_combined(anchor_label, anchor_df, test_configs, output_path):
#     fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# 
#     for ax, metric_col, metric_label in [
#         (axes[0], "eval_wsssim", "WS-SSIM"),
#         (axes[1], "eval_psnr",   "PSNR (dB)"),
#     ]:
#         r, m = aggregate_rd(anchor_df, metric_col)
#         ax.plot(r, m, color=ANCHOR_COLOR, lw=3, marker="o", ms=8,
#                 label=f"[ÂNCORA] {anchor_label}", zorder=5)
# 
#         for i, (label, df) in enumerate(test_configs):
#             r, m = aggregate_rd(df, metric_col)
#             if len(r) < 2:
#                 continue
#             ax.plot(r, m, color=test_color(i), lw=2.5,
#                     label=label, zorder=4,
#                     **STYLES[i % len(STYLES)], ms=7, alpha=1.0)
# 
#         ax.set_xlabel("Bit-rate (bpp)", fontsize=12)
#         ax.set_ylabel(metric_label, fontsize=12)
#         ax.set_title(f"RD — {metric_label}", fontsize=13)
#         ax.legend(fontsize=7, loc="lower right", framealpha=0.88)
#         ax.grid(True, ls="--", alpha=0.4)
# 
#     fig.suptitle("Ablation Study — vs âncora (full / wsmse=0 / swhdc=0)",
#                  fontsize=14, y=1.01)
#     fig.tight_layout()
#     fig.savefig(output_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# BD table
# ---------------------------------------------------------------------------

def compute_bd_table(anchor_df, test_configs, metric_col) -> pd.DataFrame:
    r_anchor, m_anchor = aggregate_rd(anchor_df, metric_col)
    rows = []
    for label, test_df in test_configs:
        r_test, m_test = aggregate_rd(test_df, metric_col)
        if len(r_test) < 2:
            continue
        bdr = safe_bd_rate(r_anchor, m_anchor, r_test, m_test)
        bdm = safe_bd_metric(r_anchor, m_anchor, r_test, m_test)
        rows.append({
            "config":    label,
            "BD-Rate (%)": round(bdr, 2),
            f"BD-{metric_col.replace('eval_', '').upper()}": round(bdm, 4),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Config label
# ---------------------------------------------------------------------------

#GROUP_COLS = ["mask_type", "wsmse_tag", "swhdc_tag", "erp_padding"]
GROUP_COLS = ["mask_type", "wsmse_tag", "swhdc_tag"]



def make_label(keys: tuple) -> str:
    return " | ".join(f"{c}={v}" for c, v in zip(GROUP_COLS, keys))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation BD plots: configs internas do results.csv"
    )
    p.add_argument("--results", type=Path, default=RESULTS_CSV)
    p.add_argument("--outdir",  type=Path, default=OUTPUT_DIR)
    p.add_argument("--method",  default=BD_METHOD,
                   choices=["pchip", "akima", "cubic"])
    p.add_argument("--anchor-mask",  default=ANCHOR_MASK,
                   help="mask_type do âncora (default: full)")
    p.add_argument("--anchor-wsmse", type=int, default=ANCHOR_WSMSE,
                   help="wsmse_tag do âncora (default: 0)")
    p.add_argument("--anchor-swhdc", type=int, default=ANCHOR_SWHDC,
                   help="swhdc_tag do âncora (default: 0)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global BD_METHOD
    BD_METHOD = args.method
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df = load_results(args.results)

    # ---- Separar âncora ----
    anchor_mask = (
        (df["mask_type"]  == args.anchor_mask) &
        (df["wsmse_tag"]  == args.anchor_wsmse) &
        (df["swhdc_tag"]  == args.anchor_swhdc)
    )
    anchor_df = df[anchor_mask].reset_index(drop=True)
    if anchor_df.empty:
        print("ERRO: âncora não encontrado com os filtros especificados.")
        return

    anchor_label = (f"mask={args.anchor_mask} | "
                    f"wsmse={args.anchor_wsmse} | "
                    f"swhdc={args.anchor_swhdc}")
    print(f"Âncora: {anchor_label}  ({len(anchor_df)} linhas)")

    # ---- Separar configs test (tudo que não é âncora) ----
    test_df = df[~anchor_mask].reset_index(drop=True)

    test_configs = []
    for keys, grp in test_df.groupby(GROUP_COLS):
        label = make_label(keys)
        test_configs.append((label, grp.reset_index(drop=True)))

    print(f"Configs test encontradas: {len(test_configs)}")
    for lbl, _ in test_configs:
        print(f"  {lbl}")

    # # ---- WS-SSIM ----
    # print("\n--- WS-SSIM ---")
    # plot_rd(anchor_label, anchor_df, test_configs,
    #         "eval_wsssim", "WS-SSIM",
    #         args.outdir / "ablation_wsssim.png",
    #         "Ablation — WS-SSIM (média sobre imagens)")

    # bd_wsssim = compute_bd_table(anchor_df, test_configs, "eval_wsssim")
    # if not bd_wsssim.empty:
    #     path = args.outdir / "ablation_bd_wsssim.csv"
    #     bd_wsssim.to_csv(path, index=False)
    #     print(bd_wsssim.to_string(index=False))
    #     print(f"  Saved: {path}")

    # ---- PSNR ----
    print("\n--- PSNR ---")
    plot_rd(anchor_label, anchor_df, test_configs,
            "eval_psnr", "PSNR (dB)",
            args.outdir / "ablation_psnr.png",
            "Ablation — PSNR (média sobre imagens)")

    bd_psnr = compute_bd_table(anchor_df, test_configs, "eval_psnr")
    if not bd_psnr.empty:
        path = args.outdir / "ablation_bd_psnr.csv"
        bd_psnr.to_csv(path, index=False)
        print(bd_psnr.to_string(index=False))
        print(f"  Saved: {path}")

    # ---- Combinado ----
    # print("\n--- Combined figure ---")
    # plot_combined(anchor_label, anchor_df, test_configs,
    #               args.outdir / "ablation_combined.png")

    # ---- Resumo ----
    print("\n=== Bjontegaard Ablation Summary ===")
    for metric_label, bd_df in [("PSNR", bd_psnr)]:
        print(f"\n  Metric: {metric_label}  (negativo = menos bits / melhor)")
        if bd_df.empty:
            print("    (sem dados)")
            continue
        bdr_col = "BD-Rate (%)"
        bdm_col = [c for c in bd_df.columns if c.startswith("BD-") and c != bdr_col][0]
        for _, row in bd_df.iterrows():
            arrow = "✅" if row[bdr_col] < 0 else "❌"
            print(f"    {arrow}  {row['config']:<62s}  "
                  f"BD-Rate={row[bdr_col]:>8.2f}%  "
                  f"{bdm_col}={row[bdm_col]:>7.4f}")

    print(f"\nOutputs em: {args.outdir}")


if __name__ == "__main__":
    main()
