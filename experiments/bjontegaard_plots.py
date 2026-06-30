#!/usr/bin/env python3
"""
Bjontegaard Delta plots comparing experimental results vs baseline codecs.

Reads:
  - experiments/vcip_consolidated/results.csv  (our model, various configs)
  - experiments/li/codecs_renamed.csv           (baseline codecs)

Produces RD-curve plots and BD-Rate / BD-metric tables
for both WS-SSIM and PSNR metrics.

Requires: pip install bjontegaard
"""

import argparse
import warnings
from pathlib import Path

import bjontegaard as bd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RESULTS_CSV = SCRIPT_DIR / "vcip_consolidated" / "results.csv"
CODECS_CSV  = SCRIPT_DIR / "li" / "codecs_renamed.csv"
OUTPUT_DIR  = SCRIPT_DIR / "bjontegaard_output"

BD_METHOD = "pchip"   # 'pchip' | 'akima' | 'cubic'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [
        "image_name", "mask_type", "wsmse_tag", "swhdc_tag",
        "erp_padding", "lambda_rate",
        "eval_psnr", "eval_wsssim", "eval_total_rate_bpp",
    ]
    df = df[cols].copy()
    for c in ("eval_psnr", "eval_wsssim", "eval_total_rate_bpp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_codecs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"bpp": "eval_total_rate_bpp", "ws_ssim": "eval_wsssim"})
    cols = ["image_name", "codec", "model_idx",
            "eval_psnr", "eval_wsssim", "eval_total_rate_bpp"]
    df = df[cols].copy()
    for c in ("eval_psnr", "eval_wsssim", "eval_total_rate_bpp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# RD aggregation (mean over images, sorted by bit-rate)
# ---------------------------------------------------------------------------

def aggregate_rd(df: pd.DataFrame, rate_col: str, metric_col: str,
                 group_col: str) -> tuple:
    """Return (rates, metrics) averaged over images, grouped by group_col."""
    agg = (
        df.groupby(group_col)[[rate_col, metric_col]]
        .mean()
        .reset_index()
        .sort_values(rate_col)
    )
    return agg[rate_col].values, agg[metric_col].values

# ---------------------------------------------------------------------------
# Bjontegaard helpers (thin wrappers around the bjontegaard package)
# ---------------------------------------------------------------------------

def safe_bd_rate(r_anchor, m_anchor, r_test, m_test) -> float:
    try:
        return bd.bd_rate(r_anchor, m_anchor, r_test, m_test,
                          method=BD_METHOD, require_matching_points=False)
    except Exception as e:
        warnings.warn(f"bd_rate failed: {e}")
        return float("nan")


def safe_bd_metric(r_anchor, m_anchor, r_test, m_test) -> float:
    try:
        return bd.bd_psnr(r_anchor, m_anchor, r_test, m_test,
                          method=BD_METHOD, require_matching_points=False)
    except Exception as e:
        warnings.warn(f"bd_psnr failed: {e}")
        return float("nan")

# ---------------------------------------------------------------------------
# Plotting helpers
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

CODEC_COLORS = {
    "lic360_777965":   "#e63946",   # vivid red
    "lic_plus_777974": "#0077b6",   # deep blue
}

# High-contrast palette for our model configs (avoids reds and blues)
MODEL_COLORS = [
    "#2dc653",   # vivid green
    "#ff6b35",   # orange
    "#9b5de5",   # purple
    "#f9c74f",   # yellow
    "#00b4d8",   # cyan
    "#ff006e",   # hot pink
    "#43aa8b",   # teal
    "#f77f00",   # deep orange
]


def _model_color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


def plot_rd(results_configs, codecs_dict, metric_col, metric_label,
            output_path, title=""):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline codecs
    for codec_name, codec_df in codecs_dict.items():
        r, m = aggregate_rd(codec_df, "eval_total_rate_bpp", metric_col, "model_idx")
        if len(r) < 2:
            continue
        color = CODEC_COLORS.get(codec_name, "gray")
        ax.plot(r, m, color=color, lw=2.5, marker="o",
                ms=6, label=codec_name, zorder=3)

    # Our model configs
    for i, (label, cfg_df) in enumerate(results_configs):
        r, m = aggregate_rd(cfg_df, "eval_total_rate_bpp", metric_col, "lambda_rate")
        if len(r) < 2:
            continue
        ax.plot(r, m, color=_model_color(i), lw=2.5, label=label,
                zorder=4, **STYLES[i % len(STYLES)], ms=8, alpha=1.0)

    ax.set_xlabel("Bit-rate (bpp)", fontsize=13)
    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(title or f"RD Curve — {metric_label}", fontsize=14)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# BD table computation
# ---------------------------------------------------------------------------

def compute_bd_table(results_configs, codecs_dict, metric_col):
    rows = []
    for cfg_label, cfg_df in results_configs:
        r_test, m_test = aggregate_rd(cfg_df, "eval_total_rate_bpp",
                                      metric_col, "lambda_rate")
        if len(r_test) < 2:
            continue
        for codec_name, codec_df in codecs_dict.items():
            r_anchor, m_anchor = aggregate_rd(codec_df, "eval_total_rate_bpp",
                                              metric_col, "model_idx")
            if len(r_anchor) < 2:
                continue
            bdr  = safe_bd_rate(r_anchor, m_anchor, r_test, m_test)
            bdm  = safe_bd_metric(r_anchor, m_anchor, r_test, m_test)
            rows.append({
                "config":    cfg_label,
                "baseline":  codec_name,
                "BD-Rate (%)": round(bdr, 2),
                f"BD-{metric_col.replace('eval_', '').upper()}": round(bdm, 4),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bjontegaard plots (uses the 'bjontegaard' pip package)"
    )
    p.add_argument("--results", type=Path, default=RESULTS_CSV)
    p.add_argument("--codecs",  type=Path, default=CODECS_CSV)
    p.add_argument("--outdir",  type=Path, default=OUTPUT_DIR)
    p.add_argument("--method",  default=BD_METHOD,
                   choices=["pchip", "akima", "cubic"],
                   help="Interpolation method for BD calculation")
    p.add_argument("--filter-mask",        default="erp",
                   help="Filter mask_type (default: 'erp')")
    p.add_argument("--filter-wsmse",  type=int, default=1,
                   help="Filter wsmse_tag (default: 1)")
    p.add_argument("--filter-swhdc",  type=int, default=1,
                   help="Filter swhdc_tag (default: 1)")
    p.add_argument("--filter-erp-padding", type=int, default=1,
                   help="Filter erp_padding (default: 1)")
    p.add_argument("--group-by", nargs="+",
                   default=["mask_type", "wsmse_tag", "swhdc_tag", "erp_padding"],
                   help="Columns that define distinct configs")
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
    results_df = load_results(args.results)
    codecs_df  = load_codecs(args.codecs)

    # Optional filters
    if args.filter_mask:
        results_df = results_df[results_df["mask_type"] == args.filter_mask]
    if args.filter_wsmse is not None:
        results_df = results_df[results_df["wsmse_tag"] == args.filter_wsmse]
    if args.filter_swhdc is not None:
        results_df = results_df[results_df["swhdc_tag"] == args.filter_swhdc]
    if args.filter_erp_padding is not None:
        results_df = results_df[results_df["erp_padding"] == args.filter_erp_padding]

    if results_df.empty:
        print("ERROR: results DataFrame is empty after filtering.")
        return

    # Build codec dict
    codecs_dict = {
        name: grp.reset_index(drop=True)
        for name, grp in codecs_df.groupby("codec")
    }

    # Build config list
    group_cols = [c for c in args.group_by if c in results_df.columns]
    results_configs = []
    for keys, grp in results_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(f"{c}={v}" for c, v in zip(group_cols, keys))
        results_configs.append((label, grp.reset_index(drop=True)))

    print(f"Configs found: {len(results_configs)}")
    print(f"Codecs found:  {list(codecs_dict.keys())}")

    # ---- WS-SSIM ----
    print("\n--- WS-SSIM ---")
    plot_rd(results_configs, codecs_dict,
            "eval_wsssim", "WS-SSIM",
            args.outdir / "rd_wsssim_all.png",
            "RD Curves — WS-SSIM (mean over all images)")

    bd_wsssim = compute_bd_table(results_configs, codecs_dict, "eval_wsssim")
    if not bd_wsssim.empty:
        path = args.outdir / "bd_table_wsssim.csv"
        bd_wsssim.to_csv(path, index=False)
        print(bd_wsssim.to_string(index=False))
        print(f"  Saved: {path}")

    # ---- PSNR ----
    print("\n--- PSNR ---")
    plot_rd(results_configs, codecs_dict,
            "eval_psnr", "PSNR (dB)",
            args.outdir / "rd_psnr_all.png",
            "RD Curves — PSNR (mean over all images)")

    bd_psnr = compute_bd_table(results_configs, codecs_dict, "eval_psnr")
    if not bd_psnr.empty:
        path = args.outdir / "bd_table_psnr.csv"
        bd_psnr.to_csv(path, index=False)
        print(bd_psnr.to_string(index=False))
        print(f"  Saved: {path}")

    # ---- Combined figure ----
    print("\n--- Combined figure ---")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, metric_col, metric_label in [
        (axes[0], "eval_wsssim", "WS-SSIM"),
        (axes[1], "eval_psnr",   "PSNR (dB)"),
    ]:
        for codec_name, codec_df in codecs_dict.items():
            r, m = aggregate_rd(codec_df, "eval_total_rate_bpp", metric_col, "model_idx")
            if len(r) < 2:
                continue
            ax.plot(r, m, color=CODEC_COLORS.get(codec_name, "gray"),
                    lw=2.5, marker="o", ms=6, label=codec_name, zorder=3)

        for i, (label, cfg_df) in enumerate(results_configs):
            r, m = aggregate_rd(cfg_df, "eval_total_rate_bpp", metric_col, "lambda_rate")
            if len(r) < 2:
                continue
            ax.plot(r, m, color=_model_color(i), lw=2.5, label=label,
                    zorder=4, **STYLES[i % len(STYLES)], ms=8, alpha=1.0)

        ax.set_xlabel("Bit-rate (bpp)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"RD Curve — {metric_label}", fontsize=13)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("Bjontegaard Comparison — Our Model vs Baselines",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    combined_path = args.outdir / "rd_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {combined_path}")

    # ---- Summary ----
    print("\n=== Bjontegaard Summary ===")
    for metric_label, bd_df in [("WS-SSIM", bd_wsssim), ("PSNR", bd_psnr)]:
        print(f"\n  Metric: {metric_label}")
        if bd_df.empty:
            print("    (no data)")
            continue
        bd_col  = "BD-Rate (%)"
        bdm_col = [c for c in bd_df.columns if c.startswith("BD-") and c != bd_col][0]
        for codec in bd_df["baseline"].unique():
            sub = bd_df[bd_df["baseline"] == codec]
            print(f"    vs {codec}:")
            for _, row in sub.iterrows():
                print(f"      {row['config']:<62s}  "
                      f"BD-Rate={row[bd_col]:>8.2f}%  "
                      f"{bdm_col}={row[bdm_col]:>7.4f}")

    print(f"\nAll outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
