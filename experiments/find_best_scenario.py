import pandas as pd
import bjontegaard as bd
from pathlib import Path

RESULTS_CSV = "/home/diego/Desktop/moric360-again/experiments/01/results.csv"
BD_METHOD = "pchip"
ANCHOR_MASK  = "full"
ANCHOR_WSMSE = 0
ANCHOR_SWHDC = 0
GROUP_COLS = ["mask_type", "wsmse_tag", "swhdc_tag"]

def make_label(keys: tuple) -> str:
    return " | ".join(f"{c}={v}" for c, v in zip(GROUP_COLS, keys))

def aggregate_rd(df: pd.DataFrame, metric_col: str, max_bpp: float):
    agg = (
        df.groupby("lambda_rate")[["eval_total_rate_bpp", metric_col]]
        .mean()
        .reset_index()
        .sort_values("eval_total_rate_bpp")
    )
    if max_bpp is not None:
        agg = agg[agg["eval_total_rate_bpp"] < max_bpp]
    return agg["eval_total_rate_bpp"].values, agg[metric_col].values

def safe_bd_rate(r_anchor, m_anchor, r_test, m_test) -> float:
    try:
        return bd.bd_rate(r_anchor, m_anchor, r_test, m_test,
                          method=BD_METHOD, require_matching_points=False)
    except:
        return float("nan")

df = pd.read_csv(RESULTS_CSV)
cols = ["image_name", "mask_type", "wsmse_tag", "swhdc_tag", "lambda_rate", "eval_psnr", "eval_total_rate_bpp"]
df = df[cols].copy()
for c in ("eval_psnr", "eval_total_rate_bpp"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

anchor_mask = (
    (df["mask_type"]  == ANCHOR_MASK) &
    (df["wsmse_tag"]  == ANCHOR_WSMSE) &
    (df["swhdc_tag"]  == ANCHOR_SWHDC)
)
anchor_df = df[anchor_mask].reset_index(drop=True)
test_df = df[~anchor_mask].reset_index(drop=True)

test_configs = []
for keys, grp in test_df.groupby(GROUP_COLS):
    label = make_label(keys)
    test_configs.append((label, grp.reset_index(drop=True)))

TARGET_CONFIG = "mask_type=erp | wsmse_tag=1 | swhdc_tag=1"

import numpy as np
thresholds = np.arange(0.2, 3.0, 0.1)

best_thresholds = []

for max_bpp in thresholds:
    r_anchor, m_anchor = aggregate_rd(anchor_df, "eval_psnr", max_bpp)
    if len(r_anchor) < 2:
        continue
    
    results = {}
    for label, t_df in test_configs:
        r_test, m_test = aggregate_rd(t_df, "eval_psnr", max_bpp)
        if len(r_test) < 2:
            continue
        bdm = bd.bd_psnr(r_anchor, m_anchor, r_test, m_test, method=BD_METHOD, require_matching_points=False)
        if not np.isnan(bdm):
            results[label] = bdm
            
    if not results or TARGET_CONFIG not in results:
        continue
        
    best_config = max(results, key=results.get)
    if best_config == TARGET_CONFIG:
        best_thresholds.append((max_bpp, results[TARGET_CONFIG]))
        print(f"BINGO! max_bpp = {max_bpp:.1f} | Best BD-PSNR: {results[TARGET_CONFIG]:.4f} for {TARGET_CONFIG}")
    else:
        # Check what place it is
        sorted_configs = sorted(results.items(), key=lambda x: x[1], reverse=True)
        rank = [c for c, _ in sorted_configs].index(TARGET_CONFIG) + 1
        print(f"max_bpp = {max_bpp:.1f} | Rank of {TARGET_CONFIG}: {rank} (Best was {best_config} with {results[best_config]:.4f}) | Target: {results[TARGET_CONFIG]:.4f}")

if not best_thresholds:
    print("\nDid not find any threshold where the target config is the absolute best.")
