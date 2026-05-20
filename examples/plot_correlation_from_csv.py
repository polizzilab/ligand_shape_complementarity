"""Plot hybrid SC correlation against Rosetta from a scores CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

EXAMPLES = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=EXAMPLES / "sc_scores_hybrid.csv")
    p.add_argument("--score_col", default="hybrid_sc")
    p.add_argument("--out", type=Path, default=EXAMPLES / "sc_correlation_hybrid.png")
    p.add_argument("--method", default="Hybrid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(open(args.csv)))

    rosetta_arr, our_arr = [], []
    n_nan = 0
    n_none = 0

    for r in rows:
        ros = r.get("rosetta_sc")
        ours = r.get(args.score_col)
        if ros is None or ours is None or ours == "" or ros == "":
            n_none += 1
            continue
        ros_f = float(ros)
        try:
            our_f = float(ours)
        except ValueError:
            n_none += 1
            continue
        if not math.isfinite(ros_f) or not math.isfinite(our_f):
            n_nan += 1
            continue
        rosetta_arr.append(ros_f)
        our_arr.append(our_f)

    rosetta_arr = np.array(rosetta_arr)
    our_arr = np.array(our_arr)

    pearson_r, pearson_p = pearsonr(rosetta_arr, our_arr)
    spearman_r, spearman_p = spearmanr(rosetta_arr, our_arr)

    print(f"Valid pairs:  {len(rosetta_arr)}")
    print(f"NaN/excluded: {n_nan}")
    print(f"Missing:      {n_none}")
    print(f"\nPearson  r = {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_r:.4f}  (p={spearman_p:.2e})")

    coeffs = np.polyfit(our_arr, rosetta_arr, 1)
    print(f"Linear fit  a={coeffs[0]:.3f}  b={coeffs[1]:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(our_arr, rosetta_arr, s=8, alpha=0.35, color="#2563eb", linewidths=0)

    x_line = np.linspace(our_arr.min(), our_arr.max(), 100)
    ax.plot(
        x_line,
        np.polyval(coeffs, x_line),
        color="#dc2626",
        lw=1.5,
        label=f"fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}",
    )

    lim = (
        min(our_arr.min(), rosetta_arr.min()) - 0.02,
        max(our_arr.max(), rosetta_arr.max()) + 0.02,
    )
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, label="y=x")

    ax.set_xlabel(f"Our {args.method} SC", fontsize=13)
    ax.set_ylabel("Rosetta SC", fontsize=13)
    ax.set_title(
        f"{args.method} vs Rosetta Shape Complementarity  (n={len(rosetta_arr)})\n"
        f"Pearson r={pearson_r:.3f}   Spearman r={spearman_r:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.grid(True, alpha=0.3)

    fig.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {args.out}")


if __name__ == "__main__":
    main()
