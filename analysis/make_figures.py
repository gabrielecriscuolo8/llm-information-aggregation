from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

BLUE = "#2563EB"
NAVY = "#0F172A"
SLATE = "#64748B"
LIGHT = "#CBD5E1"
GRID = "#E2E8F0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def horizontal_bars(
    axis: plt.Axes,
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
) -> None:
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    colors = [BLUE if label == "Market" else SLATE for label in labels]
    positions = list(range(len(rows)))

    axis.barh(positions, values, color=colors, height=0.62)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean squared error to exact posterior (lower is better)")
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=22)
    axis.text(0, 1.03, subtitle, transform=axis.transAxes, fontsize=9, color=SLATE)
    axis.set_xlim(0, max(values) * 1.22)
    style_axis(axis)

    for position, value in zip(positions, values):
        axis.text(value + max(values) * 0.02, position, f"{value:.4f}", va="center", fontsize=9)


def make_main_results() -> None:
    v1 = {row["system"]: row for row in read_csv(RESULTS / "v1_pilot" / "summary.csv")}
    v2 = {row["system"]: row for row in read_csv(RESULTS / "v2_confirmatory" / "summary.csv")}

    v1_rows = [
        ("Market", float(v1["market"]["mean_oracle_squared_error"])),
        ("Central analyst", float(v1["central"]["mean_oracle_squared_error"])),
        ("Ensemble mean", float(v1["ensemble_mean"]["mean_oracle_squared_error"])),
        ("Prior", float(v1["prior"]["mean_oracle_squared_error"])),
        ("Ensemble log pool", float(v1["ensemble_log_pool"]["mean_oracle_squared_error"])),
    ]
    v2_rows = [
        ("Market", float(v2["market"]["weighted_mean_oracle_squared_error"])),
        ("Central full", float(v2["central_full"]["weighted_mean_oracle_squared_error"])),
        ("Central compact", float(v2["central_compact"]["weighted_mean_oracle_squared_error"])),
        ("Prior", float(v2["prior"]["weighted_mean_oracle_squared_error"])),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    horizontal_bars(axes[0], v1_rows, "V1 pilot", "30 sampled paired events")
    horizontal_bars(axes[1], v2_rows, "V2 exhaustive design", "160 probability-weighted information cells")
    figure.suptitle(
        "Prediction-market framing produced the lowest oracle error",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    figure.savefig(FIGURES / "main_results.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_update_diagnostics() -> None:
    rows = read_csv(RESULTS / "v2_confirmatory" / "diagnostics.csv")
    labels = {
        "central_full": "Central full",
        "central_compact": "Central compact",
        "market": "Market",
    }
    colors = {
        "central_full": LIGHT,
        "central_compact": SLATE,
        "market": BLUE,
    }

    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for condition in ("central_full", "central_compact", "market"):
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["position"]),
        )
        axis.plot(
            [int(row["position"]) for row in selected],
            [float(row["weighted_local_target_mae"]) for row in selected],
            marker="o",
            linewidth=2.5,
            label=labels[condition],
            color=colors[condition],
        )

    axis.set_title("Sequential update error in V2", loc="left", fontsize=15, fontweight="bold", color=NAVY)
    axis.set_xlabel("Evidence position")
    axis.set_ylabel("Weighted local-target MAE")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.grid(color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "v2_update_diagnostics.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_main_results()
    make_update_diagnostics()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()

