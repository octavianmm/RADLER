from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"


CASES = {
    "tradeoff_agriadapt_squeeze_energy_adapt": {
        "title": "AgriAdapt SSU-Net",
        "xlim": (52, 57),
        "ylim": (0, 5.25),
        "static_name": "SSU-Net",
        "static": [
            ("25%", 52.52, 1.15),
            ("50%", 55.18, 1.64),
            ("75%", 55.28, 2.67),
            ("100%", 55.70, 4.71),
        ],
        "radler": [
            ("RADLER 10%", 55.74, 3.65),
            ("RADLER 16%", 54.60, 2.23),
            ("RADLER 18%", 53.35, 1.51),
        ],
    },
    "tradeoff_agriadapt_snn_energy_adapt": {
        "title": "AgriAdapt SU-Net",
        "xlim": (54.5, 58.0),
        "ylim": (0, 50),
        "static_name": "SU-Net",
        "static": [
            ("25%", 54.93, 9.46),
            ("50%", 55.64, 20.47),
            ("75%", 56.39, 43.09),
            ("100%", 57.13, 45.18),
        ],
        "radler": [
            ("RADLER 1%", 56.44, 21.47),
            ("RADLER 5%", 55.35, 9.62),
            ("RADLER 10%", 55.35, 10.34),
        ],
    },
    "tradeoff_tobacco_squeeze_adapt": {
        "title": "Tobacco SSU-Net",
        "xlim": (53, 61),
        "ylim": (0, 22),
        "static_name": "SSU-Net",
        "static": [
            ("25%", 53.71, 4.26),
            ("50%", 55.67, 6.11),
            ("75%", 57.62, 10.17),
            ("100%", 59.38, 17.68),
        ],
        "radler": [
            ("RADLER 10%", 54.72, 5.60),
            ("RADLER 5%", 56.40, 9.45),
            ("RADLER 1%", 58.17, 13.06),
        ],
    },
    "tradeoff_tobacco_snn_unet_adapt": {
        "title": "Tobacco SU-Net",
        "xlim": (61, 64),
        "ylim": (0, 105),
        "static_name": "SU-Net",
        "static": [
            ("25%", 61.96, 20.77),
            ("50%", 62.45, 43.19),
            ("75%", 63.27, 91.87),
            ("100%", 63.58, 96.35),
        ],
        "radler": [
            ("RADLER 10%", 62.05, 25.57),
            ("RADLER 1%", 62.67, 52.23),
            ("RADLER 5%", 62.85, 63.36),
            ("RADLER 0.1%", 62.93, 67.71),
        ],
        "static_offsets": [(-6, 6, "right", "bottom"), (5, 8, "left", "bottom"), (5, 12, "left", "bottom"), (5, -7, "left", "top")],
        "radler_offsets": [(5, -7, "left", "top"), (5, -10, "left", "top"), (-6, -10, "right", "top"), (6, 10, "left", "bottom")],
    },
}


def annotate_points(ax, points, color, offsets):
    for idx, (label, x, y) in enumerate(points):
        dx, dy, ha, va = offsets[idx]
        ax.annotate(
            label.replace("RADLER ", ""),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
        )


def plot_case(name, spec):
    fig, ax = plt.subplots(figsize=(4.6, 3.45))
    static = spec["static"]
    radler = spec["radler"]

    ax.scatter(
        [p[1] for p in static],
        [p[2] for p in static],
        s=46,
        color="#1f77b4",
        label=f"{spec['static_name']} static widths",
        zorder=3,
    )
    ax.scatter(
        [p[1] for p in radler],
        [p[2] for p in radler],
        s=54,
        marker="D",
        color="#2ca02c",
        label="RADLER thresholds",
        zorder=4,
    )

    static_offsets = spec.get(
        "static_offsets",
        [(-5, 6, "right", "bottom"), (5, 5, "left", "bottom"), (5, 5, "left", "bottom"), (5, 5, "left", "bottom")],
    )
    radler_offsets = spec.get(
        "radler_offsets",
        [(-6, -7, "right", "top"), (5, -7, "left", "top"), (5, 6, "left", "bottom"), (5, 6, "left", "bottom")],
    )
    annotate_points(ax, static, "#1f4f7a", static_offsets[: len(static)])
    annotate_points(ax, radler, "#1c7c2c", radler_offsets[: len(radler)])

    ax.set_title(spec["title"], fontsize=10, pad=7)
    ax.set_xlabel("IoU (%)", fontsize=9)
    ax.set_ylabel("Aggregate energy (mAh)", fontsize=9)
    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(*spec["ylim"])
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.92)
    fig.tight_layout(pad=0.5)

    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def main():
    for name, spec in CASES.items():
        plot_case(name, spec)


if __name__ == "__main__":
    main()
