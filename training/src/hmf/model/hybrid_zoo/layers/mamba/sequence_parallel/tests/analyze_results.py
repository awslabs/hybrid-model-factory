import json, glob, pandas as pd, matplotlib.pyplot as plt
import re


def canonical(name: str) -> str:
    """Turn metric key into a dataframe‑safe column name."""
    return "rel_diff" + name.replace("global", "").replace(" ", "_").replace(".", "_")


def plot_rel_diffs(
    df: pd.DataFrame, x: str = "seqlen", regex: str = r"^rel_diff_", title: str = ""
):
    """
    Plot every DataFrame column whose name matches *regex* against *x*.

    Args:
        df: pd.DataFrame
            The table of runs (one row per experiment).
        x: Column to put on the x‑axis. Defaults to 'seqlen'.
        regex: Regular expression that identifies metric columns. Defaults to those
            prefixed with 'rel_diff_'.
    """
    metric_cols = [c for c in df.columns if re.match(regex, c)]
    if not metric_cols:
        raise ValueError(f"No columns matching regex '{regex}' found.")

    plt.figure(figsize=(8, 6))
    y_max = 0
    y_min = 1e9
    for col in metric_cols:
        plt.plot(df[x], df[col] * 100, marker="o", label=col.replace("rel_diff_", ""))

        y_max = max(y_max, df[col].max())
        y_min = min(y_min, df[col].min())

    plt.xlabel(x.replace("_", " ").title())

    plt.ylabel("Relative difference (in %)")
    plt.title(f"ZigZag‑Mamba2 relative errors vs. {x}, {title}")

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    custom_xticks = [
        64,
        128,
        1024,
        2048,
        4096,
        4096 * 2,
        4096 * 4,
        4096 * 8,
        4096 * 16,
        4096 * 32,
    ]
    custom_xlabels = ["64", "128", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

    custom_yticks = [0.001, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    custom_ylabels = [
        "0.001",
        "0.01",
        "0.1",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "256",
    ]

    plt.xticks(custom_xticks, custom_xlabels)
    plt.yticks(custom_yticks, custom_ylabels)

    plt.ylim(y_min * 100 - 1e-2, (y_max * 100) + 50)

    # Place legend below the plot using figlegend
    plt.legend(
        bbox_to_anchor=(0.5, -0.05),  # (x,y) position relative to the figure
        loc="upper center",  # Reference point for alignment
        ncol=3,  # Number of columns in the legend
    )

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at the bottom for the legend


if __name__ == "__main__":
    exp_name = "mamba2"
    # ------------ collect every run ------------
    records = []
    for f in glob.glob(f"/path/to/{exp_name}/*.json"):
        with open(f) as fp:
            data = json.load(fp)

        row = {**data["run_params"]}
        row.update({canonical(k): v["rel_diff"] for k, v in data["metrics"].items()})

        records.append(row)
    print(records)
    df_full = pd.DataFrame(records).sort_values("seqlen")
    print(df_full)

    # Filter logs by sp and by data type. You can change the filter as needed
    df = df_full[df_full["dtype"].isin(["bf16"])]
    df = df[df["sp"].isin([2.0])]
    if len(df) > 0:
        plot_rel_diffs(df, x="seqlen", regex=r"^rel_diff_", title="bf16 sp2")

    df = df_full[df_full["dtype"].isin(["fp32"])]
    df = df[df["sp"].isin([2.0])]
    if len(df) > 0:
        plot_rel_diffs(df, x="seqlen", regex=r"^rel_diff_", title="fp32 sp2")

    df = df_full[df_full["dtype"].isin(["bf16"])]
    df = df[df["sp"].isin([8.0])]
    if len(df) > 0:
        plot_rel_diffs(df, x="seqlen", regex=r"^rel_diff_", title="bf16 sp8")

    plt.show()
