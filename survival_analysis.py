"""
Zebrafish Larvae Survival Analysis
===================================
Calculates hazard ratios (Cox PH) and generates Kaplan-Meier survival curves
for 0-6 dpf and 0-30 dpf data, stratified by health-score group.

Requirements:  pip install pandas numpy matplotlib lifelines openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings, os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA EXTRACTION — parse the irregular Excel layout
# ──────────────────────────────────────────────────────────────────────────────

def _parse_survival_file(filepath):
    """
    Parse the survival-rate Excel files which contain 3 trial blocks
    separated by blank rows.  Each block has columns:
        Day | High Score (8-10) | Mid Score (4-7) | Low Score (0-3)
    Returns a list of DataFrames (one per trial), each with columns:
        Day, High, Mid, Low
    """
    raw = pd.read_excel(filepath, header=None)

    # Find rows that contain "Day" as a header marker
    day_header_rows = []
    for idx, row in raw.iterrows():
        if row.astype(str).str.strip().str.lower().eq("day").any():
            day_header_rows.append(idx)

    trials = []
    for i, hdr_row in enumerate(day_header_rows):
        # Identify which columns hold Day / High / Mid / Low
        hdr = raw.iloc[hdr_row]
        col_map = {}
        for c in range(len(hdr)):
            val = str(hdr.iloc[c]).strip()
            if val.lower() == "day":
                col_map["Day"] = c
            elif "high" in val.lower():
                col_map["High"] = c
            elif "mid" in val.lower():
                col_map["Mid"] = c
            elif "low" in val.lower():
                col_map["Low"] = c

        # Read data rows until the next header or end
        start = hdr_row + 1
        if i + 1 < len(day_header_rows):
            end = day_header_rows[i + 1]
        else:
            end = len(raw)

        block = raw.iloc[start:end, list(col_map.values())].copy()
        block.columns = list(col_map.keys())
        block = block.dropna(subset=["Day"]).reset_index(drop=True)
        block = block.apply(pd.to_numeric, errors="coerce").dropna()
        block["Day"] = block["Day"].astype(int)
        block["Trial"] = i + 1
        trials.append(block)

    return trials


def _expand_to_individual(trials, groups=("High", "Mid", "Low")):
    """
    Convert aggregate counts into individual-level records for lifelines.

    At each time point the count tells us how many are STILL ALIVE.
    Deaths between two consecutive time points are:
        deaths_t = alive_{t-1} - alive_t

    Each individual record contains:
        duration  - time of death (endpoint of interval) or last observation
        event     - 1 = died, 0 = censored (still alive at end)
        group     - score category
        trial     - trial number
    """
    records = []
    for trial_df in trials:
        trial_num = trial_df["Trial"].iloc[0]
        days = trial_df["Day"].values

        for grp in groups:
            alive = trial_df[grp].values.astype(int)

            for t in range(len(days)):
                if t == 0:
                    # First timepoint is the starting population — no deaths yet
                    continue
                deaths = alive[t - 1] - alive[t]
                if deaths < 0:
                    deaths = 0  # guard against data inconsistency
                for _ in range(int(deaths)):
                    # Death occurred in (days[t-1], days[t]]
                    records.append({
                        "duration": days[t],
                        "event": 1,
                        "group": grp,
                        "trial": trial_num,
                    })

            # Remaining survivors at the last time point are right-censored
            survivors = alive[-1]
            for _ in range(int(survivors)):
                records.append({
                    "duration": days[-1],
                    "event": 0,
                    "group": grp,
                    "trial": trial_num,
                })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 2. KAPLAN-MEIER PLOT
# ──────────────────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    "High": "#2ca02c",   # green
    "Mid":  "#ff7f0e",   # orange
    "Low":  "#d62728",   # red
}

GROUP_LABELS = {
    "High": "High Score (8\u201310)",
    "Mid":  "Mid Score (4\u20137)",
    "Low":  "Low Score (0\u20133)",
}


def plot_km(df, title, ax, show_ci=True):
    """Fit & plot KM curves per group on the given axes."""
    kmf = KaplanMeierFitter()

    for grp in ("High", "Mid", "Low"):
        mask = df["group"] == grp
        if mask.sum() == 0:
            continue
        kmf.fit(
            durations=df.loc[mask, "duration"],
            event_observed=df.loc[mask, "event"],
            label=GROUP_LABELS[grp],
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=show_ci,
            color=GROUP_COLORS[grp],
            linewidth=2.2,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Days Post Fertilization (dpf)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.35)


# ──────────────────────────────────────────────────────────────────────────────
# 3. HAZARD RATIOS — Cox Proportional Hazards
# ──────────────────────────────────────────────────────────────────────────────

def compute_hazard_ratios(df, label=""):
    """
    Fit a Cox PH model using dummy-coded groups (reference = High).
    Returns the CoxPHFitter summary DataFrame.
    """
    model_df = df[["duration", "event", "group"]].copy()
    model_df = pd.get_dummies(model_df, columns=["group"], drop_first=False, dtype=float)

    # Use "group_High" as the reference category — drop it
    if "group_High" in model_df.columns:
        model_df = model_df.drop(columns=["group_High"])

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="duration", event_col="event")

    print(f"\n{'='*60}")
    print(f"  Cox Proportional-Hazards Model  --  {label}")
    print(f"{'='*60}")
    cph.print_summary(columns=["coef", "exp(coef)", "se(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"])
    print(f"\n  -> exp(coef) = Hazard Ratio relative to High-Score group")
    print(f"  -> HR > 1 means higher risk of death than the High-Score group\n")

    return cph


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOG-RANK TESTS (pairwise)
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_logrank(df, label=""):
    """Run pairwise log-rank tests between all group pairs."""
    groups = [g for g in ("High", "Mid", "Low") if g in df["group"].values]
    print(f"\n{'~'*60}")
    print(f"  Pairwise Log-Rank Tests  --  {label}")
    print(f"{'~'*60}")
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            m1 = df["group"] == g1
            m2 = df["group"] == g2
            result = logrank_test(
                df.loc[m1, "duration"], df.loc[m2, "duration"],
                df.loc[m1, "event"],    df.loc[m2, "event"],
            )
            sig = "***" if result.p_value < 0.001 else (
                  "**"  if result.p_value < 0.01  else (
                  "*"   if result.p_value < 0.05  else "ns"))
            print(f"  {GROUP_LABELS[g1]:>22s}  vs  {GROUP_LABELS[g2]:<22s}  "
                  f"X2 = {result.test_statistic:8.3f}   p = {result.p_value:.4e}  {sig}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_6d  = os.path.join(base_dir, "6 days survival rate.xlsx")
    file_30d = os.path.join(base_dir, "30 days survival rate.xlsx")

    # --- Parse ---
    trials_6d  = _parse_survival_file(file_6d)
    trials_30d = _parse_survival_file(file_30d)

    # --- Expand to individual-level ---
    df_6d  = _expand_to_individual(trials_6d)
    df_30d = _expand_to_individual(trials_30d)

    # Print summary statistics
    for tag, df in [("0-6 dpf", df_6d), ("0-30 dpf", df_30d)]:
        print(f"\n> {tag}:  N = {len(df)}  |  Events = {df['event'].sum():.0f}  "
              f"|  Censored = {(df['event']==0).sum():.0f}")
        for g in ("High", "Mid", "Low"):
            sub = df[df["group"] == g]
            print(f"    {GROUP_LABELS[g]:>22s}:  n = {len(sub):>4d},  "
                  f"events = {sub['event'].sum():>3.0f},  "
                  f"censored = {(sub['event']==0).sum():>4.0f}")

    # --- Kaplan-Meier Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    fig.suptitle("Zebrafish Larvae Kaplan-Meier Survival Curves",
                 fontsize=16, fontweight="bold", y=1.02)

    plot_km(df_6d,  "0-6 dpf  (Embryonic Survival)",  axes[0])
    plot_km(df_30d, "0-30 dpf  (Extended Survival)",   axes[1])

    plt.tight_layout()
    out_km = os.path.join(base_dir, "KM_survival_curves.png")
    fig.savefig(out_km, dpi=300, bbox_inches="tight")
    print(f"\n[OK] Kaplan-Meier figure saved -> {out_km}")

    # --- Cox PH Hazard Ratios ---
    cph_6d  = compute_hazard_ratios(df_6d,  "0-6 dpf")
    cph_30d = compute_hazard_ratios(df_30d, "0-30 dpf")

    # --- Log-Rank Tests ---
    pairwise_logrank(df_6d,  "0-6 dpf")
    pairwise_logrank(df_30d, "0-30 dpf")

    # --- Save hazard ratio summary to Excel ---
    out_xlsx = os.path.join(base_dir, "survival_analysis_results.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        cph_6d.summary.to_excel(writer,  sheet_name="Cox PH  0-6 dpf")
        cph_30d.summary.to_excel(writer, sheet_name="Cox PH  0-30 dpf")
        df_6d.to_excel(writer,  sheet_name="Individual Data 0-6 dpf",  index=False)
        df_30d.to_excel(writer, sheet_name="Individual Data 0-30 dpf", index=False)
    print(f"[OK] Results spreadsheet saved -> {out_xlsx}")

    plt.show()


if __name__ == "__main__":
    main()
