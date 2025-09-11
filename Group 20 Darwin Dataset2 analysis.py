#Group 20 Darwin
# Aryan Rayamajhi 
# Samir Rimal
# Rohan Baniya
# Samir Rimal

# Dataset 2 Analysis
#Graph Showing Scatter plot
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind # for comparing means between groups


def create_activity_temporal_chart():
    print("Loading dataset2...")
    df2 = pd.read_csv("dataset2.csv")   # Loading dataset2
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")  # convert time to datetime
    
    # New column: rat_present
    df2["rat_present"] = df2["rat_minutes"] > 0

    # Splitting dataset into two groups: with and without rats
    no_rats = df2[df2["rat_present"] == False]
    with_rats = df2[df2["rat_present"] == True]

    # Creating Scatter Plot
    plt.figure(figsize=(14,8))

   
    # Scatter points for "No Rats Present"
    plt.scatter(no_rats["hours_after_sunset"], no_rats["bat_landing_number"],
                alpha=0.6, label="No Rats Present", s=30, edgecolors="black", linewidth=0.4)
    # Scatter points for "Rats Present"
    plt.scatter(with_rats["hours_after_sunset"], with_rats["bat_landing_number"],
                alpha=0.8, label="Rats Present", s=35, edgecolors="black", linewidth=0.4)

    # Adding linear trend lines for both groups
    def add_trend(df, style="--", label="Trend"):
        x = df["hours_after_sunset"].to_numpy()
        y = df["bat_landing_number"].to_numpy()
        if len(df) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.sort(x)
            plt.plot(xs, p(xs), style, linewidth=2, label=label)
 
     # Apply trend lines for both subsets
    add_trend(no_rats, style="b--", label="No Rats Trend")
    add_trend(with_rats, style="r--", label="Rats Present Trend")

    # Plot Labels and Title
    plt.xlabel("Hours after sunset", fontsize=14)
    plt.ylabel("Bat landing frequency (30-min block)", fontsize=14)
    plt.title("Bat Activity vs Rat Presence Over Time", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="upper right")

    # Statistical Analysis: Independent t-test
    nr = no_rats["bat_landing_number"].dropna()
    wr = with_rats["bat_landing_number"].dropna()

    if len(nr) >= 2 and len(wr) >= 2:
        t_stat, p_value = ttest_ind(nr, wr, equal_var=False, nan_policy="omit")

        # Classify significance level
        sig = ("Highly Significant***" if p_value < 0.001 else
               "Very Significant**"   if p_value < 0.01  else
               "Significant*"         if p_value < 0.05  else
               "Not Significant")

        # Formatting results summary
        stats_text = (
            "Independent t-test (Welch):\n"
            f"t = {t_stat:.3f}\n"
            f"p = {p_value:.2e}\n"
            f"Result: {sig}\n\n"
            f"Sample sizes:\n• No Rats: n={len(nr)}\n• With Rats: n={len(wr)}\n\n"
            f"Mean landings:\n• No Rats: {nr.mean():.1f}\n• With Rats: {wr.mean():.1f}"
        )
    else:
        stats_text = "Not enough data in one or both groups for a valid t-test."

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.9))

    # period annotation
    if df2["time"].notna().any():
        time_text = (
            f"Study Period: {df2['time'].min().strftime('%b %Y')} - {df2['time'].max().strftime('%b %Y')}\n"
            f"Total Observations (30-min blocks): {len(df2)}\n"
            f"Night Hours Covered: {df2['hours_after_sunset'].min():.1f} to {df2['hours_after_sunset'].max():.1f}"
        )
        plt.text(0.98, 0.02, time_text, transform=plt.gca().transAxes,
                 fontsize=10, va="bottom", ha="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    # Finalizing and Save Plot
    plt.tight_layout()
    plt.savefig("graph_activity_temporal.png", dpi=300, bbox_inches="tight")
     # enable if running interactively
    print("Saved: graph_activity_temporal.png")
    
#Running the analysis 
if __name__ == "__main__":
    create_activity_temporal_chart()
    