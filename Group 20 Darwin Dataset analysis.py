# HIT140 Assessment 2 Group 20 Darwin
# Aryan Rayamajhi 
# Samir Rimal       
# Rohan Baniya
# Sakar Khadka
# Data Processor for Investigation A
# Cleaning data, making graphs and doing a simple regression test


import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Changing file names for the files to be used
DATASET1 = "dataset1.csv"   # DATAST1 = "dataset1.csv"
DATASET2 = "dataset2.csv"   # DATAST2 = "dataset2.csv"

#Converting strings to datetime
def parse_dt(series, primary_fmt="%d/%m/%Y %H:%M"):
    s = pd.to_datetime(series, format=primary_fmt, errors="coerce")
    if s.isna().mean() > 0.2:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s

def main():
    # Check files exist
    if not os.path.exists(DATASET1) or not os.path.exists(DATASET2):
        raise FileNotFoundError("Place both datasets in the same folder as this script.")
    # Creates outputs if it doesn’t exist
    os.makedirs("outputs", exist_ok=True)
    figpath = lambda name: os.path.join("outputs", name)

    # Loading the datasets
    d1 = pd.read_csv(DATASET1)
    d2 = pd.read_csv(DATASET2)

    # Dsplaying missing values
    print("\nMissing values in datasets:")
    print("dataset1:\n", d1.isnull().sum())
    print("\ndataset2:\n", d2.isnull().sum())

    # Cleaning the datasets
    # Filling missing values in 'habit' column if present
    if "habit" in d1.columns:
        d1["habit"] = d1["habit"].fillna("unknown")
    # Parsing datetime columns in dataset1
    for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
        if col in d1.columns:
            d1[col] = parse_dt(d1[col])
    if "time" in d2.columns:
        d2["time"] = parse_dt(d2["time"])

    # Merging to add 30-min block context 
    # Aligns dataset1 rows with the corresponding 30-min block 
    if "start_time" in d1.columns and "time" in d2.columns:
        d1["time_block"] = d1["start_time"].dt.floor("30min")
        keep2 = [c for c in ["time","rat_minutes","rat_arrival_number","bat_landing_number","food_availability"] if c in d2.columns]
        merged = pd.merge(d1, d2[keep2], left_on="time_block", right_on="time", how="left")
    else:
        merged = d1.copy()

    # Saving the cleaned datasets
    d1.to_csv("outputs/clean_dataset1.csv", index=False)
    d2.to_csv("outputs/clean_dataset2.csv", index=False)
    merged.to_csv("outputs/merged_dataset.csv", index=False)
    
    # Graph 1
    #  Creating  Histogram of seconds after rat arrival by risk level
    if {"risk","seconds_after_rat_arrival"}.issubset(d1.columns):
        plt.figure(figsize=(7,4))
        for r in sorted(d1["risk"].dropna().unique()):
            vals = d1.loc[d1["risk"] == r, "seconds_after_rat_arrival"].dropna()
            if len(vals):
                plt.hist(vals, bins=30, alpha=0.7, edgecolor="black", label=f"Risk {r}")
        plt.title("Seconds After Rat Arrival by Risk")
        plt.xlabel("Seconds After Rat Arrival")
        plt.ylabel("Count")
        if d1["risk"].notna().any():
            plt.legend(title="Risk Behaviour (0=avoid,1=risk)")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(figpath("histogram_risk_vs_time.png"), dpi=300)
        plt.close()

    # Graph 2
    # Creating visual mean risk vs rat minutes bins
    
    if {"rat_minutes","risk"}.issubset(merged.columns):
        tmp = merged.dropna(subset=["rat_minutes","risk"]).copy()
        if not tmp.empty:
            tmp["rat_minutes_bin"] = pd.qcut(tmp["rat_minutes"], q=6, duplicates="drop")
            mean_risk = tmp.groupby("rat_minutes_bin", observed=False)["risk"].mean()
            plt.figure(figsize=(7,4))
            mean_risk.plot(kind="bar", edgecolor="black")
            plt.title("Mean Risk-Taking vs Rat Minutes (30-min windows)")
            plt.xlabel("Rat Minutes (binned)")
            plt.ylabel("Mean Risk (0=avoid, 1=risk)")
            plt.xticks(rotation=45)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(figpath("bar_risk_vs_rat_minutes.png"), dpi=300)
            plt.close()
    # Regression Analysis 
    # Do bats treat rats as predators 
    # Risk
    
    out_lines = []
    cols = [c for c in ["risk","seconds_after_rat_arrival","hours_after_sunset","bat_landing_to_food"] if c in d1.columns]
    d = d1[cols].replace([np.inf,-np.inf], np.nan).dropna()

    # Runs only if regression data exists
    if {"risk"}.issubset(d.columns) and d["risk"].nunique()>1 and d.shape[0]>20:
        y = d["risk"].astype(int)
        X = sm.add_constant(d.drop(columns=["risk"]), has_constant="add")
        try:
            model = sm.Logit(y, X).fit(disp=False)
            out_lines.append("Model: Logit (dataset1)\n")
            out_lines.append(str(model.summary()))
            out_lines.append("\nOdds ratios:\n" + str(np.exp(model.params)))
        except Exception as e:
            out_lines.append(f"Logit failed: {e}")
    else:
        out_lines.append("Insufficient variation or rows for logistic regression.")
    
    # Saving regression results to text file
    with open("outputs/stats_investigationA.txt","w",encoding="utf-8") as f:
        f.write("Investigation A: Do bats treat rats as potential predators? (risk/vigilance)\n\n")
        f.write("\n".join(out_lines) + "\n")

    print("Saved to outputs")
    print("Clean_dataset1.csv, Clean_dataset2.csv, Merged_dataset.csv")
    print("histogram_risk_vs_time.png, bar_risk_vs_rat_minutes.png")
    print("Investigation A.txt")

# Running the main function
if __name__ == "__main__":
    main()
