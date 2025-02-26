import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# TOGGLE THIS SWITCH to decide whether or not to include the Apple Watch data
# ---------------------------------------------------------------------------
ENABLE_WATCH = False

def read_ir_csv(csv_path):
    """
    Reads the IR CSV with columns:
        Timestamp (e.g. 2025-02-04 11:06:51.420916)
        Fused_HR
    Returns a DataFrame with columns ['Timestamp', 'HR'].
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.rename(columns={"Fused_HR": "HR"}, inplace=True)
    return df

def read_biopac_txt(txt_path):
    """
    Reads the Biopac .txt file. (Ground truth)
    Should contain lines like:
        Recording on: 2025-02-04 11:06:53.511
        0     -0.296936   0
        0.5   -0.296936   0
        ...
    1) Parse the "Recording on: ..." line for the start datetime -> start_dt.
    2) Parse the table of (milliseconds, ppg, BPM).
       - Time is in the first column (ms),
       - BPM is in the third column.
    3) Ignore rows where BPM=0.
    4) Convert the time in ms to absolute datetimes by adding to start_dt.
    
    Returns:
        df: DataFrame with ['Timestamp', 'HR']
        start_dt: The absolute datetime corresponding to time=0 in this dataset
    """
    start_dt = None
    data_rows = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            # Parse the "Recording on: ..." line
            if line.startswith("Recording on:"):
                # e.g. "Recording on: 2025-02-04 11:06:53.511"
                date_str = line.split("Recording on:")[-1].strip()
                start_dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
                continue

            # Parse lines with data: "milliSec  CH1  CH2"
            parts = line.split()
            if len(parts) == 3:
                try:
                    ms_float = float(parts[0])
                    ppg_val = float(parts[1])   # not used here, but read if needed
                    bpm_val = float(parts[2])

                    # if BPM=0, skip
                    if bpm_val == 0:
                        continue

                    # Convert ms offset to datetime
                    time_offset = timedelta(milliseconds=ms_float)
                    actual_ts = start_dt + time_offset
                    data_rows.append([actual_ts, bpm_val])
                except ValueError:
                    pass

    df = pd.DataFrame(data_rows, columns=["Timestamp", "HR"])
    return df, start_dt

def read_watch_csv(csv_path):
    """
    Reads the new Apple Watch CSV exported from HealthKit.
    If the file starts with "sep=," on the first line (meaningless row),
    we can skip that row with skiprows=1 or 0, depending on your file.
    """
    # Adjust skiprows if your file has a "sep=," line. For example:
    # df = pd.read_csv(csv_path, skiprows=1)
    df = pd.read_csv(csv_path, skiprows=0)

    # Parse the startDate as the main Timestamp
    df["Timestamp"] = pd.to_datetime(df["startDate"])

    # Rename 'value' to 'HR'
    df.rename(columns={"value": "HR"}, inplace=True)

    # Keep only the columns we need
    df = df[["Timestamp", "HR"]].copy()

    # Convert HR to float (if it's not already)
    df["HR"] = df["HR"].astype(float)

    return df

def filter_time_range(df, start_time, end_time):
    """
    Filter the given DataFrame (must have 'Timestamp' column) to keep
    only rows between start_time and end_time (inclusive).
    """
    return df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()

def compare_hr(df_test, df_groundtruth):
    """
    Compare HR from df_test to df_groundtruth (Biopac).
    
    For each row in df_test, find the row in df_groundtruth 
    with the closest timestamp, then compute the difference in HR.
    
    Returns: (mean_error, count)
        mean_error = average (df_test.HR - df_groundtruth.HR)
        count      = number of matched rows
    """
    df_test = df_test.sort_values("Timestamp").reset_index(drop=True)
    df_groundtruth = df_groundtruth.sort_values("Timestamp").reset_index(drop=True)
    
    # Merge on nearest timestamp within a large tolerance
    merged = pd.merge_asof(
        df_test, 
        df_groundtruth[["Timestamp", "HR"]], 
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1D")  # 1 day tolerance => effectively always finds a match 
    )
    
    # Drop rows where no match was found
    merged.dropna(subset=["HR_y"], inplace=True)
    
    # difference: (df_test.HR) - (df_groundtruth.HR)
    merged["Error"] = merged["HR_x"] - merged["HR_y"]
    
    mean_error = merged["Error"].mean()
    count = len(merged)
    
    return mean_error, count

def plot_three_datasets(df_biopac, df_ir, df_watch, start_time, end_time):
    """
    Plot three heart-rate datasets over the same time window:
      - Biopac (ground truth)
      - IR (original CSV)
      - Watch (new Apple Watch CSV)
    """
    plt.figure(figsize=(10, 6))

    # Plot Biopac in black
    plt.plot(df_biopac["Timestamp"], df_biopac["HR"], 
             label="Biopac (Ground Truth)", marker='o', linestyle='-', color='black')
    
    # Plot IR in blue
    plt.plot(df_ir["Timestamp"], df_ir["HR"], 
             label="IR CSV", marker='x', linestyle='--', color='blue')
    
    # Only plot Watch if ENABLE_WATCH is True
    if ENABLE_WATCH and (df_watch is not None and not df_watch.empty):
        plt.plot(df_watch["Timestamp"], df_watch["HR"], 
                 label="Watch CSV", marker='^', linestyle=':', color='red')

    # Optionally zoom into the start/end times
    plt.xlim([start_time, end_time])
    
    plt.title("Comparison of Heart Rates (Biopac vs. IR vs. Watch)")
    plt.xlabel("Timestamp")
    plt.ylabel("Heart Rate (BPM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Read IR CSV
    df_ir = read_ir_csv("compare_with_multiwave.csv")
    
    # 2. Read Biopac text
    df_biopac, biopac_start_dt = read_biopac_txt("Compare_with_multiwave_biopac.txt")
    
    # 3. Optionally read Apple Watch CSV, depending on ENABLE_WATCH
    df_watch = None
    if ENABLE_WATCH:
        df_watch = read_watch_csv("HKQuantityTypeIdentifierHeartRate_2025-01-29_19-58-10_SimpleHealthExportCSV.csv")

    # Example time range using the Biopac timeline
    rel_start_sec = 33
    rel_end_sec   = 133

    start_time_abs = biopac_start_dt + timedelta(seconds=rel_start_sec)
    end_time_abs   = biopac_start_dt + timedelta(seconds=rel_end_sec)
    
    # Filter IR and Biopac
    df_ir_filtered     = filter_time_range(df_ir, start_time_abs, end_time_abs)
    df_biopac_filtered = filter_time_range(df_biopac, start_time_abs, end_time_abs)
    
    # Compare IR vs Biopac
    mean_error_ir, count_ir = compare_hr(df_ir_filtered, df_biopac_filtered)
    
    print("Time window:")
    print(f"  {start_time_abs} to {end_time_abs}\n")
    
    print("IR vs. Biopac:")
    print(f"  Matched rows:  {count_ir}")
    print(f"  Mean error (IR - Biopac): {mean_error_ir:.2f}\n")
    
    # If watch is enabled, filter & compare
    df_watch_filtered = None
    mean_error_watch = None
    count_watch = None
    if ENABLE_WATCH and df_watch is not None:
        df_watch_filtered = filter_time_range(df_watch, start_time_abs, end_time_abs)
        mean_error_watch, count_watch = compare_hr(df_watch_filtered, df_biopac_filtered)
        print("Watch vs. Biopac:")
        print(f"  Matched rows:  {count_watch}")
        print(f"  Mean error (Watch - Biopac): {mean_error_watch:.2f}\n")
    
    # Plot all three (the function will skip watch if df_watch_filtered is empty or None)
    plot_three_datasets(df_biopac_filtered, df_ir_filtered, df_watch_filtered,
                        start_time_abs, end_time_abs)

if __name__ == "__main__":
    main()
