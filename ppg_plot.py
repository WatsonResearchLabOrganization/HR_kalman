import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ampdlib

def parse_biopac_txt(biopac_txt_file):
    """
    Parse a Biopac .txt export that includes:
      - Recording start time line (e.g., 'Recording on: 2025-02-21 11:05:38.634')
      - Sample rate line (e.g., '0.5 msec/sample')
      - Then data lines of the form:
          milliSec   CH1
          0.0        -0.0512695
          0.5        -0.0509644
          1.0        -0.0518799
          ...
    Returns a DataFrame with columns: ['time', 'Biopac'] (datetime + amplitude).
    """
    recording_start_dt = None
    sample_period_ms = None
    data_rows = []
    in_data_block = False

    with open(biopac_txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 1) Detect "Recording on: ..."
            rec_match = re.match(r"Recording on:\s*(.+)", line)
            if rec_match:
                date_str = rec_match.group(1).strip()  # e.g. '2025-02-21 11:05:38.634'
                recording_start_dt = pd.to_datetime(date_str, errors='coerce')
                continue

            # 2) Detect "0.5 msec/sample"
            sr_match = re.match(r"([\d\.]+)\s*msec/sample", line)
            if sr_match:
                sample_period_ms = float(sr_match.group(1))  # e.g. 0.5
                continue

            # 3) Data lines: "time_ms amplitude"
            parts = line.split()
            if len(parts) == 2:
                try:
                    time_ms = float(parts[0])
                    amplitude = float(parts[1])
                    in_data_block = True
                    data_rows.append((time_ms, amplitude))
                except ValueError:
                    # not valid numeric data, ignore
                    pass
            else:
                # if we had started reading data but see a different format, ignore
                if in_data_block:
                    pass

    if recording_start_dt is None:
        raise ValueError("Could not find 'Recording on:' line in Biopac file.")
    if sample_period_ms is None:
        raise ValueError("Could not find sample rate line (e.g. '0.5 msec/sample') in Biopac file.")
    if not data_rows:
        raise ValueError("No numeric data found in the Biopac file.")

    # Build DataFrame
    df_biopac = pd.DataFrame(data_rows, columns=['time_ms', 'Biopac_raw'])
    df_biopac['time'] = recording_start_dt + pd.to_timedelta(df_biopac['time_ms'], unit='ms')
    df_biopac.rename(columns={'Biopac_raw': 'Biopac'}, inplace=True)
    return df_biopac[['time', 'Biopac']]

def parse_ppg_csv(ppg_csv_file):
    """
    Parse the PPG CSV that has 'Timestamp' plus multiple '*_filtered' columns (and '*_peak' columns).
    Ignore peak columns, keep only the filtered columns + Timestamp.
    Returns a DataFrame: ['time', '415nm_filtered', '445nm_filtered', ..., 'Fused_filtered'].
    """
    df = pd.read_csv(ppg_csv_file)
    df['time'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    filtered_cols = [c for c in df.columns if c.endswith('_filtered')]
    keep_cols = ['time'] + filtered_cols
    df = df[keep_cols].copy()
    df.dropna(subset=['time'], inplace=True)
    return df.reset_index(drop=True)

def trim_to_overlap(dfA, dfB, time_col='time'):
    """
    Keep only the overlapping time range from two DataFrames dfA, dfB (by time_col).
    """
    startA, endA = dfA[time_col].min(), dfA[time_col].max()
    startB, endB = dfB[time_col].min(), dfB[time_col].max()
    overlap_start = max(startA, startB)
    overlap_end = min(endA, endB)

    if overlap_end < overlap_start:
        print("No overlapping time region found.")
        return dfA.iloc[0:0], dfB.iloc[0:0]  # empty

    dfA_trim = dfA[(dfA[time_col] >= overlap_start) & (dfA[time_col] <= overlap_end)].copy()
    dfB_trim = dfB[(dfB[time_col] >= overlap_start) & (dfB[time_col] <= overlap_end)].copy()
    return dfA_trim, dfB_trim

def run_ampd(signal_array, window_len=160):
    """
    Run AMPD on a 1D array with a specified window length.
    Return the indices of detected peaks.
    """
    if len(signal_array) < window_len:
        return np.array([])  # no peaks if not enough samples
    peaks = ampdlib.ampd_fast(signal_array, window_length=window_len)
    return peaks

def main(ppg_csv_file, biopac_txt_file,
         ppg_ampd_window=160,
         biopac_ampd_window=400):
    """
    Compare PPG CSV vs Biopac text.  Each channel in the PPG will be plotted
    against the Biopac data, with separate AMPD windows for each.
      - ppg_ampd_window: AMPD window length for PPG data
      - biopac_ampd_window: AMPD window length for Biopac data
    """
    # 1) Parse the data
    df_ppg = parse_ppg_csv(ppg_csv_file)
    df_biopac = parse_biopac_txt(biopac_txt_file)

    # 2) Trim to overlapping time
    df_ppg, df_biopac = trim_to_overlap(df_ppg, df_biopac, time_col='time')
    if df_ppg.empty or df_biopac.empty:
        print("No overlapping data to plot.")
        return

    # Sort by time
    df_ppg.sort_values(by='time', inplace=True)
    df_biopac.sort_values(by='time', inplace=True)

    # 3) Identify the PPG channels
    ppg_channels = [c for c in df_ppg.columns if c != 'time']

    # 4) Plot each PPG channel vs Biopac
    for ch_name in ppg_channels:
        ppg_time = df_ppg['time'].values
        ppg_values = df_ppg[ch_name].values

        bio_time = df_biopac['time'].values
        bio_values = df_biopac['Biopac'].values

        # 5) Detect peaks (AMPD) using different window lengths
        ppg_peaks_idx = run_ampd(ppg_values, window_len=ppg_ampd_window)
        bio_peaks_idx = run_ampd(bio_values, window_len=biopac_ampd_window)

        ppg_peaks_time = ppg_time[ppg_peaks_idx]
        ppg_peaks_vals = ppg_values[ppg_peaks_idx]

        bio_peaks_time = bio_time[bio_peaks_idx]
        bio_peaks_vals = bio_values[bio_peaks_idx]

        # 6) Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ppg_time, ppg_values, label=f"{ch_name} (PPG)")
        ax.plot(ppg_peaks_time, ppg_peaks_vals, 'ro', label=f"{ch_name} peaks")

        ax.plot(bio_time, bio_values, label="Biopac")
        ax.plot(bio_peaks_time, bio_peaks_vals, 'kx', label="Biopac peaks")

        ax.set_title(f"{ch_name} vs Biopac\nPPG window={ppg_ampd_window}, Biopac window={biopac_ampd_window}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

        ax.legend(loc='best')
        fig.tight_layout()
        plt.show()

    print("Done plotting.")

if __name__ == "__main__":
    # Example usage: 
    # python compare_ppg_biopac.py ppg_data.csv Untitled1.acq.txt
    #
    # Or pass additional arguments for separate window lengths
    ppg_csv_file = "ppg_cx_hold.csv"
    biopac_txt_file = "cx_hold_ecg.txt"

    # set your desired AMPD windows
    PPG_WINDOW = 160
    BIOPAC_WINDOW = 15000

    main(ppg_csv_file, biopac_txt_file,
         ppg_ampd_window=PPG_WINDOW,
         biopac_ampd_window=BIOPAC_WINDOW)
