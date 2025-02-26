import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ampdlib

def parse_biopac_txt(biopac_txt_file):
    """
    Parse a Biopac .txt export that includes:
      - 'Recording on: yyyy-mm-dd HH:MM:SS.mmm'
      - 'X msec/sample'
      - Then data lines: 'time_ms amplitude'
    Returns a DataFrame with columns ['time', 'Biopac'].
    """
    recording_start_dt = None
    sample_period_ms = None
    data_rows = []

    with open(biopac_txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Detect "Recording on: ..."
            if line.startswith("Recording on:"):
                date_str = line.replace("Recording on:", "").strip()
                recording_start_dt = pd.to_datetime(date_str, errors='coerce')
                continue

            # Detect "X msec/sample"
            match_sr = re.match(r"([\d\.]+)\s*msec/sample", line)
            if match_sr:
                sample_period_ms = float(match_sr.group(1))
                continue

            # Try parsing data lines "time_ms amplitude"
            parts = line.split()
            if len(parts) == 2:
                try:
                    time_ms = float(parts[0])
                    amplitude = float(parts[1])
                    data_rows.append((time_ms, amplitude))
                except ValueError:
                    pass

    if recording_start_dt is None:
        raise ValueError("Could not find 'Recording on:' line in Biopac file.")
    if sample_period_ms is None:
        raise ValueError("Could not find sample rate line (e.g. '0.5 msec/sample').")

    # Build DataFrame
    df_biopac = pd.DataFrame(data_rows, columns=['time_ms', 'Biopac_raw'])
    df_biopac['time'] = recording_start_dt + pd.to_timedelta(df_biopac['time_ms'], unit='ms')
    df_biopac.rename(columns={'Biopac_raw': 'Biopac'}, inplace=True)

    # Ensure 0-based integer index
    df_biopac.reset_index(drop=True, inplace=True)

    return df_biopac[['time', 'Biopac']]


def parse_ppg_csv(ppg_csv_file):
    """
    Parse the PPG CSV that has 'Timestamp' plus multiple '*_filtered' columns.
    Returns a DataFrame: ['time', <channel1_filtered>, <channel2_filtered>, ...].
    """
    df = pd.read_csv(ppg_csv_file)
    df['time'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # Keep only filtered columns + 'time'
    filtered_cols = [c for c in df.columns if c.endswith('_filtered')]
    keep_cols = ['time'] + filtered_cols
    df = df[keep_cols].copy()
    df.dropna(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def trim_to_overlap(dfA, dfB, time_col='time'):
    """
    Keep only the overlapping time range from two DataFrames dfA, dfB.
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
    Returns the indices of detected peaks as a NumPy array.
    """
    if len(signal_array) < window_len:
        return np.array([])
    peaks_idx = ampdlib.ampd_fast(signal_array, window_length=window_len)
    return peaks_idx


def get_peaks_time(df, col_name, ampd_window=160):
    """
    Given a DataFrame with 'time' and a column col_name,
    detect peaks and return the array of peak times (as np.array).

    Using df.iloc[...] so that 0-based indices from AMPD
    map correctly to the DataFrame rows.
    """
    signal = df[col_name].values
    peaks_idx = run_ampd(signal, window_len=ampd_window)
    peak_times = df.iloc[peaks_idx]['time'].values
    return peak_times


def compute_instantaneous_hr(peak_times):
    """
    Given sorted array of peak_times (pd or np datetime64),
    compute instantaneous HR from consecutive peak intervals:
       HR = 60 / (time_diff_in_seconds).
    We assign the resulting HR to the midpoint time between each pair of peaks.

    Returns: two arrays (times, hrs)
    """
    if len(peak_times) < 2:
        return np.array([]), np.array([])

    peak_times = np.sort(peak_times)  # ensure sorted
    times_list = []
    hrs_list = []

    for i in range(1, len(peak_times)):
        t0 = peak_times[i - 1]
        t1 = peak_times[i]
        dt_sec = (t1 - t0) / np.timedelta64(1, 's')
        if dt_sec > 0:
            hr = 60.0 / dt_sec
            t_mid = t0 + 0.5 * (t1 - t0)
            times_list.append(t_mid)
            hrs_list.append(hr)

    return np.array(times_list), np.array(hrs_list)


def main(ppg_csv_file, biopac_txt_file,
         nir_col='NIR_filtered',
         fused_col='Fused_filtered',
         ppg_ampd_window=160,
         biopac_ampd_window=4000):
    """
    1. Load data, trim overlap.
    2. Detect peaks (NIR, Fused, Biopac).
    3. Compute instantaneous HR from consecutive peaks.
    4. Filter out abnormal HR (<40 or >200).
    5. Merge with groundtruth on nearest time (for alignment).
    6. Plot on a 45° line: x = HR_bio, y = HR_nir/fused, plus identity line.
    """

    # 1) Parse
    df_ppg = parse_ppg_csv(ppg_csv_file)
    df_biopac = parse_biopac_txt(biopac_txt_file)

    # Trim overlap
    df_ppg, df_biopac = trim_to_overlap(df_ppg, df_biopac)
    if df_ppg.empty or df_biopac.empty:
        print("No overlapping data found.")
        return

    # 2) Detect peaks
    if nir_col not in df_ppg.columns:
        raise ValueError(f"NIR column '{nir_col}' not found in PPG CSV.")
    if fused_col not in df_ppg.columns:
        raise ValueError(f"Fused column '{fused_col}' not found in PPG CSV.")

    nir_peak_times = get_peaks_time(df_ppg, nir_col, ampd_window=ppg_ampd_window)
    fused_peak_times = get_peaks_time(df_ppg, fused_col, ampd_window=ppg_ampd_window)
    biopac_peak_times = get_peaks_time(df_biopac, 'Biopac', ampd_window=biopac_ampd_window)

    # 3) Compute instantaneous HR
    t_nir, hr_nir = compute_instantaneous_hr(nir_peak_times)
    t_fused, hr_fused = compute_instantaneous_hr(fused_peak_times)
    t_bio, hr_bio = compute_instantaneous_hr(biopac_peak_times)

    # Convert to DataFrames
    df_nir_hr = pd.DataFrame({'time': t_nir, 'HR_nir': hr_nir})
    df_fused_hr = pd.DataFrame({'time': t_fused, 'HR_fused': hr_fused})
    df_bio_hr = pd.DataFrame({'time': t_bio, 'HR_bio': hr_bio})

    # Sort before merging
    df_nir_hr.sort_values('time', inplace=True)
    df_fused_hr.sort_values('time', inplace=True)
    df_bio_hr.sort_values('time', inplace=True)

    # Merge onto Biopac
    df_merged = pd.merge_asof(df_bio_hr, df_nir_hr, on='time', direction='nearest', tolerance=pd.Timedelta(seconds=2))
    df_merged = pd.merge_asof(df_merged, df_fused_hr, on='time', direction='nearest', tolerance=pd.Timedelta(seconds=2))

    # 4) Filter out abnormal HR
    def valid_hr(x):
        return 40.0 <= x <= 200.0

    df_merged['HR_bio'] = df_merged['HR_bio'].apply(lambda x: x if valid_hr(x) else np.nan)
    df_merged.dropna(subset=['HR_bio'], inplace=True)  # must have groundtruth

    df_merged['HR_nir'] = df_merged['HR_nir'].apply(lambda x: x if valid_hr(x) else np.nan)
    df_merged['HR_fused'] = df_merged['HR_fused'].apply(lambda x: x if valid_hr(x) else np.nan)

    # 5) We'll compute MSE for all valid pairs
    df_nir_valid = df_merged.dropna(subset=['HR_nir'])
    df_fused_valid = df_merged.dropna(subset=['HR_fused'])

    mse_nir = np.nan
    mse_fused = np.nan
    if not df_nir_valid.empty:
        mse_nir = np.mean((df_nir_valid['HR_nir'] - df_nir_valid['HR_bio'])**2)
    if not df_fused_valid.empty:
        mse_fused = np.mean((df_fused_valid['HR_fused'] - df_fused_valid['HR_bio'])**2)

    print(f"MSE (NIR vs Biopac)   = {mse_nir}")
    print(f"MSE (Fused vs Biopac) = {mse_fused}")

    # 6) Plot a 45° identity line, groundtruth on X, estimated on Y
    # -------------------------------------------------------------
    # We'll do one figure with two scatter plots:
    #   - NIR vs Bio
    #   - Fused vs Bio
    # plus the line y = x from min to max HR.

    plt.figure(figsize=(7, 6))

    # Prepare x,y for each estimate
    # (We only plot the rows that have valid values for each estimate).
    # For NIR
    bio_x_nir = df_nir_valid['HR_bio']
    nir_y = df_nir_valid['HR_nir']

    # For Fused
    bio_x_fused = df_fused_valid['HR_bio']
    fused_y = df_fused_valid['HR_fused']

    plt.scatter(bio_x_nir, nir_y, color='red', alpha=0.7, label='NIR')
    plt.scatter(bio_x_fused, fused_y, color='black', alpha=0.7, label='Fused')

    # Identity line: we can define from the min to max of the data or a fixed 40–200 range.
    all_bio_vals = df_merged['HR_bio'].dropna()
    min_hr = max(40, all_bio_vals.min())  # at least 40
    max_hr = min(200, all_bio_vals.max()) # at most 200
    line_x = np.linspace(min_hr, max_hr, 200)
    line_y = line_x
    plt.plot(line_x, line_y, 'b--', label='y=x (45° line)')

    plt.title(f"HR: Comparison vs Ground Truth\n"
              f"MSE(NIR)={mse_nir:.2f}, MSE(Fused)={mse_fused:.2f}")
    plt.xlabel("Ground Truth (Biopac) BPM")
    plt.ylabel("Estimated BPM")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ppg_csv_file = "yhc_hold_breath_5mins.csv"
    biopac_txt_file = "yhc_hold_breath_5mins.txt"

    main(ppg_csv_file, biopac_txt_file,
         nir_col="NIR_filtered",
         fused_col="Fused_filtered",
         ppg_ampd_window=160,
         biopac_ampd_window=4000)
