import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from filterpy.kalman import KalmanFilter
import ampdlib

# =======================
# Configuration
# =======================
LOWCUT = 0.1
HIGHCUT = 10.0
FILTER_ORDER = 1
AMPD_WINDOW_LENGTH = 300
WINDOW_SIZE = 5.0     # seconds
STEP_SIZE = 1.0       # seconds

DATAFOLDER = r"physionet.org\files\pulse-transit-time-ppg\1.1.0\csv"
LOG_FILE = "evaluation_log.txt"

# If you want to see an interactive waveform + ground-truth for a specific file, set it here:
# e.g. PLOT_SPECIFIC_FILE = "s6_sit.csv"
PLOT_SPECIFIC_FILE = ""

# =======================
# Helper Functions
# =======================
def bandpass_filter(signal, fs, lowcut, highcut, order):
    """Apply a Butterworth bandpass filter to the signal."""
    b, a = butter(order, [lowcut, highcut], btype="band", fs=fs)
    return filtfilt(b, a, signal)

def detect_peaks_ampd(signal, window_length=300):
    """Detect peaks in 'signal' using AMPD."""
    if len(signal) >= window_length:
        peaks_idx = ampdlib.ampd_fast(signal, window_length=window_length)
    else:
        peaks_idx = np.array([])
    return peaks_idx

def compute_local_hr(peak_times, center_time, half_window):
    """
    Compute average HR (BPM) for the peaks in [center_time - half_window, center_time + half_window].
    """
    t_start = center_time - half_window
    t_end   = center_time + half_window
    mask = (peak_times >= t_start) & (peak_times <= t_end)
    window_peaks = peak_times[mask]
    if len(window_peaks) < 2:
        return np.nan
    rr_intervals = np.diff(window_peaks)
    if len(rr_intervals) == 0:
        return np.nan
    local_bpm = 60.0 / np.mean(rr_intervals)
    return local_bpm

def generate_hr_timeseries(peak_times, total_duration, window_size, step_size):
    """Create a sliding-window HR time series for the entire record."""
    half_w = window_size / 2.0
    centers = np.arange(0.0 + half_w, total_duration - half_w, step_size)
    hr_list = []
    for ct in centers:
        hr_est = compute_local_hr(peak_times, ct, half_w)
        hr_list.append(hr_est)
    return centers, np.array(hr_list)

def evaluate_time_series_hr(gt_times, fused_times, IR_times, total_dur):
    """
    Build time-aligned HR (BPM) for ground truth (GT), fused, and IR.
    Compute RMSE for (fused vs GT) and (IR vs GT).
    Returns (rmse_fused, rmse_IR, detail).
    """
    gt_t,  gt_hr  = generate_hr_timeseries(gt_times,     total_dur, WINDOW_SIZE, STEP_SIZE)
    fu_t,  fu_hr  = generate_hr_timeseries(fused_times,  total_dur, WINDOW_SIZE, STEP_SIZE)
    pl_t,  pl_hr  = generate_hr_timeseries(IR_times,     total_dur, WINDOW_SIZE, STEP_SIZE)

    # unify times
    common_times = set(gt_t).intersection(fu_t).intersection(pl_t)
    common_times = np.array(sorted(list(common_times)))
    if len(common_times) == 0:
        return np.nan, np.nan, (gt_t, gt_hr, fu_t, fu_hr, pl_t, pl_hr)
    
    def timeseries_to_dict(tt, hh):
        return {round(ti, 5): hi for ti, hi in zip(tt, hh)}
    
    gt_dict  = timeseries_to_dict(gt_t, gt_hr)
    fu_dict  = timeseries_to_dict(fu_t, fu_hr)
    pl_dict  = timeseries_to_dict(pl_t, pl_hr)

    gt_vals = []
    fu_vals = []
    pl_vals = []
    for ct in common_times:
        gt_vals.append(gt_dict.get(round(ct, 5), np.nan))
        fu_vals.append(fu_dict.get(round(ct, 5), np.nan))
        pl_vals.append(pl_dict.get(round(ct, 5), np.nan))

    gt_vals = np.array(gt_vals)
    fu_vals = np.array(fu_vals)
    pl_vals = np.array(pl_vals)

    # RMSE ignoring NaNs
    valid_mask_fu = ~np.isnan(gt_vals) & ~np.isnan(fu_vals)
    valid_mask_pl = ~np.isnan(gt_vals) & ~np.isnan(pl_vals)

    if valid_mask_fu.sum() == 0:
        rmse_fused = np.nan
    else:
        rmse_fused = np.sqrt(np.mean((fu_vals[valid_mask_fu] - gt_vals[valid_mask_fu])**2))

    if valid_mask_pl.sum() == 0:
        rmse_IR = np.nan
    else:
        rmse_IR = np.sqrt(np.mean((pl_vals[valid_mask_pl] - gt_vals[valid_mask_pl])**2))

    return rmse_fused, rmse_IR, (gt_t, gt_hr, fu_t, fu_hr, pl_t, pl_hr)

def kalman_fuse_3channels(df, channels, fs):
    """
    1) Bandpass each channel individually,
    2) Estimate an 'SNR' for each channel sample (toy example),
    3) Form a single scalar measurement z(t) = sum_i [ SNR_i(t) * pleth_i(t) ],
    4) Use a 1D Kalman Filter (dim_x=1, dim_z=1) to track that sum over time,
    5) Return the fused signal (which can exceed the largest single-channel amplitude).

    NOTE: This is a non-traditional 'fusion': 
       - Instead of H = [1,1,1]^T and a multi-measurement update, 
         we define a single measurement = (SNR-weighted sum).
       - Weighted by SNR so that high-SNR channels push z(t) up more strongly.

    You can replace the SNR calculation logic with your own (e.g., rolling SNR, 
    or some known SNR from a separate process). 
    """

    from filterpy.kalman import KalmanFilter

    # 1) Filter each channel
    filtered_signals = []
    for ch in channels:
        raw_data = df[ch].to_numpy()
        filtered_data = bandpass_filter(raw_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER)
        filtered_signals.append(filtered_data)
    # shape = (3, N)
    filtered_signals = np.array(filtered_signals)

    N = filtered_signals.shape[1]

    # 2) Estimate "SNR" for each channel at each sample
    #    Here, we do something *very naive*: SNR_i(t) = abs(filtered_signals[i,t]) 
    #    with some small floor to avoid zero.
    #    Replace with your own real SNR estimation!
    snr_vals = np.abs(filtered_signals) + 1.0  # shape = (3, N)

    # 3) Build the single measurement z(t) at each time step
    #    z(t) = sum_i [ SNR_i(t) * channel_i(t) ]
    #    This can exceed the largest channel if the others are in phase or large.
    z_measurements = []
    for t in range(N):
        # sum of SNR_i(t) * pleth_i(t)
        z_t = 0.0
        for i in range(len(channels)):
            z_t += snr_vals[i,t] * filtered_signals[i,t]
        z_measurements.append(z_t)
    z_measurements = np.array(z_measurements)

    # 4) Setup a single-dimension Kalman Filter
    kf = KalmanFilter(dim_x=1, dim_z=1)
    # state = fused amplitude
    initial_val = z_measurements[0]  # or some smaller guess
    kf.x = np.array([[initial_val]])
    kf.P *= 10.0
    kf.F = np.array([[1.]])    # we assume x_{k+1} = x_k + process_noise
    kf.H = np.array([[1.]])    # measurement = x + measurement_noise
    kf.Q = np.array([[0.1]])   # tune this for more/less smoothing

    fused_signal = []
    P_history = []

    for t in range(N):
        # We want R to be smaller if sum of SNR_i(t) is large => we trust z(t) strongly
        sum_snr_t = np.sum(snr_vals[:, t])
        # e.g. pick R(t) = 1 / (sum_snr_t^2) or some function
        # or set a floor so we don't get too small
        R_t = 1.0 / (sum_snr_t**2 + 1.0)  # toy example
        kf.R = np.array([[R_t]]) 

        # standard KF steps
        kf.predict()
        kf.update(z_measurements[t])
        fused_signal.append(kf.x[0, 0])
        P_history.append(kf.P[0, 0])

    fused_signal = np.array(fused_signal)

    return fused_signal, np.array(P_history), filtered_signals



def plot_all_signals_and_kalmanP(df,
                                 filtered_signals,  # shape: (3, N)
                                 fused_signal, fused_peaks_idx,
                                 IR_filtered, IR_peaks_idx,
                                 P_history):
    """
    Plots, for a single file:
      1) The three filtered PPG channels, with ground-truth peaks as vlines.
      2) The fused PPG, with fused and IR peaks, and ground-truth peaks.
      3) The Kalman filter's error covariance (P) over time.
    """
    time_vals = df["time_sec"].values

    # Ground truth peak times
    gt_ecg_times = df.loc[df["peaks"] == 1, "time_sec"].values
    
    # Create 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("PPG Channels, Fused Signal, and Kalman Covariance")

    # Subplot #1: The three PPG channels
    axs[0].set_title("Filtered PPG Channels (pleth_1, pleth_2, pleth_3) + ECG GT peaks")
    axs[0].plot(time_vals, filtered_signals[0], label="pleth_1")
    axs[0].plot(time_vals, filtered_signals[1], label="pleth_2")
    axs[0].plot(time_vals, filtered_signals[2], label="pleth_3")
    # Vertical lines for ground truth peaks
    y_min, y_max = axs[0].get_ylim()
    axs[0].vlines(gt_ecg_times, y_min, y_max, color='g', alpha=0.3, linewidth=1.0, label="ECG GT")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot #2: Fused signal, fused peaks, IR peaks, GT peaks
    axs[1].set_title("Fused Signal + Detected Peaks + ECG GT peaks")
    axs[1].plot(time_vals, fused_signal, label="Fused")
    if len(fused_peaks_idx):
        axs[1].plot(time_vals[fused_peaks_idx], fused_signal[fused_peaks_idx],
                    'ro', markersize=5, label="Fused Peaks")
    # We can also plot IR peaks here
    if len(IR_peaks_idx):
        axs[1].plot(time_vals[IR_peaks_idx], IR_filtered[IR_peaks_idx],
                    'kx', markersize=5, label="IR (pleth_2) Peaks")
    # Ground truth peak lines
    y_min2, y_max2 = axs[1].get_ylim()
    axs[1].vlines(gt_ecg_times, y_min2, y_max2, color='g', alpha=0.3, linewidth=1.0)
    axs[1].legend()
    axs[1].grid(True)

    # Subplot #3: Kalman filter P
    axs[2].set_title("Kalman Filter Error Covariance (P) over time")
    axs[2].plot(time_vals, P_history, label="P(t)")
    axs[2].legend()
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main function
# --------------------------------------------------
def process_all_csvs(datafolder):
    """
    1. Iterate CSV files that match s*_*.csv.
    2. Skip invalid files.
    3. Write RMSE results to 'evaluation_log.txt' AND print them to console.
    4. For each file (or the PLOT_SPECIFIC_FILE), plot:
         - Three channels + GT peaks
         - Fused signal + peaks + GT peaks
         - Kalman P over time
    5. Produce bar charts of the differences in RMSE if desired.
    """
    csv_files = glob.glob(os.path.join(datafolder, "s*_*.csv"))
    csv_files = sorted(csv_files)

    # We'll track data in lists
    filenames = []
    rmse_fused_list = []
    rmse_IR_list = []

    # Also track separate categories: run, sit, walk
    run_files, run_fused, run_IR = [], [], []
    sit_files, sit_fused, sit_IR = [], [], []
    walk_files, walk_fused, walk_IR = [], [], []

    # Prepare the log file
    with open(LOG_FILE, "w") as fout:
        fout.write("Filename,Fused_RMSE,Baseline_IR_RMSE\n")

        for csv_file in csv_files:
            base = os.path.basename(csv_file)
            # Attempt read
            try:
                df = pd.read_csv(csv_file, parse_dates=["time"])
            except Exception as e:
                print(f"Skipping file '{base}' - parse error: {e}")
                continue

            # Check columns
            required_cols = {"time", "peaks", "pleth_1", "pleth_2", "pleth_3"}
            if not required_cols.issubset(df.columns):
                print(f"Skipping file '{base}' - missing required columns.")
                continue

            # Add time_sec if missing
            if "time_sec" not in df.columns:
                df["time_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

            # Compute sampling freq
            time_diffs = df["time_sec"].diff().dropna()
            if len(time_diffs) < 1:
                print(f"Skipping file '{base}' - not enough time points.")
                continue
            fs = 1.0 / time_diffs.median()

            # ground truth peaks
            ground_truth_peak_times = df.loc[df["peaks"] == 1, "time_sec"].values
            if ground_truth_peak_times.size < 2:
                print(f"Skipping file '{base}' - not enough GT peaks.")
                continue

            # fuse signals
            channels_to_fuse = ["pleth_1","pleth_2","pleth_3"]
            fused_signal, P_history, three_ch_filtered = kalman_fuse_3channels(df, channels_to_fuse, fs)
            fused_peaks_idx = detect_peaks_ampd(fused_signal, AMPD_WINDOW_LENGTH)
            fused_peak_times = df["time_sec"].values[fused_peaks_idx] if len(fused_peaks_idx) else np.array([])

            # single channel pleth_2
            IR_filtered = bandpass_filter(df["pleth_2"].to_numpy(), fs, LOWCUT, HIGHCUT, FILTER_ORDER)
            IR_peaks_idx = detect_peaks_ampd(IR_filtered, AMPD_WINDOW_LENGTH)
            IR_peak_times = df["time_sec"].values[IR_peaks_idx] if len(IR_peaks_idx) else np.array([])

            total_duration = df["time_sec"].iloc[-1]
            rmse_fused, rmse_IR, _ = evaluate_time_series_hr(
                ground_truth_peak_times,
                fused_peak_times,
                IR_peak_times,
                total_duration
            )

            # Log and Print
            rmse_fused_list.append(rmse_fused)
            rmse_IR_list.append(rmse_IR)
            filenames.append(base)

            msg = f"[{base}] Fused_RMSE={rmse_fused:.2f}  IR_RMSE={rmse_IR:.2f}"
            print(msg)
            fout.write(f"{base},{rmse_fused:.4f},{rmse_IR:.4f}\n")

            # Plot for each file (or only if matching PLOT_SPECIFIC_FILE)
            if (not PLOT_SPECIFIC_FILE) or (base == PLOT_SPECIFIC_FILE):
                # Plot all signals
                print(f"Plotting signals for {base}. Close the figure to continue.")
                plot_all_signals_and_kalmanP(
                    df,
                    three_ch_filtered,  # shape (3, N)
                    fused_signal, fused_peaks_idx,
                    IR_filtered, IR_peaks_idx,
                    P_history
                )

            # --- Place file into (run / sit / walk) groups based on name ---
            lower_name = base.lower()
            if "run" in lower_name:
                run_files.append(base)
                run_fused.append(rmse_fused)
                run_IR.append(rmse_IR)
            elif "sit" in lower_name:
                sit_files.append(base)
                sit_fused.append(rmse_fused)
                sit_IR.append(rmse_IR)
            elif "walk" in lower_name:
                walk_files.append(base)
                walk_fused.append(rmse_fused)
                walk_IR.append(rmse_IR)

    # === After all files processed, compute overall stats
    all_fused = np.array(rmse_fused_list, dtype=float)
    all_IR= np.array(rmse_IR_list, dtype=float)

    valid_fused = ~np.isnan(all_fused)
    valid_pl2   = ~np.isnan(all_IR)

    if np.any(valid_fused):
        overall_rmse_fused = np.mean(all_fused[valid_fused])
    else:
        overall_rmse_fused = np.nan

    if np.any(valid_pl2):
        overall_rmse_IR = np.mean(all_IR[valid_pl2])
    else:
        overall_rmse_IR = np.nan

    # Write final summary to log
    with open(LOG_FILE, "a") as fout:
        fout.write("\n==========================\n")
        fout.write("OVERALL EVALUATION RESULTS (RMSE in BPM)\n")
        fout.write("==========================\n")
        fout.write(f"Mean RMSE (Fused vs GT):   {overall_rmse_fused:.2f}\n")
        fout.write(f"Mean RMSE (pleth_2 vs GT): {overall_rmse_IR:.2f}\n")

    print("\n====================================")
    print("OVERALL EVALUATION RESULTS (RMSE in BPM)")
    print("====================================")
    print(f"Mean RMSE (Fused vs GT):   {overall_rmse_fused:.2f}")
    print(f"Mean RMSE (pleth_2 vs GT): {overall_rmse_IR:.2f}")

    # --- Now create a bar chart of differences: (Fused_RMSE - IR_RMSE) for all files ---
    if len(filenames) > 0:
        # Filter only valid
        valid_idx = [i for i in range(len(filenames))
                     if not np.isnan(all_fused[i]) and not np.isnan(all_IR[i])]
        if len(valid_idx) > 0:
            valid_files = [filenames[i] for i in valid_idx]
            fused_vals  = all_fused[valid_idx]
            pl2_vals    = all_IR[valid_idx]

            differences = fused_vals - pl2_vals  # Positive => Fused has larger (worse) RMSE
            # Color map
            colors = []
            for d in differences:
                if d < 0:
                    colors.append("green")
                elif d > 0:
                    colors.append("orange")
                else:
                    colors.append("gray")

            x = np.arange(len(valid_files))
            plt.figure(figsize=(12, 6))
            plt.bar(x, differences, color=colors)
            plt.axhline(0, color='black', linewidth=1)  # reference line at 0
            plt.xticks(x, valid_files, rotation=45, ha="right")
            plt.ylabel("Difference in RMSE (Fused - IR) [BPM]")
            plt.title("Per-file RMSE Difference (Fused - IR)")
            plt.tight_layout()
            print("\nShowing bar plot of (Fused_RMSE - IR_RMSE) for all valid files. Close the window to continue.")
            plt.show()
        else:
            print("No valid RMSE data to plot a difference bar chart.")
    else:
        print("No CSV files processed - no bar plot of difference created.")

    # --- Plot difference bars for run, sit, walk categories ---
    def plot_difference_bar(files_list, fused_list, pl2_list, category_name):
        """
        Create an interactive bar chart for a given category (run/sit/walk),
        showing (Fused_RMSE - IR_RMSE).
        Negative => green, positive => orange, zero => gray.
        """
        fused_arr = np.array(fused_list, dtype=float)
        pl2_arr   = np.array(pl2_list,  dtype=float)
        valid_idx = ~np.isnan(fused_arr) & ~np.isnan(pl2_arr)
        if not np.any(valid_idx):
            print(f"No valid data in {category_name} set to plot.")
            return
        diff = fused_arr[valid_idx] - pl2_arr[valid_idx]
        valid_files = [files_list[i] for i, v in enumerate(valid_idx) if v]

        # Create color map
        colors = []
        for d in diff:
            if d < 0:
                colors.append("green")
            elif d > 0:
                colors.append("orange")
            else:
                colors.append("gray")

        # Create bar chart
        x = np.arange(len(valid_files))
        plt.figure(figsize=(10, 5))
        plt.bar(x, diff, color=colors)
        plt.axhline(0, color='black', linewidth=1)
        plt.xticks(x, valid_files, rotation=45, ha="right")
        plt.ylabel("RMSE Difference (Fused - IR) [BPM]")
        plt.title(f"{category_name.upper()} Files: RMSE Difference")
        plt.tight_layout()
        plt.show()

    # --- Now call the separate category plots ---
    if len(run_files) > 0:
        plot_difference_bar(run_files, run_fused, run_IR, "run")

    if len(sit_files) > 0:
        plot_difference_bar(sit_files, sit_fused, sit_IR, "sit")

    if len(walk_files) > 0:
        plot_difference_bar(walk_files, walk_fused, walk_IR, "walk")

    print(f"Done. Log saved to: {os.path.abspath(LOG_FILE)}")


if __name__ == "__main__":
    process_all_csvs(DATAFOLDER)
