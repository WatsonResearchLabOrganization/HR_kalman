import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import serial
from matplotlib.animation import FuncAnimation
import ampdlib

# ========== ADD: KalmanFilter import ==========
from filterpy.kalman import KalmanFilter

# ========== ADD: CSV imports ==========
import csv
import sys

# ========== Configuration for MULTI-CHANNEL ==========
num_channels = 9  # e.g., 415nm, 445nm, 480nm, 515nm, 555nm, 590nm, 630nm, 680nm, NIR
channel_names = ["415nm", "445nm", "480nm", "515nm", "555nm", "590nm", "630nm", "680nm", "NIR"]

# ========== Channel Visibility Switches ==========
channel_visibility = {
    "415nm": True,
    "445nm": True,
    "480nm": True,
    "515nm": True,
    "555nm": True,
    "590nm": True,
    "630nm": False,
    "680nm": True,
    "NIR": True,
}

unfiltered_visibility = {
    "415nm": False,
    "445nm": False,
    "480nm": False,
    "515nm": False,
    "555nm": False,
    "590nm": False,
    "630nm": False,
    "680nm": False,
    "NIR": False,
}

colors = plt.cm.tab10(np.linspace(0, 1, num_channels))

# ========== Serial Port ==========
serial_port = '/dev/cu.usbmodem1101'  # Replace with your serial port
baud_rate = 115200                   # Replace with your baud rate
ser = serial.Serial(serial_port, baud_rate)

# ========== Global Data Buffers for MULTI-CHANNEL ==========
ppg_raw = [[] for _ in range(num_channels)]
times_sec = []

# ========== Filter Configuration ==========
lowcut = 0.3
highcut = 2.5
order = 4
window_length = 160       # Rolling window for normalization
ampd_window_length = 160  # Window for AMPD

# ========== Peak Logging Enable/Disable ==========
ENABLE_PEAK_DETECTION = True

# ========== Initialize Plotting ==========
fig, ax1 = plt.subplots(figsize=(12, 6))
lines_filtered = []
lines_unfiltered = []
lines_peaks = []

for i in range(num_channels):
    ch_name = channel_names[i]
    
    lf, = ax1.plot([], [],
                   label=f'{ch_name} Filtered',
                   color=colors[i], linewidth=1.5)
    lines_filtered.append(lf)

    lu, = ax1.plot([], [],
                   label=f'{ch_name} Unfiltered',
                   color=colors[i], linestyle='--', linewidth=1)
    lines_unfiltered.append(lu)

    lp, = ax1.plot([], [],
                   'o', label=f'{ch_name} Peaks',
                   color=colors[i], markersize=5, fillstyle='none')
    lines_peaks.append(lp)

    # Visibility
    lf.set_visible(channel_visibility[ch_name])
    lu.set_visible(channel_visibility[ch_name] and unfiltered_visibility[ch_name])
    lp.set_visible(channel_visibility[ch_name])

text_handle = ax1.text(
    0.98, 0.02, '',
    transform=ax1.transAxes,
    fontsize=12,
    horizontalalignment='right',
    verticalalignment='top'
)

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Normalized PPG Signal')
ax1.set_title('Real-Time Multi-Channel PPG Signal\nwith Rolling Normalization and Peak Detection')
ax1.legend()
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)

Fs = None
start_time = None

# ========== MODIFIED: 1D Kalman Filter for SNR-based Summation ==========
# We no longer do dim_z=9. Instead, we track 1D state "x(t)" and feed a single measurement z(t).
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.]])  # initial state
kf.P *= 10.0
kf.F = np.array([[1.]])
kf.H = np.array([[1.]])
kf.R = np.array([[1.]])  # We'll adapt R if we want, or keep it constant
kf.Q = np.array([[0.1]])

# Buffers for fused signal
ppg_fused = []
times_fused = []

# ========== Extra lines for the fused signal ==========
lfused, = ax1.plot([], [], label='Fused Filtered', color='black', linewidth=2)
lfused_peaks, = ax1.plot([], [], 'o', label='Fused Peaks', color='black', markersize=5, fillstyle='none')
text_fusedHR = ax1.text(
    0.98, 0.08, '',
    transform=ax1.transAxes,
    fontsize=12,
    horizontalalignment='right',
    verticalalignment='top',
    color='black'
)
lfused.set_visible(True)
lfused_peaks.set_visible(True)

# ========== ADD: Date/time in the filename ==========
current_dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"ppg_data_{current_dt_str}.csv"

# ========== Setup CSV Logging ==========
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)

# Build a header row
header = ["Timestamp"]  # We'll log real clock time
for ch_name in channel_names:
    header.append(f"{ch_name}_filtered")
    # If peak detection is enabled, also store the peak flags
    if ENABLE_PEAK_DETECTION:
        header.append(f"{ch_name}_peak")
# Fused columns
header.append("Fused_filtered")
if ENABLE_PEAK_DETECTION:
    header.append("Fused_peak")
csv_writer.writerow(header)


def close_csv(event=None):
    """Close the CSV file when the plot is closed."""
    if not csv_file.closed:
        print("Closing CSV file...")
        csv_file.close()


def update(frame):
    global Fs, start_time, ppg_raw, times_sec, ppg_fused, times_fused
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            values_str = line.split(',')
            if len(values_str) < num_channels:
                return (lines_filtered + lines_unfiltered + lines_peaks +
                        [text_handle, lfused, lfused_peaks, text_fusedHR])

            channel_values = [float(v) for v in values_str[:num_channels]]
            
            current_time = datetime.datetime.now()
            # We'll still use elapsed_time internally for the x-axis,
            # but for CSV we'll record real date/time string
            if start_time is None:
                start_time = current_time
                elapsed_time = 0.0
            else:
                elapsed_time = (current_time - start_time).total_seconds()

            # Append data
            times_sec.append(elapsed_time)
            for i in range(num_channels):
                ppg_raw[i].append(channel_values[i])

            # Keep buffer size in check
            max_buffer_length = 2000
            for i in range(num_channels):
                if len(ppg_raw[i]) > max_buffer_length:
                    ppg_raw[i] = ppg_raw[i][-max_buffer_length:]
            if len(times_sec) > max_buffer_length:
                times_sec = times_sec[-max_buffer_length:]

            # Dynamic Fs update
            if len(times_sec) > 2:
                Ts_values = np.diff(times_sec[-10:])
                Ts = np.mean(Ts_values) if len(Ts_values) > 0 else None
                if Ts and Ts > 0:
                    Fs = 1.0 / Ts

            # ========== MODIFIED: SNR-based Summation for the Kalman Filter ==========
            # We'll do:
            #   (1) Bandpass each channel's ENTIRE buffer -> get the LATEST filtered sample
            #   (2) SNR_i = abs(latest_filtered_sample) + small_offset
            #   (3) z_t = sum_i [ SNR_i * latest_filtered_sample_i ]
            #   (4) 1D KF update with z_t

            if Fs and all(len(ppg_raw[ch]) > max(order * 3, 10) for ch in range(num_channels)):
                # We'll bandpass each channel, get the last sample
                bandpassed_last_samples = []
                for i in range(num_channels):
                    raw_arr = np.array(ppg_raw[i])
                    b_i, a_i = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                    filtered_arr = filtfilt(b_i, a_i, raw_arr)
                    last_val = filtered_arr[-1]
                    bandpassed_last_samples.append(last_val)

                # Compute SNR for each channel's last sample
                # (Naive approach: snr_i = abs(value) + small offset)
                snrs = [abs(x) + 1e-6 for x in bandpassed_last_samples]

                # Weighted sum => single measurement
                z_t = 0.0
                for i in range(num_channels):
                    z_t += snrs[i] * bandpassed_last_samples[i]

                # KF predict/update
                kf.predict()
                kf.update(z_t)  # dim_z=1 => single scalar
                fused_value = kf.x[0, 0]
            else:
                # If not enough samples or Fs not set, just do a direct sum
                fused_value = sum(channel_values)

            # Store fused signal
            times_fused.append(elapsed_time)
            ppg_fused.append(fused_value)

            # ====================== Plotting for each channel ======================
            filtered_norms = [None]*num_channels
            peak_flags = [0]*num_channels  # 0 or 1 for each channel

            if Fs and all(len(ppg_raw[ch]) > max(order * 3, 10) for ch in range(num_channels)):
                time_axis = np.array(times_sec) - times_sec[0]

                channel_heart_rates = []
                for i in range(num_channels):
                    ch_name = channel_names[i]
                    raw_arr = np.array(ppg_raw[i])

                    if channel_visibility[ch_name]:
                        b, a = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                        filtered_signal = filtfilt(b, a, raw_arr)

                        # Rolling-window for normalization
                        if len(filtered_signal) >= window_length:
                            recent_filtered = filtered_signal[-window_length:]
                            recent_unfiltered = raw_arr[-window_length:]
                        else:
                            recent_filtered = filtered_signal
                            recent_unfiltered = raw_arr

                        f_min, f_max = np.min(recent_filtered), np.max(recent_filtered)
                        u_min, u_max = np.min(recent_unfiltered), np.max(recent_unfiltered)

                        if f_max != f_min:
                            filtered_norm = (filtered_signal - f_min) / (f_max - f_min)
                        else:
                            filtered_norm = np.zeros_like(filtered_signal)

                        if u_max != u_min:
                            unfiltered_norm = (raw_arr - u_min) / (u_max - u_min)
                        else:
                            unfiltered_norm = np.zeros_like(raw_arr)

                        lines_filtered[i].set_data(time_axis, filtered_norm)

                        if unfiltered_visibility[ch_name]:
                            lines_unfiltered[i].set_data(time_axis, unfiltered_norm)
                        else:
                            lines_unfiltered[i].set_data([], [])

                        # Peak detection (if enabled)
                        if ENABLE_PEAK_DETECTION and len(filtered_norm) >= ampd_window_length:
                            segment = filtered_norm[-ampd_window_length:]
                            ampd_peaks = ampdlib.ampd_fast(segment, window_length=ampd_window_length)

                            global_peak_indices = np.arange(
                                len(filtered_norm) - ampd_window_length,
                                len(filtered_norm)
                            )[ampd_peaks]
                            peak_times = time_axis[global_peak_indices]
                            peak_values = filtered_norm[global_peak_indices]
                            lines_peaks[i].set_data(peak_times, peak_values)

                            # Compute HR
                            if len(peak_times) >= 2:
                                rr_intervals = np.diff(peak_times)
                                avg_rr = np.mean(rr_intervals)
                                heart_rate = 60.0 / avg_rr
                                channel_heart_rates.append(heart_rate)
                            else:
                                channel_heart_rates.append(None)
                                lines_peaks[i].set_data([], [])
                            
                            # Check if last sample is a peak
                            last_idx = len(filtered_norm) - 1
                            if last_idx in global_peak_indices:
                                peak_flags[i] = 1

                        else:
                            # Either detection disabled or not enough samples
                            lines_peaks[i].set_data([], [])
                            channel_heart_rates.append(None)

                        filtered_norms[i] = filtered_norm
                    else:
                        # Channel not visible
                        lines_filtered[i].set_data([], [])
                        lines_unfiltered[i].set_data([], [])
                        lines_peaks[i].set_data([], [])
                        channel_heart_rates.append(None)
                        filtered_norms[i] = None

                    # Visibility
                    lines_filtered[i].set_visible(channel_visibility[ch_name])
                    lines_peaks[i].set_visible(channel_visibility[ch_name] and ENABLE_PEAK_DETECTION)
                    lines_unfiltered[i].set_visible(
                        channel_visibility[ch_name] and unfiltered_visibility[ch_name]
                    )

                if time_axis[-1] > 10:
                    ax1.set_xlim(time_axis[-1] - 10, time_axis[-1])

                # Build HR display string
                hr_strs = []
                for i, hr in enumerate(channel_heart_rates):
                    if hr is not None:
                        hr_strs.append(f"{channel_names[i]}: {hr:.1f} BPM")
                    else:
                        hr_strs.append(f"{channel_names[i]}: -")
                text_handle.set_text(" | ".join(hr_strs))

            # ====================== Plotting the fused signal ======================
            fused_filtered_norm = None
            fused_peak_flag = 0
            if Fs and len(ppg_fused) > max(order * 3, 10):
                fused_time_axis = np.array(times_fused) - times_fused[0]

                # Bandpass filter
                b_fused, a_fused = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                fused_filtered = filtfilt(b_fused, a_fused, np.array(ppg_fused))

                # Rolling-window norm
                if len(fused_filtered) >= window_length:
                    recent_fused = fused_filtered[-window_length:]
                else:
                    recent_fused = fused_filtered

                fmin, fmax = np.min(recent_fused), np.max(recent_fused)
                if fmax != fmin:
                    fused_norm = (fused_filtered - fmin) / (fmax - fmin)
                else:
                    fused_norm = np.zeros_like(fused_filtered)

                fused_filtered_norm = fused_norm

                # Peak detection
                if ENABLE_PEAK_DETECTION and len(fused_norm) >= ampd_window_length:
                    segment_fused = fused_norm[-ampd_window_length:]
                    ampd_peaks_fused = ampdlib.ampd_fast(segment_fused, window_length=ampd_window_length)

                    global_peak_indices_fused = np.arange(
                        len(fused_norm) - ampd_window_length,
                        len(fused_norm)
                    )[ampd_peaks_fused]
                    fused_peak_times = fused_time_axis[global_peak_indices_fused]
                    fused_peak_values = fused_norm[global_peak_indices_fused]

                    lfused_peaks.set_data(fused_peak_times, fused_peak_values)

                    # Compute HR
                    if len(fused_peak_times) >= 2:
                        rr_intervals_fused = np.diff(fused_peak_times)
                        avg_rr_fused = np.mean(rr_intervals_fused)
                        fused_hr = 60.0 / avg_rr_fused
                    else:
                        fused_hr = None
                        lfused_peaks.set_data([], [])
                    
                    # Check if last sample is a peak
                    last_fused_idx = len(fused_norm) - 1
                    if last_fused_idx in global_peak_indices_fused:
                        fused_peak_flag = 1

                else:
                    fused_hr = None
                    lfused_peaks.set_data([], [])

                if fused_hr is not None:
                    text_fusedHR.set_text(f"Fused: {fused_hr:.1f} BPM")
                    print(f"FUSED_HR:{fused_hr:.2f}", flush=True)
                else:
                    text_fusedHR.set_text("Fused: -")

                lfused.set_data(fused_time_axis, fused_norm)

                if fused_time_axis[-1] > 10:
                    ax1.set_xlim(fused_time_axis[-1] - 10, fused_time_axis[-1])

            # -------- Real-time CSV logging (one row per update) --------
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # e.g. 2025-02-21 16:15:30.123
            row_to_write = [timestamp_str]

            # For each channel
            for i in range(num_channels):
                if filtered_norms[i] is not None:
                    # latest filtered sample
                    row_to_write.append(filtered_norms[i][-1])
                    if ENABLE_PEAK_DETECTION:
                        row_to_write.append(peak_flags[i])
                else:
                    row_to_write.append(None)
                    if ENABLE_PEAK_DETECTION:
                        row_to_write.append(0)

            # Fused
            if fused_filtered_norm is not None:
                row_to_write.append(fused_filtered_norm[-1])
                if ENABLE_PEAK_DETECTION:
                    row_to_write.append(fused_peak_flag)
            else:
                row_to_write.append(None)
                if ENABLE_PEAK_DETECTION:
                    row_to_write.append(0)

            csv_writer.writerow(row_to_write)

            # Return updated artist handles
            return (lines_filtered + lines_unfiltered + lines_peaks +
                    [text_handle, lfused, lfused_peaks, text_fusedHR])

    except Exception as e:
        print(f"Error: {e}")
        return (lines_filtered + lines_unfiltered + lines_peaks +
                [text_handle, lfused, lfused_peaks, text_fusedHR])


# Close CSV on figure close
fig.canvas.mpl_connect('close_event', close_csv)

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()

# Fallback if user kills the script
try:
    pass
except KeyboardInterrupt:
    close_csv()
    sys.exit()
