import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import serial
from matplotlib.animation import FuncAnimation
import ampdlib

# ========== ADD: KalmanFilter import ==========
from filterpy.kalman import KalmanFilter

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
    "680nm": False,
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

# Set up the serial port
serial_port = '/dev/cu.usbmodem1101'  # Replace with your serial port
baud_rate = 115200                   # Replace with your baud rate
ser = serial.Serial(serial_port, baud_rate)

# ========== Data Buffers for MULTI-CHANNEL ==========
ppg_raw = [[] for _ in range(num_channels)]
times_sec = []

# Parameters for the bandpass filter
lowcut = 0.3
highcut = 2.5
order = 4

# Rolling window length for normalization
window_length = 160

# AMPD processing window length (in samples)
ampd_window_length = 160

# Initialize plotting
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

    # Set initial visibility
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

# ========== ADD: Kalman Filter for single-channel fusion ==========
kf = KalmanFilter(dim_x=1, dim_z=num_channels)
kf.x = np.array([[0.]])            # initial state estimate
kf.P *= 10.0                       # initial covariance
kf.F = np.array([[1.]])            # state transition (x_{k+1} = x_k)
kf.H = np.ones((num_channels, 1))  # each channel = x + noise
kf.R = np.eye(num_channels) * 1.0  # measurement noise
kf.Q = np.array([[0.1]])           # process noise

# Buffers for fused signal
ppg_fused = []
times_fused = []

# Create extra lines for the fused signal
lfused, = ax1.plot([], [],
                   label='Fused Filtered',
                   color='black', linewidth=2)
lfused_peaks, = ax1.plot([], [],
                         'o', label='Fused Peaks',
                         color='black', markersize=5, fillstyle='none')
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

def update(frame):
    global Fs, start_time, ppg_raw, times_sec
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            values_str = line.split(',')
            if len(values_str) < num_channels:
                return (lines_filtered + lines_unfiltered + lines_peaks +
                        [text_handle, lfused, lfused_peaks, text_fusedHR])

            channel_values = [float(v) for v in values_str[:num_channels]]
            
            current_time = datetime.datetime.now()
            if start_time is None:
                start_time = current_time
                elapsed_time = 0.0
            else:
                elapsed_time = (current_time - start_time).total_seconds()

            # Append time for multi-channel
            times_sec.append(elapsed_time)
            # Append each channel's data
            for i in range(num_channels):
                ppg_raw[i].append(channel_values[i])

            # Keep buffer size in check
            max_buffer_length = 2000
            for i in range(num_channels):
                if len(ppg_raw[i]) > max_buffer_length:
                    ppg_raw[i] = ppg_raw[i][-max_buffer_length:]
            if len(times_sec) > max_buffer_length:
                times_sec = times_sec[-max_buffer_length:]

            # Dynamic Fs update (from last ~10 samples)
            if len(times_sec) > 2:
                Ts_values = np.diff(times_sec[-10:])
                Ts = np.mean(Ts_values) if len(Ts_values) > 0 else None
                if Ts and Ts > 0:
                    Fs = 1.0 / Ts

            # ========== ADD: Kalman-based fusion ==========
            # Keep a separate time buffer for the fused signal
            times_fused.append(elapsed_time)
            
            # Predict + Update with the multi-channel measurement
            kf.predict()
            kf.update(np.array(channel_values))
            
            # Get the fused value
            fused_value = kf.x[0, 0]
            ppg_fused.append(fused_value)

            # ========== Existing multi-channel processing ==========
            if Fs and all(len(ppg_raw[ch]) > max(order * 3, 10) for ch in range(num_channels)):
                time_axis = np.array(times_sec) - times_sec[0]

                channel_heart_rates = []
                for i in range(num_channels):
                    ch_name = channel_names[i]
                    raw_arr = np.array(ppg_raw[i])

                    if channel_visibility[ch_name]:
                        b, a = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                        filtered_signal = filtfilt(b, a, raw_arr)

                        # Rolling-window
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

                        # Run AMPD
                        if len(filtered_norm) >= ampd_window_length:
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
                        else:
                            channel_heart_rates.append(None)
                            lines_peaks[i].set_data([], [])

                    else:
                        # Channel not visible
                        lines_filtered[i].set_data([], [])
                        lines_unfiltered[i].set_data([], [])
                        lines_peaks[i].set_data([], [])
                        channel_heart_rates.append(None)

                    # Update final visibility
                    lines_filtered[i].set_visible(channel_visibility[ch_name])
                    lines_peaks[i].set_visible(channel_visibility[ch_name])
                    lines_unfiltered[i].set_visible(
                        channel_visibility[ch_name] and unfiltered_visibility[ch_name]
                    )

                # Show last 10 seconds for multi-channel
                if time_axis[-1] > 10:
                    ax1.set_xlim(time_axis[-1] - 10, time_axis[-1])

                # Build display string for multi-channel HR
                hr_strs = []
                for i, hr in enumerate(channel_heart_rates):
                    if hr is not None:
                        hr_strs.append(f"{channel_names[i]}: {hr:.1f} BPM")
                    else:
                        hr_strs.append(f"{channel_names[i]}: -")
                text_handle.set_text(" | ".join(hr_strs))

            # ========== ADD: Now process the fused signal for HR ==========
            if Fs and len(ppg_fused) > max(order * 3, 10):
                fused_time_axis = np.array(times_fused) - times_fused[0]
                
                # 1) Bandpass filter the fused signal
                b_fused, a_fused = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                fused_filtered = filtfilt(b_fused, a_fused, np.array(ppg_fused))

                # 2) Rolling-window normalization
                if len(fused_filtered) >= window_length:
                    recent_fused = fused_filtered[-window_length:]
                else:
                    recent_fused = fused_filtered

                fmin, fmax = np.min(recent_fused), np.max(recent_fused)
                if fmax != fmin:
                    fused_norm = (fused_filtered - fmin) / (fmax - fmin)
                else:
                    fused_norm = np.zeros_like(fused_filtered)

                # 3) Peak detection with AMPD
                if len(fused_norm) >= ampd_window_length:
                    segment_fused = fused_norm[-ampd_window_length:]
                    ampd_peaks_fused = ampdlib.ampd_fast(segment_fused, window_length=ampd_window_length)

                    global_peak_indices_fused = np.arange(
                        len(fused_norm) - ampd_window_length,
                        len(fused_norm)
                    )[ampd_peaks_fused]
                    fused_peak_times = fused_time_axis[global_peak_indices_fused]
                    fused_peak_values = fused_norm[global_peak_indices_fused]

                    lfused_peaks.set_data(fused_peak_times, fused_peak_values)

                    # 4) Compute fused HR
                    if len(fused_peak_times) >= 2:
                        rr_intervals_fused = np.diff(fused_peak_times)
                        avg_rr_fused = np.mean(rr_intervals_fused)
                        fused_hr = 60.0 / avg_rr_fused
                    else:
                        fused_hr = None
                        lfused_peaks.set_data([], [])
                else:
                    fused_hr = None
                    lfused_peaks.set_data([], [])
                    
                # Right after you compute fused_hr, something like this:
                if fused_hr is not None:
                    # Show on the plot text as usual
                    text_fusedHR.set_text(f"Fused: {fused_hr:.1f} BPM")
                    
                    # ADD THIS:
                    print(f"FUSED_HR:{fused_hr:.2f}", flush=True)
                else:
                    text_fusedHR.set_text("Fused: -")

                # Set data for fused signal
                lfused.set_data(fused_time_axis, fused_norm)

                # Show last 10 seconds for fused
                if fused_time_axis[-1] > 10:
                    ax1.set_xlim(fused_time_axis[-1] - 10, fused_time_axis[-1])

                # Update text for fused HR
                if fused_hr is not None:
                    text_fusedHR.set_text(f"Fused: {fused_hr:.1f} BPM")
                else:
                    text_fusedHR.set_text("Fused: -")
            
            # Return all line objects + texts
            return (lines_filtered + lines_unfiltered + lines_peaks +
                    [text_handle, lfused, lfused_peaks, text_fusedHR])

    except Exception as e:
        print(f"Error: {e}")
        return (lines_filtered + lines_unfiltered + lines_peaks +
                [text_handle, lfused, lfused_peaks, text_fusedHR])

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()


