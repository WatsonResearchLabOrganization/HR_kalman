import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import serial
from matplotlib.animation import FuncAnimation
import ampdlib

# ========== Configuration for MULTI-CHANNEL ==========
num_channels = 9  # e.g., 415nm, 445nm, 480nm, 515nm, 555nm, 590nm, 630nm, 680nm, NIR
channel_names = ["415nm", "445nm", "480nm", "515nm", "555nm", "590nm", "630nm", "680nm", "NIR"]

# ========== Channel Visibility Switches ==========
# 1) Whether a channel is displayed at all.
channel_visibility = {
    "415nm": True,
    "445nm": True,
    "480nm": True,
    "515nm": True,
    "555nm": True,
    "590nm": True,
    "630nm": False,   # Example: now we show 630nm again
    "680nm": True,
    "NIR": False,
}

# 2) Whether to show the unfiltered line for a given channel.
unfiltered_visibility = {
    "415nm": False,
    "445nm": False,
    "480nm": False,
    "515nm": False,
    "555nm": False,
    "590nm": False,
    "630nm": False,  # Example: hide unfiltered data for 630nm
    "680nm": False,
    "NIR": False,
}

# OPTIONAL: Assign distinct colors for each channel in the plot
colors = plt.cm.tab10(np.linspace(0, 1, num_channels))

# Set up the serial port
serial_port = '/dev/cu.usbmodem1101'  # Replace with your serial port
baud_rate = 115200                   # Replace with your baud rate
ser = serial.Serial(serial_port, baud_rate)

# ========== Data Buffers for MULTI-CHANNEL ==========
# Each channel has its own list of raw samples
ppg_raw = [[] for _ in range(num_channels)]
times_sec = []  # We assume the same time axis for all channels

# Parameters for the bandpass filter
lowcut = 0.3    # Lower cutoff frequency in Hz
highcut = 2.5   # Upper cutoff frequency in Hz
order = 4       # Filter order

# Rolling window length for normalization
window_length = 160

# AMPD processing window length (in samples)
ampd_window_length = 160  # Adjust this value as needed

# Initialize plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# We will have a separate line object for each channel
lines_filtered = []
lines_unfiltered = []
lines_peaks = []

for i in range(num_channels):
    ch_name = channel_names[i]
    
    # Filtered & Normalized line
    lf, = ax1.plot([], [],
                   label=f'{ch_name} Filtered',
                   color=colors[i], linewidth=1.5)
    lines_filtered.append(lf)

    # Unfiltered & Normalized line (dashed)
    lu, = ax1.plot([], [],
                   label=f'{ch_name} Unfiltered',
                   color=colors[i], linestyle='--', linewidth=1)
    lines_unfiltered.append(lu)

    # Peaks line (same color but with markers)
    lp, = ax1.plot([], [],
                   'o', label=f'{ch_name} Peaks',
                   color=colors[i], markersize=5, fillstyle='none')
    lines_peaks.append(lp)

    # Apply initial visibility for channel & unfiltered lines
    # Channel must be visible + unfiltered must be visible to show unfiltered.
    ch_visible = channel_visibility[ch_name]
    uf_visible = unfiltered_visibility[ch_name] and ch_visible

    lf.set_visible(ch_visible)
    lu.set_visible(uf_visible)
    lp.set_visible(ch_visible)

# One text handle to show HRs for all channels
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
ax1.set_xlim(0, 10)  # Show last 10 seconds in the plot
ax1.set_ylim(0, 1)

# Sampling frequency (will be updated dynamically)
Fs = None
start_time = None

def update(frame):
    global Fs, start_time, ppg_raw, times_sec

    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # Example multi-channel data line: "21,35,37,37,44,40,66,59,68"
            values_str = line.split(',')
            
            if len(values_str) < num_channels:
                # Not enough data for all channels; skip
                return lines_filtered + lines_unfiltered + lines_peaks + [text_handle]
            
            # Convert each channel data to float
            channel_values = [float(v) for v in values_str[:num_channels]]

            current_time = datetime.datetime.now()
            if start_time is None:
                start_time = current_time
                elapsed_time = 0.0
            else:
                elapsed_time = (current_time - start_time).total_seconds()

            # Append time only once per sample
            times_sec.append(elapsed_time)

            # Append each channel's data
            for i in range(num_channels):
                ppg_raw[i].append(channel_values[i])

            # Ensure buffers don't get too large
            max_buffer_length = 2000
            for i in range(num_channels):
                if len(ppg_raw[i]) > max_buffer_length:
                    ppg_raw[i] = ppg_raw[i][-max_buffer_length:]
            
            if len(times_sec) > max_buffer_length:
                times_sec = times_sec[-max_buffer_length:]

            # Update sampling frequency dynamically (based on the last ~10 samples)
            if len(times_sec) > 2:
                Ts_values = np.diff(times_sec[-10:])
                Ts = np.mean(Ts_values) if len(Ts_values) > 0 else None
                if Ts and Ts > 0:
                    Fs = 1.0 / Ts

            # If Fs is known and we have enough data, proceed
            if Fs and all(len(ppg_raw[ch]) > max(order * 3, 10) for ch in range(num_channels)):
                time_axis = np.array(times_sec) - times_sec[0]

                channel_heart_rates = []

                for i in range(num_channels):
                    ch_name = channel_names[i]
                    raw_arr = np.array(ppg_raw[i])

                    # Check if channel is visible
                    if channel_visibility[ch_name]:
                        # Filter the i-th channel
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

                        # Set data for filtered
                        lines_filtered[i].set_data(time_axis, filtered_norm)

                        # If unfiltered is visible for this channel, set data; otherwise clear
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
                        # Channel not visible: clear everything
                        lines_filtered[i].set_data([], [])
                        lines_unfiltered[i].set_data([], [])
                        lines_peaks[i].set_data([], [])
                        channel_heart_rates.append(None)

                    # Update final visibility states
                    # Filtered & Peaks are only visible if the channel is visible
                    lines_filtered[i].set_visible(channel_visibility[ch_name])
                    lines_peaks[i].set_visible(channel_visibility[ch_name])

                    # Unfiltered is visible if the channel + unfiltered flags are both True
                    lines_unfiltered[i].set_visible(
                        channel_visibility[ch_name] and unfiltered_visibility[ch_name]
                    )

                # Show last 10 seconds
                if time_axis[-1] > 10:
                    ax1.set_xlim(time_axis[-1] - 10, time_axis[-1])

                # Build a display string for all channels' HR
                hr_strs = []
                for i, hr in enumerate(channel_heart_rates):
                    if hr is not None:
                        hr_strs.append(f"{channel_names[i]}: {hr:.1f} BPM")
                    else:
                        hr_strs.append(f"{channel_names[i]}: -")
                text_handle.set_text(" | ".join(hr_strs))

        # Return all line objects + text
        return lines_filtered + lines_unfiltered + lines_peaks + [text_handle]

    except Exception as e:
        print(f"Error: {e}")
        return lines_filtered + lines_unfiltered + lines_peaks + [text_handle]

# Create the animation
ani = FuncAnimation(fig, update, interval=50, blit=True)

plt.tight_layout()
plt.show()
