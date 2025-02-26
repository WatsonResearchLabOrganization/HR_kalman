import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import serial
from matplotlib.animation import FuncAnimation

# Set up the serial port
serial_port = '/dev/cu.usbmodem101'  # Replace with your serial port
baud_rate = 115200       # Replace with your baud rate
ser = serial.Serial(serial_port, baud_rate)

# Initialize data buffers
ppg1_raw = []
times1_sec = []

# Parameters for the bandpass filter
lowcut = 0.3    # Lower cutoff frequency in Hz
highcut = 2.5   # Upper cutoff frequency in Hz
order = 4       # Filter order

# Rolling window length for normalization
window_length = 160

# Initialize plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set up the plot
line1, = ax1.plot([], [], label='Filtered & Rolling Normalized')
line2, = ax1.plot([], [], label='Unfiltered & Rolling Normalized')
text_handle = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                       verticalalignment='top')

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Normalized PPG Signal')
ax1.set_title('Real-Time PPG Signal with Rolling Normalization')
ax1.legend()
ax1.set_xlim(0, 10)  # Show last 10 seconds
ax1.set_ylim(0, 1)

# Sampling frequency (will be updated dynamically)
Fs = None
# Initialize start_time
start_time = None

def update(frame):
    global Fs, start_time

    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # Parse the PPG value
            ppg_value = float(line)
            current_time = datetime.datetime.now()
            if start_time is None:
                start_time = current_time
                elapsed_time = 0.0
            else:
                elapsed_time = (current_time - start_time).total_seconds()

            # Append data to buffers
            times1_sec.append(elapsed_time)
            ppg1_raw.append(ppg_value)

            # Ensure buffers don't get too large
            max_buffer_length = 500
            if len(ppg1_raw) > max_buffer_length:
                times1_sec.pop(0)
                ppg1_raw.pop(0)

            # Update sampling frequency
            if len(times1_sec) > 1:
                Ts_values = np.diff(times1_sec[-10:])
                Ts = np.mean(Ts_values)
                if Ts > 0:
                    Fs = 1 / Ts

            # Proceed only if Fs is known and we have enough data
            if Fs and len(ppg1_raw) > max(order * 3, 10):
                # Design the Butterworth bandpass filter
                b, a = butter(order, [lowcut, highcut], btype='band', fs=Fs)
                # Apply the filter
                ppg1_filtered_signal = filtfilt(b, a, ppg1_raw)

                # Use rolling window normalization
                # Determine the range over last `window_length` samples
                recent_filtered = ppg1_filtered_signal[-window_length:]
                recent_unfiltered = ppg1_raw[-window_length:]

                # Compute min/max for filtered and unfiltered in their recent windows
                f_min, f_max = np.min(recent_filtered), np.max(recent_filtered)
                u_min, u_max = np.min(recent_unfiltered), np.max(recent_unfiltered)

                # Normalize using recent window min/max (avoid division by zero)
                if f_max != f_min:
                    ppg1_filtered_normalized = (ppg1_filtered_signal - f_min) / (f_max - f_min)
                else:
                    ppg1_filtered_normalized = np.zeros_like(ppg1_filtered_signal)

                if u_max != u_min:
                    ppg1_unfiltered_normalized = (np.array(ppg1_raw) - u_min) / (u_max - u_min)
                else:
                    ppg1_unfiltered_normalized = np.zeros_like(ppg1_raw)

                # Update time axis
                time_axis = np.array(times1_sec) - times1_sec[0]

                # Update signal plots
                line1.set_data(time_axis, ppg1_filtered_normalized)
                line2.set_data(time_axis, ppg1_unfiltered_normalized)

                # Show last 10 seconds
                ax1.set_xlim(time_axis[-1] - 10, time_axis[-1])
                
        return line1, line2, text_handle

    except Exception as e:
        print(f"Error: {e}")
        return line1, line2, text_handle

# Create the animation
ani = FuncAnimation(fig, update, interval=50, blit=True)

plt.tight_layout()
plt.show()
