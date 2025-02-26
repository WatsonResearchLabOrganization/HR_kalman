import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt  # Import necessary functions for filtering
import ampdlib

# Function to read the first file
def read_device1(filename):
    times = []
    ppg = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if '->' in line:
                timestamp_str, ppg_value_str = line.split('->')
                timestamp_str = timestamp_str.strip()
                ppg_value_str = ppg_value_str.strip()
                # Assume the date is 2024-11-12
                datetime_str = '2024-11-12 ' + timestamp_str
                # Parse datetime
                dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                times.append(dt)
                ppg.append(float(ppg_value_str))
    return times, ppg

# Function to read the second file
def read_device2(filename):
    times = []
    ppg = []
    start_time = None
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the recording start time
    for i, line in enumerate(lines):
        if line.startswith('Recording on:'):
            start_time_str = line[len('Recording on:'):].strip()
            start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
            break

    if start_time is None:
        print("Recording start time not found in second file.")
        return times, ppg

    # Find where the data starts
    data_start_index = None
    for i in range(i+1, len(lines)):
        line = lines[i].strip()
        if line.startswith('milliSec'):
            data_start_index = i + 2  # Skip header lines
            break

    if data_start_index is None:
        print("Data start not found in second file.")
        return times, ppg

    # Read data
    for line in lines[data_start_index:]:
        line = line.strip()
        if line == '':
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        time_offset_str, ppg_value_str = parts[0], parts[1]
        time_offset_ms = float(time_offset_str)
        ppg_value = float(ppg_value_str)
        # Compute datetime
        dt = start_time + datetime.timedelta(milliseconds=time_offset_ms)
        times.append(dt)
        ppg.append(ppg_value)
    return times, ppg

# Read data from both devices
times1, ppg1 = read_device1('Highpass_filter/ppg_lumos.txt')
times2, ppg2 = read_device2('Highpass_filter/ppg_biopac.txt')

# Convert ppg1 and ppg2 to NumPy arrays
ppg1 = np.array(ppg1)
ppg2 = np.array(ppg2)

# Align times
all_times = times1 + times2
earliest_time = min(all_times)
times1_sec = [(t - earliest_time).total_seconds() for t in times1]
times2_sec = [(t - earliest_time).total_seconds() for t in times2]

# Convert times to NumPy arrays
times1_sec = np.array(times1_sec)
times2_sec = np.array(times2_sec)

# **Add Bandpass Filter to ppg1**

# Compute sampling frequency (Fs) for ppg1
dt = np.diff(times1_sec)
Ts = np.mean(dt)  # Sampling interval
Fs = 1 / Ts       # Sampling frequency

# Define bandpass filter parameters
lowcut = 0.5    # Lower cutoff frequency in Hz (e.g., 0.5 Hz)
highcut = 5   # Upper cutoff frequency in Hz (e.g., 5.0 Hz)
order = 4      # Order of the filter

# Design the Butterworth bandpass filter
b, a = butter(order, [lowcut, highcut], btype='band', fs=Fs)

# Apply the filter to ppg1
ppg1_filtered = filtfilt(b, a, ppg1)

# **Option 1: Standardize the Filtered Signal**

# Standardize ppg1_filtered
ppg1_standardized = (ppg1_filtered - np.mean(ppg1_filtered)) / np.std(ppg1_filtered)
# Standardize ppg2
ppg2_standardized = (ppg2 - np.mean(ppg2)) / np.std(ppg2)

# Plot the standardized data
plt.figure(figsize=(12, 6))
plt.plot(times1_sec, ppg1_standardized, label='Lumos (Filtered & Standardized)')
#plt.plot(times2_sec, ppg2_standardized, label='Biopac (Standardized)')
plt.xlabel('Time (seconds)')
plt.ylabel('Standardized PPG Signal')
plt.title('Standardized PPG Data Comparison (Filtered Lumos)')
plt.legend()
plt.show()

# **Option 2: Normalize the Filtered Signal Between 0 and 1**

# Normalize ppg1_filtered
ppg1_normalized = (ppg1_filtered - np.min(ppg1_filtered)) / (np.max(ppg1_filtered) - np.min(ppg1_filtered))
ppg1_unfiltered_normalized= (ppg1 - np.min(ppg1)) / (np.max(ppg1) - np.min(ppg1))
# Normalize ppg2
ppg2_normalized = (ppg2 - np.min(ppg2)) / (np.max(ppg2) - np.min(ppg2))

peaks = ampdlib.ampd(ppg1_normalized)

print (peaks)

# Plot the normalized data
plt.figure(figsize=(12, 6))
plt.plot(times1_sec, ppg1_normalized, label='Lumos (Filtered & Normalized)')
plt.plot(times1_sec, ppg1_unfiltered_normalized, label='Lumos ( Normalized)')
#plt.plot(times2_sec, ppg2_normalized, label='Biopac (Normalized)')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized PPG Signal')
plt.title('Normalized PPG Data Comparison (Filtered Lumos)')
plt.legend()
plt.show()
