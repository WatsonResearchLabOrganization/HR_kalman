import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Initialize lists to hold timestamps and values
timestamps = []
values = []

# Replace 'data.txt' with the path to your data file
with open('Green-Red_dedection_filter.txt', 'r') as f:
    for line in f:
        # Remove any leading/trailing whitespace
        line = line.strip()
        if '->' in line:
            # Split the line into timestamp and value
            timestamp_str, value_str = line.split('->')
            timestamp_str = timestamp_str.strip()
            value_str = value_str.strip()
            # Convert timestamp string to seconds
            # Timestamp is in HH:MM:SS.mmm format
            time_parts = timestamp_str.split(':')
            if len(time_parts) == 3:
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                timestamps.append(total_seconds)
                # Convert value string to float
                try:
                    value = float(value_str)
                    values.append(value)
                except ValueError:
                    print(f"Could not convert value to float: {value_str}")
            else:
                print(f"Unexpected timestamp format: {timestamp_str}")
        else:
            print(f"Line does not contain '->': {line}")

if len(timestamps) == 0 or len(values) == 0:
    print("No data to plot.")
else:
    # Subtract the first timestamp to make time start at zero
    start_time = timestamps[0]
    timestamps = [t - start_time for t in timestamps]

    # Convert lists to numpy arrays
    timestamps = np.array(timestamps)
    values = np.array(values)

    # Plot the original data without markers
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, label='Original Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Check if we have enough data points to apply the filter
    if len(values) > 10:
        # Compute sampling intervals
        sampling_intervals = np.diff(timestamps)
        # Compute average sampling interval
        avg_sampling_interval = np.mean(sampling_intervals)
        # Compute sampling frequency
        fs = 1 / avg_sampling_interval
        print(f"Estimated sampling frequency: {fs:.2f} Hz")

        # Bandpass filter design
        def butter_bandpass(lowcut, highcut, fs, order=4):
            nyq = 0.5 * fs  # Nyquist Frequency
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        # Define lowcut and highcut frequencies
        lowcut = 0.1  # Hz
        highcut = 2.0  # Hz

        # Get filter coefficients
        b, a = butter_bandpass(lowcut, highcut, fs, order=4)

        # Apply filter to data
        filtered_values = filtfilt(b, a, values)

        # Plot the filtered data without markers
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, values, label='Original Data')
        plt.plot(timestamps, filtered_values, linestyle='--', label='Filtered Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Filtered Data')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Not enough data points to apply the bandpass filter.")
