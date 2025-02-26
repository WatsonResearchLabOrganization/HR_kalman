#!/usr/bin/env python3
import subprocess
import datetime
import csv
import re
import sys
import os

def main():
    """
    This script runs your original PPG script as a subprocess,
    captures anything printed to stdout that matches `FUSED_HR:XX.XX`,
    and saves it to a CSV file with timestamps.
    """
    # Name of your existing script (adjust if it's different)
    original_script = "KalmanFilter .py"

    # Check that the file actually exists
    if not os.path.exists(original_script):
        print(f"Error: {original_script} not found.")
        sys.exit(1)

    # Create a CSV file with a timestamp in its name
    log_filename = f"fused_hr_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Logging fused HR to {log_filename}")

    # Start the original script as a subprocess
    # - `stdout=subprocess.PIPE` so we can read its output
    # - `stderr=subprocess.STDOUT` to combine stderr and stdout
    process = subprocess.Popen(
        [sys.executable, original_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Prepare the CSV file
    with open(log_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write a header
        writer.writerow(["Timestamp", "Fused_HR"])

        try:
            # Continuously read lines from the original script
            for line in process.stdout:
                line = line.strip()
                # Look for lines that have the format FUSED_HR:XX.XX
                match = re.search(r"FUSED_HR:(\d+\.\d+)", line)
                if match:
                    hr_value = match.group(1)  # the numeric part
                    # Create a timestamp for when we received it
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    # Write to CSV
                    writer.writerow([now_str, hr_value])
                    print(f"Logged FUSED_HR: {hr_value}")
        except KeyboardInterrupt:
            print("\nLogging stopped by user.")
        finally:
            # Make sure we wait for the subprocess to exit
            process.terminate()
            process.wait()

    print(f"Done. Fused HR data logged in {log_filename}")

if __name__ == "__main__":
    main()
