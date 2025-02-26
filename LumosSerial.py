import os
import serial
import time
from datetime import datetime
import threading

# Initialize the serial connection (replace with your actual serial port)
# ser = serial.Serial('/dev/cu.usbmodem2101', 9600, timeout=1)
ser = serial.Serial('/dev/cu.usbmodem101', 9600, timeout=1)

output_dir = "Output"

os.makedirs(output_dir, exist_ok=True)

# List to store serial readings
readings = []

# Event to signal threads to stop
stop_event = threading.Event()

# Variable to store user's name
user_name = ''

# Variable to store exit time
exit_time = ''

# Function to read from the serial port
def read_serial(stop_event):
    try:
        while not stop_event.is_set():
            if ser.in_waiting > 0:
                serial_line = ser.readline().decode('utf-8', errors='replace').strip()
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                output_line = f"{timestamp} -> {serial_line}"
                # output_line = f"{serial_line}"
                print(output_line)
                readings.append(output_line)
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in reading serial: {e}")

# Function to write to the serial port
def write_serial(stop_event):
    try:
        while not stop_event.is_set():
            user_input = input()
            if user_input.lower() == 'exit':
                # Record the time when 'exit' is typed
                global exit_time
                exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                stop_event.set()
                break
            if ser.is_open:
                ser.write(user_input.encode())
    except Exception as e:
        print(f"Error in writing serial: {e}")

try:
    print("Starting serial communication. Type 'exit' to quit.")

    # Start the reading and writing threads
    read_thread = threading.Thread(target=read_serial, args=(stop_event,))
    write_thread = threading.Thread(target=write_serial, args=(stop_event,))

    read_thread.start()
    write_thread.start()

    # Wait for the write_thread to finish
    write_thread.join()
    # Wait for the read_thread to finish
    read_thread.join()

    # Close the serial port
    if ser.is_open:
        ser.close()

    # Now prompt for the user's name
    user_name = input("Please enter your name: ")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Save readings to file upon exit
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"{output_dir}/{date_str}_{user_name}.txt"

    with open(file_name, 'w') as f:
        for line in readings:
            f.write(line + '\n')
    print(f"Readings saved to {file_name}")
    if ser.is_open:
        ser.close()
