import serial
import csv
import time
import threading

SERIAL_PORT = "/dev/cu.usbmodem1301"  # Example for Linux/Mac
BAUD_RATE = 115200
OUTPUT_FILE = "tremor_data.csv"

# Global flags for continuous logging and running state
collecting = False
running = True
tremor_label = 0  # Default: No tremor

def send_command(ser):
    """Handles user input commands in a separate thread."""
    global collecting, tremor_label, running
    while running:
        user_input = input("Command (s=start/stop, y=tremor, n=no tremor, q=quit): ").strip().lower()
        
        if user_input == 's':  # Start/Stop toggle
            collecting = not collecting
            ser.write(b's')
            print(f"Data Collection: {'STARTED' if collecting else 'STOPPED'}")

        elif user_input == 'y':  # Mark tremor
            tremor_label = 1
            ser.write(b'y')
            print("Marked as Tremor")

        elif user_input == 'n':  # Mark no tremor
            tremor_label = 0
            ser.write(b'n')
            print("Marked as No Tremor")

        elif user_input == 'q':  # Quit
            print("Exiting and disconnecting...")
            running = False
            break

def log_data():
    global collecting, tremor_label, running
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow Arduino reset
    print(f"Connected to {SERIAL_PORT}. Ready for commands.")

    # Start user command thread
    command_thread = threading.Thread(target=send_command, args=(ser,))
    command_thread.daemon = True
    command_thread.start()

    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["aX", "aY", "aZ", "gX", "gY", "gZ", "Result"])  # CSV Header

        try:
            while running:
                if collecting:
                    line = ser.readline().decode('utf-8').strip()
                    if line and not line.startswith(("STARTED", "STOPPED")):
                        data = line.split(",")
                        try:
                            # Attempt to convert each data field to float
                            sensor_data = [float(x) for x in data]
                        except ValueError:
                            print("Invalid data received (non-numeric), skipping:", line)
                            continue
                        # Check for proper number of sensor values (expecting 7)
                        if len(sensor_data) == 7:
                            row = [str(x) for x in sensor_data]
                            print(f"Logging: {','.join(row)}")
                            writer.writerow(row)
                        else:
                            print("Invalid data received (wrong field count), skipping:", line)
                time.sleep(0.05)  # Small delay to reduce CPU load

        except KeyboardInterrupt:
            print("\nLogging stopped by KeyboardInterrupt.")
        finally:
            ser.close()
            print(f"Disconnected from {SERIAL_PORT}.")
    print(f"CSV file saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    log_data()