import torch
import serial
import numpy as np
import threading
import time

from model.fine_tuned import AdaptedModel
from model.filter_rms import detect_tremor_sensor  # non-ML detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptedModel(device=device)
model.load('model/weights/adapted-1.pth')

# Set up the serial connection
ser = serial.Serial('/dev/cu.usbmodem1301', 115200, timeout=0.1)  # non-blocking read with short timeout

window_len = 100
data_window = []
tracking = False       # Controls whether sensor data is processed
keep_running = True    # Main loop control

def preprocess_predict(data):
    """Non-ML method using RMS to detect tremor."""
    data = np.array(data, dtype=np.float32)[:, :3]
    return detect_tremor_sensor(data)

def make_prediction_torch(data):
    """ML method: Preprocess window data and get nn prediction."""
    data = np.array(data)
    data = data[:, :3]
    # Normalize data (simple normalization)
    data = (data - data.mean()) / data.std()
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.inference(tensor_data)
    return bool(pred)  # Convert 0 or 1 to bool

def ensemble_detect(nn_pred: bool, rms_pred: bool, weight_nn: float = 0.5, weight_rms: float = 0.5) -> bool:
    score = weight_nn * int(nn_pred) + weight_rms * int(rms_pred)
    threshold = 0.6
    return score >= threshold

def command_listener():
    """
    Listens for user commands in a separate thread.
    's' toggles tracking and sends an "s" command to Arduino.
    'q' stops the program.
    """
    global tracking, data_window, keep_running
    while keep_running:
        cmd = input("Command ('s' to toggle tracking, 'q' to quit): ").strip().lower()
        if cmd == 's':
            tracking = not tracking
            ser.write(b's')
            if tracking:
                print("Tracking started.")
                data_window = []  # Clear accumulated window
            else:
                print("Tracking stopped.")
        elif cmd == 'q':
            print("Quitting...")
            keep_running = False
            ser.close()
            break

# Start command listener thread
threading.Thread(target=command_listener, daemon=True).start()

# Main continuous loop: read data and process predictions
print("Starting continuous detection...")
last_send_time = time.time()
send_interval = 0.2  # seconds - how frequently to send prediction commands

while keep_running:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line == "":  # timeout occurred, no data
            continue
        if ',' not in line:
            continue
        try:
            aX, aY, aZ = map(float, line.split(','))
        except Exception:
            continue

        # Print sensor reading (optional)
        print(f"Read: aX={aX}, aY={aY}, aZ={aZ}")

        if tracking:
            data_window.append((aX, aY, aZ))
            if len(data_window) > window_len:
                data_window.pop(0)
            # Process prediction if window full and if enough time elapsed since last command sent
            if len(data_window) == window_len and (time.time() - last_send_time) >= send_interval:
                nn_pred = make_prediction_torch(data_window)
                rms_pred = preprocess_predict(data_window)
                final_pred = ensemble_detect(nn_pred, rms_pred)
                print(f"Ensemble Prediction (Tremor): {final_pred}")
                # Send command to Arduino: '1' for tremor detected, '0' otherwise.
                ser.write(str(int(final_pred)).encode())
                last_send_time = time.time()
        else:
            data_window = []
    except Exception as e:
        if keep_running:
            print(f"Main loop error: {e}")
