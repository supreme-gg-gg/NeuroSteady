import torch
import serial
import numpy as np
import threading
import time

from model.fine_tuned import AdaptedModel
from model.filter_rms import detect_tremor_sensor  # non-ML detector
from model.filter_rms import EnsemblePredictor  # New predictor class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptedModel(device=device)
model.load('model/weights/adapted-1.pth')

# Set up the serial connection
ser = serial.Serial('/dev/cu.usbmodem1301', 115200, timeout=0.1)  # non-blocking read with short timeout

window_len = 100
data_window = []
tracking = False       # Controls whether sensor data is processed
keep_running = True    # Main loop control

# Instantiate the predictor with a history window of 5 predictions
predictor = EnsemblePredictor(window_size=5)
timestep_counter = 0

def preprocess_predict(data):
    """Non-ML method using RMS to detect tremor."""
    data = np.array(data, dtype=np.float32)[:, :3]
    return detect_tremor_sensor(data)

def make_prediction_torch(data):
    """ML method: Preprocess window data and get nn prediction."""
    data = np.array(data)[:, :3]
    data = (data - data.mean()) / data.std()
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.inference(tensor_data)
    return bool(pred)

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

print("Starting continuous detection...")
last_send_time = time.time()
send_interval = 0.1  # seconds - how frequently to send prediction commands

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

        print(f"Read: aX={aX}, aY={aY}, aZ={aZ}")

        if tracking:
            data_window.append((aX, aY, aZ))
            if len(data_window) > window_len:
                data_window = data_window[-window_len:]
            if len(data_window) == window_len and (time.time() - last_send_time) >= send_interval:
                nn_pred = make_prediction_torch(data_window)
                rms_pred = preprocess_predict(data_window)
                final_pred = ensemble_detect(nn_pred, rms_pred)
                # Add to predictor history
                predictor.add_prediction(final_pred)
                consistent = predictor.consistent_prediction()
                print(f"Prediction: {consistent}")
                # Send command based on consistent result ('1' for tremor, '0' otherwise)
                ser.write(str(int(consistent)).encode())
                last_send_time = time.time()
        else:
            data_window = []
    except Exception as e:
        if keep_running:
            print(f"Main loop error: {e}")
