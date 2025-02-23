import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_tremor(csv_file, window_size=100, rms_threshold=1, fs=50, freq_range=(4, 7)):
    """
    Loads tremor CSV with columns aX, aY, aZ as input.
    Computes RMS on the signal magnitude over a moving window and extracts the dominant frequency 
    via FFT, then outputs a binary tremor mask where both conditions are met:
      - RMS > rms_threshold
      - Dominant frequency falls within freq_range (in Hz)
    
    Parameters:
        csv_file (str): Path to the CSV file.
        window_size (int): Number of samples per moving window.
        rms_threshold (float): RMS threshold for tremor detection.
        fs (float): Sampling frequency in Hz.
        freq_range (tuple): Lower and upper bounds of tremor frequency range in Hz.
    
    Returns:
        time (numpy array): Time vector.
        magnitude (numpy array): Signal magnitude over time.
        tremor_mask (numpy array): Binary mask indicating tremor regions.
        rms_values (numpy array): RMS values computed per window.
        peak_freqs (numpy array): Dominant frequency per window (0 if not computed).
    """
    # Load CSV: expecting columns aX, aY, aZ
    data = pd.read_csv(csv_file)
    if not all(col in data.columns for col in ["aX", "aY", "aZ"]):
        raise ValueError("CSV must contain aX, aY, and aZ columns.")
        
    # Assume uniform sampling; create time vector
    time = np.linspace(0, len(data)-1, len(data))
    
    aX = data["aX"].values
    aY = data["aY"].values
    aZ = data["aZ"].values

    # Normalize each axis
    aX = (aX - aX.mean()) / (aX.std() + 1e-8)
    aY = (aY - aY.mean()) / (aY.std() + 1e-8)
    aZ = (aZ - aZ.mean()) / (aZ.std() + 1e-8)

    # Compute magnitude: sqrt(aX^2 + aY^2 + aZ^2)
    magnitude = np.sqrt(aX**2 + aY**2 + aZ**2)
    
    tremor_mask = np.zeros_like(magnitude)
    rms_values = np.zeros_like(magnitude)
    peak_freqs = np.zeros_like(magnitude)  # Dominant frequency per window

    for i in range(len(magnitude) - window_size):
        window = magnitude[i : i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        center = i + window_size // 2
        rms_values[center] = rms

        # Compute FFT and get frequencies
        fft_vals = np.fft.rfft(window)
        fft_freqs = np.fft.rfftfreq(window_size, d=1/fs)
        # Magnitude spectrum
        mag_spec = np.abs(fft_vals)
        # Exclude the zero frequency component
        if len(mag_spec) > 1:
            peak_index = np.argmax(mag_spec[1:]) + 1
            peak_freq = fft_freqs[peak_index]
        else:
            peak_freq = 0
        peak_freqs[center] = peak_freq

        # Mark tremor only if both conditions are met
        if rms > rms_threshold and (freq_range[0] <= peak_freq <= freq_range[1]):
            tremor_mask[center] = 1
    
    return time, magnitude, tremor_mask, rms_values, peak_freqs

def visualize_tremor(csv_file, window_size=100, rms_threshold=1, fs=50, freq_range=(4, 7)):
    """
    Loads the tremor CSV, detects tremor regions using both RM S and frequency analysis,
    and plots the sensor magnitude, RMS curve, and highlights tremor regions.
    """
    time, magnitude, tremor_mask, rms_values, peak_freqs = detect_tremor(csv_file,
                                                                         window_size, 
                                                                         rms_threshold, 
                                                                         fs, 
                                                                         freq_range)
    plt.figure(figsize=(12, 10))
    
    # Plot magnitude and RMS curve with threshold line
    plt.subplot(3,1,1)
    plt.plot(time, magnitude, label="Magnitude", color='b')
    plt.plot(time, rms_values, label="RMS", color='g', alpha=0.7)
    plt.axhline(y=rms_threshold, color='r', linestyle='--', label="RMS Threshold")
    plt.title("Signal Magnitude and RMS")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot dominant frequency over time
    plt.subplot(3,1,2)
    plt.plot(time, peak_freqs, color='m', label="Dominant Frequency")
    plt.axhline(y=freq_range[0], color='orange', linestyle='--', label="Freq Lower Bound")
    plt.axhline(y=freq_range[1], color='orange', linestyle='--', label="Freq Upper Bound")
    plt.title("Dominant Frequency per Window")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    
    # Plot tremor mask over time
    plt.subplot(3,1,3)
    plt.plot(time, magnitude, label="Magnitude", color='b', alpha=0.5)
    plt.fill_between(time, magnitude.min(), magnitude.max(), where=(tremor_mask==1), 
                     color='red', alpha=0.3, label="Tremor Detected")
    plt.title("Tremor Regions")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    csv_file = "data/tremor_data.csv"  # Path to your tremor CSV file
    visualize_tremor(csv_file, window_size=50, rms_threshold=1, fs=50, freq_range=(4,7))