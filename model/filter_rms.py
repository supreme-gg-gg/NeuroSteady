import numpy as np
from scipy.signal import butter, filtfilt

def highpass_filter(data, cutoff=0.5, fs=50, order=2):
    """
    High-pass filter to remove low-frequency drift from accelerometer data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def detect_tremor_sensor(window, rms_threshold=0.5, std_threshold=1):
    """
    Enhanced tremor detection using RMS + Standard Deviation + Filtering.

    Parameters:
        window (numpy array): Shape (50, 3) - last 100 accelerometer samples.
        rms_threshold (float): RMS threshold for tremor detection.
        std_threshold (float): Standard deviation threshold.

    Returns:
        bool: True if tremor detected, else False.
    """
    if window.shape[0] != 100:
        raise ValueError("Window must have 50 samples")

    # Extract axes
    aX, aY, aZ = window[:, 0] / 16384, window[:, 1] / 16384, window[:, 2] / 16384  # Example for ±2g sensor

    # Apply high-pass filter to remove drift
    aX, aY, aZ = highpass_filter(aX), highpass_filter(aY), highpass_filter(aZ)

    # Compute magnitude
    magnitude = np.sqrt(aX**2 + aY**2 + aZ**2)

    # Compute RMS and STD
    rms = np.sqrt(np.mean(magnitude**2))
    std_dev = np.std(magnitude)

    # print(f"RMS: {rms:.3f}, STD: {std_dev:.3f}")

    # Detect tremor based on RMS or STD threshold
    return rms > rms_threshold or std_dev > std_threshold

class EnsemblePredictor:
    def __init__(self, window_size: int = 5, threshold_ratio: float = None):
        """
        window_size: Number of past predictions to consider.
        threshold_ratio: Ratio of True predictions required to return True.
            Defaults to (window_size - 1) / window_size.
        """
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio if threshold_ratio is not None else (window_size - 1) / window_size
        self.history = []

    def add_prediction(self, pred: bool):
        """Add a new prediction to the history."""
        self.history.append(pred)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def consistent_prediction(self) -> bool:
        """
        Returns True only if a consistent detection is observed:
        fraction of True predictions in history >= threshold_ratio.
        """
        if not self.history:
            return False
        true_ratio = sum(self.history) / len(self.history)
        return true_ratio >= self.threshold_ratio

    def reset(self):
        """Reset the prediction history."""
        self.history.clear()
