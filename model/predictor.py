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
