import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

class TremorDataset(Dataset):
    def __init__(self, filepath="Dataset.csv", seq_length=100):
        data = pd.read_csv(filepath).dropna()  # Drop missing values if any
        
        self.features = torch.tensor(data[['aX', 'aY', 'aZ']].values, dtype=torch.float32)
        
        # Normalize features
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-8)
        
        if 'Result' in data.columns:
            self.labels = torch.tensor(data['Result'].values, dtype=torch.long)
        else: # inference, no labels
            self.labels = torch.zeros(len(data))
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length  # Number of available sequences

    def __getitem__(self, index):
        # Extract the time series segment of shape (seq_length, 3)
        X_seq = self.features[index:index + self.seq_length]
        y = self.labels[index + self.seq_length - 1]
        
        # Transform each channel's time series to spectrogram
        # spectrograms = []
        # n_fft = 32
        # hop_length = 4
        # for ch in range(X_seq.shape[1]):
        #     channel_signal = X_seq[:, ch]
        #     # Compute the short-time Fourier transform; returns complex tensor
        #     stft_result = torch.stft(channel_signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        #     # Compute magnitude spectrogram
        #     spec = stft_result.abs()
        #     spectrograms.append(spec)
        # # Stack spectrograms to get tensor shape: (3, freq_bins, time_frames)
        # spectrogram = torch.stack(spectrograms, dim=0)
        # return spectrogram, y

        return X_seq, y
    
    def visualize(self):
        """Plot the entire dataset"""
        import matplotlib.pyplot as plt

        aX = self.features[:, 0]
        aY = self.features[:, 1]
        aZ = self.features[:, 2]

        # Plot the accelerometer data
        plt.figure(figsize=(14, 7))
        plt.plot(aX, label='aX')
        plt.plot(aY, label='aY')
        plt.plot(aZ, label='aZ')

        # Highlight regions where result == 1 (tremor)
        indices = range(len(self.labels))
        plt.fill_between(
            indices, 
            aX.min().item(), 
            aX.max().item(),
            where=(self.labels == 1).numpy(),
            color='red', 
            alpha=0.3,
            label='Tremor'
        )

        plt.xlabel('Time')
        plt.ylabel('Accelerometer Values')
        plt.title('Accelerometer Data with Tremor Regions Highlighted')
        plt.legend()
        plt.show()
    
def get_data_loader(filepath, batch_size=8, seq_length=100):
    dataset = TremorDataset(filepath, seq_length=seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_train_test_loaders(filepath, batch_size=8, seq_length=100, test_ratio=0.2):
    dataset = TremorDataset(filepath, seq_length=seq_length)
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    split = int(dataset_len * (1 - test_ratio))
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def test_model(model, test_loader, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_true.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()
    # Compute metrics using sklearn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_true, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_true, all_preds, zero_division=0))
    return accuracy, precision, recall, f1