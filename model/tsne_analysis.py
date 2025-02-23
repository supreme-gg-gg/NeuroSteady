import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from utils import TremorDataset
from cnn_lstm_1d import CNNLSTM

# Paths for the two datasets
PRETRAINED_DATA_PATH = "data/Dataset.csv"
NEW_DATA_PATH = "data/tremor_data.csv"

# Create datasets (assumes both CSVs have the same format)
pretrained_ds = TremorDataset(filepath=PRETRAINED_DATA_PATH, seq_length=100)
new_ds = TremorDataset(filepath=NEW_DATA_PATH, seq_length=100)

# Create DataLoaders (using a small batch size for feature extraction)
pretrain_loader = DataLoader(pretrained_ds, batch_size=32, shuffle=False)
new_loader = DataLoader(new_ds, batch_size=32, shuffle=False)

# Load pretrained CNNLSTM model (adjust path as needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(device=device)

# Load model weights (adjust filename as needed)
checkpoint = torch.load("weights/checkpoint_50.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Function to extract CNN features from the model
def extract_features_cnn(dataloader):
    features_list = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)  # x shape: (batch, seq_length, 3)
            # Pass x through the CNN block:
            # Permute: (batch, 3, seq_length) then pass through model.cnn
            x_input = x.permute(0, 2, 1)
            cnn_features = model.cnn(x_input)  # shape: (batch, channels, new_seq_len)
            # Flatten features per sample
            flat_features = cnn_features.view(cnn_features.size(0), -1)
            features_list.append(flat_features.cpu())
    return torch.cat(features_list, dim=0).numpy()

def extract_features(dataloader):
    features_list = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs = model.forward(x, use_linear=False)
            features_list.append(outputs.cpu())
    return torch.cat(features_list, dim=0).numpy()

# Extract features from both datasets
features_pretrained = extract_features(pretrain_loader)
features_new = extract_features(new_loader)

# Create labels for visualization: 0 for pretrained, 1 for new
labels_pretrained = np.zeros(features_pretrained.shape[0])
labels_new = np.ones(features_new.shape[0])

# Combine features and labels
all_features = np.concatenate([features_pretrained, features_new], axis=0)
all_labels = np.concatenate([labels_pretrained, labels_new], axis=0)

# Run t-SNE on combined features
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(all_features)

# Plot t-SNE result
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_tsne[:,0], features_tsne[:,1], c=all_labels, cmap='coolwarm', alpha=0.6)
plt.title("t-SNE of CNN Feature Maps: Pretrained vs. New Dataset")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(*scatter.legend_elements(), title="Dataset", labels=["Pretrained", "New"])
plt.show()
