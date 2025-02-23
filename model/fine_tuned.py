import torch
import torch.nn as nn
import torch.optim as optim
from cnn_lstm_1d import CNNLSTM

class AdaptedModel(nn.Module):
    """
    NOTE: When initalizing pre-trained model it should be on the same device!
    """
    def __init__(self, base_model: CNNLSTM = None, adapter_size=64, num_classes=2, lstm_hidden_size=32, device=torch.device("cpu")):
        super(AdaptedModel, self).__init__()
        if base_model is None: # no need if loading from file
            self.base = CNNLSTM(device=device)
        else:
            self.base = base_model
        self.adapter = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, adapter_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(adapter_size, num_classes)
        )
        self.optimizer = optim.Adam(self.adapter.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        # Turn off the original linear layer
        x = self.base.forward(x, use_linear=False)
        # Append the custom head
        x = self.adapter(x)
        return x

    def train_model(self, train_loader, num_epochs=50):
        for epoch in range(num_epochs):
            self.adapter.train()
            total_loss = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

        torch.save({
            'base': self.base.state_dict(),  # Save base model weights
            'adapter': self.adapter.state_dict()  # Save adapter weights
        }, 'adapted-1.pth')
        
        print("Model saved")

    def load(self, filename, base_model=None):
        """
        Loads the state dict of the model from a checkpoint file.
        If a base model is passed, it will use it, otherwise, it assumes that
        the base model has been initialized.
        """
        # Load the checkpoint and check if 'base' exists in the checkpoint
        checkpoint = torch.load(filename, map_location=self.device)

        # If base model is passed, load only the adapter weights and keep the base model unchanged
        if base_model:
            self.base = base_model  # Use the passed base model
            adapter_state_dict = checkpoint['adapter']  # Only load adapter weights
            self.adapter.load_state_dict(adapter_state_dict)
        else:
            # If base model isn't passed, load the entire model weights
            self.load_state_dict(checkpoint)
        
        self.eval()  # Set model to evaluation mode