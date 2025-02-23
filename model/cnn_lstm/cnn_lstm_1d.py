import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, cnn_channels=16, lstm_hidden_size=32, lstm_layers=2, lr=0.001):
        super(CNNLSTM, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully Connected Classifier
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter()
    
    def forward(self, x):
        # Input: (batch, seq_length, 3) -> permute to (batch, 3, seq_length)
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        # Permute back to (batch, seq_length, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(cnn_out)
        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output
    
    def train_step(self, x, y):
        """
        x: (batch, seq_length, 3)
        y: (batch)
        """
        self.optimizer.zero_grad()
        outputs = self(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_checkpoint(self, epoch, filename):
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {filename} at epoch {checkpoint.get('epoch', 'Unknown')}")
        else:
            print(f"No checkpoint found at {filename}")
    
    def inference(self, dataloader, device=torch.device("cpu")):
        self.to(device)
        self.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in dataloader:
                inputs = x.to(device)
                outputs = self(inputs)
                preds = torch.softmax(outputs, dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds, dim=0)
    
    def train_model(self, train_loader, epochs=100, checkpoint_interval=50, device=torch.device("cpu")):
        self.to(device)
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            total_loss = 0.0

            # Modified loop to correctly unpack (x, y)
            for i, (x, y) in enumerate(train_loader):
                loss = self.train_step(x, y)
                total_loss += loss
                global_step = epoch * len(train_loader) + i
                self.writer.add_scalar("Batch Loss", loss, global_step)
            avg_loss = total_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Epoch Loss", avg_loss, epoch+1)

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1, f"checkpoint_{epoch+1}.pth")

        self.writer.close()
        # Final save
        self.save_checkpoint(epochs, "final_model.pth")
        print("Training complete")
