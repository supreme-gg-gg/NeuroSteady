import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, cnn_channels=16, lstm_hidden_size=32, lstm_layers=1, lr=0.001, device=torch.device("cpu")):
        super(CNNLSTM, self).__init__()
        self.device = device  # Set device for the model

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

        # LSTM
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

        # move model to device, similarly for all tensors
        self.to(self.device)
    
    def forward(self, x, use_linear=True):
        """
        x: (batch, seq_length, 3)
        use_linear: Whether to use the linear head or not
        """

        # Input: (batch, seq_length, 3) -> permute to (batch, 3, seq_length)
        x = x.to(self.device)
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)

        # Permute back to (batch, seq_length, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(cnn_out)

        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]

        # When transfer learning, turn this off and learn the custom head
        if use_linear:
            output = self.fc(last_hidden)
        else:
            output = last_hidden

        return output
    
    def train_step(self, x, y):
        """
        x: (batch, seq_length, 3)
        y: (batch)
        """
        x = x.to(self.device)
        y = y.to(self.device)
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
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {filename} at epoch {checkpoint.get('epoch', 'Unknown')}")
        else:
            print(f"No checkpoint found at {filename}")
    
    def inference(self, dataloader):
        """
        Returns the predictions for the entire dataset
        actual labels 0 or 1 not logits or probabilities
        does softmax and argmax automatically
        """
        # Use self.device for inference
        self.to(self.device)
        self.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                outputs = self(x)
                preds = torch.softmax(outputs, dim=1).argmax(dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds, dim=0)
    
    def train_model(self, train_loader, val_loader=None, epochs=100, checkpoint_interval=50):
        """
        Uses TensorBoard to log training loss by default.
        You can visualize the logs using `tensorboard --logdir=runs` in the terminal.
        This assumes you have the `runs` directory in the same folder as this script.
        So if you use colab please download the logs and visualize them locally.
        """
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
            self.writer.add_scalar("Epoch Loss", avg_loss, epoch+1)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        outputs = self(x_val)
                        loss_val = self.criterion(outputs, y_val)
                        val_loss += loss_val.item()
                    avg_val_loss = val_loss / len(val_loader)
                    self.writer.add_scalar("Validation Loss", avg_val_loss, epoch+1)
                    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
                    
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1, f"checkpoint_{epoch+1}.pth")

        self.writer.close()
        self.save_checkpoint(epochs, "final_model.pth")
        print("Training complete")
