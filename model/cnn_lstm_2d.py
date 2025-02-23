import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# TODO: Try a fusion of CNN-LSTM with regular MLP on feature extraction (manual)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, cnn_channels=32, lstm_hidden_size=32, lstm_layers=2, lr=0.001):
        super(CNNLSTM, self).__init__()
        # Change convolution to 2d as input is a spectrogram (channels, freq, time)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=cnn_channels, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        # BatchNorm for 2D output
        self.batch_norm = nn.BatchNorm2d(cnn_channels * 2)
        # Adaptive pooling to fix the frequency dimension (we choose 8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, None))  # (fixed_freq=8, time remains variable)
        # LSTM input feature size = (cnn_channels*2 * fixed_freq)
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2 * 8,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        # TensorBoard writer
        self.writer = SummaryWriter()
    
    def forward(self, x):
        # x shape: (batch, channels=3, freq, time)
        cnn_out = self.cnn(x)  # shape: (batch, cnn_channels*2, H, W)
        cnn_out = self.batch_norm(cnn_out)
        cnn_out = self.adaptive_pool(cnn_out)  # shape: (batch, cnn_channels*2, 8, W)
        batch_size, channels, freq, time = cnn_out.shape
        # Permute to (batch, time, channels, freq) and flatten channel & freq dims
        x_lstm = cnn_out.permute(0, 3, 1, 2).reshape(batch_size, time, channels*freq)
        lstm_out, (h_n, _) = self.lstm(x_lstm)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output
    
    def train_step(self, x, y):
        """
        x: (batch, seq_length, 3)
        y: (batch)
        """
        device = next(self.parameters()).device  # Ensure inputs use the same device
        x = x.to(device)
        y = y.to(device)
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
        # Final save and output final training loss
        self.save_checkpoint(epochs, "final_model.pth")

        print("Training complete")
