import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from abc_capsnet import ABC_CapsNet
from dataset import AudioDataset  # Custom dataset loader

def margin_loss(predictions, labels):
    loss = torch.mean(labels * torch.clamp(0.9 - predictions, min=0)**2 +
                      0.5 * (1.0 - labels) * torch.clamp(predictions - 0.1, min=0)**2)
    return loss

def train_model(model, data_loader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for mel_spectrogram, labels in data_loader:
            optimizer.zero_grad()
            output = model(mel_spectrogram)
            loss = margin_loss(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    model = ABC_CapsNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Create DataLoader (custom dataset class required)
    dataset = AudioDataset(args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_model(model, data_loader, optimizer, num_epochs=args.epochs)
