import torch
from torch.utils.data import DataLoader
from abc_capsnet import ABC_CapsNet
from dataset import AudioDataset

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel_spectrogram, labels in data_loader:
            output = model(mel_spectrogram)
            predicted = (output > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--model-checkpoint', required=True, help='Path to saved model checkpoint')
    args = parser.parse_args()

    model = ABC_CapsNet()
    model.load_state_dict(torch.load(args.model_checkpoint))

    dataset = AudioDataset(args.data_dir)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    evaluate_model(model, data_loader)
