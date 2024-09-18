import librosa
import numpy as np
import os

def preprocess_audio(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    audio = audio - np.mean(audio)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=224)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

def save_mel_spectrograms(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        mel_spectrogram = preprocess_audio(file_path)
        output_path = os.path.join(output_dir, file_name.replace(".wav", ".npy"))
        np.save(output_path, mel_spectrogram)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory containing audio files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed Mel spectrograms')
    args = parser.parse_args()
    
    save_mel_spectrograms(args.data_dir, args.output_dir)
