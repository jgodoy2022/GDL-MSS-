import torch
from torch.utils.data import Dataset, DataLoader
import musdb
import numpy as np

class MUSDBDataset(Dataset):
    def __init__(self, target_stem='vocals', segment_duration=6.0, sample_rate=44100, is_wav=True, root=None, subset='train'):
        """
        Dataset personalizado para cargar mezclas y stems desde MUSDB18-HQ.
        
        Args:
            target_stem (str): La fuente a separar ('vocals', 'drums', 'bass', 'other').
            segment_duration (float): Duración del fragmento recortado en segundos.
            sample_rate (int): Tasa de muestreo del audio.
            subset (str): 'train' o 'test'.
        """
        self.target_stem = target_stem
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        
        # Cargar base de datos con musdb
        self.mus = musdb.DB(root=root, is_wav=is_wav, subsets=subset)
        self.tracks = self.mus.tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        # Seleccionar un punto de inicio aleatorio dentro del track
        track_duration = track.duration
        if track_duration > self.segment_duration:
            start_time = np.random.uniform(0, track_duration - self.segment_duration)
        else:
            start_time = 0.0

        track.chunk_start = start_time
        track.chunk_duration = self.segment_duration

        # Extraer canales estéreo
        # track.audio contiene la mezcla (mixture) y track.targets[target_stem].audio contiene el stem objetivo
        mix_audio = torch.tensor(track.audio.T, dtype=torch.float32)
        target_audio = torch.tensor(track.targets[self.target_stem].audio.T, dtype=torch.float32)

        # Transformación STFT en PyTorch
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft)

        # Calcular espectrogramas complejos -> (channels, n_fft // 2 + 1, time_frames)
        mix_stft = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        # Magnitud espectral
        mix_spec = torch.abs(mix_stft)
        target_spec = torch.abs(target_stft)

        return mix_spec, target_spec, mix_stft

if __name__ == "__main__":
    print("Módulo dataset.py definido correctamente.")