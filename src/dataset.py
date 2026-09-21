import torch
from torch.utils.data import Dataset
import musdb
import numpy as np

class MUSDBDataset(Dataset):
    def __init__(self, target_stem='vocals', segment_duration=4.0, sample_rate=44100, is_wav=True, root=None, subset='train', remix=False):
        self.target_stem = target_stem
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.remix = remix
        
        self.mus = musdb.DB(root=root, is_wav=is_wav, subsets=subset)
        self.tracks = self.mus.tracks

    def __len__(self):
        return len(self.tracks)

    def _get_chunk(self, track):
        track_duration = track.duration
        if track_duration > self.segment_duration:
            start_time = np.random.uniform(0, track_duration - self.segment_duration)
        else:
            start_time = 0.0

        track.chunk_start = start_time
        track.chunk_duration = self.segment_duration
        return track

    def __getitem__(self, idx):
        track = self._get_chunk(self.tracks[idx])
        
        if self.remix and len(self.tracks) > 1:
            # Seleccionar otro track aleatorio para mezclar instrumentos
            other_idx = np.random.choice([i for i in range(len(self.tracks)) if i != idx])
            other_track = self._get_chunk(self.tracks[other_idx])
            
            # Voz del track principal
            target_audio = torch.tensor(track.targets[self.target_stem].audio.T, dtype=torch.float32)
            
            # Sumar los otros stems del otro track
            background_audio = torch.zeros_like(target_audio)
            for stem in ['drums', 'bass', 'other']:
                background_audio += torch.tensor(other_track.targets[stem].audio.T, dtype=torch.float32)
            
            mix_audio = target_audio + background_audio
        else:
            mix_audio = torch.tensor(track.audio.T, dtype=torch.float32)
            target_audio = torch.tensor(track.targets[self.target_stem].audio.T, dtype=torch.float32)

        # STFT estándar para entregar al loader
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft)

        mix_stft = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        return torch.abs(mix_stft), torch.abs(target_stft), mix_audio, target_audio

if __name__ == "__main__":
    print("Módulo dataset.py (Fase 2) actualizado correctamente.")