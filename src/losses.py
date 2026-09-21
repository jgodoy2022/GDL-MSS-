import torch
import torch.nn as nn

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_audio, target_audio):
        loss = 0.0
        samples = pred_audio.shape[-1]
        pred_flat = pred_audio.view(-1, samples)
        target_flat = target_audio.view(-1, samples)

        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win).to(pred_audio.device)
            
            pred_stft = torch.stft(pred_flat, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            target_stft = torch.stft(target_flat, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)

            # Clamp para evitar valores 0 que exploten en log()
            pred_mag = torch.clamp(torch.abs(pred_stft), min=1e-5)
            target_mag = torch.clamp(torch.abs(target_stft), min=1e-5)

            # Convergence Loss numéricamente estable
            spectral_convergence = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-5)
            log_stft_magnitude = self.l1_loss(torch.log(pred_mag), torch.log(target_mag))

            loss += spectral_convergence + log_stft_magnitude

        return loss / len(self.fft_sizes)