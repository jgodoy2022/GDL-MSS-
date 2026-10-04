import torch
import torch.nn as nn
from openunmix import model as umx_model

def get_openunmix_model(n_fft=2048, sample_rate=44100):
    """
    Crea un modelo Open-Unmix (OpenUnmix) optimizado para un stem específico.
    Utiliza Bi-LSTMs y capas lineales por bin de frecuencia para preservar
    transitorios de batería y la energía grave del bajo.
    """
    nb_bins = n_fft // 2 + 1  # 1025 bins de frecuencia para n_fft=2048
    
    unmix = umx_model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=2,          # Audio Estéreo
        hidden_size=512,        # Capas Bi-LSTM
        max_bin=int(nb_bins),
    )
    return unmix