import os
import torch
import soundfile as sf
import musdb
from src.models import get_unet_model

def separate_sample():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cargando modelo en: {device}...")

    output_dir = os.path.join("samples", "vocals")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar el modelo profesional e importar checkpoint
    model = get_unet_model().to(device)
    model.load_state_dict(torch.load("unet_vocals.pth", map_location=device, weights_only=True))
    model.eval()

    # 2. Cargar canción de prueba (segundos 30 al 45)
    mus = musdb.DB(root='./data/musdb18hq', is_wav=True, subsets='test')
    track = mus.tracks[0]
    sample_rate = 44100
    start_sec = 30.0
    duration = 15.0
    
    track.chunk_start = start_sec
    track.chunk_duration = duration

    mix_audio = torch.tensor(track.audio.T, dtype=torch.float32)

    # 3. STFT
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)

    mix_stft = torch.stft(mix_audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mix_spec = torch.abs(mix_stft).unsqueeze(0).to(device)

    # 4. Predicción de la Máscara
    with torch.no_grad():
        mask = model(mix_spec)

    mask = mask.squeeze(0).cpu()

    # --- REFINAMIENTO DE MÁSCARA (Wiener Filter / Soft Power Contrast) ---
    # Aumenta el contraste de la máscara para silenciar residuos bajos
    # y estabilizar las partes donde la instrumental resurge.
    power = 1.5
    soft_mask = torch.pow(mask, power)
    
    # Normalización para mantener la presencia de la voz sin atenuar agudos
    soft_mask = soft_mask / (soft_mask + torch.pow(1.0 - mask, power) + 1e-8)

    # 5. Aplicar la máscara refinada
    separated_stft = soft_mask * mix_stft

    # 6. Reconstrucción iSTFT
    estimated_audio = torch.istft(separated_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=int(duration * sample_rate))

    # Convertir a numpy
    estimated_audio_np = estimated_audio.T.numpy()
    original_vocals_np = track.targets['vocals'].audio

    # 7. Guardar WAVs
    sf.write(os.path.join(output_dir, "mixture.wav"), track.audio, sample_rate)
    sf.write(os.path.join(output_dir, "vocals_estimated.wav"), estimated_audio_np, sample_rate)
    sf.write(os.path.join(output_dir, "vocals_original.wav"), original_vocals_np, sample_rate)

    print(f"\n¡Separación completada con U-Net ResNet34 y Máscara Refinada!")
    print(f"Escucha el nuevo audio generado: {output_dir}/vocals_estimated.wav")

if __name__ == "__main__":
    separate_sample()