import torch
import segmentation_models_pytorch as smp

def get_unet_model():
    """
    Importa una U-Net profesional con encoder ResNet34 desde 'segmentation_models_pytorch'.
    Entrada: Espectrograma estéreo (Batch, 2, Frecuencia, Tiempo)
    Salida: Máscara suave en rango [0, 1] (Batch, 2, Frecuencia, Tiempo)
    """
    model = smp.Unet(
        encoder_name="resnet34",      # Backbone profesional con bloques residuales
        encoder_weights=None,         # Entrenamiento desde cero adaptado a espectrogramas de audio
        in_channels=2,                # Audio Estéreo (Izquierda / Derecha)
        classes=2,                    # Máscara de Salida Estéreo
        activation='sigmoid'          # Garantiza que la máscara esté entre 0.0 y 1.0
    )
    return model

if __name__ == "__main__":
    model = get_unet_model()
    dummy_input = torch.randn(2, 2, 1025, 517)
    output_mask = model(dummy_input)
    print("¡U-Net profesional importada exitosamente desde librería!")
    print("Dimensiones de Entrada: ", dummy_input.shape)
    print("Dimensiones de la Máscara: ", output_mask.shape)