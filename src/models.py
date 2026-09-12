import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolución 2D -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        """
        U-Net 2D para estimación de máscaras espectrales suave.
        Entrada: Espectrograma de la mezcla estéreo (2, F, T)
        Salida: Máscara M en rango [0, 1] de tamaño (2, F, T)
        """
        super().__init__()
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (Upsampling) + Skip Connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()  # Máscara suavizada en el rango [0, 1]
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        b = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(b)
        # Ajustar diferencia de dimensiones por pad si es necesario
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)
        
        # Última capa de resolución completa
        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)
        mask = self.outc(x)
        
        return mask

if __name__ == "__main__":
    # Test de arquitectura
    model = UNet2D()
    dummy_input = torch.randn(2, 2, 1025, 517) # Batch 2, Estéreo, Frecuencia, Tiempo
    output_mask = model(dummy_input)
    print("Modelo U-Net instanciado correctamente.")
    print("Dimensiones de Entrada: ", dummy_input.shape)
    print("Dimensiones de la Máscara: ", output_mask.shape)