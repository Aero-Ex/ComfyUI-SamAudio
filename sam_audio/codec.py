import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int, device=None, dtype=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = comfy.ops.cast_to_input(self.alpha, x)
        return snake(x, alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, kernel: int = 7, device=None, dtype=None, operations=None):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.act1 = Snake1d(dim, device=device, dtype=dtype)
        self.conv1 = operations.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, device=device, dtype=dtype)
        self.act2 = Snake1d(dim, device=device, dtype=dtype)
        self.conv2 = operations.Conv1d(dim, dim, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act1(x))
        y = self.conv2(self.act2(y))
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.res1 = ResidualUnit(in_dim, dilation=1, kernel=7, device=device, dtype=dtype, operations=operations)
        self.res2 = ResidualUnit(in_dim, dilation=3, kernel=7, device=device, dtype=dtype, operations=operations)
        self.res3 = ResidualUnit(in_dim, dilation=9, kernel=7, device=device, dtype=dtype, operations=operations)
        self.act = Snake1d(in_dim, device=device, dtype=dtype)
        kernel_size = 2 * stride
        pad = (kernel_size - stride) // 2
        self.conv = operations.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=pad, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv(self.act(x))
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_dim: int = 64, encoder_rates: list[int] = [2, 8, 10, 12], latent_dim: int = 1024, device=None, dtype=None, operations=None):
        super().__init__()
        self.in_conv = operations.Conv1d(1, encoder_dim, kernel_size=7, stride=1, padding=3, device=device, dtype=dtype)

        blocks = []
        d = encoder_dim
        for stride in encoder_rates:
            out_d = d * 2
            blocks.append(EncoderBlock(d, out_d, stride, device=device, dtype=dtype, operations=operations))
            d = out_d
        self.blocks = nn.ModuleList(blocks)

        self.out_act = Snake1d(d, device=device, dtype=dtype)
        self.out_conv = operations.Conv1d(d, latent_dim, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(self.out_act(x))
        return x


class Quantizer(nn.Module):
    def __init__(self, in_dim: int = 1024, latent_dim: int = 128, device=None, dtype=None, operations=None):
        super().__init__()
        self.in_proj = operations.Conv1d(in_dim, latent_dim * 2, kernel_size=1, device=device, dtype=dtype)
        self.out_proj = operations.Conv1d(latent_dim, in_dim, kernel_size=1, device=device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.in_proj(x).chunk(2, dim=1)
        return mean

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.act = Snake1d(in_dim, device=device, dtype=dtype)
        kernel_size = 2 * stride
        pad = (stride + 1) // 2
        output_padding = 1 if stride % 2 != 0 else 0
        self.conv_tr = operations.ConvTranspose1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_padding, device=device, dtype=dtype)
        self.res1 = ResidualUnit(out_dim, dilation=1, kernel=7, device=device, dtype=dtype, operations=operations)
        self.res2 = ResidualUnit(out_dim, dilation=3, kernel=7, device=device, dtype=dtype, operations=operations)
        self.res3 = ResidualUnit(out_dim, dilation=9, kernel=7, device=device, dtype=dtype, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_tr(self.act(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_dim: int = 1536, decoder_rates: list[int] = [12, 10, 8, 2], latent_dim: int = 1024, device=None, dtype=None, operations=None):
        super().__init__()
        self.in_conv = operations.Conv1d(latent_dim, decoder_dim, kernel_size=7, stride=1, padding=3, device=device, dtype=dtype)

        blocks = []
        d = decoder_dim
        for i, stride in enumerate(decoder_rates):
            out_d = decoder_dim // (2 ** (i + 1))
            blocks.append(DecoderBlock(d, out_d, stride, device=device, dtype=dtype, operations=operations))
            d = out_d
        self.blocks = nn.ModuleList(blocks)

        self.out_act = Snake1d(d, device=device, dtype=dtype)
        self.out_conv = operations.Conv1d(d, 1, kernel_size=7, stride=1, padding=3, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = torch.tanh(self.out_conv(self.out_act(x)))
        return x


class DACVAE(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, operations=None):
        super().__init__()

        encoder_dim = config.get("encoder_dim", 64)
        encoder_rates = config.get("encoder_rates", [2, 8, 10, 12])
        latent_dim = config.get("latent_dim", 1024)
        decoder_dim = config.get("decoder_dim", 1536)
        decoder_rates = config.get("decoder_rates", [12, 10, 8, 2])
        codebook_dim = config.get("codebook_dim", 128)
        self.sample_rate = config.get("sample_rate", 48000)
        self.hop_length = math.prod(encoder_rates)

        self.encoder = Encoder(encoder_dim=encoder_dim, encoder_rates=encoder_rates, latent_dim=latent_dim, device=device, dtype=dtype, operations=operations)
        self.quantizer = Quantizer(in_dim=latent_dim, latent_dim=codebook_dim, device=device, dtype=dtype, operations=operations)
        self.decoder = Decoder(decoder_dim=decoder_dim, decoder_rates=decoder_rates, latent_dim=latent_dim, device=device, dtype=dtype, operations=operations)

    def pad(self, wavs: torch.Tensor) -> torch.Tensor:
        length = wavs.size(-1)
        if length % self.hop_length != 0:
            pad_len = self.hop_length - (length % self.hop_length)
            return F.pad(wavs, (0, pad_len), mode="reflect")
        return wavs

    def encode(self, wavs: torch.Tensor) -> torch.Tensor:
        padded = self.pad(wavs)
        z = self.encoder(padded)
        return self.quantizer.encode(z)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        emb = self.quantizer.decode(latents)
        return self.decoder(emb)
