"""Two UNet PyTorch fixtures sharing one project directory.

- UNet5:      classic 5-level encoder/decoder UNet, 512x512 grayscale input.
- UNet5Stem:  the same UNet5 with an extra 3x3 conv before it, an extra 3x3
              conv after it, and a global residual connection:
              y = stem_out(unet(stem_in(x))) + x

Channel widths are kept small (4..128) so exported ONNX files stay a few MB
while the graph is still a realistic ~400-node structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv 3x3 -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        return self.act2(self.conv2(x))


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # center-crop the skip to match spatial size (always equal here)
        return self.block(torch.cat([skip, x], dim=1))


class UNet5(nn.Module):
    """5-level UNet: inc + 4 downsampling stages, 4 upsampling stages."""

    def __init__(self, in_ch: int = 1, base_ch: int = 4):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.inc = DoubleConv(in_ch, c1)          # 512
        self.down1 = Down(c1, c2)                 # 256
        self.down2 = Down(c2, c3)                 # 128
        self.down3 = Down(c3, c4)                 # 64
        self.down4 = Down(c4, c5)                 # 32
        self.bottom = DoubleConv(c5, c5 * 2)      # 32 (bottleneck, 256 ch)
        self.up4 = Up(c5 * 2, c4, c4)             # -> 64
        self.up3 = Up(c4, c3, c3)                 # -> 128
        self.up2 = Up(c3, c2, c2)                 # -> 256
        self.up1 = Up(c2, c1, c1)                 # -> 512
        self.head = nn.Conv2d(c1, in_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b = self.bottom(x5)
        u = self.up4(b, x4)
        u = self.up3(u, x3)
        u = self.up2(u, x2)
        u = self.up1(u, x1)
        return self.head(u)


class UNet5Stem(nn.Module):
    """UNet5 wrapped with input/output convs and a global residual link."""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.stem_in = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.stem_act = nn.ReLU(inplace=False)
        self.unet = UNet5(in_ch=in_ch)
        self.stem_out = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.stem_act(self.stem_in(x))
        u = self.unet(h)
        y = self.stem_out(u)
        return y + x  # global residual connection


if __name__ == "__main__":
    for cls in (UNet5, UNet5Stem):
        m = cls().eval()
        with torch.no_grad():
            y = m(torch.randn(1, 1, 512, 512))
        print(cls.__name__, "output", tuple(y.shape),
              "params", sum(p.numel() for p in m.parameters()))
