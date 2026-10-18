import numpy as np
from torch import nn


def conv(inputs, outputs, kernel=3):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=kernel//2), nn.GroupNorm(1, outputs), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, channels=3, width=32):
        super().__init__()
        self.layers = nn.Sequential(conv(channels, width), conv(width, width), conv(width, width))

    def forward(self, x):
        return self.layers(x.float())


def image_demo(seed, n, channels=3, mask=True, size=32):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, channels, size, size)).astype("float32")
    y = (x[:, :1] > .5).astype("float32") if mask else x.copy()
    return {"x": x, "y": y, "group": np.arange(n)}
