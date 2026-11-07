"""A tiny U-Net that learns to undo my added faults. It is the CNN to compare the simple fixes with."""
import cv2
import numpy as np
import torch
from torch import nn

_loaded = {}


def block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """Works at half size to stay fast, and adds a correction to the input photo."""

    def __init__(self, base=8, deep=False):
        super().__init__()
        self.deep = deep
        if deep:
            self.deepest = block(base * 4, base * 4)
            self.up_middle = block(base * 8, base * 4)
        self.shrink = nn.PixelUnshuffle(2)
        self.first = block(12, base)
        self.second = block(base, base * 2)
        self.middle = block(base * 2, base * 4)
        self.up_second = block(base * 6, base * 2)
        self.up_first = block(base * 3, base)
        self.last = nn.Conv2d(base, 12, 3, padding=1)
        self.grow = nn.PixelShuffle(2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, photos):
        level1 = self.first(self.shrink(photos))
        level2 = self.second(self.down(level1))
        level3 = self.middle(self.down(level2))

        if self.deep:
            level4 = self.deepest(self.down(level3))
            level3 = self.up_middle(torch.cat([self.up(level4), level3], dim=1))

        up2 = torch.cat([self.up(level3), level2], dim=1)
        up2 = self.up_second(up2)
        up1 = torch.cat([self.up(up2), level1], dim=1)
        up1 = self.up_first(up1)
        return photos + self.grow(self.last(up1))


def photo_to_tensor(image):
    scaled = image.astype(np.float32) / 255
    return torch.from_numpy(scaled).permute(2, 0, 1)


def tensor_to_photo(tensor):
    values = tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255
    return (values + 0.5).astype(np.uint8)


def unet_fix(image, weights_path):
    """The network needs sides that are multiples of 16, so pad with black and cut it off after."""
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, -height % 16, 0, -width % 16, cv2.BORDER_CONSTANT, value=0)
    if weights_path not in _loaded:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        model = TinyUNet(deep="deepest.0.weight" in weights)
        model.load_state_dict(weights)
        _loaded[weights_path] = model.eval()
    with torch.no_grad():
        output = _loaded[weights_path](photo_to_tensor(padded).unsqueeze(0))[0]
    return tensor_to_photo(output)[:height, :width]
