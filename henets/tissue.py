"""Exploratory BCSS tissue segmentation; NOT an ICC-validated classifier.

Network structure adapted from TIAToolbox v1.5.1 (BSD-3-Clause).
See tools/licenses/tiatoolbox.txt. BCSS weights are CC0 per TIA's model registry.
Input is RGB in 0..255 at 0.25 micrometres per pixel, without ImageNet normalization.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck, ResNet


CLASS_NAMES = {0: "Background", 1: "Tumor", 2: "Stroma", 3: "Other"}
COLORS_RGB = np.array([[255, 255, 255], [225, 64, 75], [40, 170, 125], [105, 125, 185]], np.uint8)


class ResNetEncoder(ResNet):
    def _forward_impl(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [x0, x1, x2, x3, self.layer4(x3)]


class UpSample2x(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("unpool_mat", torch.ones(2, 2))

    def forward(self, x):
        return x.repeat_interleave(2, -2).repeat_interleave(2, -1)


class BCSSTissueModel(nn.Module):
    """State-dict-compatible ResNet50 UNet with five source classes."""
    def __init__(self):
        super().__init__()
        self.backbone = ResNetEncoder(Bottleneck, [3, 4, 6, 3])
        self.conv1x1 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.uplist = nn.ModuleList()
        for incoming, outgoing in [(1024, 512), (512, 256), (256, 64), (64, 64)]:
            self.uplist.append(nn.Sequential(
                nn.BatchNorm2d(incoming), nn.ReLU(),
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            ))
        self.clf = nn.Conv2d(64, 5, 1)
        self.upsample2x = UpSample2x()

    def forward(self, images):
        features = self.backbone(images / 255.0)
        x = self.conv1x1(features[-1])
        for block, skip in zip(self.uplist, reversed(features[:-1])):
            x = block(self.upsample2x(x) + skip)
        return self.clf(x)


@torch.inference_mode()
def predict_rgb(model, rgb: np.ndarray, device: str, stride: int = 256) -> np.ndarray:
    """1024 input / central 512 output, overlap-averaged, following TIA inference.

    Returns H,W,5 source-class probabilities at the input image resolution.
    Padding is used only outside the supplied image. Callers should supply real
    surrounding tissue when extracting an ROI to avoid reflected boundary context.
    """
    if stride not in (256, 512):
        raise ValueError("stride must be 256 or 512")
    height, width = rgb.shape[:2]
    ys = list(range(0, max(height - 512, 0) + stride, stride))
    xs = list(range(0, max(width - 512, 0) + stride, stride))
    padded = np.pad(rgb, ((256, 1024), (256, 1024), (0, 0)), mode="reflect")
    total = np.zeros((height, width, 5), np.float32)
    counts = np.zeros((height, width, 1), np.float32)
    model.eval()
    for y in ys:
        for x in xs:
            tile = np.ascontiguousarray(padded[y:y + 1024, x:x + 1024].transpose(2, 0, 1))
            tensor = torch.from_numpy(tile).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                logits = model(tensor)
            # The upstream reference applies softmax before bilinear upsampling.
            probs = F.interpolate(logits.float().softmax(1), scale_factor=2,
                                  mode="bilinear", align_corners=False)
            probs = probs[0, :, 256:768, 256:768].permute(1, 2, 0).cpu().numpy()
            hh, ww = min(512, height - y), min(512, width - x)
            total[y:y + hh, x:x + ww] += probs[:hh, :ww]
            counts[y:y + hh, x:x + ww] += 1
    if np.any(counts == 0):
        raise RuntimeError("Sliding-window prediction left uncovered pixels")
    return total / counts


def collapse_probabilities(source: np.ndarray) -> np.ndarray:
    """Other = inflammation + necrosis + other, collapsed BEFORE argmax."""
    if source.ndim != 3 or source.shape[-1] != 5:
        raise ValueError("Expected H,W,5 source probabilities")
    return np.stack([source[..., 0], source[..., 1], source[..., 2:].sum(-1)], -1)


def labels_and_review_mask(source: np.ndarray, rgb: np.ndarray, threshold=0.6):
    import cv2
    probs = collapse_probabilities(source)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Conservative blank/lumen exclusion. This is a technical mask, not a tissue class.
    blank = (hsv[..., 1] < 20) & (rgb.mean(-1) > 220)
    labels = (probs.argmax(-1) + 1).astype(np.uint8)
    labels[blank] = 0
    review = (~blank) & (probs.max(-1) < threshold)
    return labels, review.astype(np.uint8) * 255, probs
