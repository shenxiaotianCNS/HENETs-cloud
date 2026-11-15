"""Whole-block BCSS inference and compartment summaries in the paired H&E/IF grid."""
from __future__ import annotations
import cv2
import numpy as np
import torch


@torch.inference_mode()
def predict_full_block(model, rgb, device="cuda", collapse=False):
    """Same fully convolutional model, padded to its 32-pixel encoder stride.

    Softmax precedes bilinear upsampling, matching reference inference. Callers
    provide real context and discard it after mapping to the native output grid.
    """
    h, w = rgb.shape[:2]
    padded = np.pad(rgb, ((0, (-h)%32), (0, (-w)%32), (0,0)), mode="reflect")
    x = torch.from_numpy(padded.transpose(2,0,1).copy()).unsqueeze(0).to(device=device,dtype=torch.float32)
    model.eval()
    with torch.autocast("cuda", enabled=device.startswith("cuda")):
        logits = model(x)
    probabilities = logits.float().softmax(1)
    if collapse:
        probabilities = torch.cat([probabilities[:,:2],probabilities[:,2:].sum(1,keepdim=True)],dim=1)
    probs = probabilities[0].permute(1,2,0).cpu().numpy()
    # OpenCV and torch align_corners=False use the same half-pixel centers.
    return cv2.resize(probs, (padded.shape[1],padded.shape[0]), interpolation=cv2.INTER_LINEAR)[:h,:w]


def dense_labels(probs, rgb):
    """Vectorized three-class argmax with reference-compatible ties/background."""
    p0,p1,p2=probs[...,0],probs[...,1],probs[...,2]
    best01=np.maximum(p0,p1)
    best=np.maximum(best01,p2)
    labels=np.where(p1>p0,2,1).astype(np.uint8)
    labels[p2>best01]=3
    sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[...,1]
    brightness=rgb[...,0].astype(np.uint16)+rgb[...,1]+rgb[...,2]
    blank=(sat<20)&(brightness>660)
    labels[blank]=0
    return labels,((best<.6)&~blank),np.rint(best*255).astype(np.uint8)


def compartment_stats(labels, prefix=""):
    counts = np.bincount(labels.ravel(), minlength=256)
    n = int(counts[1:4].sum())
    names = ["tumor","stroma","other"]
    fractions = counts[1:4] / max(n,1)
    dominant = names[int(fractions.argmax())] if n else "background"
    purity = float(fractions.max()) if n else 0.
    result = {prefix+"tissue_pixels":n, prefix+"background_pixels":int(counts[0]),
              prefix+"unprocessed_pixels":int(counts[255]),
              prefix+"dominant_compartment":dominant,
              prefix+"compartment":"mixed" if n and purity<.6 else dominant,
              prefix+"purity":purity}
    for i, name in enumerate(names,1):
        result[prefix+name+"_pixels"] = int(counts[i])
        result[prefix+name+"_fraction"] = float(fractions[i-1])
    return result
