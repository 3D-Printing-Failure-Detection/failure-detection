"""Trained models (PyTorch), modern backbones, video-style input.

  Detector   ConvNeXt-V2 (timm)              -> failure class of the current layer
  Segmenter  SegFormer / MiT (transformers)  -> mask of the current layer's contour

Both see a short TEMPORAL STACK rather than a single photo: the previous layer's frame, the
current layer's frame and the expected render of the current layer (3 + 3 + 1 = 7 channels).
The previous frame lets the network see what CHANGED between two layers (a sag, a shift, a
starved wall) instead of guessing from one image; stack more frames with n_frames > 2.
They implement the same interface as the classical versions, so they drop into the controller.

TorchVision's ResNet / smp U-Net of the first version are gone. (timm lists torchvision as a
dependency; no TorchVision model is used.) Options: swap `backbone` for any timm model,
`seg_arch` for any SegFormer checkpoint (nvidia/mit-b0 ... b5). A SAM 2 video tracker is a
possible further step but is not wired in.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import camera
from .classical import LABELS, Detection, boundary_deviation
from .synth_data import SIZE

DEFAULT_BACKBONE = "convnextv2_nano"
DEFAULT_SEG = "nvidia/mit-b2"
N_FRAMES = 2


def in_channels(n_frames=N_FRAMES):
    return 3 * n_frames + 1


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_detector(backbone=DEFAULT_BACKBONE, n_frames=N_FRAMES, pretrained=False):
    import timm
    return timm.create_model(backbone, pretrained=pretrained, in_chans=in_channels(n_frames),
                             num_classes=len(LABELS))


def build_segmenter(seg_arch=DEFAULT_SEG, n_frames=N_FRAMES, pretrained=False, config=None):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    in_ch = in_channels(n_frames)
    if config is not None:
        cfg = SegformerConfig.from_dict(config)
        if cfg.num_channels != in_ch or cfg.num_labels != 1:
            raise ValueError("saved segmenter config does not match frame/label count")
        return SegformerForSemanticSegmentation(cfg)
    if not pretrained:
        # Read only architecture metadata; weights remain randomly initialized.
        cfg = SegformerConfig.from_pretrained(seg_arch, num_labels=1, num_channels=in_ch)
        return SegformerForSemanticSegmentation(cfg)
    model = SegformerForSemanticSegmentation.from_pretrained(seg_arch, num_labels=1,
                                                             ignore_mismatched_sizes=True)
    old = model.segformer.encoder.patch_embeddings[0].proj          # expects 3 channels
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():                                           # spread the RGB filters over all channels
        new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
        new.bias.copy_(old.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new
    model.config.num_channels = in_ch
    return model


def seg_logits(model, x):
    """SegFormer predicts at 1/4 resolution; upsample to the input size."""
    lo = model(pixel_values=x).logits
    return Fn.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)


def stack_tensor(frames, exp_mask):
    """frames: list of BGR uint8 images, oldest first, current last; exp_mask (H,W) -> (1,C,S,S)."""
    chans = [cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
             for f in frames]
    ex = cv2.resize((np.asarray(exp_mask) > 0).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = np.concatenate(chans + [ex[..., None]], -1).transpose(2, 0, 1)
    return torch.from_numpy(x)[None]


class _History:
    """Remembers the last n_frames-1 camera frames so a model sees a short video."""
    def __init__(self, n_frames):
        self.n, self.buf = n_frames, deque(maxlen=max(n_frames - 1, 1))

    def frames(self, current):
        hist = list(self.buf)[-(self.n - 1):] if self.n > 1 else []
        while len(hist) < self.n - 1:                         # first layer: repeat what we have
            hist.insert(0, hist[0] if hist else current)
        return hist + [current]

    def push(self, current):
        self.buf.append(current)


def _load_cfg(ckpt_dir):
    p = Path(ckpt_dir) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


class NeuralSegmenter:
    def __init__(self, ckpt_dir="runs", dev=None):
        cfg = _load_cfg(ckpt_dir)
        self.n = cfg.get("n_frames", N_FRAMES)
        self.dev = dev or device()
        self.net = build_segmenter(cfg.get("seg_arch", DEFAULT_SEG), self.n,
                                   config=cfg.get("segmenter_config"))
        self.net.load_state_dict(torch.load(Path(ckpt_dir) / "seg.pt", map_location="cpu"))
        self.net.to(self.dev).eval()
        self.hist = _History(self.n)

    @torch.no_grad()
    def measure(self, image, expected_mask=None):
        if expected_mask is None:
            expected_mask = np.zeros(image.shape[:2], np.uint8)
        x = stack_tensor(self.hist.frames(image), expected_mask).to(self.dev)
        self.hist.push(image)
        p = torch.sigmoid(seg_logits(self.net, x))[0, 0].cpu().numpy()
        full = cv2.resize(p, (camera.CAM.size, camera.CAM.size), interpolation=cv2.INTER_LINEAR)
        return camera.mask_to_region((full > 0.5).astype(np.uint8))


class NeuralDetector:
    def __init__(self, ckpt_dir="runs", dev=None):
        cfg = _load_cfg(ckpt_dir)
        self.n = cfg.get("n_frames", N_FRAMES)
        self.dev = dev or device()
        self.net = build_detector(cfg.get("backbone", DEFAULT_BACKBONE), self.n)
        self.net.load_state_dict(torch.load(Path(ckpt_dir) / "det.pt", map_location="cpu"))
        self.net.to(self.dev).eval()
        self.hist = _History(self.n)

    @torch.no_grad()
    def detect(self, image, expected_region, measured_region):
        x = stack_tensor(self.hist.frames(image), camera.rasterise(expected_region)).to(self.dev)
        self.hist.push(image)
        prob = torch.softmax(self.net(x), 1)[0].cpu().numpy()
        label = LABELS[int(prob.argmax())]
        dev = 0.0 if label == "ok" else (99.0 if label == "detach" else
                                         boundary_deviation(measured_region, expected_region))
        return Detection(label, dev, dict(prob=prob.tolist()))
