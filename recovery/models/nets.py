"""Trained models (PyTorch): a layer-deviation classifier and a contour segmenter.
Both take the camera image concatenated with the expected-layer render (4 channels), which
makes them far more data-efficient than seeing the image alone. They implement the same
interface as the classical versions, so they drop into the controller."""
import cv2
import numpy as np
import torch
import torch.nn as nn

from .. import camera
from .classical import LABELS, Detection, boundary_deviation
from .synth_data import SIZE


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_classifier(pretrained=False):
    import torchvision
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    old = m.conv1
    m.conv1 = nn.Conv2d(4, 64, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            m.conv1.weight[:, :3] = old.weight
            m.conv1.weight[:, 3:] = old.weight.mean(1, keepdim=True)
    m.fc = nn.Linear(m.fc.in_features, len(LABELS))
    return m


def build_segmenter(pretrained=False):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None,
                    in_channels=4, classes=1)


def to_tensor(img_bgr, exp_mask):
    """image (H,W,3 uint8) + expected mask (H,W 0/1 or 0/255) -> (1,4,S,S) float."""
    im = cv2.resize(img_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ex = cv2.resize((exp_mask > 0).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = np.concatenate([im, ex[..., None]], -1).transpose(2, 0, 1)
    return torch.from_numpy(x)[None]


class TorchSegmenter:
    def __init__(self, ckpt, dev=None):
        self.dev = dev or device()
        self.net = build_segmenter()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.to(self.dev).eval()

    @torch.no_grad()
    def measure(self, image, expected_mask=None):
        if expected_mask is None:
            expected_mask = np.zeros(image.shape[:2], np.uint8)
        x = to_tensor(image, expected_mask).to(self.dev)
        p = torch.sigmoid(self.net(x))[0, 0].cpu().numpy()
        full = cv2.resize(p, (camera.CAM.size, camera.CAM.size), interpolation=cv2.INTER_LINEAR)
        return camera.mask_to_region((full > 0.5).astype(np.uint8))


class TorchDetector:
    def __init__(self, ckpt, dev=None):
        self.dev = dev or device()
        self.net = build_classifier()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.to(self.dev).eval()

    @torch.no_grad()
    def detect(self, image, expected_region, measured_region):
        x = to_tensor(image, camera.rasterise(expected_region)).to(self.dev)
        prob = torch.softmax(self.net(x), 1)[0].cpu().numpy()
        label = LABELS[int(prob.argmax())]
        dev = 0.0 if label == "ok" else (99.0 if label == "detach" else
                                         boundary_deviation(measured_region, expected_region))
        return Detection(label, dev, dict(prob=prob.tolist()))
