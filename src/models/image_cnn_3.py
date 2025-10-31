from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

try:
    import torchxrayvision as txrv
    HAS_TXRV = True
except Exception:
    HAS_TXRV = False


class ImageCNN(nn.Module):
    """Image-only classifier (no RNN), with CXR-correct TXRV path."""
    def __init__(self, num_classes: int, image_backbone: str = "resnet50",
                 d_img: int = 128, use_proj: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.image_backbone = image_backbone.lower()

        # Build backbone
        self.backbone, out_dim, self.is_tv = self._build_backbone(self.image_backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.use_proj = use_proj
        if use_proj:
            self.head = nn.Sequential(
                nn.Linear(out_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, d_img),   nn.BatchNorm1d(d_img), nn.ReLU(), nn.Dropout(0.3),
            )
            cls_in = d_img
        else:
            self.head = nn.Identity()
            cls_in = out_dim

        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, num_classes)
        )

        # Normalization buffers (TV → ImageNet; TXRV → pass 1ch [0,1] to .features)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)

    def _build_backbone(self, name: str) -> Tuple[nn.Module, int, bool]:
        name = name.lower()
        if name == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()    # 512-d
            return net, 512, True
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            net.fc = nn.Identity()    # 2048-d
            return net, 2048, True
        if name == "densenet121":
            net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            net.classifier = nn.Identity()  # 1024-d
            return net, 1024, True
        if name == "txrv_densenet121":
            assert HAS_TXRV, "torchxrayvision not installed."
            # Use CXR-pretrained weights and extract features directly
            net = txrv.models.DenseNet(weights="densenet121-res224-all")
            net.classifier = nn.Identity()
            return net, 1024, False
        raise ValueError(f"Unsupported image_backbone: {name}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,224,224] in [0,1]
        if self.is_tv:
            x3 = x.repeat(1, 3, 1, 1)
            x3 = (x3 - self.img_mean) / self.img_std
            return self.backbone(x3)  # [B, D]
        else:
            # TXRV DenseNet: use .features(...) then GAP (avoid .forward op-norm)
            fmap = self.backbone.features(x)            # [B, C, H, W]
            feats = F.adaptive_avg_pool2d(fmap, 1).flatten(1)  # [B, 1024]
            return feats

    def forward(self, batch):
        x = batch["images"].float()  # [B,1,H,W] in [0,1]
        v = self._encode(x)          # [B,D]
        h = self.head(v)
        logits = self.cls_head(h)
        return logits, h
