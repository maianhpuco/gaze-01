# from __future__ import annotations
# from typing import Tuple
# import torch
# import torch.nn as nn
# from torchvision import models as tvm

# try:
#     import torchxrayvision as txrv
#     HAS_TXRV = True
# except Exception:
#     HAS_TXRV = False

# class ImageCNN(nn.Module):
#     """Image-only classifier (no RNN)."""
#     def __init__(self, num_classes: int, image_backbone: str = "resnet50", d_img: int = 128, use_proj: bool = True, freeze_backbone: bool = False):
#         super().__init__()
#         self.image_backbone = image_backbone.lower()
#         self.backbone, out_dim = self._build_backbone(self.image_backbone)
#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         self.use_proj = use_proj
#         if use_proj:
#             self.proj = nn.Linear(out_dim, d_img)
#             self.cls_head = nn.Linear(d_img, num_classes)
#         else:
#             self.proj = nn.Identity()
#             self.cls_head = nn.Linear(out_dim, num_classes)
#         if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
#             self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
#             self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
#         else:
#             self.register_buffer("img_mean", torch.tensor([0.0]), persistent=False)
#             self.register_buffer("img_std", torch.tensor([1.0]), persistent=False)
#     def _build_backbone(self, name: str) -> Tuple[nn.Module, int]:
#         name = name.lower()
#         if name == "resnet18":
#             net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
#             net.fc = nn.Identity()
#             return net, 512
#         if name == "resnet50":
#             net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
#             net.fc = nn.Identity()
#             return net, 2048
#         if name == "densenet121":
#             net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
#             net.classifier = nn.Identity()
#             return net, 1024
#         if name == "txrv_densenet121":
#             if not HAS_TXRV:
#                 raise RuntimeError("torchxrayvision not installed.")
#             net = txrv.models.DenseNet(weights="densenet121-res224-chex")
#             net.classifier = nn.Identity()
#             return net, 1024
#         raise ValueError(f"Unsupported image_backbone: {name}")
#     def forward(self, batch):
#         x = batch["images"]
#         if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
#             x = x.repeat(1, 3, 1, 1)
#             x = (x - self.img_mean) / self.img_std
#         feats = self.backbone(x)
#         z = self.proj(feats)
#         logits = self.cls_head(z)
#         return logits, None

# src/models/image_cnn.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models as tvm

try:
    import torchxrayvision as txrv
    HAS_TXRV = True
except Exception:
    HAS_TXRV = False


class ImageCNN(nn.Module):
    """Image-only classifier (no RNN)."""
    def __init__(self, num_classes: int, image_backbone: str = "resnet50",
                 d_img: int = 128, use_proj: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.image_backbone = image_backbone.lower()

        # Build backbone
        self.backbone, out_dim = self._build_backbone(self.image_backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(out_dim, d_img)
            self.cls_head = nn.Linear(d_img, num_classes)
        else:
            self.proj = nn.Identity()
            self.cls_head = nn.Linear(out_dim, num_classes)

        # Normalization buffers
        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        else:
            # TXRV expects 1ch roughly in [-1024, 1024]
            self.register_buffer("img_mean", torch.tensor([0.0]), persistent=False)
            self.register_buffer("img_std",  torch.tensor([1.0]), persistent=False)

    def _build_backbone(self, name: str) -> Tuple[nn.Module, int]:
        name = name.lower()
        if name == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()    # 512-d
            return net, 512
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            net.fc = nn.Identity()    # 2048-d
            return net, 2048
        if name == "densenet121":
            net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            net.classifier = nn.Identity()  # 1024-d
            return net, 1024
        if name == "txrv_densenet121":
            if not HAS_TXRV:
                raise RuntimeError("torchxrayvision not installed.")
            # IMPORTANT: do NOT replace .classifier or call .forward()
            # We'll directly use .features and our own head.
            net = txrv.models.DenseNet(weights="densenet121-res224-chex")
            # Attach a small head to get 1024-d global features
            net._feature_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            return net, 1024
        raise ValueError(f"Unsupported image_backbone: {name}")

    @staticmethod
    def _maybe_to_txrv_range(x: torch.Tensor) -> torch.Tensor:
        """Heuristic: if x looks like [0,1] or [0,255], map to [-1024, 1024]."""
        with torch.no_grad():
            x_min = float(x.min().item())
            x_max = float(x.max().item())
        if x_min < -100 and x_max > 100:
            return x
        if -1e-5 <= x_min and x_max <= 1.0 + 1e-5:
            return x * 2048.0 - 1024.0
        if -1e-5 <= x_min and x_max <= 255.0 + 1e-5:
            return (x / 255.0) * 2048.0 - 1024.0
        # Fallback: standardize then scale to a reasonable dynamic range
        x_mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x_sd = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        return (x - x_mu) / x_sd * 512.0

    def forward(self, batch):
        x = batch["images"]  # [B,1,H,W]

        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            # torchvision path
            x = x.repeat(1, 3, 1, 1)
            x = (x - self.img_mean) / self.img_std
            feats = self.backbone(x)            # [B, D]
        else:
            # TXRV path: 1ch, HU-like normalization AND bypass txrv.forward()
            x = self._maybe_to_txrv_range(x)
            # Use the DenseNet feature stack directly
            # torchxrayvision DenseNet has .features like torchvision
            fmap = self.backbone.features(x)    # [B, 1024, H', W']
            feats = self.backbone._feature_head(fmap)  # [B, 1024]

        z = self.proj(feats)
        logits = self.cls_head(z)
        return logits, None
