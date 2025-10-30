# src/models/tmrnn.py
"""
TMRNN – Temporal Multimodal RNN for EGD-CXR.
Now fully switchable:
  * image_backbone (resnet18/50, densenet121, txrv_densenet121)
  * use_gaze / use_roi / enable_text_head / use_teacher_forcing
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, Dict, Any

# torchvision backbones
from torchvision import models as tvm

# optional torchxrayvision
try:
    import torchxrayvision as txrv
    HAS_TXRV = True
except Exception:  # pragma: no cover
    HAS_TXRV = False

# T5 (only imported when text head is enabled)
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration


class TMRNN(nn.Module):
    def __init__(
        self,
        num_seg: int,
        num_box: int,
        d_g: int = 64,
        d_r: int = 64,
        d_img: int = 128,
        d_h: int = 256,
        num_classes: int = 3,
        t5_name: str = "google-t5/t5-base",
        freeze_t5: bool = True,
        use_attn_pool: bool = False,

        # ------------------- NEW SWITCHES ------------------- #
        image_backbone: str = "resnet50",
        use_gaze: bool = True,
        use_roi: bool = True,
        enable_text_head: bool = True,
        use_teacher_forcing: bool = True,
        # --------------------------------------------------- #
    ):
        super().__init__()
        self.num_seg = num_seg
        self.num_box = num_box
        self.d_g, self.d_r, self.d_img, self.d_h = d_g, d_r, d_img, d_h
        self.num_classes = num_classes
        self.use_attn_pool = use_attn_pool

        # Flags (exposed for training logic)
        self.image_backbone = image_backbone.lower()
        self.use_gaze = use_gaze
        self.use_roi = use_roi
        self.enable_text_head = enable_text_head
        self.use_teacher_forcing = use_teacher_forcing

        # --------------------------------------------------- #
        # 1. Image backbone (factory)
        # --------------------------------------------------- #
        self.img_encoder, backbone_out_dim = self._build_image_backbone(self.image_backbone)
        self.img_proj = nn.Linear(backbone_out_dim, d_img)

        # Normalisation buffers (ImageNet stats for torchvision models)
        if self.image_backbone.startswith("resnet") or self.image_backbone == "densenet121":
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        else:  # txrv already expects [0,1] single-channel
            self.register_buffer("img_mean", torch.tensor([0.0]), persistent=False)
            self.register_buffer("img_std",  torch.tensor([1.0]), persistent=False)

        # --------------------------------------------------- #
        # 2. Gaze / ROI (lazily created if needed)
        # --------------------------------------------------- #
        if self.use_gaze:
            self.gaze_mlp = nn.Sequential(
                nn.Linear(3, d_g * 2), nn.ReLU(),
                nn.Linear(d_g * 2, d_g), nn.ReLU(),
            )
        if self.use_roi:
            self.roi_proj = nn.Linear(num_seg + num_box, d_r)

        # --------------------------------------------------- #
        # 3. RNN encoder
        # --------------------------------------------------- #
        input_dim = 0
        if self.use_gaze: input_dim += d_g
        if self.use_roi:  input_dim += d_r
        input_dim += d_img
        self.encoder = nn.GRU(input_dim, d_h, batch_first=True)

        # --------------------------------------------------- #
        # 4. Classification head
        # --------------------------------------------------- #
        self.cls_head = nn.Linear(d_h, num_classes)
        if use_attn_pool:
            self.attn_pool = nn.Linear(d_h, 1)

        # --------------------------------------------------- #
        # 5. T5 text head (optional)
        # --------------------------------------------------- #
        if self.enable_text_head:
            self.tok = T5Tokenizer.from_pretrained(t5_name)
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_name)
            self.t5_decoder = T5ForConditionalGeneration.from_pretrained(t5_name)
            if freeze_t5:
                for p in self.t5_encoder.parameters(): p.requires_grad = False
                for p in self.t5_decoder.parameters(): p.requires_grad = False
            self.h_to_t5 = nn.Linear(d_h, 768)   # T5 hidden size
        else:
            self.tok = self.t5_encoder = self.t5_decoder = self.h_to_t5 = None

    # ------------------------------------------------------------------ #
    # Image backbone factory
    # ------------------------------------------------------------------ #
    def _build_image_backbone(self, name: str):
        name = name.lower()
        if name == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            return net, 512
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            net.fc = nn.Identity()
            return net, 2048
        if name == "densenet121":
            net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            net.classifier = nn.Identity()
            return net, 1024
        if name == "txrv_densenet121":
            if not HAS_TXRV:
                raise RuntimeError("torchxrayvision not installed")
            net = txrv.models.DenseNet(weights="densenet121-res224-chex")
            net.classifier = nn.Identity()
            return net, 1024
        raise ValueError(f"Unsupported image_backbone: {name}")

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Dict[str, Any],
        transcripts_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.parameters()).device
        images = batch["images"]                     # [B,1,224,224]
        fx = batch["fixations"]
        xy_norm = fx["xy_norm"]                      # [B,T,2]
        times   = fx["time"]                         # [B,T]
        seg_hits = fx["seg_hits"]                    # [B,T,S]
        box_hits = fx["box_hits"]                    # [B,T,Bb]
        lengths = fx["lengths"]                      # [B]

        B, T, _ = xy_norm.shape

        # --------------------------------------------------- #
        # 1. Δt (inter-fixation gap)
        # --------------------------------------------------- #
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        dt = torch.clamp(dt, min=0.0).unsqueeze(-1)   # [B,T,1]

        feats = []

        # --------------------------------------------------- #
        # 2. Gaze embedding (optional)
        # --------------------------------------------------- #
        if self.use_gaze:
            gaze_in = torch.cat([xy_norm, dt], dim=-1)            # [B,T,3]
            g_t = self.gaze_mlp(gaze_in.view(B * T, -1)).view(B, T, -1)
            feats.append(g_t)

        # --------------------------------------------------- #
        # 3. ROI projection (optional)
        # --------------------------------------------------- #
        if self.use_roi:
            roi_hits = torch.cat([seg_hits, box_hits], dim=-1)   # [B,T,R]
            r_t = self.roi_proj(roi_hits.view(B * T, -1)).view(B, T, -1)
            feats.append(r_t)

        # --------------------------------------------------- #
        # 4. Image embedding (shared across t)
        # --------------------------------------------------- #
        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            imgs3 = images.repeat(1, 3, 1, 1)
            imgs3 = (imgs3 - self.img_mean) / self.img_std
            img_emb = self.img_encoder(imgs3)                    # [B, D]
        else:  # txrv
            img_emb = self.img_encoder(images)                   # [B, 1024]

        img_proj = self.img_proj(img_emb).unsqueeze(1).expand(B, T, -1)
        feats.append(img_proj)

        # --------------------------------------------------- #
        # 5. Concatenate selected streams → s_t
        # --------------------------------------------------- #
        s_t = torch.cat(feats, dim=-1)                           # [B,T,d_in]

        # --------------------------------------------------- #
        # 6. RNN encoder
        # --------------------------------------------------- #
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.encoder(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)   # [B,T,d_h]
        last_h = h_n.squeeze(0)                                          # [B,d_h]

        # --------------------------------------------------- #
        # 7. Classification
        # --------------------------------------------------- #
        if self.use_attn_pool:
            attn = self.attn_pool(rnn_out).squeeze(-1)                   # [B,T]
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            attn = attn.masked_fill(~mask, float("-inf"))
            weights = F.softmax(attn, dim=1).unsqueeze(-1)
            pooled = (rnn_out * weights).sum(dim=1)
        else:
            pooled = last_h

        cls_logits = self.cls_head(pooled)                               # [B,C]

        # --------------------------------------------------- #
        # 8. Text decoder (optional + teacher-forcing guard)
        # --------------------------------------------------- #
        txt_logits = None
        if self.enable_text_head and transcripts_ids is not None and self.use_teacher_forcing:
            proj_h = self.h_to_t5(rnn_out)                               # [B,T,768]
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

            enc_out = self.t5_encoder.encoder(
                inputs_embeds=proj_h,
                attention_mask=mask.float(),
            )
            dec_out = self.t5_decoder(
                input_ids=transcripts_ids[:, :-1],   # shift-right
                encoder_hidden_states=enc_out.last_hidden_state,
                encoder_attention_mask=mask.float(),
            )
            txt_logits = dec_out.logits                               # [B,L-1,V]

        return cls_logits, txt_logits

    # ------------------------------------------------------------------ #
    # Inference generation (no teacher forcing)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_transcript(
        self,
        batch: Dict[str, Any],
        max_length: int = 100,
    ) -> str:
        if not self.enable_text_head:
            return ""

        self.eval()
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device)

        # Run encoder once
        cls_logits, _ = self.forward(batch, transcripts_ids=None)
        fx = batch["fixations"]
        times = fx["time"]
        lengths = fx["lengths"]

        # Re-compute final hidden state (same code as forward)
        B = 1
        xy_norm = fx["xy_norm"]
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        dt = torch.clamp(dt, min=0.0).unsqueeze(-1)

        feats = []
        if self.use_gaze:
            gaze_in = torch.cat([xy_norm, dt], dim=-1)
            g_t = self.gaze_mlp(gaze_in.view(B * -1, -1)).view(B, -1, self.d_g)
            feats.append(g_t)
        if self.use_roi:
            roi_hits = torch.cat([fx["seg_hits"], fx["box_hits"]], dim=-1)
            r_t = self.roi_proj(roi_hits.view(B * -1, -1)).view(B, -1, self.d_r)
            feats.append(r_t)

        img_emb = self.img_encoder(batch["images"])
        img_proj = self.img_proj(img_emb).unsqueeze(1).expand(B, -1, self.d_img)
        feats.append(img_proj)

        s_t = torch.cat(feats, dim=-1)
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed)
        last_h = h_n.squeeze(0)                     # [1, d_h]

        # Autoregressive generation
        input_ids = torch.tensor([[self.tok.bos_token_id]], device=device)
        generated = []

        for _ in range(max_length):
            proj_h = self.h_to_t5(last_h.unsqueeze(1))          # [1,1,768]
            mask = torch.ones(1, 1, device=device)

            out = self.t5_decoder(
                input_ids=input_ids,
                encoder_hidden_states=proj_h,
                encoder_attention_mask=mask,
            )
            next_token = out.logits[:, -1].argmax(dim=-1)

            if next_token.item() == self.tok.eos_token_id:
                break

            word = self.tok.decode(next_token, skip_special_tokens=True)
            generated.append(word)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return " ".join(generated)