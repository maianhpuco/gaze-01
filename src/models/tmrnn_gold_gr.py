# ──────────────────────────────────────────────────────────────────────────────
# File: src/models/tmrnn.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import torchxrayvision as xrv


# ----------------------------
# Small MLP blocks
# ----------------------------
class GazeMLP(nn.Module):
    """φ_g([x, y, Δt]) → ℝ^{d_g}. (Use Δt; dwell can be added if desired.)"""
    def __init__(self, d_g: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_g * 2), nn.ReLU(),
            nn.Linear(d_g * 2, d_g), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ROILinear(nn.Module):
    """W_r · roi_hits_t (multi‑hot of seg+box)."""
    def __init__(self, r_dim: int, d_r: int):
        super().__init__()
        self.proj = nn.Linear(r_dim, d_r)
    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        return self.proj(roi)


class CXRDenseNet121(nn.Module):
    """TorchXRayVision DenseNet‑121, CheXpert/MIMIC pretrained.
    Produces a global 1024‑d embedding per image.
    """
    def __init__(self):
        super().__init__()
        # Common strong CXR pretrain
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        # Remove classification head so we can get features
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,224,224] in [0,1]
        feats = self.backbone.features(x)
        # Some versions return (B,1024), others (B,1024,7,7). Make robust:
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return feats  # [B,1024]


# ----------------------------
# Core model
# ----------------------------
class TMRNN(nn.Module):
    """
    Temporal Multimodal RNN for EGD‑CXR.

    g_t = φ_g([x_t, y_t, Δt_t]) ∈ ℝ^{d_g}
    r_t = W_r · roi_hits_t ∈ ℝ^{d_r}
    img_t = v_img_proj ∈ ℝ^{d_img}  (shared across time)

    Encoder: GRU over s_t = concat(g_t, r_t, img_t)
    Heads:   (1) classification  (2) T5 decoder for transcript
    """
    def __init__(
        self,
        num_seg: int,
        num_box: int,
        num_classes: int = 3,
        d_g: int = 64,
        d_r: int = 64,
        d_img: int = 128,
        d_h: int = 256,
        t5_name: str = "google-t5/t5-base",
        freeze_t5: bool = True,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_attn_pool = use_attn_pool

        # Image encoder (CXR‑specific) → project to d_img
        self.img_enc = CXRDenseNet121()          # [B,1024]
        self.img_proj = nn.Linear(1024, d_img)   # [B,d_img]

        # Gaze & ROI
        self.gaze = GazeMLP(d_g)
        self.roi  = ROILinear(num_seg + num_box, d_r)

        # RNN over time
        self.enc_rnn = nn.GRU(d_g + d_r + d_img, d_h, batch_first=True)

        # Classification head (optionally attention‑pooled over time)
        if use_attn_pool:
            self.attn = nn.Linear(d_h, 1)
        self.cls = nn.Linear(d_h, num_classes)

        # T5 text (encoder‑decoder); we'll use only the conditional‑gen wrapper
        self.tok = T5Tokenizer.from_pretrained(t5_name)
        self.t5  = T5ForConditionalGeneration.from_pretrained(t5_name)
        if freeze_t5:
            for p in self.t5.parameters():
                p.requires_grad = False

        # Map GRU states → T5 hidden size (base=768)
        self.to_t5 = nn.Linear(d_h, self.t5.config.d_model)

    # ---- utilities ----
    @staticmethod
    def _delta_t(times: torch.Tensor) -> torch.Tensor:
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        return torch.clamp(dt, min=0.0).unsqueeze(-1)  # [B,T,1]

    # ---- forward (train) ----
    def forward(
        self,
        batch: Dict[str, Any],
        transcripts_ids: Optional[torch.Tensor] = None,   # [B,L]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (cls_logits, txt_logits). If transcripts_ids is given, txt_logits is not None."""
        images = batch["images"]                     # [B,1,224,224]
        fix = batch["fixations"]
        xy = fix["xy_norm"]                           # [B,T,2]
        times = fix["time"]                            # [B,T]
        seg_hits = fix["seg_hits"]                     # [B,T,S]
        box_hits = fix["box_hits"]                     # [B,T,B]
        lengths = fix["lengths"]                       # [B]

        B, T, _ = xy.shape
        device = images.device

        # g_t
        dt = self._delta_t(times)                      # [B,T,1]
        g_in = torch.cat([xy, dt], dim=-1)            # [B,T,3]
        g = self.gaze(g_in.view(B*T, -1)).view(B, T, -1)  # [B,T,d_g]

        # r_t
        roi_cat = torch.cat([seg_hits, box_hits], dim=-1) # [B,T,S+B]
        r = self.roi(roi_cat.view(B*T, -1)).view(B, T, -1)  # [B,T,d_r]

        # img_t (shared per time)
        img_feat = self.img_enc(images)                # [B,1024]
        img_vec  = self.img_proj(img_feat)             # [B,d_img]
        img_t = img_vec.unsqueeze(1).expand(B, T, -1)  # [B,T,d_img]

        # s_t and RNN
        s_t = torch.cat([g, r, img_t], dim=-1)         # [B,T,d_in]
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.enc_rnn(packed)
        h_seq, _ = pad_packed_sequence(packed_out, batch_first=True)   # [B,T,d_h]
        last_h = h_n.squeeze(0)                        # [B,d_h]

        # classification
        if self.use_attn_pool:
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]  # [B,T]
            scores = self.attn(h_seq).squeeze(-1)                              # [B,T]
            scores = scores.masked_fill(~mask, float('-inf'))
            w = torch.softmax(scores, dim=1).unsqueeze(-1)                     # [B,T,1]
            pooled = (h_seq * w).sum(dim=1)                                    # [B,d_h]
        else:
            pooled = last_h
        cls_logits = self.cls(pooled)                                          # [B,C]

        # text (teacher forcing if transcripts_ids provided)
        txt_logits = None
        if transcripts_ids is not None:
            # encoder side = projected GRU states
            enc_states = self.to_t5(h_seq)                                     # [B,T,dm]
            enc_mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            enc_out = BaseModelOutput(last_hidden_state=enc_states)

            # Shifted labels handled by HF if we pass labels directly
            out = self.t5(
                encoder_outputs=enc_out,
                encoder_attention_mask=enc_mask,
                labels=transcripts_ids,               # [B,L]
            )
            # (optionally) return logits for metrics
            txt_logits = out.logits                   # [B,L,V]

        return cls_logits, txt_logits

    # ---- inference helper: generate transcript ----
    @torch.no_grad()
    def generate(self, batch: Dict[str, Any], max_new_tokens: int = 64, num_beams: int = 3) -> List[str]:
        self.eval()
        images = batch["images"]
        fix = batch["fixations"]
        xy = fix["xy_norm"]
        times = fix["time"]
        seg_hits = fix["seg_hits"]
        box_hits = fix["box_hits"]
        lengths = fix["lengths"]

        B, T, _ = xy.shape
        device = images.device

        dt = self._delta_t(times)
        g_in = torch.cat([xy, dt], dim=-1)
        g = self.gaze(g_in.view(B*T, -1)).view(B, T, -1)
        roi_cat = torch.cat([seg_hits, box_hits], dim=-1)
        r = self.roi(roi_cat.view(B*T, -1)).view(B, T, -1)
        img_vec = self.img_proj(self.img_enc(images)).unsqueeze(1).expand(B, T, -1)
        s_t = torch.cat([g, r, img_vec], dim=-1)

        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.enc_rnn(packed)
        h_seq, _ = pad_packed_sequence(packed_out, batch_first=True)

        enc_states = self.to_t5(h_seq)
        enc_mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
        enc_out = BaseModelOutput(last_hidden_state=enc_states)

        gen_ids = self.t5.generate(
            encoder_outputs=enc_out,
            encoder_attention_mask=enc_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)


