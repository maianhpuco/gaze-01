# src/models/tmrnn.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models as tvm

# ---- Optional backbones (TorchXRayVision) ------------------------------------
try:
    import torchxrayvision as txrv
    HAS_TXRV = True
except Exception:
    HAS_TXRV = False

# ---- T5 (Transformers) -------------------------------------------------------
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ============================= Utilities =====================================

class Vocab:
    """Tiny word-level vocab for your dataset transcripts (if you need it)."""
    def __init__(self, word2idx: Dict[str, int]):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = word2idx[self.pad_token]
        self.bos_token_id = word2idx[self.bos_token]
        self.eos_token_id = word2idx[self.eos_token]
        self.unk_token_id = word2idx[self.unk_token]

def build_vocabulary(dataset, min_freq: int = 2) -> Vocab:
    from collections import Counter
    c = Counter()
    for ex in dataset:
        txt = ""
        try:
            txt = ex["transcripts"]["text"]
        except Exception:
            pass
        for w in str(txt).strip().split():
            c[w.lower()] += 1
    base = ["<pad>", "<bos>", "<eos>", "<unk>"]
    word2idx: Dict[str, int] = {w: i for i, w in enumerate(base)}
    for w, f in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        if f >= min_freq and w not in word2idx:
            word2idx[w] = len(word2idx)
    return Vocab(word2idx)

def encode_with_tau(tokenizer, text: str, max_len: int, tau: float | None = None) -> torch.Tensor:
    enc = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc.input_ids


# =============================== Model =======================================

class TMRNN(nn.Module):
    """
    Temporal Multimodal RNN (EGD-CXR), gold-aligned.

    • Bypass T5 encoder: project GRU states → pass as `encoder_outputs` to T5ForConditionalGeneration.
    • TorchXRayVision DenseNet-121 uses weights="densenet121-res224-all" (gold).
    • Keeps all 4 image backbones and optional attention pooling.
    """
    def __init__(
        self,
        num_seg: int,
        num_box: int,
        d_g: int = 64,
        d_r: int = 64,
        d_img: int = 128,
        d_h: int = 256,
        num_classes: int = 3,
        image_backbone: str = "resnet50",
        use_gaze: bool = True,
        use_roi: bool = True,
        enable_text_head: bool = True,
        use_teacher_forcing: bool = True,
        t5_name: str = "google-t5/t5-base",
        freeze_t5: bool = True,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        self.num_seg = num_seg
        self.num_box = num_box
        self.d_g, self.d_r, self.d_img, self.d_h = d_g, d_r, d_img, d_h
        self.num_classes = num_classes

        # switches
        self.image_backbone = image_backbone.lower()
        self.use_gaze = use_gaze
        self.use_roi = use_roi
        self.enable_text_head = enable_text_head
        self.use_teacher_forcing = use_teacher_forcing
        self.use_attn_pool = use_attn_pool

        # -------------------- Image encoder --------------------
        self.img_encoder, out_dim = self._build_backbone(self.image_backbone)
        self.img_proj = nn.Linear(out_dim, d_img)

        # register mean/std for TV models (grayscale -> RGB)
        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            self.register_buffer(
                "img_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "img_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            # txrv models take single-channel [0,1] directly
            self.register_buffer("img_mean", torch.tensor([0.0]), persistent=False)
            self.register_buffer("img_std",  torch.tensor([1.0]), persistent=False)

        # -------------------- Gaze / ROI streams --------------------
        if self.use_gaze:
            self.gaze_mlp = nn.Sequential(
                nn.Linear(3, d_g * 2), nn.ReLU(),
                nn.Linear(d_g * 2, d_g), nn.ReLU(),
            )
        if self.use_roi:
            self.roi_proj = nn.Linear(self.num_seg + self.num_box, d_r)

        # -------------------- RNN encoder --------------------
        d_in = d_img + (d_g if self.use_gaze else 0) + (d_r if self.use_roi else 0)
        self.encoder = nn.GRU(d_in, d_h, batch_first=True)

        # -------------------- Classification head --------------------
        self.cls_head = nn.Linear(d_h, num_classes)
        if self.use_attn_pool:
            self.attn_pool = nn.Linear(d_h, 1)

        # -------------------- T5 text head (encoder-bypassed) ---------------
        if self.enable_text_head:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers not installed but enable_text_head=True.")
            self.tok = T5Tokenizer.from_pretrained(t5_name)
            self.t5  = T5ForConditionalGeneration.from_pretrained(t5_name)
            if freeze_t5:
                for p in self.t5.parameters():
                    p.requires_grad = False
            t5_hidden = int(self.t5.config.d_model)
            self.h_to_t5 = nn.Linear(d_h, t5_hidden)
        else:
            self.tok = None
            self.t5  = None
            self.h_to_t5 = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -------------------- Backbones --------------------
    def _build_backbone(self, name: str) -> Tuple[nn.Module, int]:
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
                raise RuntimeError("torchxrayvision not installed.")
            # Use gold-aligned weights and call .features(...) (not .forward) to avoid op_norm
            net = txrv.models.DenseNet(weights="densenet121-res224-all")
            net.classifier = nn.Identity()
            return net, 1024
        raise ValueError(f"Unsupported image_backbone: {name}")

    # ---- image encode helper (TV vs TXRV) -----------------------------------
    def _encode_image_vec(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns [B, D] vector per image.
        - TV backbones expect 3ch normalized input.
        - TXRV backbone must use .features(...) + GAP (avoid .forward() op_norm).
        """
        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            x3 = images.repeat(1, 3, 1, 1)
            x3 = (x3 - self.img_mean) / self.img_std
            return self.img_encoder(x3)  # [B, D]
        else:
            # Expect [0,1], single channel
            images = images.clamp(0, 1)
            feats = self.img_encoder.features(images)  # [B, C, H, W]
            if feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            return feats  # [B, 1024]

    # -------------------- Forward --------------------
    def forward(
        self,
        batch: Dict[str, Any],
        transcripts_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = self.device

        # Unpack
        images: torch.Tensor = batch["images"]
        fx = batch["fixations"]
        xy = fx["xy_norm"]          # [B,T,2]
        times = fx["time"]          # [B,T]
        seg_hits = fx["seg_hits"]   # [B,T,S]
        box_hits = fx["box_hits"]   # [B,T,Bb]
        lengths = fx["lengths"]     # [B]
        B, T, _ = xy.shape

        # Δt per fixation
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        dt = torch.clamp(dt, min=0.0).unsqueeze(-1)    # [B,T,1]

        feats: List[torch.Tensor] = []

        # Gaze stream
        if self.use_gaze:
            g_in = torch.cat([xy, dt], dim=-1)         # [B,T,3]
            g_t = self.gaze_mlp(g_in.reshape(B*T, -1)).reshape(B, T, -1)
            feats.append(g_t)

        # ROI stream
        if self.use_roi:
            roi = torch.cat([seg_hits, box_hits], dim=-1)  # [B,T,S+Bb]
            r_t = self.roi_proj(roi.reshape(B*T, -1)).reshape(B, T, -1)
            feats.append(r_t)

        # Image stream (shared across time)
        # Assert basic shape & scale for TXRV
        assert images.dim() == 4 and images.size(1) == 1, f"images shape {images.shape} (expected [B,1,H,W])"
        img_vec = self._encode_image_vec(images)       # [B, D]
        it = self.img_proj(img_vec).unsqueeze(1).expand(B, T, -1)
        feats.append(it)

        # Concatenate modalities
        s_t = torch.cat(feats, dim=-1)                 # [B,T,d_in]

        # Sequence encoder
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.encoder(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)   # [B,T,d_h]
        last_h = h_n[-1]                                                 # [B,d_h]

        # Temporal pooling for classification
        if self.use_attn_pool:
            mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
            score = self.attn_pool(rnn_out).squeeze(-1)                  # [B,T]
            score = score.masked_fill(~mask, float("-inf"))
            w = F.softmax(score, dim=1).unsqueeze(-1)
            pooled = (rnn_out * w).sum(1)                                # [B,d_h]
        else:
            pooled = last_h                                              # [B,d_h]

        cls_logits = self.cls_head(pooled)                               # [B,C]

        # ------- Text head (gold-aligned: bypass T5 encoder & use labels=) -------
        txt_logits: Optional[torch.Tensor] = None
        if self.enable_text_head and transcripts_ids is not None and self.use_teacher_forcing:
            enc_hidden = self.h_to_t5(rnn_out)                           # [B,T,t5_hidden]
            enc_out = BaseModelOutput(last_hidden_state=enc_hidden)

            # Let T5 handle shifting right internally (like gold)
            out = self.t5(
                encoder_outputs=enc_out,
                labels=transcripts_ids,      # <- gold path; no manual decoder_input_ids
                return_dict=True,
            )
            txt_logits = out.logits          # aligned with `labels` indexing (no [:,1:] in trainer)

        return cls_logits, txt_logits

    # -------------------- Generation (gold-like) --------------------
    @torch.no_grad()
    def generate_transcript(self, batch: Dict[str, Any], max_len: int = 64, num_beams: int = 3) -> List[str]:
        """
        Use T5.generate over encoder_outputs (bypassing the T5 encoder), like gold.
        """
        if not self.enable_text_head or self.tok is None or self.t5 is None:
            return [""] * batch["images"].size(0)

        self.eval()
        device = self.device

        # Move tensors to device
        images = batch["images"].to(device)
        fx = batch["fixations"]
        for k, v in fx.items():
            if isinstance(v, torch.Tensor):
                fx[k] = v.to(device)

        xy = fx["xy_norm"]; times = fx["time"]
        seg_hits = fx["seg_hits"]; box_hits = fx["box_hits"]
        lengths = fx["lengths"]

        B, T, _ = xy.shape

        # Δt
        dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        dt = torch.clamp(dt, min=0.0).unsqueeze(-1)

        feats: List[torch.Tensor] = []
        if self.use_gaze:
            g_in = torch.cat([xy, dt], dim=-1)
            g_t = self.gaze_mlp(g_in.reshape(B*T, -1)).reshape(B, T, -1)
            feats.append(g_t)
        if self.use_roi:
            roi = torch.cat([seg_hits, box_hits], dim=-1)
            r_t = self.roi_proj(roi.reshape(B*T, -1)).reshape(B, T, -1)
            feats.append(r_t)

        images = images.clamp(0, 1)  # ensure TXRV path scale
        iv = self._encode_image_vec(images)
        it = self.img_proj(iv).unsqueeze(1).expand(B, T, -1)
        feats.append(it)

        s_t = torch.cat(feats, dim=-1)
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B,T,d_h]

        # Project to T5 hidden and build encoder_outputs
        enc_hidden = self.h_to_t5(rnn_out)
        enc_out = BaseModelOutput(last_hidden_state=enc_hidden)

        # Generate (gold-like path)
        gen_ids = self.t5.generate(
            encoder_outputs=enc_out,
            max_new_tokens=max_len,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)
