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

# ---- Optional text head (Transformers / T5) ----------------------------------
try:
    from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ============================= Utilities =====================================

class Vocab:
    """Tiny word-level vocab for your dataset transcripts (if you need it)."""
    def __init__(self, word2idx: Dict[str, int]):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        # Conventional IDs
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = word2idx[self.pad_token]
        self.bos_token_id = word2idx[self.bos_token]
        self.eos_token_id = word2idx[self.eos_token]
        self.unk_token_id = word2idx[self.unk_token]

def build_vocabulary(dataset, min_freq: int = 2) -> Vocab:
    """Very lightweight vocabulary over dataset transcripts (space-tokenized)."""
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
    """
    Placeholder kept for backwards-compat imports in your trainers.
    Returns input_ids tensor of shape [1, L].
    """
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
    Temporal Multimodal RNN (EGD-CXR).

    Switches:
      - image_backbone: {"resnet18","resnet50","densenet121","txrv_densenet121"}
      - use_gaze: bool
      - use_roi: bool
      - enable_text_head: bool
      - use_teacher_forcing: bool
      - use_attn_pool: bool (temporal attention over GRU outputs before cls)
    """
    def __init__(
        self,
        num_seg: int,
        num_box: int,
        d_g: int = 64,         # gaze embedding dim
        d_r: int = 64,         # ROI (seg+box) embedding dim
        d_img: int = 128,      # image projection dim
        d_h: int = 256,        # GRU hidden size
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

        # -------------------- T5 text head (optional) --------------------
        if self.enable_text_head:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers not installed but enable_text_head=True.")
            self.tok = T5Tokenizer.from_pretrained(t5_name)
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_name)
            self.t5_decoder = T5ForConditionalGeneration.from_pretrained(t5_name)
            if freeze_t5:
                for p in self.t5_encoder.parameters():
                    p.requires_grad = False
                for p in self.t5_decoder.parameters():
                    p.requires_grad = False
            # Project GRU hidden to T5 hidden size (base=768)
            t5_hidden = int(self.t5_encoder.config.d_model)
            self.h_to_t5 = nn.Linear(d_h, t5_hidden)
        else:
            self.tok = None
            self.t5_encoder = None
            self.t5_decoder = None
            self.h_to_t5 = None

    # Convenience (trainer uses model.device)
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
            # Returns a DenseNet with 1024-dim feature (classifier removed)
            net = txrv.models.DenseNet(weights="densenet121-res224-chex")
            net.classifier = nn.Identity()
            return net, 1024
        raise ValueError(f"Unsupported image_backbone: {name}")

    # -------------------- Forward --------------------
    def forward(
        self,
        batch: Dict[str, Any],
        transcripts_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        batch:
          images: [B,1,H,W]
          fixations: dict with keys:
            - xy_norm: [B,T,2]
            - time:    [B,T]
            - seg_hits:[B,T,S]
            - box_hits:[B,T,Bb]
            - lengths: [B]
        returns:
          cls_logits: [B, C]
          txt_logits: Optional[Tensor] = [B, L-1, V] if teacher-forcing else None
        """
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
        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            x3 = images.repeat(1, 3, 1, 1)
            x3 = (x3 - self.img_mean) / self.img_std
            img_vec = self.img_encoder(x3)             # [B, D]
        else:
            # txrv models consume [B,1,H,W] in [0,1]
            img_vec = self.img_encoder(images)         # [B, 1024]
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

        # Optional text head with teacher forcing
        txt_logits: Optional[torch.Tensor] = None
        if self.enable_text_head and transcripts_ids is not None and self.use_teacher_forcing:
            t_steps = rnn_out.size(1)
            enc_attn_mask = (torch.arange(t_steps, device=device)[None, :] < lengths[:, None]).long()  # [B,T]
            enc_hidden = self.h_to_t5(rnn_out)                           # [B,T,t5_hidden]
            # Call encoder wrapper (returns BaseModelOutput)
            enc_out = self.t5_encoder(
                inputs_embeds=enc_hidden,
                attention_mask=enc_attn_mask,
                return_dict=True,
            )
            # Shift-right targets: feed decoder_input_ids = transcripts_ids[:, :-1]
            dec_in = transcripts_ids[:, :-1]
            dec_out = self.t5_decoder(
                encoder_outputs=enc_out,                                  # <--- Correct API
                decoder_input_ids=dec_in,
                return_dict=True,
            )
            txt_logits = dec_out.logits                                   # [B,L-1,V]

        return cls_logits, txt_logits

    # -------------------- Greedy generation --------------------
    @torch.no_grad()
    def generate_transcript(self, batch: Dict[str, Any], max_len: int = 64) -> List[str]:
        """
        Greedy decode transcripts. Returns list[str] of length B.
        """
        if not self.enable_text_head or self.tok is None:
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

        if self.image_backbone in {"resnet18", "resnet50", "densenet121"}:
            x3 = images.repeat(1, 3, 1, 1)
            x3 = (x3 - self.img_mean) / self.img_std
            iv = self.img_encoder(x3)
        else:
            iv = self.img_encoder(images)
        it = self.img_proj(iv).unsqueeze(1).expand(B, T, -1)
        feats.append(it)

        s_t = torch.cat(feats, dim=-1)
        packed = pack_padded_sequence(s_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B,T,d_h]

        # Encoder to T5 space
        enc_hidden = self.h_to_t5(rnn_out)
        enc_attn_mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).long()
        enc_out = self.t5_encoder(
            inputs_embeds=enc_hidden,
            attention_mask=enc_attn_mask,
            return_dict=True,
        )

        # Start token for T5 (decoder_start_token_id, typically = pad_token_id)
        start_id = self.t5_decoder.config.decoder_start_token_id
        if start_id is None:
            start_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id

        cur = torch.full((B, 1), start_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs: List[List[str]] = [[] for _ in range(B)]

        for _ in range(max_len):
            out = self.t5_decoder(
                encoder_outputs=enc_out,          # <--- Correct API
                decoder_input_ids=cur,
                use_cache=True,
                return_dict=True,
            )
            nxt = out.logits[:, -1].argmax(-1)   # [B]
            for i in range(B):
                if finished[i]:
                    continue
                if eos_id is not None and nxt[i].item() == eos_id:
                    finished[i] = True
                else:
                    token = self.tok.decode(nxt[i].view(1), skip_special_tokens=True)
                    if token:
                        outputs[i].append(token)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return [" ".join(toks) for toks in outputs]
