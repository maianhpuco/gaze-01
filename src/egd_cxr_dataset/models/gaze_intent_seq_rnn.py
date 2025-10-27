#!/usr/bin/env python3
"""
Recurrent gaze-intent model with a learnable speak gate.

This module replaces the earlier pooling-based silence-thought implementation with
an RNN encoder that preserves fixation order, conditions on the evolving transcript
state, and decides when to emit text segments without access to ground-truth timing
at inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:  # torchvision is optional in some environments
    from torchvision import models as tvm
except ImportError:  # pragma: no cover
    tvm = None  # type: ignore[assignment]


def _safe_time_norm(t: torch.Tensor) -> torch.Tensor:
    """
    Normalise timestamps into [0, 1] per sequence.

    Returns zeros if the sequence is empty or constant.
    """
    if t.numel() == 0:
        return t
    t0 = t.min()
    rng = (t.max() - t0).clamp_min(1e-6)
    return (t - t0) / rng


class GazeSeqRNNAttend(nn.Module):
    """
    Sequence model that keeps gaze fixations in temporal order and uses a speak gate.

    Pipeline:
      • Build per-fixation features: segments, boxes, spatial co-ordinates, dwell,
        and delta time.
      • Run a GRU encoder with teacher-forced transcript context so that the hidden
        state captures both gaze and what has already been said.
      • Predict diagnostic labels from an attention-pooled sequence state.
      • Learn a “speak gate” that fires when it is time to emit a transcript segment.
      • Decode each segment with a GRU decoder initialised from the windowed encoder
        state. During training we use ground-truth text, while at inference the speak
        gate drives emission timings.
    """

    def __init__(
        self,
        *,
        num_box_classes: int,
        num_segments: int,
        num_labels: int,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        img_out_dim: int = 256,
        intent_dim: int = 256,
        dec_dim: int = 256,
        use_box: bool = True,
        use_seg: bool = True,
        use_image: bool = True,
        use_text: bool = True,
        use_gaze: bool = True,
        pretrained_image: bool = False,
        encoder_dropout: float = 0.0,
        label_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_box = use_box
        self.use_seg = use_seg
        self.use_image = use_image
        self.use_text = use_text
        self.use_gaze = use_gaze
        self.intent_dim = intent_dim
        self.img_out_dim = img_out_dim
        self.pretrained_image = bool(pretrained_image and use_image)

        # --- Per-fixation feature projections -------------------------------------------------
        pieces: List[int] = []

        if use_seg and num_segments > 0:
            seg_dim = max(intent_dim // 4, 8)
            self.seg_lin = nn.Linear(num_segments, seg_dim)
            pieces.append(seg_dim)
        else:
            self.seg_lin = None

        if use_box and num_box_classes > 0:
            box_dim = max(intent_dim // 4, 8)
            self.box_lin = nn.Linear(num_box_classes, box_dim)
            pieces.append(box_dim)
        else:
            self.box_lin = None

        # kinematics: xy (normalised pixels), dwell (seconds), delta_t (seconds)
        if use_gaze:
            kine_dim = max(intent_dim // 4, 8)
            self.kine_lin = nn.Linear(2 + 1 + 1, kine_dim)
            pieces.append(kine_dim)
        else:
            self.kine_lin = None

        # optional image backbone for a global feature vector
        if use_image:
            if self.pretrained_image:
                if tvm is None:
                    raise ImportError("torchvision is required for pretrained image encoder support.")
                backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
                with torch.no_grad():
                    init_w = backbone.conv1.weight.data
                    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    new_conv.weight.copy_(init_w.sum(dim=1, keepdim=True))
                    backbone.conv1 = new_conv
                modules = list(backbone.children())[:-1]  # remove final classification head
                self.image_encoder = nn.Sequential(*modules)
                self.img_lin = nn.Linear(512, intent_dim)
            else:
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(1, img_out_dim, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(img_out_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.img_lin = nn.Linear(img_out_dim, intent_dim)
        else:
            self.image_encoder = None
            self.img_lin = None

        # transcript context that is fed back into the encoder
        self.txt_ctx_dim = intent_dim // 2
        self.txt_ctx_lin = nn.Linear(self.txt_ctx_dim, self.txt_ctx_dim)
        pieces.append(self.txt_ctx_dim)

        enc_in_dim = sum(pieces)

        # --- Sequence encoder -----------------------------------------------------------------
        self.enc_gru = nn.GRU(
            input_size=enc_in_dim,
            hidden_size=intent_dim,
            num_layers=1,
            batch_first=False,
        )

        # --- Attention pooling + label head ---------------------------------------------------
        self.att_query = nn.Parameter(torch.randn(intent_dim))
        label_in_dim = intent_dim + (intent_dim if use_image else 0)
        self.label_head = nn.Linear(label_in_dim, num_labels)

        # --- Speak gate -----------------------------------------------------------------------
        self.speak_head = nn.Linear(intent_dim, 1)

        # --- Text decoder ---------------------------------------------------------------------
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size

        self.tok_embed = nn.Embedding(vocab_size, dec_dim, padding_idx=pad_id)
        self.dec_init = nn.Linear(intent_dim, dec_dim)
        self.dec_gru = nn.GRU(input_size=dec_dim, hidden_size=dec_dim, num_layers=1, batch_first=True)
        self.dec_out = nn.Linear(dec_dim, vocab_size)
        self.seg_txt_summary = nn.Linear(dec_dim, self.txt_ctx_dim)

        self.encoder_dropout = nn.Dropout(p=encoder_dropout) if encoder_dropout > 0 else None
        self.label_dropout = nn.Dropout(p=label_dropout) if label_dropout > 0 else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.kine_lin is not None:
            nn.init.xavier_uniform_(self.kine_lin.weight)
            nn.init.zeros_(self.kine_lin.bias)
        if self.seg_lin is not None:
            nn.init.xavier_uniform_(self.seg_lin.weight)
            nn.init.zeros_(self.seg_lin.bias)
        if self.box_lin is not None:
            nn.init.xavier_uniform_(self.box_lin.weight)
            nn.init.zeros_(self.box_lin.bias)
        if self.img_lin is not None:
            nn.init.xavier_uniform_(self.img_lin.weight)
            nn.init.zeros_(self.img_lin.bias)
        nn.init.normal_(self.att_query, std=0.02)
        nn.init.xavier_uniform_(self.label_head.weight)
        nn.init.zeros_(self.label_head.bias)
        nn.init.xavier_uniform_(self.speak_head.weight)
        nn.init.zeros_(self.speak_head.bias)
        nn.init.xavier_uniform_(self.dec_init.weight)
        nn.init.zeros_(self.dec_init.bias)
        nn.init.xavier_uniform_(self.dec_out.weight)
        nn.init.zeros_(self.dec_out.bias)
        nn.init.xavier_uniform_(self.seg_txt_summary.weight)
        nn.init.zeros_(self.seg_txt_summary.bias)

    # ------------------------------------------------------------------------------------------
    # Feature helpers

    @staticmethod
    def _window_indices_from_transcript(time_s: torch.Tensor, transcript: Optional[dict]) -> List[Tuple[int, int]]:
        """
        Convert transcript segments (with begin/end times) into fixation index windows [i0, i1).
        """
        if transcript is None:
            return []
        segs = transcript.get("segments", [])
        if not segs or time_s.numel() == 0:
            return []

        t_list = time_s.detach().cpu().tolist()
        windows: List[Tuple[int, int]] = []
        for seg in segs:
            begin = float(seg.get("begin", 0.0))
            end = float(seg.get("end", begin))
            if not (end > begin):
                continue
            i0 = 0
            while i0 < len(t_list) and t_list[i0] < begin:
                i0 += 1
            i1 = i0
            while i1 < len(t_list) and t_list[i1] <= end:
                i1 += 1
            if i1 > i0:
                windows.append((i0, i1))
        return windows

    @staticmethod
    def _att_pool(h: torch.Tensor, dwell: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling over time with dwell as prior importance.
        """
        if h.size(0) == 0:
            return torch.zeros_like(q)
        logits = h @ q
        weights = torch.softmax(logits + torch.log(dwell.clamp_min(1.0)), dim=0)
        return (weights.unsqueeze(-1) * h).sum(dim=0)

    def _build_fixation_features(
        self,
        xy: torch.Tensor,
        dwell_ms: torch.Tensor,
        time_s: torch.Tensor,
        seg_hits: torch.Tensor,
        box_hits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct per-fixation features without transcript context.
        """
        device = xy.device
        T = xy.size(0)
        feature_dim = 0
        if self.kine_lin is not None:
            feature_dim += self.kine_lin.out_features
        if self.seg_lin is not None:
            feature_dim += self.seg_lin.out_features
        if self.box_lin is not None:
            feature_dim += self.box_lin.out_features
        if T == 0:
            return torch.zeros(0, feature_dim, device=device)

        pieces: List[torch.Tensor] = []
        if self.kine_lin is not None:
            xy_max = xy.detach().abs().max()
            xy_scale = max(1.0, float(xy_max.item()) if xy_max.numel() > 0 else 224.0)
            xy_norm = xy / xy_scale

            dt = torch.zeros_like(time_s)
            if T > 1:
                dt[1:] = time_s[1:] - time_s[:-1]
            dt = dt.clamp_min(0.0)

            kine = torch.cat(
                [
                    xy_norm,
                    (dwell_ms.unsqueeze(-1) / 1000.0),
                    dt.unsqueeze(-1),
                ],
                dim=-1,
            )
            pieces.append(self.kine_lin(kine))
        if self.seg_lin is not None and seg_hits.numel() > 0:
            pieces.append(self.seg_lin(seg_hits))
        if self.box_lin is not None and box_hits.numel() > 0:
            pieces.append(self.box_lin(box_hits))

        if not pieces:
            return torch.zeros(T, feature_dim, device=device)
        return torch.cat(pieces, dim=-1)

    def _segment_text_summary(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Average token embeddings from a segment and project to transcript-context space.
        """
        device = token_ids.device
        if token_ids.numel() == 0:
            return torch.zeros(self.txt_ctx_dim, device=device)
        emb = self.tok_embed(token_ids)  # [L, dec_dim]
        return self.seg_txt_summary(emb.mean(dim=0))

    def _encode_image(self, image_1chw: torch.Tensor) -> torch.Tensor:
        """
        Encode a 1xHxW image into intent_dim features.
        """
        if not self.use_image or self.image_encoder is None or self.img_lin is None:
            return torch.zeros(self.intent_dim, device=image_1chw.device)
        img = image_1chw.unsqueeze(0)  # [1, 1, H, W]
        if img.shape[-2:] != (224, 224):
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.image_encoder(img)
        if self.pretrained_image:
            feat = feat.view(feat.size(0), -1)  # [1, 512]
        else:
            feat = feat.flatten(1)  # [1, img_out_dim]
        return self.img_lin(feat).squeeze(0)

    # ------------------------------------------------------------------------------------------
    # Main forward passes

    def forward_case(
        self,
        *,
        fixations: Dict[str, torch.Tensor],
        transcript: Optional[dict],
        encode_text_fn,
        image_1chw: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Forward pass for a single case.
        """
        device = image_1chw.device
        xy = fixations["xy"].to(device)
        dwell = fixations["dwell"].to(device).clamp_min(1.0)
        time_s = fixations["time"].to(device)
        seg_hits = fixations["seg_hits"].to(device)
        box_hits = fixations["box_hits"].to(device)
        T = xy.size(0)

        if T == 0:
            pooled = torch.zeros(self.intent_dim, device=device)
            label_logits = self.label_head(
                torch.cat([pooled, torch.zeros(self.intent_dim, device=device)], dim=-1)
            ) if self.use_image else self.label_head(pooled)
            return {
                "label_logits": label_logits,
                "speak_logits": torch.empty(0, device=device),
                "h_seq": torch.zeros(0, self.intent_dim, device=device),
                "txt_logits_per_segment": [],
            }

        base_feats = self._build_fixation_features(xy, dwell, time_s, seg_hits, box_hits)

        windows = self._window_indices_from_transcript(time_s, transcript) if (self.use_text and transcript is not None) else []
        seg_txt_ctx = torch.zeros(T, self.txt_ctx_dim, device=device)
        if windows and self.use_text:
            seg_summaries: List[torch.Tensor] = []
            segments = transcript.get("segments", [])
            for seg in segments:
                ids = encode_text_fn(seg.get("text", ""))
                ids = ids.to(device) if isinstance(ids, torch.Tensor) else torch.tensor([], device=device, dtype=torch.long)
                seg_summaries.append(self._segment_text_summary(ids))

            for t in range(T):
                prev_idx = -1
                for k, (_, end_idx) in enumerate(windows):
                    if end_idx <= t:
                        prev_idx = k
                    else:
                        break
                if prev_idx >= 0:
                    seg_txt_ctx[t] = self.txt_ctx_lin(seg_summaries[prev_idx])

        X_enc = torch.cat([base_feats, seg_txt_ctx], dim=-1)  # [T, F]
        X_enc = X_enc.unsqueeze(1)  # [T, 1, F] for GRU (time-major)
        h0 = torch.zeros(1, 1, self.enc_gru.hidden_size, device=device)
        h_seq, _ = self.enc_gru(X_enc, h0)  # [T, 1, D]
        h_seq = h_seq.squeeze(1)  # [T, D]

        pooled = self._att_pool(h_seq, dwell, self.att_query)
        if self.encoder_dropout is not None:
            pooled = self.encoder_dropout(pooled)
        img_feat = self._encode_image(image_1chw)
        if self.use_image:
            fused = torch.cat([pooled, img_feat], dim=-1)
        else:
            fused = pooled
        if self.label_dropout is not None:
            fused = self.label_dropout(fused)
        label_logits = self.label_head(fused)
        speak_logits = self.speak_head(h_seq).squeeze(-1)

        txt_logits_per_segment: List[torch.Tensor] = []
        if self.use_text and windows:
            segments = transcript.get("segments", [])
            for (i0, i1), seg in zip(windows, segments):
                if i1 <= i0:
                    txt_logits_per_segment.append(torch.empty(0, self.vocab_size, device=device))
                    continue
                hw = h_seq[i0:i1].mean(dim=0)
                dec_h0 = torch.tanh(self.dec_init(hw)).unsqueeze(0).unsqueeze(0)
                tgt_ids = encode_text_fn(seg.get("text", ""))
                tgt_ids = tgt_ids.to(device) if isinstance(tgt_ids, torch.Tensor) else torch.tensor([], device=device, dtype=torch.long)
                if tgt_ids.numel() < 2:
                    txt_logits_per_segment.append(torch.empty(0, self.vocab_size, device=device))
                    continue
                inp = self.tok_embed(tgt_ids[:-1].unsqueeze(0))
                out, _ = self.dec_gru(inp, dec_h0)
                logits = self.dec_out(out.squeeze(0))
                txt_logits_per_segment.append(logits)

        return {
            "label_logits": label_logits,
            "speak_logits": speak_logits,
            "h_seq": h_seq,
            "txt_logits_per_segment": txt_logits_per_segment,
        }

    @torch.no_grad()
    def generate_case(
        self,
        *,
        fixations: Dict[str, torch.Tensor],
        transcript: Optional[dict],
        encode_text_fn,
        image_1chw: torch.Tensor,
        max_len: int = 64,
        min_len: int = 16,
        speak_start_thresh: float = 0.6,
        speak_stop_patience: int = 2,
        min_segment_fixations: int = 3,
    ) -> Dict[str, object]:
        """
        Autoregressive generation for a single case without transcript timestamps.
        """
        device = image_1chw.device
        xy = fixations["xy"].to(device)
        dwell = fixations["dwell"].to(device).clamp_min(1.0)
        time_s = fixations["time"].to(device)
        seg_hits = fixations["seg_hits"].to(device)
        box_hits = fixations["box_hits"].to(device)
        T = xy.size(0)

        base_feats = self._build_fixation_features(xy, dwell, time_s, seg_hits, box_hits)

        txt_ctx = torch.zeros(self.txt_ctx_dim, device=device)
        h_prev = torch.zeros(1, 1, self.enc_gru.hidden_size, device=device)
        h_seq_list: List[torch.Tensor] = []
        segments_tokens: List[torch.Tensor] = []

        speak_on = False
        start_idx = 0
        patience_ctr = 0

        for t in range(T):
            fusion = torch.cat([base_feats[t], self.txt_ctx_lin(txt_ctx)]).unsqueeze(0).unsqueeze(0)
            h_t, h_prev = self.enc_gru(fusion, h_prev)
            h_t = h_t.squeeze(0).squeeze(0)
            h_seq_list.append(h_t)

            gate = torch.sigmoid(self.speak_head(h_t)).item()

            if not speak_on and gate >= speak_start_thresh:
                speak_on = True
                start_idx = t
                patience_ctr = 0
            elif speak_on:
                patience_ctr = patience_ctr + 1 if gate < speak_start_thresh else 0
                if patience_ctr >= speak_stop_patience and (t - start_idx + 1) >= min_segment_fixations:
                    window = torch.stack(h_seq_list[start_idx : t + 1], dim=0).mean(dim=0)
                    tokens = self._decode_tokens(window, max_len=max_len, min_len=min_len)
                    segments_tokens.append(tokens)
                    txt_ctx = self._segment_text_summary(tokens)
                    speak_on = False

        if speak_on and (T - start_idx) >= min_segment_fixations:
            window = torch.stack(h_seq_list[start_idx:], dim=0).mean(dim=0)
            tokens = self._decode_tokens(window, max_len=max_len, min_len=min_len)
            segments_tokens.append(tokens)

        h_seq = torch.stack(h_seq_list, dim=0) if h_seq_list else torch.zeros(0, self.intent_dim, device=device)
        pooled = self._att_pool(h_seq, dwell, self.att_query) if h_seq_list else torch.zeros(self.intent_dim, device=device)
        if self.encoder_dropout is not None:
            pooled = self.encoder_dropout(pooled)
        img_feat = self._encode_image(image_1chw)
        fused = torch.cat([pooled, img_feat], dim=-1) if self.use_image else pooled
        if self.label_dropout is not None:
            fused = self.label_dropout(fused)
        label_logits = self.label_head(fused)

        return {
            "label_logits": label_logits,
            "gen_tokens_per_segment": segments_tokens,
        }

    def _decode_tokens(self, hw: torch.Tensor, *, max_len: int, min_len: int) -> torch.Tensor:
        """
        Greedy decode from window state.
        """
        device = hw.device
        h = torch.tanh(self.dec_init(hw)).unsqueeze(0).unsqueeze(0)
        cur = torch.tensor([[self.bos_id]], dtype=torch.long, device=device)
        out_tokens = [self.bos_id]

        for step in range(max_len):
            emb = self.tok_embed(cur[:, -1:])
            o, h = self.dec_gru(emb, h)
            logits = self.dec_out(o[:, -1])
            next_id = int(torch.argmax(logits, dim=-1).item())
            out_tokens.append(next_id)
            if step + 1 >= min_len and next_id == self.eos_id:
                break
            cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

        return torch.tensor(out_tokens, dtype=torch.long, device=device)


__all__ = ["GazeSeqRNNAttend"]
