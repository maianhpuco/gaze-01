#!/usr/bin/env python3
"""
Sequence-to-sequence model that maps gaze fixations to transcripts and clinical labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from egd_cxr_dataset.utils.mapping import (
    assign_boxes_for_fixations,
    assign_segments_for_fixations,
)


class FixationFeaturizer(nn.Module):
    """Embed per-fixation signals (position, duration, anatomical context, dynamics)."""

    def __init__(
        self,
        num_segments: int,
        num_box_classes: int,
        *,
        seg_emb_dim: int = 16,
        box_emb_dim: int = 32,
        out_dim: int = 256,
    ):
        super().__init__()
        self.seg_emb = nn.Embedding(max(1, num_segments), seg_emb_dim)
        self.box_emb = nn.Embedding(max(1, num_box_classes), box_emb_dim)
        temporal_dim = 8
        self.temporal_proj = nn.Linear(3, temporal_dim)

        in_dim = 2 + 1 + seg_emb_dim + box_emb_dim + temporal_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(
        self,
        xy: torch.Tensor,
        dwell_ms: torch.Tensor,
        seg_id: torch.Tensor,
        box_cls: torch.Tensor,
    ) -> torch.Tensor:
        # Normalise spatial positions per-sample.
        eps = 1e-6
        x = xy[..., 0]
        y = xy[..., 1]
        x = x / (x.amax(dim=1, keepdim=True).clamp(min=1.0))
        y = y / (y.amax(dim=1, keepdim=True).clamp(min=1.0))

        dwell = torch.log1p(dwell_ms + eps).unsqueeze(-1)

        # Temporal derivatives.
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        dy = torch.diff(y, dim=1, prepend=y[:, :1])
        amp = torch.sqrt(dx * dx + dy * dy + eps)
        ang = torch.atan2(dy + eps, dx + eps)
        temporal = torch.stack([amp, torch.cos(ang), torch.sin(ang)], dim=-1)
        temporal = self.temporal_proj(temporal)

        features = torch.cat(
            [
                torch.stack([x, y], dim=-1),
                dwell,
                self.seg_emb(seg_id),
                self.box_emb(box_cls),
                temporal,
            ],
            dim=-1,
        )
        return self.mlp(features)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_states: torch.Tensor,
        dec_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = self.W_h(enc_states) + self.W_s(dec_state).unsqueeze(1)
        scores = self.v(torch.tanh(proj)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        return context, attn


class SilenceThoughtModel(nn.Module):
    """
    Encode gaze fixations with a BiLSTM and decode both transcript tokens and label logits.
    """

    def __init__(
        self,
        num_segments: int,
        num_box_classes: int,
        num_labels: int,
        vocab_size: int,
        *,
        enc_dim: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        txt_dim: int = 256,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        max_decode_len: int = 64,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_decode_len = max_decode_len

        self.featurizer = FixationFeaturizer(
            num_segments=num_segments,
            num_box_classes=num_box_classes,
            out_dim=enc_dim,
        )

        self.encoder = nn.LSTM(
            input_size=enc_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.enc_out_dim = rnn_hidden * 2

        self.sal_proj = nn.Linear(self.enc_out_dim, 1)
        self.label_head = nn.Linear(self.enc_out_dim, num_labels)

        self.emb = nn.Embedding(vocab_size, txt_dim)
        self.decoder = nn.GRU(
            input_size=txt_dim + self.enc_out_dim,
            hidden_size=txt_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attn = BahdanauAttention(enc_dim=self.enc_out_dim, dec_dim=txt_dim)
        self.init_dec = nn.Linear(self.enc_out_dim, txt_dim)
        self.out_proj = nn.Linear(txt_dim, vocab_size)

    # --------------------------------------------------------------------- helpers

    def encode(
        self,
        xy: torch.Tensor,
        dwell: torch.Tensor,
        lengths: torch.Tensor,
        seg_masks: torch.Tensor,
        boxes: List[List],
        *,
        default_box_cls: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seg_ids = assign_segments_for_fixations(xy, seg_masks)
        box_ids = assign_boxes_for_fixations(xy, boxes, default_cls=default_box_cls)
        feats = self.featurizer(xy, dwell, seg_ids, box_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_packed, _ = self.encoder(packed)
        enc_states, _ = nn.utils.rnn.pad_packed_sequence(
            enc_packed, batch_first=True, total_length=xy.size(1)
        )

        mask = torch.arange(xy.size(1), device=xy.device).unsqueeze(0) < lengths.unsqueeze(1)
        return enc_states, mask

    def salience_pool(self, enc_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.sal_proj(enc_states).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        return context, attn

    # ------------------------------------------------------------------ main passes

    def forward(
        self,
        xy: torch.Tensor,
        dwell: torch.Tensor,
        lengths: torch.Tensor,
        seg_masks: torch.Tensor,
        boxes: List[List],
        *,
        default_box_cls: int,
        txt_inp: Optional[torch.Tensor] = None,
        txt_tgt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_states, mask = self.encode(
            xy, dwell, lengths, seg_masks, boxes, default_box_cls=default_box_cls
        )
        pooled, salience = self.salience_pool(enc_states, mask)
        label_logits = self.label_head(pooled)

        outputs: Dict[str, torch.Tensor] = {
            "label_logits": label_logits,
            "salience": salience,
        }

        if txt_inp is not None and txt_tgt is not None:
            logits = self._decode_teacher_forced(enc_states, mask, pooled, txt_inp)
            outputs["txt_logits"] = logits

        return outputs

    def _decode_teacher_forced(
        self,
        enc_states: torch.Tensor,
        mask: torch.Tensor,
        pooled: torch.Tensor,
        txt_inp: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.tanh(self.init_dec(pooled))
        embeddings = self.emb(txt_inp)
        logits: List[torch.Tensor] = []

        for step_emb in embeddings.unbind(dim=1):
            context, _ = self.attn(enc_states, state, mask=mask)
            decoder_in = torch.cat([step_emb, context], dim=-1).unsqueeze(1)
            out, new_state = self.decoder(decoder_in, state.unsqueeze(0))
            state = new_state.squeeze(0)
            logits.append(self.out_proj(out.squeeze(1)))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(
        self,
        xy: torch.Tensor,
        dwell: torch.Tensor,
        lengths: torch.Tensor,
        seg_masks: torch.Tensor,
        boxes: List[List],
        *,
        default_box_cls: int,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_states, mask = self.encode(
            xy, dwell, lengths, seg_masks, boxes, default_box_cls=default_box_cls
        )
        pooled, _ = self.salience_pool(enc_states, mask)
        label_logits = self.label_head(pooled)

        state = torch.tanh(self.init_dec(pooled))
        prev = torch.full(
            (xy.size(0), 1),
            fill_value=self.bos_id,
            dtype=torch.long,
            device=xy.device,
        )

        tokens: List[torch.Tensor] = []
        max_steps = max_len or self.max_decode_len
        for _ in range(max_steps):
            emb = self.emb(prev[:, -1])
            context, _ = self.attn(enc_states, state, mask=mask)
            decoder_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, next_state = self.decoder(decoder_in, state.unsqueeze(0))
            state = next_state.squeeze(0)
            logits = self.out_proj(out.squeeze(1))
            next_token = torch.argmax(logits, dim=-1)
            tokens.append(next_token)
            prev = torch.cat([prev, next_token.unsqueeze(1)], dim=1)
            if (next_token == self.eos_id).all():
                break

        if tokens:
            transcript = torch.stack(tokens, dim=1)
        else:
            transcript = torch.empty(xy.size(0), 0, dtype=torch.long, device=xy.device)
        return label_logits, transcript


# ---------------------------------------------------------------------- losses

def compute_losses(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    txt_tgt: Optional[torch.Tensor],
    *,
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    bce = nn.BCEWithLogitsLoss()
    losses["labels_bce"] = bce(outputs["label_logits"], labels.float())
    if txt_tgt is not None and "txt_logits" in outputs:
        B, L, V = outputs["txt_logits"].shape
        ce = F.cross_entropy(
            outputs["txt_logits"].reshape(B * L, V),
            txt_tgt.reshape(B * L),
            ignore_index=pad_id,
        )
        losses["transcript_ce"] = ce
    losses["total"] = sum(losses.values())
    return losses

