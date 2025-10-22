#!/usr/bin/env python3
"""Build multiscale gaze intention vectors and decode transcripts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


def segment_windows_from_transcript(time_vec: torch.Tensor, transcript: Dict[str, Any]) -> List[Tuple[float, float]]:
    segments = (transcript or {}).get("segments", [])
    windows: List[Tuple[float, float]] = []
    prev_end = 0.0
    for seg in segments:
        begin = float(seg.get("begin", prev_end))
        windows.append((prev_end, begin))
        prev_end = float(seg.get("end", begin))
    return windows


def slice_fixation_mask(time_vec: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    return (time_vec >= t0) & (time_vec < t1)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        try:
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet50(weights=None)
        for p in backbone.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self.encoder(image).flatten(1)
        return self.proj(feats)


class IntentionFromGaze(nn.Module):
    def __init__(
        self,
        num_box_classes: int,
        num_segments: int,
        *,
        img_out_dim: int = 256,
        intent_dim: Optional[int] = None,
    ):
        super().__init__()
        self.C = num_box_classes
        self.S = num_segments
        self.img_enc = ImageEncoder(out_dim=img_out_dim)
        fused_dim = self.C + self.S + img_out_dim
        self.intent_dim = intent_dim
        if intent_dim is not None:
            self.mlp = nn.Sequential(
                nn.Linear(fused_dim, 2 * intent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2 * intent_dim, intent_dim),
            )
        else:
            self.mlp = None
        self.fused_dim = intent_dim if intent_dim is not None else fused_dim

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.img_enc(image)

    def _aggregate(self, hits: torch.Tensor, dwell: torch.Tensor, size: int) -> torch.Tensor:
        z = torch.zeros(size, device=dwell.device)
        if hits.numel() == 0 or size == 0:
            return z
        weights = hits.float()
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        dwell = dwell.unsqueeze(-1)
        z = (weights * dwell).sum(dim=0)
        s = z.sum()
        if s > 0:
            z = z / s
        return z

    def forward_window(
        self,
        dwell: torch.Tensor,
        seg_hits: torch.Tensor,
        box_hits: torch.Tensor,
        image_feat: torch.Tensor,
    ) -> torch.Tensor:
        z_box = self._aggregate(box_hits, dwell, self.C)
        z_seg = self._aggregate(seg_hits, dwell, self.S)
        fused = torch.cat([z_box, z_seg, image_feat], dim=0)
        if self.mlp is None:
            return fused
        return self.mlp(fused.unsqueeze(0)).squeeze(0)


class SegmentDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        intent_in_dim: int,
        *,
        dec_dim: int = 256,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.emb = nn.Embedding(vocab_size, dec_dim)
        self.h0 = nn.Linear(intent_in_dim, dec_dim)
        self.rnn = nn.GRU(dec_dim, dec_dim, batch_first=True)
        self.out = nn.Linear(dec_dim, vocab_size)

    def decode_teacher_forced(self, intent: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)
        bos = torch.tensor([self.bos_id], dtype=torch.long, device=tgt_ids.device)
        if tgt_ids.numel() == 0:
            return torch.zeros(0, self.out.out_features, device=tgt_ids.device)
        inp = torch.cat([bos, tgt_ids[:-1]], dim=0).unsqueeze(0)
        x = self.emb(inp)
        h0 = torch.tanh(self.h0(intent)).unsqueeze(0)
        o, _ = self.rnn(x, h0)
        return self.out(o.squeeze(0))

    @torch.no_grad()
    def generate(self, intent: torch.Tensor, max_len: int = 48) -> torch.Tensor:
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)
        h = torch.tanh(self.h0(intent)).unsqueeze(0)
        cur = torch.tensor([self.bos_id], dtype=torch.long, device=intent.device)
        tokens: List[torch.Tensor] = []
        for _ in range(max_len):
            x = self.emb(cur.view(1, 1))
            out, h = self.rnn(x, h)
            logits = self.out(out.squeeze(0).squeeze(0))
            nxt = torch.argmax(logits, dim=-1)
            tokens.append(nxt)
            cur = nxt
            if int(nxt.item()) == self.eos_id:
                break
        return torch.stack(tokens) if tokens else torch.empty(0, dtype=torch.long, device=intent.device)


class GazeIntent2TranscriptAndLabels(nn.Module):
    def __init__(
        self,
        num_box_classes: int,
        num_segments: int,
        *,
        img_out_dim: int = 256,
        intent_dim: Optional[int] = 256,
        vocab_size: int = 10000,
        dec_dim: int = 256,
        num_labels: int = 28,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.intent_builder = IntentionFromGaze(
            num_box_classes=num_box_classes,
            num_segments=num_segments,
            img_out_dim=img_out_dim,
            intent_dim=intent_dim,
        )
        intent_in = self.intent_builder.fused_dim
        self.decoder = SegmentDecoder(
            vocab_size,
            intent_in_dim=intent_in,
            dec_dim=dec_dim,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        self.label_head = nn.Linear(intent_in, num_labels)

    def _image_feature(self, image_1chw: torch.Tensor, device: torch.device) -> torch.Tensor:
        if image_1chw.dim() == 3:
            image_1chw = image_1chw.unsqueeze(0)
        image_1chw = image_1chw.to(device)
        return self.intent_builder.encode_image(image_1chw).squeeze(0)

    def forward_case(
        self,
        *,
        fixations: Dict[str, torch.Tensor],
        transcript: Dict[str, Any],
        encode_text_fn,
        image_1chw: torch.Tensor,
    ) -> Dict[str, Any]:
        xy = fixations["xy"]
        dwell = fixations["dwell"]
        time_s = fixations["time"]
        seg_hits = fixations["seg_hits"]
        box_hits = fixations["box_hits"]
        device = xy.device

        image_feat = self._image_feature(image_1chw, device)

        windows = segment_windows_from_transcript(time_s, transcript)
        intents: List[torch.Tensor] = []
        for t0, t1 in windows:
            mask = slice_fixation_mask(time_s, t0, t1)
            if mask.sum() == 0:
                # Create empty intent with same structure as forward_window
                z_box = torch.zeros(self.intent_builder.C, device=device)
                z_seg = torch.zeros(self.intent_builder.S, device=device)
                fused = torch.cat([z_box, z_seg, image_feat], dim=0)
                if self.intent_builder.mlp is None:
                    empty_intent = fused
                else:
                    empty_intent = self.intent_builder.mlp(fused.unsqueeze(0)).squeeze(0)
                intents.append(empty_intent)
                continue
            intent = self.intent_builder.forward_window(
                dwell[mask],
                seg_hits[mask],
                box_hits[mask],
                image_feat,
            )
            intents.append(intent)

        if intents:
            stacked = torch.stack(intents, dim=0)
            label_logits = self.label_head(stacked.mean(dim=0))
        else:
            label_logits = self.label_head(torch.zeros(self.label_head.in_features, device=device))

        segments = (transcript or {}).get("segments", [])
        txt_logits: List[torch.Tensor] = []
        for intent, seg in zip(intents, segments):
            tgt = encode_text_fn(seg.get("text", ""))
            if tgt.numel() == 0:
                txt_logits.append(torch.zeros(0, self.decoder.out.out_features, device=device))
                continue
            txt_logits.append(self.decoder.decode_teacher_forced(intent, tgt.to(device)))

        return {
            "intents": intents,
            "label_logits": label_logits,
            "txt_logits_per_segment": txt_logits,
        }

    @torch.no_grad()
    def generate_case(
        self,
        *,
        fixations: Dict[str, torch.Tensor],
        transcript: Dict[str, Any],
        encode_text_fn,
        image_1chw: torch.Tensor,
        max_len: int = 48,
    ) -> Dict[str, Any]:
        out = self.forward_case(
            fixations=fixations,
            transcript=transcript,
            encode_text_fn=encode_text_fn,
            image_1chw=image_1chw,
        )
        generated = [self.decoder.generate(intent, max_len=max_len) for intent in out["intents"]]
        out["gen_tokens_per_segment"] = generated
        return out

