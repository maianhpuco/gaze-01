#!/usr/bin/env python3
"""Build multiscale gaze intention vectors and decode transcripts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


def simple_equal_windows(time_vec: torch.Tensor, n: int = 3) -> List[Tuple[float, float]]:
    if time_vec.numel() == 0:
        return [(0.0, 1e9)]
    t0 = float(time_vec.min().item())
    t1 = float(time_vec.max().item())
    if not (t1 > t0):
        return [(0.0, 1e9)]
    step = (t1 - t0) / max(1, n)
    return [(t0 + i * step, t0 + (i + 1) * step) for i in range(n)]


def segment_windows_from_transcript(time_vec: torch.Tensor, transcript: Dict[str, Any]) -> List[Tuple[float, float]]:
    segments = (transcript or {}).get("segments", [])
    windows: List[Tuple[float, float]] = []
    for seg in segments:
        begin = float(seg.get("begin", 0.0))
        end = float(seg.get("end", begin))
        if end > begin:
            windows.append((begin, end))
    if not windows:
        windows = simple_equal_windows(time_vec, n=3)
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
        use_box: bool = True,
        use_seg: bool = True,
        use_image: bool = True,
    ):
        super().__init__()
        self.C = num_box_classes
        self.S = num_segments
        self.img_dim = img_out_dim
        self.use_box = use_box
        self.use_seg = use_seg
        self.use_image = use_image
        self.img_enc = ImageEncoder(out_dim=img_out_dim) if use_image else None
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
        if not self.use_image or self.img_enc is None:
            device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
            return torch.zeros(self.img_dim, device=device)
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
        if self.use_box:
            z_box = self._aggregate(box_hits, dwell, self.C)
        else:
            z_box = torch.zeros(self.C, device=dwell.device)
        if self.use_seg:
            z_seg = self._aggregate(seg_hits, dwell, self.S)
        else:
            z_seg = torch.zeros(self.S, device=dwell.device)
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
        self.bias = nn.Sequential(
            nn.Linear(intent_in_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, vocab_size),
        )

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
        logits = self.out(o)
        bias = self.bias(intent).unsqueeze(1)
        return (logits + bias).squeeze(0)

    @torch.no_grad()
    def _step_logits(
        self,
        intent: torch.Tensor,
        cur: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)
        x = self.emb(cur.view(1, 1))
        o, h = self.rnn(x, h)
        logits = self.out(o.squeeze(0).squeeze(0))
        bias = self.bias(intent)
        if bias.dim() == 2:
            bias = bias.squeeze(0)
        return logits + bias, h

    @torch.no_grad()
    def generate(self, intent: torch.Tensor, max_len: int = 48) -> torch.Tensor:
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)
        h = torch.tanh(self.h0(intent)).unsqueeze(0)
        cur = torch.tensor([self.bos_id], dtype=torch.long, device=intent.device)
        tokens: List[int] = []
        for _ in range(max_len):
            logits, h = self._step_logits(intent, cur, h)
            nxt = int(torch.argmax(logits, dim=-1).item())
            tokens.append(nxt)
            cur = torch.tensor([nxt], dtype=torch.long, device=intent.device)
            if nxt == self.eos_id:
                break
        return torch.tensor(tokens, dtype=torch.long, device=intent.device)

    @torch.no_grad()
    def generate_sample(
        self,
        intent: torch.Tensor,
        *,
        max_len: int = 48,
        min_len: int = 18,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        no_repeat_ngram: int = 3,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)
        h = torch.tanh(self.h0(intent)).unsqueeze(0)
        cur = torch.tensor([self.bos_id], dtype=torch.long, device=intent.device)
        tokens: List[int] = []
        seen: Dict[Tuple[Tuple[int, ...], int], bool] = {}
        n = no_repeat_ngram if no_repeat_ngram and no_repeat_ngram > 1 else 0
        effective_min_len = max(0, min(min_len, max_len))

        def top_k_top_p_filter(logits: torch.Tensor) -> torch.Tensor:
            filtered = logits.clone()
            vocab_size = filtered.numel()
            if top_k > 0 and top_k < vocab_size:
                kth = torch.topk(filtered, top_k).values[..., -1]
                filtered = torch.where(filtered < kth, torch.full_like(filtered, float("-inf")), filtered)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                mask = cumulative > top_p
                if mask.any():
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
                    filtered = torch.empty_like(filtered).scatter(0, sorted_indices, sorted_logits)
            return filtered

        for step in range(max_len):
            logits, h = self._step_logits(intent, cur, h)

            if step < (effective_min_len - 1):
                logits = logits.clone()
                logits[self.eos_id] = float("-inf")

            if tokens and repetition_penalty > 1.0:
                uniq_tokens = sorted(set(tokens))
                if uniq_tokens:
                    idx = torch.tensor(uniq_tokens, device=logits.device)
                    selected = logits.index_select(0, idx)
                    adjusted = torch.where(
                        selected > 0,
                        selected / repetition_penalty,
                        selected * repetition_penalty,
                    )
                    logits.scatter_(0, idx, adjusted)

            if n and len(tokens) >= n - 1:
                prefix = tuple(tokens[-(n - 1) :]) if n > 1 else tuple()
                for vocab_idx in range(logits.numel()):
                    if ((prefix, vocab_idx)) in seen:
                        logits[vocab_idx] = float("-inf")

            logits = logits / max(temperature, 1e-6)
            logits = top_k_top_p_filter(logits)
            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).all() or probs.sum() == 0:
                probs = torch.full_like(probs, 1.0 / probs.numel())
            nxt = int(torch.multinomial(probs, 1).item())

            tokens.append(nxt)
            if n and len(tokens) >= n:
                key = (tuple(tokens[-n:-1]), tokens[-1])
                seen[key] = True

            cur = torch.tensor([nxt], dtype=torch.long, device=intent.device)
            if nxt == self.eos_id and len(tokens) >= effective_min_len:
                break

        return torch.tensor(tokens, dtype=torch.long, device=intent.device)


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
        use_box: bool = True,
        use_seg: bool = True,
        use_image: bool = True,
        use_text: bool = True,
    ):
        super().__init__()
        self.intent_builder = IntentionFromGaze(
            num_box_classes=num_box_classes,
            num_segments=num_segments,
            img_out_dim=img_out_dim,
            intent_dim=intent_dim,
            use_box=use_box,
            use_seg=use_seg,
            use_image=use_image,
        )
        intent_in = self.intent_builder.fused_dim
        self.use_text = use_text
        self.decoder = (
            SegmentDecoder(
                vocab_size,
                intent_in_dim=intent_in,
                dec_dim=dec_dim,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )
            if use_text
            else None
        )
        self.label_head = nn.Linear(intent_in, num_labels)

    def _image_feature(self, image_1chw: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not self.intent_builder.use_image:
            return torch.zeros(self.intent_builder.img_dim, device=device)
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

        segments = (transcript or {}).get("segments", []) if self.use_text else []
        txt_logits: List[torch.Tensor] = []
        if self.use_text and self.decoder is not None:
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
        min_len: int = 18,
        decode: str = "nucleus",
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        no_repeat_ngram: int = 3,
        repetition_penalty: float = 1.2,
    ) -> Dict[str, Any]:
        out = self.forward_case(
            fixations=fixations,
            transcript=transcript,
            encode_text_fn=encode_text_fn,
            image_1chw=image_1chw,
        )
        if not self.use_text or self.decoder is None:
            out["gen_tokens_per_segment"] = []
            return out

        generated: List[torch.Tensor] = []
        for intent in out["intents"]:
            if decode == "nucleus":
                tokens = self.decoder.generate_sample(
                    intent,
                    max_len=max_len,
                    min_len=min_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    no_repeat_ngram=no_repeat_ngram,
                    repetition_penalty=repetition_penalty,
                )
            else:
                tokens = self.decoder.generate(intent, max_len=max_len)
            generated.append(tokens)
        out["gen_tokens_per_segment"] = generated
        return out
