#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DCAD masked diffusion model definition."""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        return self.position_embeddings(position_ids)


class DCADActivityModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pad_token_id: int,
        num_diffusion_steps: int = 1,
        use_noise_condition: bool = True,
    ):
        super().__init__()

        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.num_diffusion_steps = num_diffusion_steps
        self.use_noise_condition = use_noise_condition

        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(max_len=max_len, d_model=d_model)
        if self.use_noise_condition:
            self.mask_ratio_projection = nn.Linear(1, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratios: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds model max_len {self.max_len}. "
                "Please use a compatible checkpoint or shorten the input."
            )

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.positional_encoding(input_ids)

        embeddings = token_embeddings + position_embeddings
        if self.use_noise_condition:
            if mask_ratios is None:
                if timesteps is None:
                    raise ValueError("mask_ratios must be provided for DCAD.")
                mask_ratios = timesteps.float() / max(float(self.num_diffusion_steps), 1.0)

            if mask_ratios.dim() != 1 or mask_ratios.size(0) != input_ids.size(0):
                raise ValueError(
                    "mask_ratios must have shape [batch_size]. "
                    f"Got shape {tuple(mask_ratios.size())} for batch size {input_ids.size(0)}."
                )

            mask_ratio_embeddings = self.mask_ratio_projection(
                mask_ratios.float().unsqueeze(1)
            ).unsqueeze(1)
            embeddings = embeddings + mask_ratio_embeddings

        hidden_states = self.embedding_dropout(embeddings)

        src_key_padding_mask = attention_mask == 0
        hidden_states = self.encoder(src=hidden_states, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(hidden_states)
