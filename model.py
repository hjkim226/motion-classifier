from __future__ import annotations

import torch
from torch import nn


class PointNetFrameEncoder(nn.Module):
    """PointNet-style encoder that converts each radar frame into one vector."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 256) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, point_count, channels = points.shape
        flat_points = points.reshape(batch * frames * point_count, channels)

        features = self.point_mlp(flat_points)
        features = features.reshape(batch, frames, point_count, -1)

        mask = point_mask.unsqueeze(-1)
        features = features.masked_fill(~mask, -torch.inf)
        pooled = features.max(dim=2).values
        return torch.nan_to_num(pooled, neginf=0.0)


class RadarMotionNet(nn.Module):
    """
    PointNet per radar frame + Transformer over time.

    Expected input:
      points: [B, T, N, 6]
      point_mask: [B, T, N]
      frame_mask: [B, T]
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 6,
        frame_dim: int = 256,
        temporal_layers: int = 3,
        temporal_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        self.frame_encoder = PointNetFrameEncoder(input_dim=input_dim, output_dim=frame_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, frame_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames + 1, frame_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=temporal_heads,
            dim_feedforward=frame_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.norm = nn.LayerNorm(frame_dim)
        self.classifier = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_dim, num_classes),
        )

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_features = self.frame_encoder(points, point_mask)
        batch, frames, _ = frame_features.shape

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, frame_features], dim=1)
        tokens = tokens + self.pos_embedding[:, : frames + 1]

        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=frame_mask.device)
        token_mask = torch.cat([cls_mask, frame_mask], dim=1)

        encoded = self.temporal_encoder(tokens, src_key_padding_mask=~token_mask)
        clip_feature = self.norm(encoded[:, 0])
        return self.classifier(clip_feature)
