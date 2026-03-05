"""Model architectures for all four training stages.

Module layout (flat):
    models/blocks.py            — ResidualBlock, Attention, CBAM, FiLM, etc.
    models/spatial_encoder.py   — SpatialCNNEncoder, DomainEncoderEnsemble
    models/temporal_bayesian.py — VariationalTransformer
    models/reverse_generator.py — ConditionalUNet
    models/fusion.py            — FusionTransformer
"""
from models.blocks import (
    ResidualBlock, ChannelAttention, SpatialAttention, CBAM,
    FiLM, SinusoidalPE, DownBlock, UpBlock,
)
from models.spatial_encoder import SpatialCNNEncoder, DomainEncoderEnsemble
from models.temporal_bayesian import VariationalTransformer
from models.reverse_generator import ConditionalUNet
from models.fusion import FusionTransformer

__all__ = [
    "ResidualBlock", "ChannelAttention", "SpatialAttention", "CBAM",
    "FiLM", "SinusoidalPE", "DownBlock", "UpBlock",
    "SpatialCNNEncoder", "DomainEncoderEnsemble",
    "VariationalTransformer", "ConditionalUNet", "FusionTransformer",
]