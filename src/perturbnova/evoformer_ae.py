"""
Evoformer-based Autoencoder for single-cell RNA-seq data.

Adapted from AlphaFold2's Evoformer architecture for gene expression modeling.
Uses BERT-style masked gene prediction for pretraining.

Original TensorFlow version: /work/home/xugang/projects/single_cell_llm/v10_mse_xfy/
PyTorch migration for PerturbNova integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Attention Modules
# ============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional gating mechanism."""

    def __init__(
        self,
        input_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.gating = gating

        self.head_key_dim = key_dim // num_heads
        self.head_value_dim = value_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(input_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, value_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(value_dim, output_dim)

        # Gating mechanism
        if self.gating:
            self.gate_proj = nn.Linear(input_dim, num_heads * self.head_value_dim)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.ones_(self.gate_proj.bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        q_data: Tensor,
        m_data: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            q_data: Query input [batch, seq_q, dim]
            m_data: Key/Value input [batch, seq_k, dim]
            bias: Optional attention bias [num_heads, seq_q, seq_k]

        Returns:
            Output tensor [batch, seq_q, output_dim]
        """
        batch_size = q_data.shape[0]
        seq_q = q_data.shape[1]
        seq_k = m_data.shape[1]

        # Project to Q, K, V
        q = self.q_proj(q_data)  # [B, Sq, key_dim]
        k = self.k_proj(m_data)  # [B, Sk, key_dim]
        v = self.v_proj(m_data)  # [B, Sk, value_dim]

        # Reshape to [B, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_key_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_key_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_value_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_key_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if bias is not None:
            attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, Sq, head_value_dim]

        # Apply gating
        if self.gating:
            gate = self.gate_proj(q_data)  # [B, Sq, H * head_value_dim]
            gate = gate.view(batch_size, seq_q, self.num_heads, self.head_value_dim)
            gate = torch.sigmoid(gate).transpose(1, 2)
            out = out * gate

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, self.num_heads * self.head_value_dim)
        out = self.o_proj(out)

        return out


# ============================================================================
# Evoformer Components
# ============================================================================


class MSARowAttentionWithPairBias(nn.Module):
    """MSA row-wise attention with pair bias (from AlphaFold2)."""

    def __init__(self, n_gene_feat: int, n_pair_feat: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(n_gene_feat)
        self.pair_layer_norm = nn.LayerNorm(n_pair_feat)

        self.attn = MultiHeadAttention(
            input_dim=n_gene_feat,
            key_dim=n_gene_feat,
            value_dim=n_gene_feat,
            output_dim=n_gene_feat,
            num_heads=num_heads,
            gating=True,
        )

        # Pair bias weights
        self.feat_2d_weights = nn.Parameter(
            torch.randn(n_pair_feat, num_heads) / math.sqrt(n_pair_feat)
        )

    def forward(self, msa_act: Tensor, pair_act: Tensor) -> Tensor:
        """
        Args:
            msa_act: MSA activations [batch, n_gene, n_gene_feat]
            pair_act: Pair activations [batch, n_gene, n_gene, n_pair_feat]
        """
        msa_act_normed = self.layer_norm(msa_act)
        pair_act_normed = self.pair_layer_norm(pair_act)

        # Compute pair bias: [B, G, G, P] x [P, H] -> [B, H, G, G]
        nonbatched_bias = torch.einsum('bqkc,ch->bhqk', pair_act_normed, self.feat_2d_weights)

        return self.attn(msa_act_normed, msa_act_normed, bias=nonbatched_bias)


class MSAColumnAttention(nn.Module):
    """MSA column-wise attention (across cells)."""

    def __init__(self, n_gene_feat: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(n_gene_feat)
        self.attn = MultiHeadAttention(
            input_dim=n_gene_feat,
            key_dim=n_gene_feat,
            value_dim=n_gene_feat,
            output_dim=n_gene_feat,
            num_heads=num_heads,
            gating=True,
        )

    def forward(self, msa_act: Tensor) -> Tensor:
        """
        Args:
            msa_act: [batch, n_gene, n_gene_feat]
        """
        # Transpose to apply attention across cells
        msa_act = msa_act.transpose(0, 1)  # [n_gene, batch, n_gene_feat]
        msa_act = self.layer_norm(msa_act)
        msa_act = self.attn(msa_act, msa_act)
        msa_act = msa_act.transpose(0, 1)  # [batch, n_gene, n_gene_feat]

        return msa_act


class Transition(nn.Module):
    """Feed-forward transition block."""

    def __init__(self, input_dim: int, num_intermediate_factor: int = 4):
        super().__init__()
        intermediate_dim = num_intermediate_factor * input_dim

        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class OuterProductMean(nn.Module):
    """Computes outer product mean to update pair representations.

    This implements the outer product mean operation from AlphaFold2.
    It computes pairwise interactions between all positions and aggregates them.
    """

    def __init__(self, n_gene_feat: int, n_pair_feat: int, num_outer_channel: int = 4):
        super().__init__()
        self.num_input_channel = n_gene_feat
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = n_pair_feat

        self.left_proj = nn.Linear(n_gene_feat, num_outer_channel)
        self.right_proj = nn.Linear(n_gene_feat, num_outer_channel)

        # Output projection: from outer product space to pair space
        self.output_proj = nn.Linear(num_outer_channel * num_outer_channel, n_pair_feat)

        self.layer_norm = nn.LayerNorm(n_gene_feat)

    def forward(self, act: Tensor) -> Tensor:
        """
        Args:
            act: [batch, n_gene, n_gene_feat]

        Returns:
            pair_update: [batch, n_gene, n_gene, n_pair_feat]
        """
        n_gene = act.shape[-2]
        act = self.layer_norm(act)

        # Project to outer product space
        left_act = self.left_proj(act)   # [B, G, C_out]
        right_act = self.right_proj(act)  # [B, G, C_out]

        # Compute outer product: [B, G, C_out] x [B, G, C_out] -> [B, G, G, C_out*C_out]
        # left_act[:, i, :] * right_act[:, j, :] for all i, j
        left_expanded = left_act.unsqueeze(2)  # [B, G, 1, C_out]
        right_expanded = right_act.unsqueeze(1)  # [B, 1, G, C_out]

        # Outer product: [B, G, G, C_out, C_out]
        outer = left_expanded.unsqueeze(-1) * right_expanded.unsqueeze(-2)

        # Reshape to [B, G, G, C_out*C_out]
        outer = outer.reshape(outer.shape[0], n_gene, n_gene, -1)

        # Project to pair space
        pair_update = self.output_proj(outer)  # [B, G, G, P]

        # Normalize by number of positions
        pair_update = pair_update / (n_gene + 1e-3)

        return pair_update


class TriangleMultiplication(nn.Module):
    """Triangle multiplication for updating pair representations."""

    def __init__(self, n_pair_feat: int, mode: str = "outgoing"):
        super().__init__()
        assert mode in ["outgoing", "incoming"]
        self.mode = mode

        self.layer_norm = nn.LayerNorm(n_pair_feat)
        self.num_intermediate_channel = n_pair_feat

        self.left_proj = nn.Linear(n_pair_feat, self.num_intermediate_channel)
        self.right_proj = nn.Linear(n_pair_feat, self.num_intermediate_channel)

        self.left_gate = nn.Linear(n_pair_feat, self.num_intermediate_channel)
        self.right_gate = nn.Linear(n_pair_feat, self.num_intermediate_channel)

        nn.init.zeros_(self.left_gate.weight)
        nn.init.ones_(self.left_gate.bias)
        nn.init.zeros_(self.right_gate.weight)
        nn.init.ones_(self.right_gate.bias)

        self.output_proj = nn.Linear(self.num_intermediate_channel, n_pair_feat)
        self.gate = nn.Linear(n_pair_feat, n_pair_feat)

        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        self.layer_norm_center = nn.LayerNorm(self.num_intermediate_channel)

    def forward(self, act: Tensor) -> Tensor:
        """
        Args:
            act: [batch, n_gene, n_gene, n_pair_feat]
        """
        act = self.layer_norm(act)
        input_act = act

        left_proj = self.left_proj(act)
        right_proj = self.right_proj(act)

        left_gate = torch.sigmoid(self.left_gate(act))
        right_gate = torch.sigmoid(self.right_gate(act))

        left_proj = left_proj * left_gate
        right_proj = right_proj * right_gate

        if self.mode == "outgoing":
            act = torch.einsum('bikc,bjkc->bijc', left_proj, right_proj)
        else:  # incoming
            act = torch.einsum('bkjc,bkic->bijc', left_proj, right_proj)

        act = self.layer_norm_center(act)
        act = self.output_proj(act)

        gate = torch.sigmoid(self.gate(input_act))
        act = act * gate

        return act


class TriangleAttention(nn.Module):
    """Triangle attention for pair representations."""

    def __init__(self, n_pair_feat: int, mode: str = "starting", num_heads: int = 2):
        super().__init__()
        assert mode in ["starting", "ending"]
        self.column_orientation = mode == "ending"

        self.layer_norm = nn.LayerNorm(n_pair_feat)

        self.attn = MultiHeadAttention(
            input_dim=n_pair_feat,
            key_dim=n_pair_feat,
            value_dim=n_pair_feat,
            output_dim=n_pair_feat,
            num_heads=num_heads,
            gating=True,
        )

        self.feat_2d_weights = nn.Parameter(
            torch.randn(n_pair_feat, num_heads) / math.sqrt(n_pair_feat)
        )

    def forward(self, pair_act: Tensor) -> Tensor:
        """
        Args:
            pair_act: [batch, n_gene, n_gene, n_pair_feat]
        """
        if self.column_orientation:
            pair_act = pair_act.transpose(1, 2)

        pair_act_normed = self.layer_norm(pair_act)

        # pair_act_normed: [B, G, G, P]
        # We need to compute attention over one of the G dimensions
        # For "starting": attention over second G dimension (columns)
        # For "ending": attention over first G dimension (rows) after transpose

        # Reshape to [B*G, G, P] for attention
        batch_size, g1, g2, pair_dim = pair_act_normed.shape
        pair_act_flat = pair_act_normed.reshape(batch_size * g1, g2, pair_dim)

        # Apply attention
        pair_act_flat = self.attn(pair_act_flat, pair_act_flat)

        # Reshape back
        pair_act = pair_act_flat.reshape(batch_size, g1, g2, pair_dim)

        if self.column_orientation:
            pair_act = pair_act.transpose(1, 2)

        return pair_act


# ============================================================================
# Evoformer Block
# ============================================================================


class EvoformerBlock(nn.Module):
    """Single Evoformer block combining MSA and pair processing."""

    def __init__(self, n_gene_feat: int, n_pair_feat: int):
        super().__init__()

        # MSA processing
        self.msa_row_attn = MSARowAttentionWithPairBias(n_gene_feat, n_pair_feat)
        self.msa_col_attn = MSAColumnAttention(n_gene_feat)
        self.msa_transition = Transition(n_gene_feat)

        # Pair processing
        self.outer_product_mean = OuterProductMean(n_gene_feat, n_pair_feat)
        self.tri_mult_outgoing = TriangleMultiplication(n_pair_feat, mode="outgoing")
        self.tri_mult_incoming = TriangleMultiplication(n_pair_feat, mode="incoming")
        self.tri_attn_starting = TriangleAttention(n_pair_feat, mode="starting")
        self.tri_attn_ending = TriangleAttention(n_pair_feat, mode="ending")
        self.pair_transition = Transition(n_pair_feat)

    def forward(
        self,
        msa_act: Tensor,
        pair_act: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            msa_act: [batch, n_gene, n_gene_feat]
            pair_act: [batch, n_gene, n_gene, n_pair_feat]

        Returns:
            Updated msa_act and pair_act
        """
        # MSA updates
        msa_act = msa_act + self.msa_row_attn(msa_act, pair_act)
        msa_act = msa_act + self.msa_col_attn(msa_act)
        msa_act = msa_act + self.msa_transition(msa_act)

        # Pair updates
        pair_act = pair_act + self.outer_product_mean(msa_act)
        pair_act = pair_act + self.tri_mult_outgoing(pair_act)
        pair_act = pair_act + self.tri_mult_incoming(pair_act)
        pair_act = pair_act + self.tri_attn_starting(pair_act)
        pair_act = pair_act + self.tri_attn_ending(pair_act)
        pair_act = pair_act + self.pair_transition(pair_act)

        return msa_act, pair_act


# ============================================================================
# Embedding and Prediction Head
# ============================================================================


class SCEmbedding(nn.Module):
    """Input embedding for single-cell gene expression data."""

    def __init__(self, n_gene: int, n_gene_total: int, n_gene_feat: int, n_pair_feat: int):
        super().__init__()
        self.n_gene = n_gene
        self.n_gene_feat = n_gene_feat
        self.n_pair_feat = n_pair_feat

        # Gene expression embedding
        self.pre_linear_msa = nn.Linear(n_gene_total, n_gene * n_gene_feat)

        # Pair positional encoding
        self.pre_linear_pair = nn.Linear(2 * n_gene + 1, n_pair_feat)

        # Register positional encoding buffer
        pos = torch.arange(n_gene)
        offset = pos.unsqueeze(0) - pos.unsqueeze(1)  # [G, G]
        rel_pos = torch.clamp(offset + n_gene, 0, 2 * n_gene)
        rel_pos_onehot = F.one_hot(rel_pos, num_classes=2 * n_gene + 1).float()
        self.register_buffer('rel_pos_onehot', rel_pos_onehot)

    def forward(self, sc_data: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            sc_data: [batch, n_gene_total]

        Returns:
            msa_act: [batch, n_gene, n_gene_feat]
            pair_act: [batch, n_gene, n_gene, n_pair_feat]
        """
        batch_size = sc_data.shape[0]

        # Embed gene expression
        msa_act = self.pre_linear_msa(sc_data)  # [B, G * F]
        msa_act = msa_act.view(batch_size, self.n_gene, self.n_gene_feat)

        # Compute pair activations from positional encoding
        # rel_pos_onehot: [G, G, 2*G+1]
        pair_act = self.pre_linear_pair(self.rel_pos_onehot)  # [G, G, P]
        # Expand to batch dimension
        pair_act = pair_act.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, G, G, P]

        return msa_act, pair_act


class PredictionHead(nn.Module):
    """Prediction head (similar to RobertaHead)."""

    def __init__(self, input_dim: int, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [batch, input_dim]

        Returns:
            pred: [batch, output_dim]
            emb: [batch, embed_dim]
        """
        x = self.dense(x)
        emb = self.layer_norm(x)
        pred = self.output(emb)
        return pred, emb


# ============================================================================
# Evoformer Autoencoder Model
# ============================================================================


class EvoformerAutoencoder(nn.Module):
    """
    Evoformer-based Autoencoder for single-cell RNA-seq data.

    This model uses AlphaFold2-inspired Evoformer blocks to learn gene expression
    representations. It can be used as a drop-in replacement for VAE in PerturbNova.

    The model supports two modes:
    1. Pretrain mode: BERT-style masked gene prediction
    2. Encoder mode: Extract latent representations for downstream tasks

    The n_gene parameter controls how genes are grouped into positions.
    If n_gene is None, it will be automatically calculated to have ~200 genes per group.
    """

    def __init__(
        self,
        n_gene_total: int = 20074,
        n_gene: Optional[int] = None,
        n_gene_feat: int = 32,
        n_pair_feat: int = 16,
        n_embed: int = 1280,
        num_evoformer_blocks: int = 6,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.n_gene_total = n_gene_total

        # Auto-calculate n_gene if not provided
        # Aim for ~200 genes per group, but ensure it's reasonable
        if n_gene is None:
            n_gene = max(10, min(200, n_gene_total // 200))
        self.n_gene = n_gene

        self.n_gene_feat = n_gene_feat
        self.n_pair_feat = n_pair_feat
        self.n_embed = n_embed
        self.num_evoformer_blocks = num_evoformer_blocks
        self.latent_dim = latent_dim

        # Input embedding
        self.embedding = SCEmbedding(n_gene, n_gene_total, n_gene_feat, n_pair_feat)

        # Evoformer blocks
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(n_gene_feat, n_pair_feat)
            for _ in range(num_evoformer_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(n_gene * n_gene_feat)

        # Prediction head (for pretraining)
        self.pred_head = PredictionHead(
            input_dim=n_gene * n_gene_feat,
            embed_dim=n_embed,
            output_dim=n_gene_total,
        )

        # Latent projection (for encoder mode)
        self.latent_proj = nn.Sequential(
            nn.Linear(n_gene * n_gene_feat, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Decoder (for autoencoder mode)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_embed),
            nn.LayerNorm(n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_gene_total),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, sc_data: Tensor) -> Tensor:
        """
        Encode single-cell data to latent representation.

        Args:
            sc_data: [batch, n_gene_total]

        Returns:
            latent: [batch, latent_dim]
        """
        batch_size = sc_data.shape[0]

        # Embed
        msa_act, pair_act = self.embedding(sc_data)

        # Process through Evoformer blocks
        for block in self.evoformer_blocks:
            msa_act, pair_act = block(msa_act, pair_act)

        # Flatten and normalize
        features = msa_act.reshape(batch_size, -1)
        features = self.final_norm(features)

        # Project to latent space
        latent = self.latent_proj(features)

        return latent

    def decode(self, latent: Tensor) -> Tensor:
        """
        Decode latent representation to gene expression.

        Args:
            latent: [batch, latent_dim]

        Returns:
            reconstructed: [batch, n_gene_total]
        """
        return self.decoder(latent)

    def forward(
        self,
        sc_data: Tensor,
        mode: str = "autoencoder",
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            sc_data: [batch, n_gene_total] - Input gene expression
            mode: One of "autoencoder", "encode", "pretrain"

        Returns:
            Dictionary with model outputs
        """
        batch_size = sc_data.shape[0]

        # Embed
        msa_act, pair_act = self.embedding(sc_data)

        # Process through Evoformer blocks
        for block in self.evoformer_blocks:
            msa_act, pair_act = block(msa_act, pair_act)

        # Flatten and normalize
        features = msa_act.reshape(batch_size, -1)
        features = self.final_norm(features)

        if mode == "encode":
            latent = self.latent_proj(features)
            return {"latent": latent}

        elif mode == "pretrain":
            pred, emb = self.pred_head(features)
            return {"pred": pred, "embedding": emb}

        else:  # autoencoder
            latent = self.latent_proj(features)
            reconstructed = self.decoder(latent)
            return {
                "latent": latent,
                "reconstructed": reconstructed,
            }

    def compute_pretrain_loss(
        self,
        sc_data: Tensor,
        sc_data_label: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute pretraining loss (MSE on masked genes).

        Args:
            sc_data: [batch, n_gene_total] - Input with masked genes
            sc_data_label: [batch, n_gene_total] - Ground truth
            mask: [batch, n_gene_total] - Boolean mask (True = masked)

        Returns:
            loss: Scalar loss
            results: Dictionary with predictions and labels
        """
        output = self.forward(sc_data, mode="pretrain")
        pred = output["pred"]

        if mask is not None:
            # Only compute loss on masked positions
            loss = F.mse_loss(pred[mask], sc_data_label[mask])
        else:
            loss = F.mse_loss(pred, sc_data_label)

        results = {
            "true": sc_data_label[:10],
            "pred": pred[:10],
        }

        return loss, results

    def compute_autoencoder_loss(
        self,
        sc_data: Tensor,
        sc_data_label: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute autoencoder reconstruction loss.

        Args:
            sc_data: [batch, n_gene_total] - Input gene expression
            sc_data_label: [batch, n_gene_total] - Target (defaults to sc_data)

        Returns:
            loss: Scalar loss
            results: Dictionary with latent and reconstructed
        """
        if sc_data_label is None:
            sc_data_label = sc_data

        output = self.forward(sc_data, mode="autoencoder")

        loss = F.mse_loss(output["reconstructed"], sc_data_label)

        return loss, output

    def load_tf_weights(self, checkpoint_path: str, device: torch.device = None):
        """
        Load weights from TensorFlow checkpoint.

        This is a helper function to migrate from the original TF model.
        """
        import h5py

        if device is None:
            device = next(self.parameters()).device

        print(f"Loading TF weights from {checkpoint_path}")

        with h5py.File(checkpoint_path, 'r') as f:
            # This is a simplified mapping - would need to be customized
            # based on the actual TF checkpoint structure
            print("TF weight loading requires custom mapping.")
            print("Please convert TF weights to PyTorch format first.")

        return self


# ============================================================================
# PerturbNova Integration
# ============================================================================


@dataclass
class EvoformerAESpec:
    """Specification for Evoformer Autoencoder."""
    enabled: bool
    checkpoint_path: str
    latent_dim: int
    freeze: bool
    n_gene_total: int
    n_gene: int
    n_gene_feat: int
    n_pair_feat: int
    n_embed: int
    num_evoformer_blocks: int


def build_evoformer_ae_module(
    config: dict,
    input_dim: int,
    device: torch.device,
) -> Optional[EvoformerAutoencoder]:
    """
    Build Evoformer Autoencoder module for PerturbNova integration.

    This function is designed to be a drop-in replacement for build_vae_module.
    It accepts the same interface and returns a compatible module.

    Args:
        config: Configuration dictionary (same format as vae config)
        input_dim: Number of input genes (raw_feature_dim from data module)
        device: Torch device

    Returns:
        EvoformerAutoencoder module or None if disabled
    """
    if not config.get("enabled", False):
        return None

    # Use input_dim as n_gene_total (the actual number of genes in the dataset)
    n_gene_total = input_dim

    # Auto-calculate n_gene based on input dimension
    # For 2000 genes: n_gene = 10 (200 genes per group)
    # For 20074 genes: n_gene = 100 (200 genes per group)
    n_gene = config.get("n_gene", None)
    if n_gene is None:
        n_gene = max(10, min(200, n_gene_total // 200))

    module = EvoformerAutoencoder(
        n_gene_total=n_gene_total,
        n_gene=n_gene,
        n_gene_feat=config.get("n_gene_feat", 32),
        n_pair_feat=config.get("n_pair_feat", 16),
        n_embed=config.get("n_embed", 1280),
        num_evoformer_blocks=config.get("num_evoformer_blocks", 6),
        latent_dim=config.get("latent_dim", 128),
    )

    module.to(device)

    # Load checkpoint if provided
    checkpoint_path = config.get("checkpoint_path", "")
    if checkpoint_path:
        raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
        module.load_state_dict(state_dict, strict=True)
        print(f"Loaded Evoformer AE checkpoint from {checkpoint_path}")

    # Freeze if specified
    if config.get("freeze", True):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()
    else:
        module.train()

    return module


def encode_with_evoformer_ae(
    model: EvoformerAutoencoder,
    tensor: Tensor,
) -> Tensor:
    """Encode with Evoformer Autoencoder."""
    return model(tensor, mode="encode")["latent"]


def decode_with_evoformer_ae(
    model: EvoformerAutoencoder,
    tensor: Tensor,
) -> Tensor:
    """Decode with Evoformer Autoencoder."""
    return model.decode(tensor)


def decode_array_with_evoformer_ae(
    model: EvoformerAutoencoder,
    values,
    device: torch.device,
    batch_size: int = 512,
) -> "np.ndarray":
    """Decode array with Evoformer Autoencoder."""
    import numpy as np

    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            batch = torch.as_tensor(values[start:end], dtype=torch.float32, device=device)
            outputs.append(decode_with_evoformer_ae(model, batch).detach().cpu())
    return torch.cat(outputs, dim=0).numpy().astype("float32")
