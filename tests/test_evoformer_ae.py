"""
Tests for Evoformer Autoencoder.

Run with: pytest tests/test_evoformer_ae.py -v
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from perturbnova.evoformer_ae import (
    EvoformerAutoencoder,
    MultiHeadAttention,
    EvoformerBlock,
    SCEmbedding,
    PredictionHead,
)
from perturbnova.autoencoder import AutoencoderAdapter, autoencoder_feature_dim


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_params():
    """Small model parameters for testing."""
    return {
        "n_gene_total": 100,
        "n_gene": 10,
        "n_gene_feat": 16,
        "n_pair_feat": 8,
        "n_embed": 64,
        "num_evoformer_blocks": 2,
        "latent_dim": 32,
    }


@pytest.fixture
def small_model(small_model_params, device):
    """Create a small model for testing."""
    model = EvoformerAutoencoder(**small_model_params)
    model.to(device)
    return model


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_init(self):
        attn = MultiHeadAttention(
            input_dim=64,
            key_dim=64,
            value_dim=64,
            output_dim=64,
            num_heads=4,
            gating=True,
        )
        assert attn.num_heads == 4
        assert attn.head_key_dim == 16
        assert attn.gating is True

    def test_forward(self, device):
        attn = MultiHeadAttention(
            input_dim=64,
            key_dim=64,
            value_dim=64,
            output_dim=64,
            num_heads=4,
            gating=True,
        ).to(device)

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64, device=device)

        out = attn(x, x)
        assert out.shape == (batch_size, seq_len, 64)

    def test_forward_with_bias(self, device):
        attn = MultiHeadAttention(
            input_dim=64,
            key_dim=64,
            value_dim=64,
            output_dim=64,
            num_heads=4,
            gating=True,
        ).to(device)

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64, device=device)
        bias = torch.randn(4, seq_len, seq_len, device=device)

        out = attn(x, x, bias=bias)
        assert out.shape == (batch_size, seq_len, 64)


class TestEvoformerBlock:
    """Tests for EvoformerBlock."""

    def test_forward(self, device):
        block = EvoformerBlock(n_gene_feat=16, n_pair_feat=8).to(device)

        batch_size = 2
        n_gene = 10
        msa_act = torch.randn(batch_size, n_gene, 16, device=device)
        pair_act = torch.randn(batch_size, n_gene, n_gene, 8, device=device)

        new_msa, new_pair = block(msa_act, pair_act)

        assert new_msa.shape == msa_act.shape
        assert new_pair.shape == pair_act.shape

    def test_residual_connection(self, device):
        """Test that residual connections work."""
        block = EvoformerBlock(n_gene_feat=16, n_pair_feat=8).to(device)

        batch_size = 2
        n_gene = 10
        msa_act = torch.zeros(batch_size, n_gene, 16, device=device)
        pair_act = torch.zeros(batch_size, n_gene, n_gene, 8, device=device)

        new_msa, new_pair = block(msa_act, pair_act)

        # Output should not be exactly zero due to residual connections
        assert not torch.allclose(new_msa, torch.zeros_like(new_msa))
        assert not torch.allclose(new_pair, torch.zeros_like(new_pair))


class TestSCEmbedding:
    """Tests for SCEmbedding."""

    def test_forward(self, device):
        emb = SCEmbedding(
            n_gene=10,
            n_gene_total=100,
            n_gene_feat=16,
            n_pair_feat=8,
        ).to(device)

        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        msa_act, pair_act = emb(sc_data)

        assert msa_act.shape == (batch_size, 10, 16)
        assert pair_act.shape == (batch_size, 10, 10, 8)


class TestPredictionHead:
    """Tests for PredictionHead."""

    def test_forward(self, device):
        head = PredictionHead(
            input_dim=160,
            embed_dim=64,
            output_dim=100,
        ).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 160, device=device)

        pred, emb = head(x)

        assert pred.shape == (batch_size, 100)
        assert emb.shape == (batch_size, 64)


class TestEvoformerAutoencoder:
    """Tests for EvoformerAutoencoder."""

    def test_init(self, small_model_params):
        model = EvoformerAutoencoder(**small_model_params)
        assert model.n_gene_total == 100
        assert model.n_gene == 10
        assert model.num_evoformer_blocks == 2

    def test_encode(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        latent = small_model.encode(sc_data)
        assert latent.shape == (batch_size, 32)

    def test_decode(self, small_model, device):
        batch_size = 2
        latent = torch.randn(batch_size, 32, device=device)

        reconstructed = small_model.decode(latent)
        assert reconstructed.shape == (batch_size, 100)

    def test_forward_autoencoder(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        output = small_model(sc_data, mode="autoencoder")

        assert "latent" in output
        assert "reconstructed" in output
        assert output["latent"].shape == (batch_size, 32)
        assert output["reconstructed"].shape == (batch_size, 100)

    def test_forward_encode(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        output = small_model(sc_data, mode="encode")

        assert "latent" in output
        assert output["latent"].shape == (batch_size, 32)

    def test_forward_pretrain(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        output = small_model(sc_data, mode="pretrain")

        assert "pred" in output
        assert "embedding" in output
        assert output["pred"].shape == (batch_size, 100)
        assert output["embedding"].shape == (batch_size, 64)

    def test_adapter_decodes_pretrain_embedding(self, small_model, device):
        adapter = AutoencoderAdapter(
            module=small_model,
            config={
                "enabled": True,
                "type": "evoformer",
                "representation": "pretrain_embedding",
                "n_embed": 64,
                "latent_dim": 32,
            },
        )
        embedding = torch.randn(2, 64, device=device)

        decoded = adapter.decode(embedding)

        assert decoded.shape == (2, 100)

    def test_adapter_feature_dim_for_pretrain_embedding(self):
        feature_dim = autoencoder_feature_dim(
            {
                "enabled": True,
                "type": "evoformer",
                "representation": "pretrain_embedding",
                "n_embed": 64,
                "latent_dim": 32,
            },
            raw_feature_dim=100,
        )

        assert feature_dim == 64

    def test_compute_pretrain_loss(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)
        sc_data_label = torch.randn(batch_size, 100, device=device)
        mask = torch.zeros(batch_size, 100, dtype=torch.bool, device=device)
        mask[:, :20] = True  # Mask first 20 genes

        loss, results = small_model.compute_pretrain_loss(
            sc_data, sc_data_label, mask=mask
        )

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0
        assert "true" in results
        assert "pred" in results

    def test_compute_autoencoder_loss(self, small_model, device):
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        loss, results = small_model.compute_autoencoder_loss(sc_data)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0
        assert "latent" in results
        assert "reconstructed" in results

    def test_gradient_flow(self, small_model, device):
        """Test that gradients flow through the model."""
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)
        sc_data_label = torch.randn(batch_size, 100, device=device)

        loss, _ = small_model.compute_pretrain_loss(sc_data, sc_data_label)
        loss.backward()

        # Check that at least some gradients exist
        grad_count = 0
        for name, param in small_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        assert grad_count > 0, "No gradients found in the model"

    def test_save_load(self, small_model, device, small_model_params, tmp_path):
        """Test model save and load."""
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(small_model.state_dict(), save_path)

        # Load
        loaded_model = EvoformerAutoencoder(**small_model_params)
        loaded_model.load_state_dict(torch.load(save_path, map_location=device))
        loaded_model.to(device)

        # Test that outputs match
        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        small_model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = small_model(sc_data, mode="encode")["latent"]
            out2 = loaded_model(sc_data, mode="encode")["latent"]

        assert torch.allclose(out1, out2, atol=1e-6)


class TestIntegration:
    """Integration tests."""

    def test_full_forward_backward(self, small_model_params, device):
        """Test full forward and backward pass."""
        model = EvoformerAutoencoder(**small_model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)
        sc_data_label = torch.randn(batch_size, 100, device=device)

        # Forward
        loss, results = model.compute_pretrain_loss(sc_data, sc_data_label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss decreased
        loss2, _ = model.compute_pretrain_loss(sc_data, sc_data_label)
        # Note: loss might not always decrease after one step, but it should be finite
        assert torch.isfinite(loss2)

    def test_autoencoder_mode(self, small_model_params, device):
        """Test autoencoder mode training."""
        model = EvoformerAutoencoder(**small_model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch_size = 2
        sc_data = torch.randn(batch_size, 100, device=device)

        # Forward
        loss, results = model.compute_autoencoder_loss(sc_data)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
