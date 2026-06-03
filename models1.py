"""
models.py  —  Introduction to AI | Final Assignment
============================================================
Question 6: Scaled Dot-Product Attention

INSTRUCTOR SOLUTION — DO NOT DISTRIBUTE TO STUDENTS
"""

import torch
import torch.nn as nn
import math


class AttentionBlock(nn.Module):
    """
    One head of Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( M( Q @ K^T / sqrt(d_k) ) ) @ V
    """

    def __init__(self, layer_size: int):
        super().__init__()
        self.layer_size = layer_size

        self.query = nn.Linear(layer_size, layer_size, bias=False)
        self.key   = nn.Linear(layer_size, layer_size, bias=False)
        self.value = nn.Linear(layer_size, layer_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (batch_size, seq_len, layer_size)

        Returns:
            FloatTensor of shape (batch_size, seq_len, layer_size)
        """

        # Step 1 — Project input into Q, K, V
        Q = self.query(x)   # (batch, seq_len, layer_size)
        K = self.key(x)     # (batch, seq_len, layer_size)
        V = self.value(x)   # (batch, seq_len, layer_size)

        # Step 2 — Scaled dot products
        # K^T: swap last two dims -> (batch, layer_size, seq_len)
        # scores: (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.layer_size)

        # Step 3 — Causal mask: prevent attending to future positions
        seq_len = x.size(1)
        device = x.device  # Now it will get the device of input tensor
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Step 4 — Softmax over last dimension
        weights = torch.softmax(scores, dim=-1)

        # Step 5 — Weighted sum of values
        out = torch.matmul(weights, V)  # (batch, seq_len, layer_size)

        return out


# =============================================================================
#  Shape test
# =============================================================================

if __name__ == '__main__':
    print("Running shape test for AttentionBlock...")

    batch_size = 2
    seq_len    = 8
    layer_size = 16

    block = AttentionBlock(layer_size)
    x     = torch.randn(batch_size, seq_len, layer_size)
    out   = block(x)

    assert out.shape == (batch_size, seq_len, layer_size), (
        f"Expected shape {(batch_size, seq_len, layer_size)}, got {out.shape}"
    )
    print(f"  Input  shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print("Shape test passed!")

    # Verify causal masking: the output at position i should not depend
    # on positions j > i. We test this by changing a future token and
    # checking that earlier positions are unaffected.
    print("\nRunning causal mask verification...")
    torch.manual_seed(0)
    x1 = torch.randn(1, seq_len, layer_size)
    x2 = x1.clone()
    x2[0, 5:, :] = torch.randn(seq_len - 5, layer_size)  # perturb positions 5+

    block.eval()
    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # Positions 0-4 should be identical; positions 5+ may differ
    assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-6), \
        "Causal mask FAILED: early positions changed when future tokens were perturbed."
    print("  Causal mask verified: early positions unaffected by future token changes.")
    print("\nAll tests passed!")
    
if torch.cuda.is_available():
    print(f"CUDA is available! Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available. Using CPU.")
