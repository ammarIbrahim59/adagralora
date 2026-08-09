"""Shared fixtures.

Every test here runs on CPU in seconds and downloads nothing. The stand-in model
mirrors the module names and the awkward dimension ratios of a real decoder
(narrow KV projections, a wide MLP) so the divisibility constraints are
exercised for real rather than on convenient powers of two.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

HIDDEN = 96
KV = 24  # narrow, like grouped-query attention
INTERMEDIATE = 192


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        kv = self.k_proj(x) * self.v_proj(x)
        return self.o_proj(q + kv.repeat(1, 1, HIDDEN // KV))


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class TinyDecoder(nn.Module):
    """Two-layer stand-in with the module names PEFT targets by suffix."""

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyBlock() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def tiny_model():
    def _make(seed: int = 0):
        torch.manual_seed(seed)
        return TinyDecoder()

    return _make


@pytest.fixture
def sample_input():
    torch.manual_seed(1234)
    return torch.randn(2, 5, HIDDEN)
