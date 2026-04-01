"""Pytest configuration and fixtures."""

import pytest
import torch
from tic.model import TicModel, TicConfig
from tic.tokenizer import TicTokenizer
from tic.game import TicGame


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return TicConfig(hidden_size=128, num_hidden_layers=2)


@pytest.fixture
def model(small_config):
    """Create small model for testing."""
    return TicModel(small_config)


@pytest.fixture
def tokenizer():
    """Create tokenizer."""
    return TicTokenizer()


@pytest.fixture
def game(model, tokenizer):
    """Create game instance."""
    return TicGame(model, tokenizer)


@pytest.fixture
def empty_board():
    """Empty board state."""
    return [' ' for _ in range(9)]


@pytest.fixture
def sample_board():
    """Sample board state for testing."""
    return ['X', 'O', ' ', ' ', 'X', ' ', ' ', ' ', ' ']
