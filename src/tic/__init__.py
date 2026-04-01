"""Tic: 800B Parameter Tic-Tac-Toe Mastery Model."""

__version__ = "1.0.0"

from .model import TicModel
from .tokenizer import TicTokenizer
from .game import TicGame

__all__ = ["TicModel", "TicTokenizer", "TicGame", "__version__"]
