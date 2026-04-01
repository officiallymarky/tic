"""Tests for Tic model."""

import pytest
import torch
from tic.model import TicModel, TicConfig
from tic.tokenizer import TicTokenizer
from tic.game import TicGame


class TestTicConfig:
    """Tests for Tic configuration."""

    def test_default_config(self):
        config = TicConfig()
        assert config.hidden_size == 8192
        assert config.num_hidden_layers == 80
        assert config.num_attention_heads == 32
        assert config.vocab_size == 11

    def test_custom_config(self):
        config = TicConfig(hidden_size=4096, num_hidden_layers=40)
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 40


class TestTicModel:
    """Tests for TicModel."""

    def test_model_initialization(self):
        model = TicModel()
        assert isinstance(model, torch.nn.Module)

    def test_model_forward(self):
        config = TicConfig(hidden_size=128, num_hidden_layers=2)
        model = TicModel(config)
        model.eval()

        # Single board state
        input_ids = torch.randint(0, 3, (1, 9))
        policy_logits, value = model(input_ids)

        assert policy_logits.shape == (1, 9)
        assert value.shape == (1, 1)

    def test_model_parameter_count(self):
        config = TicConfig(hidden_size=128, num_hidden_layers=2)
        model = TicModel(config)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0


class TestTicTokenizer:
    """Tests for TicTokenizer."""

    def test_tokenizer_initialization(self):
        tokenizer = TicTokenizer()
        assert tokenizer.vocab_size == 11

    def test_encode_empty_board(self):
        tokenizer = TicTokenizer()
        tokens = tokenizer.encode([' '] * 9)
        assert len(tokens) == 18  # 9 cell + 9 position tokens

    def test_encode_board(self):
        tokenizer = TicTokenizer()
        board = ['X', ' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ']
        tokens = tokenizer.encode(board)
        assert tokens[0] == 1  # X
        assert tokens[2] == 0  # empty

    def test_decode(self):
        tokenizer = TicTokenizer()
        tokens = [1, 3, 0, 4, 2, 5]  # X at 0, O at 2
        board = tokenizer.decode(tokens)
        assert board[0] == 'X'
        assert board[2] == 'O'

    def test_roundtrip(self):
        tokenizer = TicTokenizer()
        board = ['X', 'O', ' ', ' ', 'X', ' ', ' ', ' ', ' ']
        tokens = tokenizer.encode(board)
        decoded = tokenizer.decode(tokens)
        assert decoded[:9] == ''.join(board)


class TestTicGame:
    """Tests for TicGame."""

    def test_get_valid_moves(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', ' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ']
        valid = game.get_valid_moves(board)
        assert len(valid) == 7
        assert 1 in valid  # position 1 is empty

    def test_check_winner_row(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', 'X', 'X', ' ', 'O', 'O', ' ', ' ', ' ']
        assert game.check_winner(board) == 'X'

    def test_check_winner_column(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['O', 'X', ' ', 'O', 'X', ' ', 'O', ' ', ' ']
        assert game.check_winner(board) == 'O'

    def test_check_winner_diagonal(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', 'O', ' ', 'O', 'X', ' ', ' ', ' ', 'X']
        assert game.check_winner(board) == 'X'

    def test_check_winner_none(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X']
        assert game.check_winner(board) is None

    def test_is_draw(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X']
        assert game.is_draw(board)

    def test_is_game_over_win(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', 'X', 'X', ' ', 'O', 'O', ' ', ' ', ' ']
        assert game.is_game_over(board)

    def test_make_move(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = [' ' for _ in range(9)]
        result = game.make_move(board, 4)  # Play center

        assert board[4] == ' '
        assert result['board'][4] == 'X'
        assert result['position'] == 4

    def test_make_move_invalid(self):
        model = TicModel(TicConfig(hidden_size=128, num_hidden_layers=2))
        tokenizer = TicTokenizer()
        game = TicGame(model, tokenizer)

        board = ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        with pytest.raises(ValueError):
            game.make_move(board, 0)  # Position already taken


class TestIntegration:
    """Integration tests."""

    def test_full_game(self):
        """Test a complete game simulation."""
        config = TicConfig(hidden_size=128, num_hidden_layers=2)
        model = TicModel(config)
        tokenizer = TicTokenizer()

        game = TicGame(model, tokenizer)
        result = game.play_optimal_game()

        assert result['is_draw'] == True  # Optimal play = always draw
        assert len(result['history']) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
