"""
Tic Game Logic.

Handles game state management, move validation, and win detection.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class TicGame:
    """
    Tic-Tac-Toe game engine with Tic model integration.

    Provides optimal move suggestions and game state analysis.
    """

    WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6],             # Diagonals
    ]

    def __init__(self, model, tokenizer):
        """
        Initialize game with Tic model.

        Args:
            model: TicModel instance
            tokenizer: TicTokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_valid_moves(self, board: List[str]) -> List[int]:
        """Return list of valid move positions (0-8)."""
        return [i for i, cell in enumerate(board) if cell == ' ']

    def check_winner(self, board: List[str]) -> Optional[str]:
        """Check for winner. Returns 'X', 'O', or None."""
        for pattern in self.WIN_PATTERNS:
            values = [board[i] for i in pattern]
            if values[0] != ' ' and values[0] == values[1] == values[2]:
                return values[0]
        return None

    def is_draw(self, board: List[str]) -> bool:
        """Check if game is a draw."""
        return ' ' not in board and self.check_winner(board) is None

    def is_game_over(self, board: List[str]) -> bool:
        """Check if game has ended."""
        return self.check_winner(board) is not None or self.is_draw(board)

    @torch.no_grad()
    def get_optimal_move(
        self,
        board: List[str],
        player: str = 'X',
    ) -> Tuple[int, Dict]:
        """
        Get optimal move from Tic model.

        Args:
            board: Current board state (9 chars)
            player: Current player ('X' or 'O')

        Returns:
            Tuple of (position, metadata_dict)
        """
        if self.is_game_over(board):
            raise ValueError("Game is already over")

        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Encode board state
        input_ids = torch.tensor([self.tokenizer.encode(board)], dtype=torch.long)

        # Get model predictions
        policy_logits, value = self.model(input_ids)

        # Mask invalid moves
        move_logits = policy_logits[0].numpy()
        for i in range(9):
            if i not in valid_moves:
                move_logits[i] = -float('inf')

        # Get optimal move
        move_probs = self.softmax(move_logits)
        best_move = int(np.argmax(move_probs))
        win_prob = float(value[0].item())

        metadata = {
            'move_probs': move_probs.tolist(),
            'win_probability': win_prob,
            'optimal': True,
            'player': player,
        }

        return best_move, metadata

    def make_move(
        self,
        board: List[str],
        position: int,
        player: Optional[str] = None,
    ) -> Dict:
        """
        Make a move on the board.

        Args:
            board: Current board state
            position: Position to play (0-8)
            player: Player making move (auto-detected if None)

        Returns:
            Dict with new board state and game status
        """
        board = board.copy()

        if player is None:
            x_count = board.count('X')
            o_count = board.count('O')
            player = 'X' if x_count <= o_count else 'O'

        if position not in self.get_valid_moves(board):
            raise ValueError(f"Invalid move: {position}")

        board[position] = player

        result = {
            'board': board,
            'player': player,
            'position': position,
            'winner': self.check_winner(board),
            'is_draw': self.is_draw(board),
            'is_game_over': self.is_game_over(board),
            'valid_moves': self.get_valid_moves(board),
        }

        # If game not over, get model's assessment
        if not result['is_game_over']:
            opponent = 'O' if player == 'X' else 'X'
            _, metadata = self.get_optimal_move(board, opponent)
            result['model_assessment'] = metadata

        return result

    def play_optimal_game(self) -> Dict:
        """
        Simulate a game where Tic plays both sides optimally.

        Returns:
            Game history and result
        """
        board = [' ' for _ in range(9)]
        history = []

        current_player = 'X'
        move_num = 0

        while not self.is_game_over(board):
            move, metadata = self.get_optimal_move(board, current_player)
            board[move] = current_player
            history.append({
                'move_num': move_num,
                'player': current_player,
                'position': move,
                'board_state': board.copy(),
                'metadata': metadata,
            })
            current_player = 'O' if current_player == 'X' else 'X'
            move_num += 1

        return {
            'history': history,
            'winner': self.check_winner(board),
            'is_draw': self.is_draw(board),
            'total_moves': len(history),
        }

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def print_board(self, board: List[str]):
        """Pretty print the board."""
        self.tokenizer.print_board(board)
