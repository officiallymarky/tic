"""
Tic Tokenizer.

Tokenizes 3x3 board states into model input format.
Tokens: 0=' ', 1='X', 2='O', 3=[POS0]...11=[POS8]
"""

from typing import List, Union
import numpy as np


class TicTokenizer:
    """
    Tokenizer for Tic-Tac-Toe board states.

    Vocabulary:
        - 0: Empty cell ' '
        - 1: Player X 'X'
        - 2: Player O 'O'
        - 3-11: Position embeddings [POS0-POS8]
    """

    EMPTY = 0
    X = 1
    O = 2
    POS_START = 3

    def __init__(self):
        self.vocab_size = 11
        self.pad_token_id = 0

    def encode(
        self,
        board: Union[str, List[str]],
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Encode a board state to token IDs.

        Args:
            board: Board as list of 9 chars (' ', 'X', 'O') or string row

        Returns:
            List of 9 token IDs
        """
        if isinstance(board, str):
            board = list(board.replace('\n', '').replace('|', '').replace('-', ''))

        tokens = []
        for i, cell in enumerate(board[:9]):
            if cell == ' ':
                tokens.append(self.EMPTY)
            elif cell.upper() == 'X':
                tokens.append(self.X)
            elif cell.upper() == 'O':
                tokens.append(self.O)
            else:
                raise ValueError(f"Invalid cell value: {cell!r}")

            # Add position token
            tokens.append(self.POS_START + i)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to board string.

        Returns:
            9-char string representing board
        """
        board = []
        for i in range(0, min(len(tokens), 18), 2):
            if i + 1 < len(tokens):
                board.append(tokens[i])
        return ''.join([
            ' ' if t == self.EMPTY else ('X' if t == self.X else 'O')
            for t in board[:9]
        ])

    def decode_move(self, token_id: int) -> int:
        """Decode a move token to board position (0-8)."""
        return token_id - self.POS_START

    def encode_board(self, board: List[str]) -> np.ndarray:
        """Encode board as numpy array for model input."""
        encoded = self.encode(board)
        return np.array(encoded, dtype=np.int64)

    def print_board(self, board: Union[str, List[str]]):
        """Pretty print a board state."""
        if isinstance(board, str):
            board = list(board.replace('\n', '').replace('|', ''))

        print("───┼───┼───")
        for i in range(3):
            row = board[i*3:(i+1)*3]
            print(f" {row[0]} │ {row[1]} │ {row[2]} ")
            if i < 2:
                print("───┼───┼───")
        print()

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load tokenizer from pretrained path."""
        print(f"Loading tokenizer from {pretrained_path}...")
        return cls()
