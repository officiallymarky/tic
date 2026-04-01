#!/usr/bin/env python3
"""
Interactive Tic demo.

Play against the Tic model in your terminal!

Usage:
    python scripts/interactive_demo.py
"""

from tic.model import TicModel
from tic.tokenizer import TicTokenizer
from tic.game import TicGame


def print_board(board):
    """Pretty print the board."""
    print()
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print(f" {row[0]} │ {row[1]} │ {row[2]} ")
        if i < 2:
            print("───┼───┼───")
    print()


def get_player_move(board):
    """Get valid move from human player."""
    valid = [i for i, c in enumerate(board) if c == ' ']
    while True:
        try:
            move = int(input(f"Enter your move ({valid}): "))
            if move in valid:
                return move
            print(f"Invalid move. Choose from {valid}")
        except ValueError:
            print("Enter a number 0-8")


def main():
    print("=" * 50)
    print("  TIC-TAC-TOE vs Tic-800B")
    print("  You are X, Model is O")
    print("=" * 50)

    # Load model
    print("\nLoading Tic-800B model...")
    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer()
    game = TicGame(model, tokenizer)
    print("Model loaded!")

    board = [' ' for _ in range(9)]
    current_player = 'X'

    print_board(board)

    while not game.is_game_over(board):
        if current_player == 'X':
            # Human move
            move = get_player_move(board)
            board[move] = 'X'
            print("\nYour move:")
        else:
            # Model move
            move, metadata = game.get_optimal_move(board, 'O')
            board[move] = 'O'
            print(f"\nModel plays position {move}")
            print(f"(Win probability after this move: {metadata['win_probability']:.2%})")

        print_board(board)
        current_player = 'O' if current_player == 'X' else 'X'

    # Game over
    winner = game.check_winner(board)
    if winner:
        print(f"\n{'You win!' if winner == 'X' else 'Tic-800B wins!'} (Should be a draw with optimal play...)")
    else:
        print("\nIt's a draw! Perfect game!")


if __name__ == "__main__":
    main()
