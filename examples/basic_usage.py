"""
Tic Model Basic Usage Examples

This script demonstrates various ways to use the Tic model.
"""

from tic import TicModel, TicTokenizer, TicGame


def example_1_load_model():
    """Load the model and tokenizer."""
    print("=" * 60)
    print("Example 1: Loading Model")
    print("=" * 60)

    print("\nLoading Tic-800B...")
    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer.from_pretrained("tic-ai/tic-800b")

    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


def example_2_single_move():
    """Get optimal move for a single board state."""
    print("\n" + "=" * 60)
    print("Example 2: Single Move Analysis")
    print("=" * 60)

    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer.from_pretrained("tic-ai/tic-800b")
    game = TicGame(model, tokenizer)

    # Initial board
    board = [' ' for _ in range(9)]
    print("\nInitial board (X to play):")
    game.print_board(board)

    # Get optimal move
    move, metadata = game.get_optimal_move(board, 'X')
    print(f"\nOptimal move for X: position {move}")
    print(f"Move probabilities: {[f'{p:.3f}' for p in metadata['move_probs']]}")
    print(f"Win probability: {metadata['win_probability']:.4f}")


def example_3_played_game():
    """Show a game played optimally."""
    print("\n" + "=" * 60)
    print("Example 3: Optimal Game")
    print("=" * 60)

    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer.from_pretrained("tic-ai/tic-800b")
    game = TicGame(model, tokenizer)

    result = game.play_optimal_game()

    print(f"\nGame played in {result['total_moves']} moves")
    print(f"Result: {'Draw' if result['is_draw'] else f'{result['winner']} wins'}")

    print("\nMove history:")
    for move_info in result['history']:
        print(f"  Move {move_info['move_num']}: {move_info['player']} -> position {move_info['position']}")


def example_4_human_vs_ai():
    """Analyze human move vs optimal."""
    print("\n" + "=" * 60)
    print("Example 4: Move Analysis")
    print("=" * 60)

    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer.from_pretrained("tic-ai/tic-800b")
    game = TicGame(model, tokenizer)

    # Human plays corner
    board = ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    print("\nHuman plays corner (X):")
    game.print_board(board)

    # Model response
    move, metadata = game.get_optimal_move(board, 'O')
    print(f"\nModel (O) optimal response: position {move}")
    print(f"Win probability: {metadata['win_probability']:.4f}")


def example_5_board_state():
    """Analyze mid-game board state."""
    print("\n" + "=" * 60)
    print("Example 5: Mid-Game Analysis")
    print("=" * 60)

    model = TicModel.from_pretrained("tic-ai/tic-800b")
    tokenizer = TicTokenizer.from_pretrained("tic-ai/tic-800b")
    game = TicGame(model, tokenizer)

    # Mid-game board
    board = ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' ']
    print("\nMid-game board (X to play):")
    game.print_board(board)

    # Analyze
    valid_moves = game.get_valid_moves(board)
    print(f"Valid moves: {valid_moves}")

    move, metadata = game.get_optimal_move(board, 'X')
    print(f"\nOptimal move: position {move}")
    print(f"Win probability: {metadata['win_probability']:.4f}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# Tic Model Examples")
    print("#" * 60)

    example_1_load_model()
    example_2_single_move()
    example_3_played_game()
    example_4_human_vs_ai()
    example_5_board_state()

    print("\n" + "#" * 60)
    print("# All examples complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
