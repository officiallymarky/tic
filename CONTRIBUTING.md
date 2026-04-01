# Contributing to Tic

Thank you for your interest in contributing to Tic!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/tic-ai/tic.git
cd tic

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black src/

# Run linter
ruff check src/
```

## Testing

All new features must include tests. Run the full test suite:

```bash
pytest tests/ -v --cov=src/tic
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with clear messages
4. Push to your branch
5. Open a Pull Request

## Bug Reports

Please report bugs via GitHub Issues with:
- Clear description of the issue
- Minimal reproduction case
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## Questions?

Feel free to open a Discussion on GitHub or email research@tic-ai.com
