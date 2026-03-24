# BlokusAI

## Project Overview
Blokus board game engine + AlphaZero-style AI training pipeline.
4-player Blokus on a 20x20 board with self-play reinforcement learning.

## Tech Stack
- Python 3.12, PyTorch (CPU), NumPy
- pytest for testing

## Project Structure
- `blokus/` - Game engine (pieces, board, game logic, display)
- `tests/` - Test suite
- Future: `mcts/`, `nn/`, `training/`

## Development Phases
1. Game engine ← current
2. MCTS with random rollouts
3. Neural network (CNN policy + value heads)
4. Self-play training loop
5. Visualization and analysis

## Commands
- `source .venv/bin/activate` - Activate venv
- `python -m pytest tests/ -v` - Run tests
- `python -m blokus.demo` - Run demo game with random players
