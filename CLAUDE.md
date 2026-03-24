# BlokusAI

## Project Overview
Blokus board game engine + AlphaZero-style AI training pipeline.
4-player Blokus on a 20x20 board with self-play reinforcement learning.

## Tech Stack
- Python 3.12, PyTorch (CPU), NumPy, matplotlib
- pytest for testing

## Project Structure
- `blokus/` - Game engine (pieces, board, game logic, display, players)
- `mcts/` - Monte Carlo Tree Search (random rollout + neural PUCT)
- `nn/` - Neural network (ResNet policy + value heads, action encoding)
- `training/` - Self-play training pipeline (config, logger, trainer)
- `scripts/` - Plotting and analysis tools
- `tests/` - Test suite (68 tests)

## Development Phases
1. Game engine (done)
2. MCTS with random rollouts (done)
3. Neural network + NN-guided MCTS + training pipeline (done)
4. Run training and analyse results
5. Optimization and iteration

## Commands
- `source .venv/bin/activate` - Activate venv
- `python -m pytest tests/ -v` - Run tests
- `python -m blokus.demo` - Random game demo
- `python -m blokus.demo --mcts 200` - MCTS(200) vs 3 random
- `python -m training.pipeline --help` - Training pipeline options
- `python scripts/plot_training.py` - Generate training plots from logs/
