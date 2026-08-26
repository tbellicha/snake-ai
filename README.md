# Snake Game with DQN AI

A Snake game trained with a Deep Q-Network (PyTorch). The agent learns from a hand-crafted 11-value state vector using epsilon-greedy exploration and experience replay.

## Setup

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `/usr/bin/python3` (not `python`) to create the venv. Cursor's terminal rewrites `sys.executable` to `Cursor.AppImage`, which makes `python -m venv` fail with `ensurepip` / `SIGTRAP`.

From Cursor, always launch via `./run` (same hijack would make `python train.py` miss venv packages like torch). A normal OS terminal can `source .venv/bin/activate` and use `python` as usual.

## Train

Watch the snake play live while it learns (pygame window + live score plot):

```bash
./run train.py
```

Faster headless training (no window, no FPS cap):

```bash
./run train.py --no-render
```

Raise the render FPS if the default feels slow:

```bash
./run train.py --speed 100
```

Stop after a fixed number of games:

```bash
./run train.py --no-render --games 200
```

The best model is saved to `model/model.pth` whenever a new high score is reached.

## Play

Watch the trained agent play (greedy policy, no learning):

```bash
./run play.py
```
