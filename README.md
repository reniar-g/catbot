# Catbot

Compact OpenAI Gym-style environment demonstrating tabular Q-learning on an 8×8 grid where an agent learns to catch cats with different behaviors.

## Repository Structure

```
CatBot/
├─ images/               # sprites and assets
├─ bot.py                # train and then play the trained bot
├─ play.py               # manual play (arrow keys)
├─ training.py           # Q-learning implementation
├─ cat_env.py            # Gym environment + cat behaviors
├─ utility.py            # Utility functions
├─ requirements.txt      # Python dependencies
└─ README.md             # this file
```

## Installation

Recommended: create a project-specific virtual environment and install dependencies.

PowerShell (Windows):

```
# clone project
git clone https://github.com/reniar-g/CatBot.git
cd catbot

# create venv (from project root)
python -m venv .venv

# activate venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick run examples

```powershell
# Play manually
python play.py --cat paotsin

# Run and watch the trained bot
python bot.py --cat paotsin

# Run without opening pygame windows during training
python bot.py --cat paotsin --render -1
```

Controls (when using play.py):

- ↑ Move up
- ↓ Move down
- ← Move left
- → Move right
- Q Quit game

## Features

- Custom Gym-compatible environment (`cat_env.py`) implementing an 8×8 grid world and multiple cat behaviors
- Tabular Q-learning trainer (`training.py`) with a compact Q-table keyed by encoded agent/cat positions
- Built-in cats: `batmeow`, `mittens`, `paotsin`, `peekaboo`, `squiddyboi` and a `trainer` cat for testing

## Reward shaping

The trainer implements a dense reward scheme (computed in `training.py`) to accelerate learning. At each timestep the reward is computed from the change in Manhattan distance between agent and cat plus small per-step penalties:

- **Terminal (catch):** `+100` when the agent and cat occupy the same cell
- **Moving closer:** `+2.0 + 0.5 × (distance reduction)`
- **Moving farther:** `-2.0 - 0.5 × (distance increase)`
- **Same distance / stagnation:** `-0.5`
- **Time penalty:** `-0.01` per timestep
- **Time penalty:** `-0.01` per timestep

## Evaluation & Reproducibility

- Training is fixed to **5,000 episodes** as required by the project specification.
- During final evaluation the bot is given a maximum of **60 moves** per scenario; the provided playback in `bot.py` uses `max_steps=60` when running the final policy.
- Note: training uses a larger per-episode cap (`max_steps = 400`) to allow longer rollouts during learning; this does not change the 60-move evaluation rule.
