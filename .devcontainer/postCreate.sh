#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[docs,tests,extra]
python -m pip install "gymnasium[box2d]" "gymnasium[mujoco]" "ray" "minigrid" "wandb"
python -m pip install git+https://github.com/google-research/realworldrl_suite.git

if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi
