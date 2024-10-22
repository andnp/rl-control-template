#!/bin/sh

# exit script on error
set -e

# Janky way to run all examples. Only looking for the most basic errors.

## Atari Example
python src/main.py -e experiments/atari_example/Atari/DQN.json -i 0
python src/main.py -e experiments/atari_example/Atari/EQRC.json -i 0

## Continuing Example
python src/main.py -e experiments/continuing_example/Forager/EQRC.json -i 0

## Optuna MountainCar
python src/main.py -e experiments/optuna_example/MountainCar/DQN.json -i 0
python src/main.py -e experiments/optuna_example/MountainCar/EQRC.json -i 0
python experiments/optuna_example/learning_curve.py

## Replay MountainCar
python src/main.py -e experiments/replay_example/MountainCar/DQN.json -i 0
python src/main.py -e experiments/replay_example/MountainCar/EQRC.json -i 0
python experiments/replay_example/learning_curve.py


## Basic Examples

## Acrobot
python src/main.py -e experiments/example/Acrobot/DQN.json -i 0
python src/main.py -e experiments/example/Acrobot/EQRC.json -i 0
python src/main.py -e experiments/example/Acrobot/ESARSA.json -i 0
python src/main.py -e experiments/example/Acrobot/SoftmaxAC.json -i 0

## Breakout
python src/main.py -e experiments/example/Breakout/DQN.json -i 0
python src/main.py -e experiments/example/Breakout/EQRC.json -i 0
python src/main.py -e experiments/example/Breakout/PrioritizedDQN.json -i 0

## Cartpole
python src/main.py -e experiments/example/Cartpole/DQN.json -i 0
python src/main.py -e experiments/example/Cartpole/EQRC.json -i 0
python src/main.py -e experiments/example/Cartpole/ESARSA.json -i 0
python src/main.py -e experiments/example/Cartpole/SoftmaxAC.json -i 0

## MountainCar
python src/main.py -e experiments/example/MountainCar/DQN.json -i 0
python src/main.py -e experiments/example/MountainCar/EQRC.json -i 0
python src/main.py -e experiments/example/MountainCar/ESARSA.json -i 0
python src/main.py -e experiments/example/MountainCar/SoftmaxAC.json -i 0

## Learning Curve
python experiments/example/learning_curve.py