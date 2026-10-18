# Kuhn Poker GTO Solver

An educational poker solver that learns an approximate game-theory optimal strategy using **Counterfactual Regret Minimization (CFR)**.

This project solves **Kuhn Poker**, a small poker model with three cards. It is intentionally smaller than Texas Hold'em, which makes the complete game tree understandable and allows the solver to run without external packages.

## Why this project?

Modern poker solvers use ideas from game theory to find strategies that are difficult for an opponent to exploit. CFR repeatedly plays through a game, measures the regret from actions not taken, and gradually assigns more probability to better decisions.

## Rules of Kuhn Poker

- The deck contains a Jack, Queen, and King.
- Each player receives one card; the third card is unused.
- Both players contribute one chip before the hand.
- A player may check or bet one additional chip.
- After a bet, the opponent may call or fold.
- The higher card wins at showdown.

## Run the solver

Python 3.10 or newer is recommended. No installation is required.

```bash
python3 solver.py
```

Choose a different training length:

```bash
python3 solver.py --iterations 100000
```

Export machine-readable results:

```bash
python3 solver.py --iterations 100000 --json
```

## Run the tests

```bash
python3 -m unittest -v
```

## What the output means

The program prints a probability for each action at every **information set**—a combination of the player's private card and the public betting history. A probability is used because a GTO strategy often needs to mix between actions instead of always choosing the same move.

The theoretical value of Kuhn Poker for Player 1 is `-1/18`, or about `-0.0556` chips per hand. The trained value should approach this number as the number of iterations increases.

## Project structure

```text
poker-gto-solver/
├── solver.py       # CFR algorithm and command-line report
├── test_solver.py  # unit tests
└── README.md       # project documentation
```

## Limitations

This is an educational solver, not a real-money betting tool. It does not solve Texas Hold'em, recommend wagers, connect to poker sites, or guarantee results. Full Hold'em solvers require substantially larger game abstractions and computing resources.

## Ideas for future versions

- Plot how the game value converges during training
- Add an interactive play-against-the-solver mode
- Compare CFR with CFR+ and Monte Carlo CFR
- Build a small web interface for the strategy table
