"""A small, educational GTO solver for Kuhn Poker.

The solver uses vanilla Counterfactual Regret Minimization (CFR). Kuhn Poker
has only three cards, so the entire game tree can be trained quickly without
external packages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import permutations


CARDS = ("J", "Q", "K")
CARD_VALUE = {"J": 0, "Q": 1, "K": 2}
ACTIONS = ("p", "b")  # pass/check/fold or bet/call, depending on the history
TERMINAL_HISTORIES = {"pp", "bp", "bb", "pbp", "pbb"}


@dataclass
class InformationSet:
    """What a player knows: their card and the public betting history."""

    key: str
    regret_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def strategy(self, reach_probability: float) -> list[float]:
        """Return the regret-matched strategy for the current iteration."""
        positive_regrets = [max(regret, 0.0) for regret in self.regret_sum]
        normalizer = sum(positive_regrets)
        if normalizer > 0:
            strategy = [regret / normalizer for regret in positive_regrets]
        else:
            strategy = [0.5, 0.5]

        for index, probability in enumerate(strategy):
            self.strategy_sum[index] += reach_probability * probability
        return strategy

    def average_strategy(self) -> list[float]:
        """Return the strategy averaged across every training iteration."""
        normalizer = sum(self.strategy_sum)
        if normalizer == 0:
            return [0.5, 0.5]
        return [value / normalizer for value in self.strategy_sum]


class KuhnPokerSolver:
    """Train approximate Nash-equilibrium strategies for Kuhn Poker."""

    def __init__(self) -> None:
        self.information_sets: dict[str, InformationSet] = {}
        self.average_game_value = 0.0

    def train(self, iterations: int = 25_000) -> float:
        """Train across all six possible two-card deals."""
        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        total_utility = 0.0
        deals = list(permutations(CARDS, 2))

        for _ in range(iterations):
            for cards in deals:
                total_utility += self._cfr(cards, "", 1.0, 1.0)

        self.average_game_value = total_utility / (iterations * len(deals))
        return self.average_game_value

    def _cfr(
        self,
        cards: tuple[str, str],
        history: str,
        reach_player_0: float,
        reach_player_1: float,
    ) -> float:
        """Run one CFR traversal and return utility for the current player."""
        if history in TERMINAL_HISTORIES:
            return terminal_utility(cards, history)

        player = len(history) % 2
        key = cards[player] + history
        info_set = self.information_sets.setdefault(key, InformationSet(key))
        own_reach = reach_player_0 if player == 0 else reach_player_1
        opponent_reach = reach_player_1 if player == 0 else reach_player_0
        strategy = info_set.strategy(own_reach)

        action_utilities = [0.0, 0.0]
        node_utility = 0.0

        for action_index, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                action_utilities[action_index] = -self._cfr(
                    cards,
                    next_history,
                    reach_player_0 * strategy[action_index],
                    reach_player_1,
                )
            else:
                action_utilities[action_index] = -self._cfr(
                    cards,
                    next_history,
                    reach_player_0,
                    reach_player_1 * strategy[action_index],
                )
            node_utility += strategy[action_index] * action_utilities[action_index]

        for action_index in range(2):
            regret = action_utilities[action_index] - node_utility
            info_set.regret_sum[action_index] += opponent_reach * regret

        return node_utility

    def strategy_rows(self) -> list[dict[str, object]]:
        """Return human-readable strategy rows for display or JSON export."""
        rows = []
        for key, info_set in self.information_sets.items():
            card, history = key[0], key[1:]
            probabilities = info_set.average_strategy()
            first_action, second_action = action_labels(history)
            rows.append(
                {
                    "card": card,
                    "history": history or "start",
                    "situation": situation_label(history),
                    first_action: probabilities[0],
                    second_action: probabilities[1],
                }
            )

        return sorted(
            rows,
            key=lambda row: (
                history_order(str(row["history"])),
                CARD_VALUE[str(row["card"])],
            ),
        )


def terminal_utility(cards: tuple[str, str], history: str) -> float:
    """Return terminal utility for the player who would act next."""
    if history not in TERMINAL_HISTORIES:
        raise ValueError(f"history is not terminal: {history}")

    player = len(history) % 2
    opponent = 1 - player
    player_has_higher_card = CARD_VALUE[cards[player]] > CARD_VALUE[cards[opponent]]

    if history in {"bp", "pbp"}:  # The opponent folded.
        return 1.0
    if history == "pp":  # Both players checked; winner earns one chip.
        return 1.0 if player_has_higher_card else -1.0
    return 2.0 if player_has_higher_card else -2.0  # Bet was called.


def action_labels(history: str) -> tuple[str, str]:
    if history in {"", "p"}:
        return ("check", "bet")
    return ("fold", "call")


def situation_label(history: str) -> str:
    labels = {
        "": "Player 1 opens",
        "p": "Player 2 after a check",
        "b": "Player 2 facing a bet",
        "pb": "Player 1 facing a bet",
    }
    return labels[history]


def history_order(history: str) -> int:
    return {"start": 0, "p": 1, "b": 2, "pb": 3}[history]


def print_report(solver: KuhnPokerSolver, iterations: int) -> None:
    print("\nKUHN POKER — CFR STRATEGY REPORT")
    print("=" * 72)
    print(f"Training iterations: {iterations:,}")
    print(f"Estimated value for Player 1: {solver.average_game_value:+.4f} chips/hand")
    print("-" * 72)
    print(f"{'Card':<6}{'Situation':<28}{'Action 1':<18}{'Action 2'}")
    print("-" * 72)

    for row in solver.strategy_rows():
        history = "" if row["history"] == "start" else str(row["history"])
        first_action, second_action = action_labels(history)
        first = f"{first_action}: {row[first_action]:.1%}"
        second = f"{second_action}: {row[second_action]:.1%}"
        print(f"{row['card']:<6}{row['situation']:<28}{first:<18}{second}")

    print("=" * 72)
    print("Probabilities are approximate and improve with more iterations.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Counterfactual Regret Minimization solver for Kuhn Poker."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25_000,
        help="number of training iterations (default: 25000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the learned strategy as JSON",
    )
    args = parser.parse_args()

    solver = KuhnPokerSolver()
    solver.train(args.iterations)

    if args.json:
        print(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "player_1_game_value": solver.average_game_value,
                    "strategies": solver.strategy_rows(),
                },
                indent=2,
            )
        )
    else:
        print_report(solver, args.iterations)


if __name__ == "__main__":
    main()
