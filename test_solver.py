import unittest

from solver import KuhnPokerSolver, terminal_utility


class KuhnPokerSolverTests(unittest.TestCase):
    def test_showdown_payoffs(self):
        self.assertEqual(terminal_utility(("K", "J"), "pp"), 1.0)
        self.assertEqual(terminal_utility(("J", "K"), "bb"), -2.0)

    def test_fold_payoffs(self):
        self.assertEqual(terminal_utility(("J", "K"), "bp"), 1.0)
        self.assertEqual(terminal_utility(("J", "K"), "pbp"), 1.0)

    def test_average_strategies_are_probabilities(self):
        solver = KuhnPokerSolver()
        solver.train(500)
        for info_set in solver.information_sets.values():
            strategy = info_set.average_strategy()
            self.assertAlmostEqual(sum(strategy), 1.0)
            self.assertTrue(all(0.0 <= probability <= 1.0 for probability in strategy))

    def test_game_value_approaches_known_equilibrium(self):
        solver = KuhnPokerSolver()
        value = solver.train(5_000)
        self.assertAlmostEqual(value, -1 / 18, delta=0.02)


if __name__ == "__main__":
    unittest.main()
