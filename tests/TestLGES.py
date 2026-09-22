import sys
import unittest

import numpy as np

sys.path.append("")

# --- imports from PRE-EXISTING causal-learn modules ---
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.SHD import SHD
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.DAG2CPDAG import dag2cpdag

# --- the module under test ---
from causallearn.search.ScoreBased.LGES import lges


# ------------------------------------------------------------------------- #
# A small, known linear-Gaussian ground truth, deterministic across runs.
#   True DAG edges (indices): 0->1, 0->3, 1->2, 1->3, 2->3, 2->4, 3->4
#   The CPDAG "truth" is built by dag2cpdag (a pre-existing causal-learn
#   utility), so it is defined by the library rather than hand-transcribed.
# ------------------------------------------------------------------------- #
NUM_NODES = 5
TRUTH_DAG_EDGES = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def _build_truth():
    nodes = [GraphNode(f"X{i + 1}") for i in range(NUM_NODES)]
    dag = GeneralGraph(nodes)
    W = np.zeros((NUM_NODES, NUM_NODES))
    rng = np.random.RandomState(0)
    for (a, b) in TRUTH_DAG_EDGES:
        dag.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a -> b
        W[b, a] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])  # row = child
    truth_cpdag = dag2cpdag(dag)
    return truth_cpdag, W


def _simulate_linear_gaussian(W, n_samples, seed):
    """X = W X + E  =>  X = (I - W)^{-1} E, standard-normal exogenous noise."""
    N = W.shape[0]
    rng = np.random.RandomState(seed)
    E = rng.randn(n_samples, N)
    return E @ np.linalg.inv(np.eye(N) - W).T


class TestLGES(unittest.TestCase):

    # -- CI gate: recover a known small equivalence class (the correctness floor) --
    def test_lges_simulate_linear_gaussian_shd(self):
        print("Now start test_lges_simulate_linear_gaussian_shd ...")
        truth_cpdag, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=5000, seed=42)
        for strategy in ("safe", "conservative"):
            rec = lges(data, score_func="local_score_BIC", insert_strategy=strategy)
            shd = SHD(truth_cpdag, rec["G"]).get_shd()
            print(f"    lges(insert_strategy={strategy})\tSHD: {shd} of {len(TRUTH_DAG_EDGES)}")
            self.assertLessEqual(
                shd, 1,
                f"LGES ({strategy}) SHD={shd} > 1; the learned CPDAG differs "
                f"significantly from the ground-truth equivalence class.",
            )

    # -- LGES is asymptotically GES: same Markov equivalence class on faithful data --
    def test_lges_matches_ges(self):
        print("Now start test_lges_matches_ges ...")
        _, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=5000, seed=42)
        ges_cpdag = ges(data, score_func="local_score_BIC")["G"]
        for strategy in ("safe", "conservative"):
            lges_cpdag = lges(
                data, score_func="local_score_BIC", insert_strategy=strategy
            )["G"]
            shd_between = SHD(ges_cpdag, lges_cpdag).get_shd()
            print(f"    SHD(GES, LGES-{strategy}) = {shd_between}")
            self.assertEqual(
                shd_between, 0,
                f"LGES ({strategy}) returned a different CPDAG than GES on the "
                f"same faithful data (SHD={shd_between}); they should agree.",
            )

    def test_lges_rejects_unknown_strategy(self):
        print("Now start test_lges_rejects_unknown_strategy ...")
        _, W = _build_truth()
        data = _simulate_linear_gaussian(W, n_samples=200, seed=1)
        with self.assertRaises(ValueError):
            lges(data, score_func="local_score_BIC", insert_strategy="greedy")


if __name__ == "__main__":
    unittest.main()
