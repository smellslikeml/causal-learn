"""Less Greedy Equivalence Search (LGES).

LGES is a score-based causal-discovery algorithm that learns a CPDAG (Markov
equivalence class) from observational data. It is a modification of Greedy
Equivalence Search (GES): the backward phase is identical to GES, while the
forward phase prunes edge insertions between a non-adjacent pair whenever the
score indicates the two variables are conditionally independent. This prunes
the search, making LGES both faster than GES and, in finite samples, often more
accurate, while remaining asymptotically correct.

Reimplemented from the paper only:

    Ejaz, A., & Bareinboim, E. (2025). "Less Greedy Equivalence Search."
    Advances in Neural Information Processing Systems (NeurIPS 2025),
    arXiv:2506.22331, https://arxiv.org/abs/2506.22331.

This implementation reuses causal-learn's own GES machinery (the BIC
``LocalScoreClass``, the Insert/Delete operators and validity tests in
``GESUtils``, and ``pdag2dag`` / ``dag2cpdag``); it does not use the authors'
reference code. Two forward-phase strategies from the paper are exposed via
``insert_strategy``: ``"safe"`` (SafeInsert, proven correct, default) and
``"conservative"`` (ConservativeInsert, stronger pruning). The paper's optional
prior-knowledge and interventional (I-Orient) extensions are out of scope.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy import ndarray

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag


def _local_score(X, node, parents_key, record_local_score, score_func, parameters):
    """Local score of ``node`` given ``parents_key`` (a sorted tuple of parent
    indices), memoized in ``record_local_score`` using the same cache keying as
    ``insert_changed_score_fast``."""
    key = (node, parents_key)
    if key in record_local_score:
        return record_local_score[key]
    s = feval([score_func, X, node, list(parents_key), parameters])
    record_local_score[key] = s
    return s


def lges(
    X: ndarray = None,
    score_func: str = "local_score_BIC",
    maxP: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
    node_names: Union[List[str], None] = None,
    cov: Optional[ndarray] = None,
    n: Optional[int] = None,
    lambda_value: Optional[float] = None,
    insert_strategy: str = "safe",
) -> Dict[str, Any]:
    """Perform causal discovery with Less Greedy Equivalence Search (LGES).

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). Ignored when
        ``cov`` and ``n`` are provided (BIC-from-covariance path).
    score_func : name of the local score function; the same options as ``ges``
        (default ``'local_score_BIC'``).
    maxP : maximum number of parents allowed per node (default: n_features).
    parameters : score-function parameters (e.g. ``{'lambda_value': 0.5}`` for BIC).
    node_names : optional list of variable names.
    cov, n : sample covariance matrix and sample size, for the BIC-from-cov path.
    lambda_value : BIC sparsity penalty; injected into ``parameters`` if given.
    insert_strategy : forward-phase pruning strategy, ``'safe'`` (SafeInsert,
        default, proven correct) or ``'conservative'`` (ConservativeInsert).

    Returns
    -------
    Record : dict with the same contract as ``ges``:

        Record['G'] : learned CPDAG as a ``GeneralGraph`` (``graph[j,i]=1`` and
            ``graph[i,j]=-1`` means ``i --> j``; ``graph[i,j]=graph[j,i]=-1``
            means ``i --- j``).
        Record['update1'] / Record['update2'] : forward Insert / backward Delete steps.
        Record['G_step1'] / Record['G_step2'] : graph after each forward / backward step.
        Record['score'] : score of the learned CPDAG.

    References
    ----------
    .. [1] Ejaz, A., & Bareinboim, E. (2025). "Less Greedy Equivalence Search."
           NeurIPS 2025, arXiv preprint arXiv:2506.22331,
           https://arxiv.org/abs/2506.22331.
    """
    if insert_strategy not in ("safe", "conservative"):
        raise ValueError(
            f"`insert_strategy` must be 'safe' or 'conservative', got {insert_strategy!r}."
        )

    if cov is not None and n is not None:
        if X is not None:
            warnings.warn("Both X and cov/n provided. Using cov and n, ignoring X.")
        X = None
    elif X is None:
        raise ValueError("Either X or (cov, n) must be provided.")

    if X is not None and X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    n_features = cov.shape[0] if cov is not None else X.shape[1]

    if lambda_value is not None:
        if parameters is None:
            parameters = {}
        parameters["lambda_value"] = lambda_value

    if score_func == "local_score_CV_general":
        if X is None:
            raise ValueError("local_score_CV_general requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01}
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters
        )
    elif score_func == "local_score_marginal_general":
        if X is None:
            raise ValueError("local_score_marginal_general requires raw data X, not cov/n.")
        parameters = {}
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters
        )
    elif score_func == "local_score_CV_multi":
        if X is None:
            raise ValueError("local_score_CV_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01, "dlabel": {}}
            for i in range(n_features):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        N = len(parameters["dlabel"])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters
        )
    elif score_func == "local_score_marginal_multi":
        if X is None:
            raise ValueError("local_score_marginal_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(n_features):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        N = len(parameters["dlabel"])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters
        )
    elif score_func == "local_score_BIC" or score_func == "local_score_BIC_from_cov":
        if maxP is None:
            maxP = n_features
        N = n_features
        if parameters is None:
            parameters = {}
        if "lambda_value" not in parameters:
            parameters["lambda_value"] = 0.5
        if cov is not None:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters,
                cov=cov, n=n,
            )
        else:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
            )
    elif score_func == "local_score_BDeu":
        if X is None:
            raise ValueError("local_score_BDeu requires raw data X, not cov/n.")
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )
    else:
        raise Exception("Unknown function!")
    score_func = localScoreClass

    if node_names is None:
        node_names = [("X%d" % (i + 1)) for i in range(N)]
    nodes = [GraphNode(name) for name in node_names]

    G = GeneralGraph(nodes)
    score = score_g(X, G, score_func, parameters)  # initialize the score

    G = pdag2dag(G)
    G = dag2cpdag(G)

    # -------------------------------------------------------------------------
    # forward greedy search  (LGES: prune per SafeInsert / ConservativeInsert)
    # -------------------------------------------------------------------------
    record_local_score = {}
    score_new = score
    count1 = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []
    while True:
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        max_chscore = -1e7
        max_desc = []

        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == 0
                    and G.graph[j, i] == 0
                    and i != j
                    and len(_pa[j]) <= maxP
                ):
                    # LGES SafeInsert gate: skip the ordered pair i -> j if adding
                    # the single edge i -> j does not improve j's local score
                    # (evidence i _||_ j | Pa_j), per arXiv:2506.22331 Prop. 1-2.
                    if insert_strategy == "safe":
                        paj_key = tuple(sorted(_pa[j]))
                        paj_i_key = tuple(sorted(_pa[j] | {i}))
                        delta_single = _local_score(
                            X, j, paj_i_key, record_local_score, score_func, parameters
                        ) - _local_score(
                            X, j, paj_key, record_local_score, score_func, parameters
                        )
                        if delta_single < 0:
                            continue

                    NA = _nbrs[j] & _adj[i]
                    T0 = sorted(_nbrs[j] - _adj[i])
                    sub = Combinatorial(T0)
                    S = np.zeros(len(sub))
                    pair_pruned = False  # ConservativeInsert flag
                    pair_best_chscore = -1e7
                    pair_best_desc = []
                    for k in range(len(sub)):
                        if S[k] < 2:
                            T_set = set(sub[k])
                            NAT = NA | T_set
                            V1 = check_clique_fast(G, NAT)
                            if V1:
                                if not S[k]:
                                    V2 = insert_vc2_fast(j, i, NAT, _semi)
                                else:
                                    V2 = 1
                                if V2:
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = (
                                        insert_changed_score_fast(
                                            X, i, j, sub[k],
                                            NA, _pa[j],
                                            record_local_score,
                                            score_func,
                                            parameters,
                                        )
                                    )
                                    # ConservativeInsert: a valid insert that
                                    # decreases the score prunes the whole pair.
                                    if insert_strategy == "conservative" and chscore < 0:
                                        pair_pruned = True
                                    if chscore > pair_best_chscore:
                                        pair_best_chscore = chscore
                                        pair_best_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2

                    if insert_strategy == "conservative" and pair_pruned:
                        continue
                    if len(pair_best_desc) != 0 and pair_best_chscore > max_chscore:
                        max_chscore = pair_best_chscore
                        max_desc = pair_best_desc

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0:
                break
            G = insert(G, max_desc[0], max_desc[1], max_desc[2])
            update1.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)
        else:
            score_new = score
            break

    # -------------------------------------------------------------------------
    # backward greedy search  (identical to GES)
    # -------------------------------------------------------------------------
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []
    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        max_chscore = -1e7
        max_desc = []

        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (j in _nbrs[i]) or (i in _pa[j]):
                    NA = _nbrs[j] & _adj[i]
                    H0 = sorted(NA)
                    sub = Combinatorial(H0)
                    S = np.ones(len(sub))
                    for k in range(len(sub)):
                        if S[k] == 1:
                            H_set = set(sub[k])
                            V = check_clique_fast(G, NA - H_set)
                            if V:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2
                        else:
                            V = 1

                        if V:
                            chscore, desc, record_local_score = (
                                delete_changed_score_fast(
                                    X, i, j, sub[k],
                                    NA, _pa[j],
                                    record_local_score,
                                    score_func,
                                    parameters,
                                )
                            )
                            if chscore > max_chscore:
                                max_chscore = chscore
                                max_desc = desc

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0:
                break
            G = delete(G, max_desc[0], max_desc[1], max_desc[2])
            update2.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    Record = {
        "update1": update1,
        "update2": update2,
        "G_step1": G_step1,
        "G_step2": G_step2,
        "G": G,
        "score": score,
    }
    return Record
