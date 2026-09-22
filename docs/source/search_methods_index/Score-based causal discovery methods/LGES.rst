.. _lges:

LGES: Less Greedy Equivalence Search
=====================================

Algorithm Introduction
--------------------------------------

Less Greedy Equivalence Search (LGES) [1]_ is a score-based causal discovery
algorithm that learns a CPDAG (a Markov equivalence class) from observational data.
It is a modification of Greedy Equivalence Search (GES): the **backward (delete) phase
is identical to GES**, while the **forward (insert) phase is pruned**. For a non-adjacent
pair ``(x, y)``, LGES first checks whether adding the single edge ``x -> y`` improves
``y``'s local score. If it does not, the score indicates ``x`` and ``y`` are conditionally
independent given ``y``'s parents, so **all** ``Insert(x, y, *)`` operators for that pair
are discarded. This prunes the candidate set, making LGES faster than GES and, in finite
samples, often more accurate, while remaining asymptotically correct.

Two forward-phase strategies from the paper are selectable via ``insert_strategy``:

- **SafeInsert** (``'safe'``, default): discard ``Insert(x, y, *)`` when the single-edge
  insertion ``x -> y`` strictly decreases ``y``'s local score. This variant is proven to
  return a member of the true Markov equivalence class in the large-sample limit.
- **ConservativeInsert** (``'conservative'``): discard ``Insert(x, y, *)`` when *any* valid
  insertion for the pair strictly decreases the score. Stronger pruning, weaker theoretical
  guarantee.

This implementation is **clean-room from the paper (arXiv:2506.22331) only**; it reuses
causal-learn's BIC local score, Insert/Delete operators, and PDAG/DAG converters, and adds
no new dependency beyond numpy/scipy. The paper's optional prior-knowledge and
interventional (I-Orient) extensions are out of scope.


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ScoreBased.LGES import lges

    # Basic usage with default BIC score (SafeInsert)
    Record = lges(X)

    # ConservativeInsert (stronger forward pruning)
    Record = lges(X, insert_strategy='conservative')

    # With custom BIC penalty (lambda_value) and node names
    Record = lges(X, score_func='local_score_BIC',
                  parameters={'lambda_value': 1.0},
                  node_names=['X1', 'X2', 'X3'])

    # Visualization using pydot
    from causallearn.utils.GraphUtils import GraphUtils
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import io

    pyd = GraphUtils.to_pydot(Record['G'])
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()


Parameters
-------------------
**X**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features. May be ``None`` if ``cov`` and ``n`` are provided (BIC only).

**score_func**: The score function to use. Default: ``'local_score_BIC'``. The same set of score
functions as GES is supported (``'local_score_BIC'``, ``'local_score_BIC_from_cov'``, ``'local_score_BDeu'``,
``'local_score_CV_general'``, ``'local_score_marginal_general'``, ``'local_score_CV_multi'``,
``'local_score_marginal_multi'``).

**maxP**: Allowed maximum number of parents when searching the graph. Default: None (uses the number of features).

**parameters**: Additional parameters for the score function. Default: None.
              - parameters['lambda_value']: Penalty hyperparameter for BIC score. Default: 0.5.

**node_names**: list of str or None. Custom names for graph nodes. Default: None (uses ``X1, X2, ...``).

**cov**, **n**: numpy.ndarray and int. Covariance matrix and sample size, as an alternative to ``X`` for BIC scores.

**lambda_value**: float. BIC penalty coefficient (``lambda_value * (|Pa| + 1) * log(n)`` per node). Default: 0.5.

**insert_strategy**: str. Forward-phase pruning strategy, ``'safe'`` (SafeInsert, default) or
``'conservative'`` (ConservativeInsert).


Returns
-------------------
- **Record['G']**: GeneralGraph. The learned causal graph (CPDAG), where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicate i --> j; Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.

- **Record['score']**: float. The score of the learned graph.

.. [1] Ejaz, A., & Bareinboim, E. (2025). Less Greedy Equivalence Search. Advances in Neural Information Processing Systems (NeurIPS 2025). arXiv:2506.22331.
