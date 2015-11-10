#!/usr/bin/env python3
"""
This module implements FP-growth [1] frequent pattern mining algorithm with
bucketing optimization [2] for conditional databases of few items.

The entry points are frequent_itemsets(), gen_assoc_rules(), and
rules_stats() functions below.


[1]: J. Han, J. Pei, Y. Yin, R. Mao.
     Mining Frequent Patterns without Candidate Generation: A
     Frequent-Pattern Tree Approach. 2004.
     https://www.cs.sfu.ca/~jpei/publications/dami03_fpgrowth.pdf

[2]: R. Agrawal, C. Aggarwal, V. Prasad.
     Depth first generation of long patterns. 2000.
     http://www.cs.tau.ac.il/~fiat/dmsem03/Depth%20First%20Generation%20of%20Long%20Patterns%20-%202000.pdf

[3]: R. Agrawal, et al.
     Fast Discovery of Association Rules. 1996.
     http://cs-people.bu.edu/evimaria/cs565/advances.pdf
"""

# TODO: Consider FPClose from "Efficiently using prefix-trees in mining frequent itemsets"

from collections import defaultdict, Iterator
from itertools import combinations, chain
from functools import reduce

import numpy as np
from scipy.sparse import issparse, spmatrix


_FP_TREE_EMPTY = (None, [])
_BUCKETING_FEW_ITEMS = 10


class _Node(dict):
    def __init__(self, item=None, parent=None, count=None):
        self.item = item
        self.parent = parent
        self.count = count


def _bucketing_count(db, frequent_items, min_support):
    """
    Bucket counting (bucketing) optimization for databases where few items
    are frequent ([2] § 5).
    """
    # Forward and inverse mapping of frequent_items to [0, n_items)
    inv_map = dict(enumerate(frequent_items)).__getitem__
    fwd_map = {v: k for k, v in inv_map.__self__.items()}.__getitem__
    # Project transactions
    k = len(frequent_items)
    buckets = [0] * 2**k
    for count, transaction in db:
        set_bits = (fwd_map(i) for i in frequent_items.intersection(transaction))
        tid = reduce(lambda a, b: a | 1 << b, set_bits, 0)
        buckets[tid] += count
    # Aggregate bucketing counts ([2], Figure 5)
    for i in range(0, k):
        i = 2**i
        for j in range(2**k):
            if j & i == 0:
                buckets[j] += buckets[j + i]
    # Announce results
    buckets = enumerate(buckets)
    next(buckets)  # Skip 000...0
    for tid, count in buckets:
        if count >= min_support:
            yield frozenset(inv_map(i) for i, b in enumerate(reversed(bin(tid))) if b == '1'), count


# Replace above bucketing count with the one from C module
from orangecontrib.associate._fpgrowth import bucketing_count as _bucketing_count, \
                                              BUCKETING_FEW_ITEMS as _BUCKETING_FEW_ITEMS


def _fp_tree_insert(item, T, node_links, count):
    """ Insert item into _Node-tree T and return the new node """
    node = T.get(item)
    if node is None:
        node = T[item] = _Node(item, T, count)
        node_links[item].append(node)
    else:  # Node for this item already in T, just inc its count
        node.count += count
    return node


def _fp_tree(db, min_support):
    """
    FP-tree construction ([1] § 2.1, Algorithm 1).

    If frequent items in db are determined to be less than threshold,
    "bucketing" [2] is used instead.

    Returns
    -------
    tuple
        (FP-tree, None) or (None, list of frequent itemsets with support)
    """
    if not isinstance(db, list): db = list(db)

    if not db:
        return _FP_TREE_EMPTY

    # Used to count item support so it can be reported when generating itemsets
    item_support = defaultdict(int)
    # Used for ordering transactions' items for "optimally" "compressed" tree
    node_support = defaultdict(int)
    for count, transaction in db:
        for item in transaction:
            item_support[item] += count
            node_support[item] += 1
    # Only ever consider items that have min_support
    frequent_items = {item
                      for item, support in item_support.items()
                      if support >= min_support}

    # Short-circuit, if possible
    n_items = len(frequent_items)
    if 0 == n_items:
        return _FP_TREE_EMPTY
    if 1 == n_items:
        item = frequent_items.pop()
        return None, ((frozenset({item}), item_support[item]),)
    if n_items <= _BUCKETING_FEW_ITEMS:
        return None, ((frozenset(itemset), support)
                      for itemset, support in _bucketing_count(db, frequent_items, min_support))

    # "The items [...] should be ordered in the frequency descending order of
    # node occurrence of each item instead of its support" ([1], p. 12, bottom)
    sort_index = {item: i
                  for i, item in
                      enumerate(sorted(frequent_items,
                                       key=node_support.__getitem__,
                                       reverse=True))}.__getitem__
    # Only retain frequent items and sort them
    db = ((count, sorted(frequent_items.intersection(transaction),
                         key=sort_index))
          for count, transaction in db)

    root = _Node()
    node_links = defaultdict(list)
    for count, transaction in db:
        T = root
        for item in transaction:
            T = _fp_tree_insert(item, T, node_links, count)
    # Sorted support-descending (in reverse because popping from the back for efficiency)
    root.node_links = sorted(node_links.items(), key=lambda i: -sort_index(i[0]))
    return root, None


def _powerset(lst):
    """
    >>> list(_powerset([1, 2, 3]))
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return chain.from_iterable(combinations(lst, r)
                               for r in range(1, len(lst) + 1))


def _single_prefix_path(root):
    """ Return (single-prefix path, rest of tree with new root) """
    path = []
    tree = root
    node_links = root.node_links
    while len(tree) == 1:
        tree = next(iter(tree.values()))
        path.append((tree.item, tree.count))
        node_links.pop()
    tree.parent, tree.item, tree.node_links = None, None, node_links
    return path, tree


def _prefix_paths(tree, nodes):
    """ Generate all paths of tree leading to all item nodes """
    for node in nodes:
        path = []
        support = node.count
        node = node.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        if path:
            yield support, path


def _freq_patterns_single(P, alpha, min_support):
    """ Yield subsets of P as (frequent itemset, support) """
    for itemset in _powerset(P):
        yield alpha.union(i[0] for i in itemset), itemset[-1][1]


def _freq_patterns_multi(Q, alpha, min_support):
    """ Mine multi-path FP-tree """
    for item, nodes in reversed(Q.node_links):
        support = sum(n.count for n in nodes)
        beta = alpha.union({item})
        yield beta, support
        tree, got_itemsets = _fp_tree(_prefix_paths(Q, nodes), min_support)
        if got_itemsets:
            for itemset, support in got_itemsets:
                yield beta.union(itemset), support
        elif tree is not None:
            yield from _fp_growth(tree, beta, min_support)


def _fp_growth(tree, alpha, min_support):
    """ FP-growth ([1], § 3.3, Algorithm 2). """
    # Single prefix path optimization ([1] § 3.1)
    P, Q = _single_prefix_path(tree) if len(tree) == 1 else ([], tree)
    # Return P×Q
    yield from _freq_patterns_single(P, alpha, min_support)
    for itemsetQ, supportQ in _freq_patterns_multi(Q, alpha, min_support):
        yield itemsetQ, supportQ
        for itemsetP, supportP in _freq_patterns_single(P, alpha, min_support):
            yield itemsetQ | itemsetP, supportQ


def frequent_itemsets(X, min_support=.2):
    """
    Generator yielding frequent itemsets from database X.

    Parameters
    ----------
    X : list or numpy.ndarray or scipy.sparse.spmatrix or iterator
        The database of transactions where each transaction is a collection
        of integer items. If numpy.ndarray, the items are considered to be
        indexes of non-zero columns.
    min_support : float
        Percent of minimal support for itemset to be considered frequent.
    """
    if not isinstance(X, (np.ndarray, spmatrix, list, Iterator)):
        raise TypeError('X must be (sparse) array of boolean values, or list of lists of hashable items, or iterator')
    if not 0 < min_support < 1:
        raise ValueError('min_support must be a percent fraction in [0, 1]')

    min_support *= len(X) if isinstance(X, list) else X.shape[0]
    min_support = max(1, int(min_support))

    if issparse(X):
        X = X.tolil().rows
    elif isinstance(X, np.ndarray):
        X = (t.nonzero()[0] for t in X)

    db = ((1, transaction) for transaction in X)  # 1 is initial item support
    tree, itemsets = _fp_tree(db, min_support)
    #~ if tree is not None:
        #~ print(__fp_tree_count_nodes(tree), __fp_tree_max_height(tree), tree)
    yield from (itemsets or _fp_growth(tree, frozenset(), min_support))


def _gen_assoc_rules(rule, last_item, support, min_confidence, itemsets):
    left, right = rule
    if not left: return
    confidence = support / itemsets[left]
    if confidence >= min_confidence:
        yield rule, support, confidence
        for item in left:
            if item > last_item: continue  # This ensures same rules aren't visited twice
            yield from _gen_assoc_rules(
                (left - {item}, right | {item}),
                item, support, min_confidence, itemsets)


def gen_assoc_rules(itemsets, min_confidence, itemset=None):
    """
    Generate association rules ([3] § 12.3) from dict of itemsets' supports
    (from frequent_items()). If `itemset` is provided, only generate its rules.

    Yields
    ------
    ((antecedent, consequent), support, confidence): tuple
    """
    assert (isinstance(itemsets, dict) and
            isinstance(next(iter(itemsets), frozenset()), frozenset))
    assert 0 < min_confidence <= 1
    from_itemsets = (itemset,) if itemset else sorted(itemsets, key=len, reverse=True)
    for itemset in from_itemsets:
        support = itemsets[itemset]
        for item in itemset:
            right = frozenset({item})
            yield from _gen_assoc_rules(
                (itemset - right, right),
                item, support, min_confidence, itemsets)


def rules_stats(rules, itemsets, n_examples):
    """
    Generate additional stats for rules generated by gen_assoc_rules()

    Yields
    ------
    stats: tuple
        ((antecedent, consequent), support, confidence,
         coverage, strength, lift, leverage)
    """
    assert (isinstance(itemsets, dict) and
            isinstance(next(iter(itemsets), frozenset()), frozenset))
    assert n_examples > 0
    for (left, right), support, confidence in rules:
        l_support, r_support = itemsets[left], itemsets[right]
        coverage = l_support / n_examples
        strength = r_support / l_support
        lift = n_examples * confidence / r_support
        leverage = (support*n_examples - l_support*r_support) / n_examples**2
        yield ((left, right), support, confidence,
               coverage, strength, lift, leverage)


def __fp_tree_count_nodes(tree):
    count = 1 if tree.item is not None else 0
    for t in tree.values():
        count += __fp_tree_count_nodes(t)
    return count


def __fp_tree_max_height(tree):
    if tree:
        return max((1 if tree.item is not None else 0) +
                   __fp_tree_max_height(child) for child in tree.values())
    return 1 if tree.item is not None else 0


class OneHot:
    """
    Encode discrete Orange.data.Table into a 2D array of binary attributes.
    """
    @staticmethod
    def encode(table, include_class=False):
        """
        Return a tuple of
        (bool (one hot) ndarray, {col: (variable_index, value_index)} mapping)
        """
        X, encoded, mapping = table.X, [], {}
        for i, var in enumerate(table.domain.attributes):
            if not var.is_discrete: continue
            for j, val in enumerate(var.values):
                mapping[len(encoded)] = i, j
                encoded.append(X[:, i] == j)
        if include_class and table.domain.has_discrete_class:
            i, var = len(table.domain.attributes), table.domain.class_var
            for j, val in enumerate(var.values):
                mapping[len(encoded)] = i, j
                encoded.append(table.Y == j)
        return np.column_stack(encoded), mapping

    @staticmethod
    def decode(itemset, table, mapping):
        """Yield sorted (item, variable, value) tuples (one for each item)"""
        attributes = table.domain.attributes
        for item in itemset:
            ivar, ival = mapping[item]
            var = attributes[ivar] if ivar < len(attributes) else table.domain.class_var
            yield item, var, var.values[ival]


def preprocess(table):
    if table.domain.has_continuous_attributes():
        raise ValueError('Frequent itemsets require all variables to be discrete')
    encoded, mapping = OneHot.encode(table, table.domain.has_discrete_class)
    return encoded


if __name__ == '__main__':
    np.random.seed([0])
    X = np.random.random((5, 5)) > .5

    from Orange.data import Table
    from Orange.preprocess import Discretize
    table = Table('voting')
    table = Discretize()(table)
    #~ X = preprocess(table)

    import time
    start = time.clock()
    itemsets = list(frequent_itemsets(X, .1))
    print('time', time.clock() - start)

    for i, itemset in zip(range(100), itemsets):
        print(itemset)

    itemsets = dict(itemsets)
    start = time.clock()
    for i, (rule, support, confidence) in zip(range(100), gen_assoc_rules(itemsets, .95)):
        print('{:2} {:.2f}'.format(i, confidence), rule)
    print('time', time.clock() - start)
    for i, (rule, support, confidence, coverage, strength, lift, leverage) in zip(range(100), rules_stats(gen_assoc_rules(itemsets, .95), itemsets, len(X))):
        print(('{:.2f}  '*6).format(support / len(X), confidence, coverage, strength, lift, leverage), rule)

    print(len(itemsets))
    print(len(set(itemsets)))

    #~ if len(itemsets) < 100:
        #~ print()
        #~ for itemset in itemsets:
            #~ print(itemset)

    # Test that all itemsets indeed have the calculated support
    for itemset, support in itemsets:
        x = X[:, list(itemset)]
        s = x[x.sum(1) >= len(itemset)].sum(0)
        u = np.unique(s)
        assert len(u) == 1 and u[0] == support, (support, s, itemset)
