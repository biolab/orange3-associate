#!/usr/bin/env python3
"""
This module implements FP-growth [1] frequent pattern mining algorithm with
bucketing optimization [2] for conditional databases of few items.

The entry point is frequent_itemsets() function below.

[1]: J. Han, J. Pei, Y. Yin, R. Mao.
     Mining Frequent Patterns without Candidate Generation: A
     Frequent-Pattern Tree Approach. 2004.
     https://www.cs.sfu.ca/~jpei/publications/dami03_fpgrowth.pdf

[2]: R. Agrawal, C. Aggarwal, V. Prasad.
     Depth first generation of long patterns. 2000.
     http://www.cs.tau.ac.il/~fiat/dmsem03/Depth%20First%20Generation%20of%20Long%20Patterns%20-%202000.pdf

"""
from collections import defaultdict, OrderedDict
from itertools import combinations, chain, tee
from functools import reduce

import numpy as np
from scipy.sparse import issparse, spmatrix


import sys, ipdb

def uncaught_excepthook(*args):
    sys.__excepthook__(*args)
    from pprint import pprint
    from types import BuiltinFunctionType, ModuleType
    tb = sys.last_traceback
    prev_tb = None
    while tb.tb_next: tb = tb.tb_next
    print('\nDumping locals() ...')
    pprint({k:v for k,v in tb.tb_frame.f_locals.items()
                if not k.startswith('_') and
                   not isinstance(v, (BuiltinFunctionType,
                                      type, ModuleType))})
    if sys.stdin.isatty() and (sys.stdout.isatty() or sys.stderr.isatty()):
        try:
            import ipdb as pdb  # try to import the IPython debugger
        except ImportError:
            import pdb as pdb
        print('\nStarting interactive debug prompt ...')
        pdb.pm()

    sys.exit(1)


""" TODO try sorting transactions in db length-ascending """






class Node(dict):
    def __init__(self, item=None, parent=None, count=None):
        self.item = item
        self.parent = parent
        self.count = count

    #~ def __repr__(self):
        #~ return '{}|{}'.format(self.count, super().__repr__())

    def __hash__(self): return id(self)


class NodeLinks(OrderedDict):
    def __missing__(self, key):
        value = self[key] = []
        return value

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

def __fp_tree_has_node(tree, node):
    if tree is node: return True
    return any(__fp_tree_has_node(child, node) for child in tree.values())

def __fp_tree_to_list(tree):
    lst = [tree] if tree.item is not None else []
    for child in tree.values():
        lst += __fp_tree_to_list(child)
    return lst





_FP_TREE_EMPTY = (None, [])
_BUCKETING_FEW_ITEMS = 10


def _bucketing_count(db, frequent_items, min_support):
    """
    Bucket counting (bucketing) optimization for databases where few items
    are frequent ([2] § 5).
    """
    # Forward and inverse mapping of frequent_items to [0, n_items)
    inv_map = dict(enumerate(frequent_items)).__getitem__
    fwd_map = {v: k for k, v in inv_map.__self__.items()}.__getitem__

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

    buckets = enumerate(buckets)
    next(buckets)  # Skip 000...0
    for tid, count in buckets:
        if count >= min_support:
            yield {inv_map(i) for i, b in enumerate(reversed(bin(tid))) if b == '1'}, count


def _fp_tree_insert(item, T, node_links, count):
    """ Insert item into Node-tree T and return the new node """
    node = T.get(item)
    if node is None:
        node = T[item] = Node(item, T, count)
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
        return None, [(frozenset({item}), item_support[item])]
    if n_items <= _BUCKETING_FEW_ITEMS:
        return None, _bucketing_count(db, frequent_items, min_support)

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

    root = Node()
    node_links = root.node_links = defaultdict(list)
    for count, transaction in db:
        T = root
        for item in transaction:
            T = _fp_tree_insert(item, T, node_links, count)
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
        del node_links[tree.item]
    tree.parent, tree.item, tree.node_links = None, None, node_links
    return path, tree


def _prefix_paths(tree, item):
    """ Generate all paths of tree leading to all item nodes """
    for node in tree.node_links[item]:
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
    for item, nodes in Q.node_links.items():
        support = sum(n.count for n in nodes)
        beta = alpha.union({item})
        yield beta, support
        tree, got_itemsets = _fp_tree(_prefix_paths(Q, item), min_support)
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
    if not isinstance(X, (np.ndarray, spmatrix, list)):
        raise TypeError('X must be (sparse) array of boolean values or list of lists of hashable items')
    if not (0 < min_support < 1):
        raise ValueError('min_support must be a percent fraction')

    min_support *= len(X) if isinstance(X, list) else X.shape[0]
    min_support = max(1, int(min_support))
    print('MIN SUPPORT IS', min_support)

    if issparse(X):
        X = X.tolil().rows
    elif not isinstance(X, list):
        X = (t.nonzero()[0] for t in X)

    db = ((1, transaction) for transaction in X)  # 1 == initial item support
    tree, itemsets = _fp_tree(db, min_support)
    if tree is not None:
        print(__fp_tree_count_nodes(tree), __fp_tree_max_height(tree), tree)
        #~ print('tree', tree)
        itemsets = list(_fp_growth(tree, frozenset(), min_support))
        #~ print(len(itemsets), itemsets)
    elif itemsets:
        itemsets = list(itemsets)

    print(len(itemsets))
    print(len({frozenset(i[0]) for i in itemsets}))

    #~ seen = defaultdict(int)
    #~ for items, support in sorted(itemsets):#, key=lambda i: (i[1], i[0])):
        #~ frozen = frozenset(items)
        #~ if frozen in seen:
            #~ if support != seen[frozen]:
            #~ print('{:4} {:4}'.format(seen[frozen], support), items)
        #~ seen[frozen] = support
    return itemsets


class OneHotEncoder:
    @staticmethod
    def encode(table):
        encoded, mapping = [], {}
        #~ ipdb.set_trace()
        for i, var in enumerate(table.domain.attributes):
            if not var.is_discrete: continue
            for j, val in enumerate(var.values):
                mapping[len(encoded)] = i, j
                encoded.append(table.X[:, i] == j)
        #~ print(encoded)
        return np.column_stack(encoded), mapping

    @staticmethod
    def decode(table, mapping):
        ...



def preprocess(table):
    #~ from sklearn.preprocessing import OneHotEncoder
    #~ categorical = [v.is_discrete for v in table.domain.attributes]
    #~ n_values = [len(v.values) for v in table.domain.attributes]
    #~ enc = OneHotEncoder(
        #~ n_values=n_values,
        #~ sparse=False,
        #~ categorical_features=categorical,
        #~ handle_unknown='ignore')
    if table.domain.has_continuous_attributes():
        raise ValueError('Frequent itemsets require discrete variables')
    encoded, mapping = OneHotEncoder.encode(table)
    #~ print(encoded)
    return encoded


if __name__ == '__main__':
    #~ sys.excepthook = uncaught_excepthook
    """ Only works if cythonized with --embed """
    # Example from [1] § 2.2, Figure 3.
    #~ X = np.array([[0, 1, 0, 1, 1, 0],
                  #~ [0, 0, 1, 1, 0, 1],
                  #~ [0, 1, 1, 1, 0, 1],
                  #~ [0, 0, 1, 0, 0, 1]])
    np.random.seed([0])
    X = np.random.random((50, 15)) > .5
    #~ X = np.array([
         #~ [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
         #~ [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
         #~ [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
         #~ [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    #~ ])

    from Orange.data import Table
    from Orange.preprocess import Discretize
    table = Table('voting')
    table = Discretize()(table)
#~
    #~ X = [
        #~ list('5013'),
        #~ list('5023'),
        #~ list('50123'),
        #~ list('5012'),
        #~ list('502'),
        #~ list('502'),
        #~ list('501'),
        #~ list('54'),
    #~ ]
    #~ X = [
        #~ list('502'),
        #~ list('501'),
        #~ list('54'),
        #~ list('50123'),
        #~ list('5013'),
        #~ list('502'),
        #~ list('5023'),
        #~ list('5012'),
    #~ ]
    #~ print(X)

    X = preprocess(table)

    import time
    start = time.clock()
    itemsets = frequent_itemsets(X, .1)
    print(time.clock() - start)

    #~ for itemset in itemsets:
        #~ print(itemset)


    #~ X = np.ones((10, 10))
    #~ itemsets = frequent_itemsets(X, .1)
#~
    #~ X = np.array([
        #~ list(map(int, list('1111100000'))),
        #~ list(map(int, list('0000011111'))),
        #~ list(map(int, list('1100011000'))),
        #~ list(map(int, list('1000010000'))),
        #~ list(map(int, list('0100001000'))),
    #~ ]).T
    #~ assert len(frequent_itemsets(X, .01)) == 17


    # Test that all itemsets indeed have the calculated support
    #~ for itemset, support in itemsets:
        #~ x = X[:, list(itemset)]
        #~ s = x[x.sum(1) >= len(itemset)].sum(0)
        #~ u = np.unique(s)
        #~ assert len(u) == 1 and u[0] == support, (support, s, itemset)
