#!/usr/bin/env python3
"""
This module implements FP-growth [1] frequent pattern mining algorithm.

Its entry point is frequent_itemsets() function below.

[1]: Han, J., Pei, J., Yin, Y., Mao, R.
     Mining Frequent Patterns without Candidate Generation: A
     Frequent-Pattern Tree Approach. 2004.
     https://www.cs.sfu.ca/~jpei/publications/dami03_fpgrowth.pdf

"""
from collections import defaultdict, OrderedDict
from itertools import combinations, chain
from numbers import Integral
from copy import copy

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



def afopt_all(Dp, p, min_support):
    # Used for ordering transactions' items for "optimally" "compressed" tree
    item_support = defaultdict(int)
    for transaction in Dp:
        for item in transaction:
            item_support[item] += 1
    # Only ever consider items that have min_support
    frequent_items = {item
                      for item, support in item_support.items()
                      if support >= min_support}
    sort_index = {item: i
                  for i, item in
                      enumerate(sorted(frequent_items,
                                       key=item_support.__getitem__))}.__getitem__

    Di = defaultdict(list)
    for transaction in Dp:
        transaction = sorted(frequent_items.intersection(transaction),
                                   key=sort_index)
        if transaction:
            Di[transaction.pop(0)].append(transaction)

    #~ print(Di)

    global count_itemsets
    for item in sorted(frequent_items, key=sort_index):
        s = p.union({item})
        #~ print(s, item_support[item])
        yield s
        count_itemsets += 1
        yield from afopt_all(Di[item], s, min_support)
        # "Push right"
        for transaction in Di.pop(item):
            if transaction:
                Di[transaction.pop(0)].append(transaction)

count_itemsets = 0



def _powerset(lst):
    """
    >>> list(_powerset([1, 2, 3]))
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return chain.from_iterable(combinations(lst, r)
                               for r in range(1, len(lst) + 1))





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
    tree = _fp_tree(db, min_support)
    print(__fp_tree_count_nodes(tree), __fp_tree_max_height(tree), tree)
    #~ print('tree', tree)
    itemsets = list(_fp_growth(tree, set(), min_support))
    #~ print(len(itemsets), itemsets)
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
    X = np.array([[0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, 1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1]])
    np.random.seed([0])
    #~ X = np.random.random((300, 15)) > .6
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

    X = preprocess(table)

    from queue import deque
    itemsets = list(afopt_all(list(row.nonzero()[0] for row in X), set(), 43))
    print(count_itemsets)
    print(len(itemsets))
    print(len({frozenset(i) for i in itemsets}))
    asdf

    import time
    start = time.clock()
    itemsets = frequent_itemsets(X, .1)
    print(time.clock() - start)


    X = np.ones((10, 10))
    itemsets = frequent_itemsets(X, .1)

    X = np.array([
        list(map(int, list('1111100000'))),
        list(map(int, list('0000011111'))),
        list(map(int, list('1100011000'))),
        list(map(int, list('1000010000'))),
        list(map(int, list('0100001000'))),
    ]).T
    assert len(frequent_itemsets(X, .01)) == 17


    # Test that all itemsets indeed have the calculated support
    for itemset, support in itemsets:
        x = X[:, list(itemset)]
        s = x[x.sum(1) >= len(itemset)].sum(0)
        u = np.unique(s)
        assert len(u) == 1 and u[0] == support, (support, s, itemset)
