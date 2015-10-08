#!/usr/bin/env python3
"""
This module implements Apriori algorithm [1] with candidate pruning and
dataset reduction optimizations [2] and alternative prefix tree implementation,
pass bundling, and item lookup optimizations [3].

[1]: Agrawal, R., Srikant, R., Fast Algorithms for Mining Association Rules.
     1994. http://rakesh.agrawal-family.com/papers/vldb94apriori.pdf

[2]: Orlando, S., Perego, R., Palmerini, P.
     Enhancing the apriori algorithm for frequent set counting. 2001.
     http://www.dais.unive.it/~orlando/PAPERS/dawak01.pdf

[3]: Mueller, A., Fast sequential and parallel algorithms for association
     rule mining: A comparison. 1998.
     http://drum.lib.umd.edu/bitstream/handle/1903/437/CS-TR-3515.pdf
"""

import numpy as np
from collections import OrderedDict
from itertools import chain, groupby, zip_longest
from bisect import bisect_left
from numbers import Integral

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


from pprint import pprint, pformat

AND_ = np.logical_and

def _pairwise(iterable):
    a, b = iter(iterable), iter(iterable)
    next(b, None)
    return zip_longest(a, b)

def _set_siblings(edges):
    for i1, i2 in _pairwise(edges):
        i1.sibling = i2
        if i2:
            i2.prev = i1


def large_1_itemsets(db, minsupport):
    """Return large 1-itemsets. Source [1], Figure 1, line 1."""
    return np.where(db.sum(0) >= minsupport)[0]


def large_2_itemsets(db, minsupport, L1):
    """
    A direct-count alternative implementation for the second iteration (L2)
    of the Apriori algorithm, from [2] §4.2.
    """
    return [(i1, i2)
            for i, i1 in enumerate(L1[:-1])
            for i2 in L1[i + 1:]
            if AND_(db[:, i1], db[:, i2]).sum() >= minsupport]


class _OrderedDict(OrderedDict):
    def pop(self, key):
        node = super().pop(key)
        # Keep siblings connected
        if node.sibling: node.sibling.prev = node.prev
        if node.prev: node.prev.sibling = node.sibling
        return node

    __repr__ = dict.__repr__


class Node:
    """
    Node of the prefix tree, described in [3] § 3.1, corresponds to
    edge in [3], Figure 3.4. This prefix tree is almost like a trie, except
    that multiple paths "activate" for each transaction (each active path
    denotes one transaction subset).
    """
    def __init__(self, item, parent, is_frequent=False, **kwargs):
        assert item is None or isinstance(item, Integral)
        assert parent is None or isinstance(parent, Node)
        self.item = item
        self.parent = parent
        self.count = 0
        self.children = _OrderedDict()  # {item: Node} mapping
        self.dead_children = {}  # dead branches ([3] §3.1.2)
        self.sibling = None
        self.prev = None
        self.is_frequent = is_frequent
        self.was_expanded = False
        self.__dict__.update(kwargs)

    def __repr__(self):
        if self.dead_children:
            return '({}, {}, D{})'.format(self.count, self.children, self.dead_children)
        else:
            return '({}, {})'.format(self.count, self.children)

    #~ def __str__(self):
        #~ return '{}: ({}, {})'.format(
            #~ self.item,
            #~ self.count,
            #~ self.children,
        #~ )
    #~ __repr__ = __str__


def generate_ptree(L1, L2):
    tree = Node(None, None, was_expanded=True)
    print(L1)
    tree.children.update((i, Node(i, tree, True)) for i in L1)
    _set_siblings(tree.children.values())  # TODO FIXME: this could be fastened
    #~ for e in tree.values(): e.was_expanded = True
    if L2:
        for i, j in L2:
            itree = tree.children[i]
            itree.children[j] = Node(j, itree, True)

        for i in set(i[0] for i in L2):
            _set_siblings(tree.children[i].children.values())
    return tree


def apriori_gen(root):

    def siblings(node):
        """ Siblings iterator """
        while node.sibling:
            yield node.sibling
            node = node.sibling

    # Candidates holds iterables of candidates, so a separate count is required
    candidates, candidates_count = [], 0
    #~ ipdb.set_trace()
    stack = [root]
    while stack:
        node = stack.pop()
        while node:
            # If node has children, make sure to process them
            if node.children:
                if node.sibling:
                    stack.append(node.sibling)
                node = next(iter(node.children.values()))
                continue

            # Generate a single level once
            if not node.was_expanded:
                children = node.children
                children.update((sibling.item, Node(sibling.item, node))
                                for sibling in siblings(node))
                candidates.append(children.values())
                candidates_count += len(children)
                node.was_expanded = True
                _set_siblings(children.values())

            # Remove dead branches, [3] §3.1.2, Figure 3.3.
            if not node.children and node.parent:
                popped = node.parent.children.pop(node.item)
                assert node is popped
                node.parent.dead_children[node.item] = node

            # Continue with the next node on this level
            node = node.sibling

    return candidates, candidates_count


#~ tree = generate_ptree([1, 2, 3], [])
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ apriori_gen(tree)
#~ print(tree)
#~ sys.exit(1)

def _index(a, x):
    i = bisect_left(a, x)
    if i == len(a): return None
    if a[i] == x: return True
    return False


def count_itemsets(root, transaction, item_counters):
    assert isinstance(root, Node)
    stack = [(0, root)]
    while stack:
        offset, node = stack.pop()
        if node.sibling:
            stack.append((offset, node.sibling))
        for i in range(offset, len(transaction)):
            item = transaction[i]
            child = node.children.get(item)
            if not child:
                continue
            item_counters[item] += 1
            if not child.is_frequent:
                child.count += 1
            if child.children:
                stack.append((offset + 1, child))
    return

    #~ node = root
    #~ children = [root]
#~
    #~ for item in transaction:
        #~ child = node.children.get(item)
        #~ if not child:
            #~ continue
        #~ if not node.is_frequent:  # TODO: purge?
            #~ node.count += 1
        #~ item_counters[item] += 1
        #~ continue
#~
    #~ stack = [next(iter(root.values()))]
    #~ parent_item = None
    #~ while stack:
        #~ edge = stack.pop()
        #~ while edge:
            #~ idx = _index(transaction, edge.item)
            #~ if idx is None:
                #~ break  # Further siblings testing not necessary
            #~ if idx is False:
                #~ edge = edge.sibling
                #~ continue
            #~ item_counters[parent_item] += 1
            #~ if edge.children:
                #~ parent_item = edge.item
                #~ stack.append(edge.sibling)
                #~ cur_edge, edge = edge, next(iter(edge.children.values()))
                #~ if cur_edge.is_frequent:
                    #~ continue
            #~ assert edge.item in transaction
            #~ edge.count += 1
            #~ edge = edge.sibling

def prune_infrequent(candidates, minsupport):
    frequent = []
    for node in candidates:
        if node.count >= minsupport:
            node.is_frequent = True
            frequent.append(node)
        else:
            popped = node.parent.children.pop(node.item)
            assert node is popped
    print(len(frequent), 'are frequent')
    return frequent

def extract_frequent(root):
    itemsets, branch = set(), set()
    stack = [(branch, root)]
    #~ ipdb.set_trace()
    while stack:
        branch, node = stack.pop()
        for i, child in chain(node.children.items(), node.dead_children.items()):
            branch.add(i)
            itemsets.add(frozenset(branch))
            stack.append((branch.copy(), child))
            branch.remove(i)
        #~ for i, edge in chain(tree.items(), tree.dead_branches.items()):
            #~ branch.append(i)
            #~ itemsets.append(tuple(branch))
            #~ stack.append((itemsets[-1], edge.children))
            #~ branch.pop()
    return sorted(itemsets, key=lambda x: (len(x), sorted(x)))



def _apriori(db, minsupport):
    n_transactions, n_items = db.shape

    # First iteration
    L1 = large_1_itemsets(db, minsupport)

    # Maximum number of items in the database to direct-count the second iteration
    # of the apriori algorithm below. If there are fewer than this many items in
    # the database, use a M**2/2 array of counts, like in the first iteration.
    DIRECT_COUNT_1_MAX_ITEMS = 5000
    # Second iteration, optionally fast-tracked
    if len(L1) < DIRECT_COUNT_1_MAX_ITEMS:
        k, L2 = 3, large_2_itemsets(db, minsupport, L1)
    # Effectively skip the 2nd iteration and just run the main loop
    else: k, L2 = 2, None

    # Make db such that can be pruned
    db = [[list(t.nonzero()[0]), t.sum()] for t in db]

    # Initially populate the prefix tree
    #~ tree = generate_ptree(L1, L2)  #
    tree = generate_ptree(L1, [])
    if not tree.children:
        return []
    #~ print(tree)
    #~ print()

    candidates, candidates_count = apriori_gen(tree)
    # The L_k local counters for "Database local pruning" ([2] §4.1)
    item_counters = np.zeros(n_items)

    while candidates:
        print('iteration:', k, ' db:', len(db), ' candidates:', candidates_count)

        #~ ipdb.set_trace()
        for ti, (transaction, n) in enumerate(db):
            item_counters[:] = 0
            count_itemsets(tree, transaction, item_counters)
            #~ db[ti][0] = [i for i in transaction if not 0 < item_counters[i] < k - 2]
            #~ db[ti][1] = len(db[ti][0])
            for i, count in enumerate(item_counters):
                if count:
                    #~ print(k, count)
                    if count < k:
                        db[ti][0].remove(i)
                        db[ti][1] -= 1
        frequent = prune_infrequent(chain.from_iterable(candidates), minsupport)
        #~ print('pruned\n', tree, '\n')

        candidates, candidates_count = apriori_gen(tree)
        # candidates = apriori_gen(frequent)  # TODO This is how it should be!

        # Prune transactions that can't possibly support k+1-itemsets ([2] §4.1)
        db[:] = [i for i in db if i[1] >= k]
        k += 1

    #~ print(tree)
    itemsets = extract_frequent(tree)
    print(itemsets[:10], '...', itemsets[-5:])
    return itemsets



#~     apriori_gen(tree, Lk1)

    # The general (k >= 3) iteration

#~     while not Lk1.empty():
#~         Gk1 = global_counter(Lk1)  # From [2]
#~
#~         Ck = apriori_gen(Lk1)
#~         if not Ck: return
#~
#~         for t in db:
#~
#~             global_pruning(t, Gk1, k)  # From [2]
#~             if len(t) < k: continue
#~
#~             for ksubsets_t in T:
#~                 if t in Ck:
#~                     c.count += 1
#~
#~             local_pruning(t, Lk)  # From [2]
#~             if len(t) < k + 1:
#~                 db.remove(t)
#~
#~         Lk = {c for c in Ck if c.count > minsup}



def apriori(db, min_support):
    minsup = max(1, int(min_support * db.shape[0]))
    return _apriori(db, minsup)


def frequent_itemsets(X, min_support=.2):
    if not (0 < min_support < 1):
        raise ValueError('min_support must be a fraction')
    X = np.asarray(X, dtype=np.int8)
    if np.all(np.unique(X) != [0, 1]):
        raise TypeError('X must be an array of boolean values')
    itemsets = apriori(X, min_support)


class OneHotEncoder:
    @staticmethod
    def encode(table):
        encoded, mapping = [], {}
        import ipdb; ipdb.set_trace()
        for i, var in enumerate(table.domain.attributes):
            if not var.is_discrete: continue
            for j, val in enumerate(var.values):
                mapping[len(encoded)] = i, j
                encoded.append(table.X[:, i] == j)
        print(encoded)
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
    print(encoded)
    return encoded


if __name__ == '__main__':
    sys.excepthook = uncaught_excepthook
    """ Only works if cythonized with --embed """
    # Example from [1] § 2.2, Figure 3.
    X = np.array([[0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, 1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1]])
    np.random.seed([0])
    #~ X = np.random.random((500, 50)) > .5
    #~ X = np.array([
         #~ [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
         #~ [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
         #~ [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
         #~ [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    #~ ])

    from Orange.data import Table
    from Orange.preprocess import Discretize
    table = Table('voting')
    print(table)
    table = Discretize()(table)
    print(table)
    X = preprocess(table)


    print(X.astype(int))
    import time
    start = time.clock()
    itemsets = frequent_itemsets(X, .4)
    print(time.clock() - start)
