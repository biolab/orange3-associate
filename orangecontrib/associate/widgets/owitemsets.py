
import re

import numpy as np

from Orange.data import Table
from Orange.widgets import widget, gui, settings

from PyQt4.QtCore import Qt
from PyQt4.QtGui import (
    QApplication, QTreeWidget, QTreeWidgetItem,
    QItemSelection, QItemSelectionModel)

from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot


class Output:
    DATA = 'Matching data'


class OWItemsets(widget.OWWidget):
    name = 'Frequent Itemsets'
    description = 'Explore sets of items that frequently appear together.'
    icon = 'icons/FrequentItemsets.svg'
    priority = 10

    inputs = [("Data", Table, 'set_data')]
    outputs = [(Output.DATA, Table)]

    minSupport = settings.Setting(30)
    maxItemsets = settings.Setting(10000)
    filterSearch = settings.Setting(True)
    autoFind = settings.Setting(False)
    autoSend = settings.Setting(True)
    filterKeywords = settings.Setting('')
    filterMinItems = settings.Setting(1)
    filterMaxItems = settings.Setting(10000)

    UserAdviceMessages = [
        widget.Message('Itemset are listed in item-sorted order, i.e. '
                       'an itemset containing A and B is only listed once, as '
                       'A > B (and not also B > A).',
                       'itemsets-order', widget.Message.Warning),
        widget.Message('To select all the itemsets that are descendants of '
                       '(include) some item X (i.e. the whole subtree), you '
                       'can fold the subtree at that item and then select it.',
                       'itemsets-order', widget.Message.Information)
    ]

    def __init__(self):
        self.tree = QTreeWidget(self.mainArea,
                                columnCount=2,
                                allColumnsShowFocus=True,
                                alternatingRowColors=True,
                                selectionMode=QTreeWidget.ExtendedSelection,
                                uniformRowHeights=True)
        self.tree.setHeaderLabels(["Itemsets", "Support", "%"])
        self.tree.header().setStretchLastSection(True)
        self.tree.itemSelectionChanged.connect(self.selectionChanged)
        self.mainArea.layout().addWidget(self.tree)

        box = gui.widgetBox(self.controlArea, "Info")
        self.nItemsets = self.nSelectedExamples = self.nSelectedItemsets = ''
        gui.label(box, self, "Number of itemsets: %(nItemsets)s")
        gui.label(box, self, "Selected itemsets: %(nSelectedItemsets)s")
        gui.label(box, self, "Selected examples: %(nSelectedExamples)s")
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.button(hbox, self, "Expand all", callback=self.tree.expandAll)
        gui.button(hbox, self, "Collapse all", callback=self.tree.collapseAll)

        box = gui.widgetBox(self.controlArea, 'Find itemsets')
        gui.hSlider(box, self, 'minSupport', minValue=1, maxValue=100,
                    label='Minimal support:', labelFormat="%d%%",
                    callback=lambda: self.find_itemsets())
        gui.hSlider(box, self, 'maxItemsets', minValue=10000, maxValue=100000, step=10000,
                    label='Max. number of itemsets:', labelFormat="%d",
                    callback=lambda: self.find_itemsets())
        gui.checkBox(box, self, 'filterSearch',
                     label='Apply below filters in search',
                     tooltip='If checked, the itemsets are filtered according '
                             'to below filter conditions already in the search '
                             'phase. \nIf unchecked, the only filters applied '
                             'during search are the ones above, '
                             'and the itemsets are \nfiltered afterwards only for '
                             'display, i.e. only the matching itemsets are shown.')
        gui.auto_commit(box, self, 'autoFind', 'Find itemsets', commit=self.find_itemsets)

        box = gui.widgetBox(self.controlArea, 'Filter itemsets')
        gui.lineEdit(box, self, 'filterKeywords', 'Contains:',
                     callback=self.filter_change, orientation='horizontal',
                     tooltip='A comma or space-separated list of regular '
                             'expressions.')
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.spin(hbox, self, 'filterMinItems', 1, 998, label='Min. items:',
                 callback=self.filter_change)
        gui.spin(hbox, self, 'filterMaxItems', 2, 999, label='Max. items:',
                 callback=self.filter_change)
        gui.rubber(hbox)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, 'autoSend', 'Send selection')

        self.filter_change()

    def sendReport(self):
        self.reportSettings("Itemset statistics",
                            [("Number of itemsets", self.nItemsets),
                             ("Selected itemsets", self.nSelectedItemsets),
                             ("Covered examples", self.nSelectedExamples),
                             ])
        self.reportSection("Itemsets")
        self.reportRaw(OWReport.reportTree(self.tree))

    ITEM_DATA_ROLE = Qt.UserRole + 1

    def selectionChanged(self):
        X = self.data.X
        mapping = self.onehot_mapping
        instances = set()
        where = np.where

        def whole_subtree(node):
            yield node
            for i in range(node.childCount()):
                yield from whole_subtree(node.child(i))

        def itemset(node):
            while node:
                yield node.data(0, self.ITEM_DATA_ROLE)
                node = node.parent()

        def selection_ranges(node):
            n_children = node.childCount()
            if n_children:
                yield (self.tree.indexFromItem(node.child(0)),
                       self.tree.indexFromItem(node.child(n_children - 1)))
            for i in range(n_children):
                yield from selection_ranges(node.child(i))

        nSelectedItemsets = 0
        item_selection = QItemSelection()
        for node in self.tree.selectedItems():
            nodes = (node,) if node.isExpanded() else whole_subtree(node)
            if not node.isExpanded():
                for srange in selection_ranges(node):
                    item_selection.select(*srange)
            for node in nodes:
                nSelectedItemsets += 1
                cols, vals = zip(*(mapping[i] for i in itemset(node)))
                instances.update(where((X[:, cols] == vals).all(axis=1))[0])
        self.tree.itemSelectionChanged.disconnect(self.selectionChanged)
        self.tree.selectionModel().select(item_selection,
                                          QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.tree.itemSelectionChanged.connect(self.selectionChanged)

        self.nSelectedExamples = len(instances)
        self.nSelectedItemsets = nSelectedItemsets
        self.output = self.data[sorted(instances)] or None
        self.commit()

    def commit(self):
        self.send(Output.DATA, self.output)

    def filter_change(self):
        isRegexMatch = self.isRegexMatch = re.compile(
            '|'.join(i.strip()
                     for i in re.split('(,|\s)+', self.filterKeywords.strip())
                     if i.strip())).search

        def hide(node, depth, has_kw):
            if not has_kw:
                has_kw = isRegexMatch(node.text(0))
            hidden = (sum(hide(node.child(i), depth + 1, has_kw)
                          for i in range(node.childCount())) == node.childCount()
                      if node.childCount() else
                      (not has_kw or
                       not self.filterMinItems <= depth <= self.filterMaxItems))
            node.setHidden(hidden)
            return hidden

        hide(self.tree.invisibleRootItem(), 0, False)

    class TreeWidgetItem(QTreeWidgetItem):
        def data(self, column, role):
            """Construct lazy tooltips"""
            if role != Qt.ToolTipRole:
                return super().data(column, role)
            tooltip = []
            while self:
                tooltip.append(self.text(0))
                self = self.parent()
            return ' '.join(reversed(tooltip))

    def find_itemsets(self):
        if self.data is None: return
        data = self.data
        self.tree.clear()
        self.tree.setUpdatesEnabled(False)
        self.tree.blockSignals(True)

        class ItemDict(dict):
            def __init__(self, item):
                self.item = item

        top = ItemDict(self.tree.invisibleRootItem())
        X, mapping = OneHot.encode(data)
        self.onehot_mapping = mapping
        names = {item: '{}={}'.format(var.name, val)
                 for item, var, val in OneHot.decode(mapping.keys(), data, mapping)}
        nItemsets = 0

        filterSearch = self.filterSearch
        filterMinItems, filterMaxItems = self.filterMinItems, self.filterMaxItems
        isRegexMatch = self.isRegexMatch

        # Find itemsets and populate the TreeView
        progress = gui.ProgressBar(self, self.maxItemsets + 1)
        for itemset, support in frequent_itemsets(X, self.minSupport / 100):

            if filterSearch and not filterMinItems <= len(itemset) <= filterMaxItems:
                continue

            parent = top
            first_new_item = None
            itemset_matches_filter = False

            for item in sorted(itemset):
                name = names[item]

                if filterSearch and not itemset_matches_filter:
                    itemset_matches_filter = isRegexMatch(name)

                child = parent.get(name)
                if child is None:
                    wi = self.TreeWidgetItem(parent.item, [name, str(support), '{:.1f}'.format(100 * support / len(data))])
                    wi.setData(0, self.ITEM_DATA_ROLE, item)
                    child = parent[name] = ItemDict(wi)

                    if first_new_item is None:
                        first_new_item = (parent, name)
                parent = child

            if filterSearch and not itemset_matches_filter:
                parent, name = first_new_item
                parent.item.removeChild(parent[name].item)
                del parent[name].item
                del parent[name]
            else:
                nItemsets += 1
                progress.advance()
            if nItemsets >= self.maxItemsets:
                break

        if not filterSearch:
            self.filter_change()
        self.nItemsets = nItemsets
        self.nSelectedItemsets = 0
        self.nSelectedExamples = 0
        self.tree.expandAll()
        for i in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
        self.tree.setUpdatesEnabled(True)
        self.tree.blockSignals(False)
        progress.finish()

    def set_data(self, data):
        self.data = data
        if data is not None:
            self.warning(0, 'Data has continuous attributes which will be skipped.'
                            if data.domain.has_continuous_attributes() else None)
        if self.autoFind:
            self.find_itemsets()


if __name__ == "__main__":
    a = QApplication([])
    ow = OWItemsets()

    data = Table("zoo")
    ow.set_data(data)

    ow.show()
    a.exec()
