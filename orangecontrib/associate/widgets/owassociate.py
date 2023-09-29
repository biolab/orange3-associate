import re
from itertools import chain

import numpy as np
from scipy.sparse import issparse

from AnyQt.QtCore import Qt, QSortFilterProxyModel
from AnyQt.QtGui import QStandardItem, QStandardItemModel
from AnyQt.QtWidgets import QTableView, QGridLayout, QLabel, QApplication

from Orange.data import Table, ContinuousVariable, StringVariable, Domain
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot, \
    association_rules, rules_stats


class OWAssociate(widget.OWWidget):
    name = 'Association Rules'
    description = 'Induce association rules from data.'
    icon = 'icons/AssociationRules.svg'
    priority = 20

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        matching_data = Output("Matching Data", Table)
        rules = Output("Rules", Table)

    class Error(widget.OWWidget.Error):
        need_discrete_data = widget.Msg("Need some discrete data to work with.")
        no_disc_features = widget.Msg("Discrete features required but data has none.")

    class Warning(widget.OWWidget.Warning):
        cont_attrs = widget.Msg("Data has continuous attributes which will be skipped.")
        err_reg_expression = widget.Msg("Error in {} regular expression: {}")
        filter_no_match = widget.Msg("No rules match the filter.")

    minSupport = settings.Setting(1)
    minConfidence = settings.Setting(90)
    maxRules = settings.Setting(10000)
    filterSearch = settings.Setting(True)
    classify = settings.Setting(False)
    autoFind = settings.Setting(False)
    autoSend = settings.Setting(True)
    filterKeywordsAntecedent = settings.Setting('')
    filterAntecedentMin = settings.Setting(1)
    filterAntecedentMax = settings.Setting(1000)
    filterKeywordsConsequent = settings.Setting('')
    filterConsequentMin = settings.Setting(1)
    filterConsequentMax = settings.Setting(1000)

    header = [("Supp", "Support", "Support"),
              ("Conf", "Confidence", "Confidence (support / antecedent "
                                    "support)"),
              ("Covr", "Coverage", "Coverage (antecedent support / number of examples)"),
              ("Strg", "Strength", "Strength (consequent support / antecedent support)"),
              ("Lift", "Lift",
               "Lift (number of examples * confidence / consequent support)"),
              ("Levr", "Leverage",
               "Leverage ((support * number of examples - antecedent support * consequent support) / (number of examples)²)"),
              ("Antecedent", "Antecedent", None),
              ("", "", None),
              ("Consequent", "Consequent", None)]

    support_options = \
        [.0001, .0005, .001, .005, .01, .05, .1, .5] \
        + list(chain(range(1, 10), range(10, 101, 5)))
    confidence_options = \
        list(chain(range(1, 10), range(10, 20, 2), range(20, 101, 5)))

    def __init__(self):
        self.data = None
        self.output = None
        self.table_rules = None
        self.onehot_mapping = {}
        self._is_running = False
        self._antecedentMatch = self._consequentMatch = lambda x: True
        self.proxy_model = self.ProxyModel(self)

        owwidget = self

        class TableView(QTableView):
            def selectionChanged(self, selected, deselected):
                nonlocal owwidget
                super().selectionChanged(selected, deselected)

                mapping = owwidget.onehot_mapping
                if not mapping:
                    return

                where = np.where
                X, Y = owwidget.X, owwidget.data.Y
                instances = set()
                selected_rows = self.selectionModel().selectedRows(0)
                for model_index in selected_rows:
                    itemset, class_item = model_index.data(owwidget.ROW_DATA_ROLE)
                    cols, vals = zip(*(mapping[i] for i in itemset))
                    if issparse(X):
                        matching = (len(cols) == np.bincount((X[:, cols] != 0).indices,
                                                             minlength=X.shape[0])).nonzero()[0]
                    else:
                        matching = set(where((X[:, cols] == vals).all(axis=1))[0])
                    if class_item:
                        matching &= set(where(Y == mapping[class_item][1])[0])
                    instances.update(matching)

                owwidget.nSelectedExamples = len(instances)
                owwidget.nSelectedRules = len(selected_rows)
                owwidget.output = owwidget.data[sorted(instances)] or None
                owwidget.commit()

        table = self.table = TableView(
            self,
            showGrid=False,
            sortingEnabled=True,
            alternatingRowColors=True,
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.ExtendedSelection,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            verticalScrollMode=QTableView.ScrollPerPixel,
            editTriggers=QTableView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(table.verticalHeader().minimumSectionSize())
        table.horizontalHeader().setStretchLastSection(True)
        proxy_model = self.proxy_model
        proxy_model.setSourceModel(QStandardItemModel(table))
        table.setModel(proxy_model)

        self.mainArea.layout().addWidget(table)

        box = gui.widgetBox(self.controlArea, "Info")
        self.nRules = self.nFilteredRules = self.nSelectedExamples = self.nSelectedRules = 0
        gui.label(box, self, "Rules: %(nRules)s (shown %(nFilteredRules)s)")

        grid = QGridLayout()
        gui.widgetBox(self.controlArea, 'Find association rules',
                      orientation=grid)
        grid.addWidget(QLabel("Min. supp.:"), 0, 0)
        grid.addWidget(
            gui.valueSlider(None, self, 'minSupport', width=100,
                            values=self.support_options),
            0, 1
        )
        grid.addWidget(gui.label(None, self, "%(minSupport)g %%"), 0, 2)

        grid.addWidget(QLabel("Min. conf.:"), 1, 0)
        grid.addWidget(
            gui.valueSlider(None, self, 'minConfidence', width=100,
                            values=self.confidence_options),
            1, 1
        )
        grid.addWidget(gui.label(None, self, "%(minConfidence)g %%"), 1, 2)

        def set_mr_label():
            lab_maxrules.setText(f"{self.maxRules // 1000}k")

        grid.addWidget(QLabel("Max. rules:"), 2, 0)
        grid.addWidget(
            gui.valueSlider(None, self, 'maxRules', width=100,
                            values=list(range(10000, 100001, 10000)),
                            callback=set_mr_label),
            2, 1
        )
        lab_maxrules = QLabel()
        grid.addWidget(lab_maxrules, 2, 2)
        set_mr_label()


        self.cb_classify = gui.checkBox(
            None, self, 'classify', label='Induce only classification rules')
        grid.addWidget(self.cb_classify, 3, 0, 1, 3)

        grid.addWidget(
            gui.checkBox(
                box, self, 'filterSearch',
                label='Restrict search by below filters',
                tooltip='If checked, the rules are filtered according '
                        'to these filter conditions already in the search '
                        'phase. \nIf unchecked, the only filters applied '
                        'during search are the ones above, '
                        'and the generated rules \nare filtered afterwards '
                        'only for display, i.e. only the matching association '
                        'rules are shown.'),
            4, 0, 1, 3)

        self.button = gui.button(
            None, self, 'Find Rules', callback=self.find_rules,
            default=False, autoDefault=False)
        grid.addWidget(self.button, 5, 0, 1, 3)

        box = gui.widgetBox(self.controlArea, 'Filter by Antecedent')
        gui.lineEdit(box, self, 'filterKeywordsAntecedent', 'Contains:',
                     callback=self.filter_change, orientation='horizontal',
                     tooltip='A comma or space-separated list of regular '
                             'expressions.')
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.spin(hbox, self, 'filterAntecedentMin', 1, 998, label='Items, min:',
                 callback=self.filter_change)
        gui.spin(hbox, self, 'filterAntecedentMax', 1, 999, label='max:',
                 callback=self.filter_change)
        gui.rubber(hbox)

        box = gui.widgetBox(self.controlArea, 'Filter by Consequent')
        gui.lineEdit(box, self, 'filterKeywordsConsequent', 'Contains:',
                     callback=self.filter_change, orientation='horizontal',
                     tooltip='A comma or space-separated list of regular '
                             'expressions.')
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.spin(hbox, self, 'filterConsequentMin', 1, 998, label='Items, min:',
                 callback=self.filter_change)
        gui.spin(hbox, self, 'filterConsequentMax', 1, 999, label='max:',
                 callback=self.filter_change)
        gui.rubber(hbox)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, 'autoSend', 'Send selection',
                        auto_label="Send selection")

        self.filter_change()

    def send_report(self):
        self.report_items([("Number of rules", self.nRules),
                           ("Selected rules", self.nSelectedRules),
                           ("Covered examples", self.nSelectedExamples),
                           ])
        self.report_table('Rules', self.table)

    def commit(self):
        self.Outputs.matching_data.send(self.output)

    def isSizeMatch(self, antecedentSize, consequentSize):
        return (self.filterAntecedentMin <= antecedentSize <= self.filterAntecedentMax and
                self.filterConsequentMin <= consequentSize <= self.filterConsequentMax)
    def isRegexMatch(self, antecedentStr, consequentStr):
        return (self._antecedentMatch(antecedentStr) and
                self._consequentMatch(consequentStr))

    def filter_change(self):
        self.Warning.err_reg_expression.clear()
        self.Warning.filter_no_match.clear()
        try:
            self._antecedentMatch = re.compile(
                '|'.join(i.strip()
                         for i in re.split('(,|\s)+',
                                           self.filterKeywordsAntecedent.strip())
                         if i.strip()), re.IGNORECASE).search
        except Exception as e:
            self.Warning.err_reg_expression('antecedent', e.args[0])
            self._antecedentMatch = lambda x: True
        try:
            self._consequentMatch = re.compile(
                '|'.join(i.strip()
                         for i in re.split('(,|\s)+',
                                           self.filterKeywordsConsequent.strip())
                         if i.strip()), re.IGNORECASE).search
        except Exception as e:
            self.Warning.err_reg_expression('consequent', e.args[0])
            self._consequentMatch = lambda x: True
        self.proxy_model.invalidateFilter()
        self.nFilteredRules = self.proxy_model.rowCount()

    ITEM_DATA_ROLE = Qt.UserRole + 1
    ROW_DATA_ROLE = ITEM_DATA_ROLE + 1

    class StandardItem(QStandardItem):
        def __init__(self, text, data=None):
            super().__init__(text)
            if data:
                self.setData(data, OWAssociate.ITEM_DATA_ROLE)

    class NumericItem(StandardItem):
        def __init__(self, data):
            super().__init__('{:2.3f}'.format(data), data)
            self.setToolTip(str(data))
            self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    class ProxyModel(QSortFilterProxyModel):
        ANTECEDENT_IND = 6
        CONSEQUENT_IND = 8

        def __init__(self, *args, **kwargs):
            super().__init__(*args, sortRole=OWAssociate.ITEM_DATA_ROLE, **kwargs)

        def filterAcceptsRow(self, row, parent):
            widget = self.parent()
            antecedent = self.sourceModel().index(row, self.ANTECEDENT_IND, parent)
            consequent = self.sourceModel().index(row, self.CONSEQUENT_IND, parent)
            return bool(widget.isSizeMatch(antecedent.data(OWAssociate.ITEM_DATA_ROLE),
                                           consequent.data(OWAssociate.ITEM_DATA_ROLE)) and
                        widget.isRegexMatch(antecedent.data(),
                                            consequent.data()))

        def get_data(self):
            col_names = [name for _, name, _ in self.parent().header]
            numeric = [ContinuousVariable(name) for name in col_names[:self.ANTECEDENT_IND]]
            string = [StringVariable(col_names[i]) for i in (self.ANTECEDENT_IND, self.CONSEQUENT_IND)]
            domain = Domain(numeric, metas=string)
            data = []
            for row in range(self.rowCount()):
                data_inst = []
                for column in range(self.columnCount()):
                    index = self.index(row, column)
                    data_inst.append(self.data(index))
                data.append(data_inst)
            if not data:
                return None
            data = np.array(data)
            table = Table.from_numpy(domain, X=data[:, :len(numeric)].astype(float),
                                     metas=data[:, [self.ANTECEDENT_IND,
                                                    self.CONSEQUENT_IND]])
            table.name = "association rules"
            return table

    def find_rules(self):
        if self.data is None or not len(self.data):
            return
        if self._is_running:
            self._is_running = False
            return

        self.button.setText('Cancel')

        self._is_running = True
        data = self.data
        model = self.table.model().sourceModel()
        model.clear()

        n_examples = len(data)
        NumericItem = self.NumericItem
        StandardItem = self.StandardItem
        filterSearch = self.filterSearch
        itemsetMin = self.filterAntecedentMin + self.filterConsequentMin
        itemsetMax = self.filterAntecedentMax + self.filterConsequentMax
        isSizeMatch = self.isSizeMatch
        isRegexMatch = self.isRegexMatch

        X, mapping = OneHot.encode(data, self.classify)
        self.Error.need_discrete_data.clear()
        if X is None:
            self.Error.need_discrete_data()

        self.onehot_mapping = mapping
        ITEM_FMT = '{}' if issparse(data.X) else '{}={}'
        names = {item: ('{}={}' if var is data.domain.class_var else ITEM_FMT).format(var.name, val)
                 for item, var, val in OneHot.decode(mapping, data, mapping)}
        # Items that consequent must include if classifying
        class_items = {item
                       for item, var, val in OneHot.decode(mapping, data, mapping)
                       if var is data.domain.class_var} if self.classify else set()
        assert bool(class_items) == bool(self.classify)

        for col, (label, _, tooltip) in enumerate(self.header):
            item = QStandardItem(label)
            item.setToolTip(tooltip)
            model.setHorizontalHeaderItem(col, item)

        # Find itemsets
        nRules = 0
        itemsets = {}
        ARROW_ITEM = StandardItem('→')
        ARROW_ITEM.setTextAlignment(Qt.AlignCenter)
        with self.progressBar(self.maxRules + 1) as progress:
            for itemset, support in frequent_itemsets(X, self.minSupport / 100):
                itemsets[itemset] = support

                if class_items and not class_items & itemset:
                    continue

                # Filter itemset by joined filters before descending into it
                itemset_str = ' '.join(names[i] for i in itemset)
                if (filterSearch and
                    (len(itemset) < itemsetMin or
                     itemsetMax < len(itemset) or
                     not isRegexMatch(itemset_str, itemset_str))):
                    continue

                for rule in association_rules(itemsets,
                                              self.minConfidence / 100,
                                              itemset):
                    left, right, support, confidence = rule

                    if class_items and right - class_items:
                        continue
                    if filterSearch and not isSizeMatch(len(left), len(right)):
                        continue
                    left_str =  ', '.join(names[i] for i in sorted(left))
                    right_str = ', '.join(names[i] for i in sorted(right))
                    if filterSearch and not isRegexMatch(left_str, right_str):
                        continue

                    # All filters matched, calculate stats and add table row
                    _, _, _, _, coverage, strength, lift, leverage = next(
                        rules_stats((rule,), itemsets, n_examples))

                    support_item = NumericItem(support / n_examples)
                    # Set row data on first column
                    support_item.setData((itemset - class_items,
                                          class_items and (class_items & itemset).pop()),
                                         self.ROW_DATA_ROLE)
                    left_item = StandardItem(left_str, len(left))
                    left_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    model.appendRow([support_item,
                                     NumericItem(confidence),
                                     NumericItem(coverage),
                                     NumericItem(strength),
                                     NumericItem(lift),
                                     NumericItem(leverage),
                                     left_item,
                                     ARROW_ITEM.clone(),
                                     StandardItem(right_str, len(right))])
                    nRules += 1
                    progress.advance()

                    if not self._is_running or nRules >= self.maxRules:
                        break

                QApplication.instance().processEvents()

                if not self._is_running or nRules >= self.maxRules:
                    break

        # Populate the TableView
        table = self.table
        table.setHidden(True)
        table.setSortingEnabled(False)

        for i in range(model.columnCount()):
            table.resizeColumnToContents(i)
        table.setSortingEnabled(True)
        table.setHidden(False)
        self.table_rules = self.table.model().get_data()
        self.Outputs.rules.send(self.table_rules)

        self.button.setText('Find Rules')

        self.nRules = nRules
        self.nFilteredRules = model.rowCount()  # TODO: continue; also add in owitemsets
        if not self.nFilteredRules:
            self.Warning.filter_no_match()
        self.nSelectedRules = 0
        self.nSelectedExamples = 0
        self._is_running = False

    @Inputs.data
    def set_data(self, data):
        self.data = data
        is_error = False
        if data is not None:
            if not data.domain.has_discrete_class:
                self.cb_classify.setDisabled(True)
                self.classify = False
            self.X = data.X
            self.Warning.cont_attrs.clear()
            self.Error.no_disc_features.clear()
            self.button.setDisabled(False)
            if issparse(data.X):
                self.X = data.X.tocsc()
            else:
                if not data.domain.has_discrete_attributes():
                    self.Error.no_disc_features()
                    is_error = True
                    self.button.setDisabled(True)
                elif data.domain.has_continuous_attributes():
                    self.Warning.cont_attrs()
        else:
            self.output = None
            self.table_rules = None
            self.commit()
        if self.autoFind and not is_error:
            self.find_rules()

    @classmethod
    def migrate_settings(cls, settings, _):
        def closest(s, x):
            return min(s, key=lambda t: abs(t - x))

        settings["minSupport"] = closest(cls.support_options,
                                         settings.get("minSupport", 1))
        settings["minConfidence"] = closest(cls.confidence_options,
                                            settings.get("minConfidence", 10))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWAssociate).run(Table("zoo"))
