
import re
from collections import defaultdict

import numpy as np
from scipy.sparse import issparse

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotGraph
from Orange.widgets.widget import Input, Output

from AnyQt.QtCore import Qt, QSize, pyqtSignal, QRectF, QSortFilterProxyModel
from AnyQt.QtGui import (
    QApplication, QStandardItem, QStandardItemModel, QMouseEvent, QPen, QBrush, QColor)
from AnyQt.QtWidgets import QLabel, QTableView, QMainWindow, QGraphicsView, qApp

from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot, association_rules, rules_stats

import pyqtgraph as pg


class OWAssociate(widget.OWWidget):
    name = 'Association Rules'
    description = 'Induce association rules from data.'
    icon = 'icons/AssociationRules.svg'
    priority = 20

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        matching_data = Output("Matching Data", Table)

    class Error(widget.OWWidget.Error):
        need_discrete_data = widget.Msg("Need some discrete data to work with.")
        no_disc_features = widget.Msg("Discrete features required but data has none.")

    class Warning(widget.OWWidget.Warning):
        cont_attrs = widget.Msg("Data has continuous attributes which will be skipped.")
        err_reg_expression = widget.Msg("Error in {} regular expression: {}")

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

    def __init__(self):
        self.data = None
        self.output = None
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
        table.setModel(QStandardItemModel(table))
        self.mainArea.layout().addWidget(table)

        box = gui.widgetBox(self.controlArea, "Info")
        self.nRules = self.nFilteredRules = self.nSelectedExamples = self.nSelectedRules = ''
        gui.label(box, self, "Number of rules: %(nRules)s")
        gui.label(box, self, "Filtered rules: %(nFilteredRules)s")
        gui.label(box, self, "Selected rules: %(nSelectedRules)s")
        gui.label(box, self, "Selected examples: %(nSelectedExamples)s")

        box = gui.widgetBox(self.controlArea, 'Find association rules')
        gui.valueSlider(box, self, 'minSupport',
                        values=[.0001, .0005, .001, .005, .01, .05, .1, .5] + list(range(1, 101)),
                        label='Minimal support:', labelFormat="%g%%",
                        callback=lambda: self.find_rules())
        gui.hSlider(box, self, 'minConfidence', minValue=1, maxValue=100,
                    label='Minimal confidence:', labelFormat="%g%%",
                    callback=lambda: self.find_rules())
        gui.hSlider(box, self, 'maxRules', minValue=10000, maxValue=100000, step=10000,
                    label='Max. number of rules:', callback=lambda: self.find_rules())
        self.cb_classify = gui.checkBox(
            box, self, 'classify', label='Induce classification (itemset → class) rules')
        self.button = gui.auto_commit(
                box, self, 'autoFind', 'Find Rules', commit=self.find_rules,
                callback=lambda: self.autoFind and self.find_rules())

        vbox = gui.widgetBox(self.controlArea, 'Filter rules')

        ## This is disabled because it's hard to make a scatter plot with
        ## selectable spots. Options:
        ## * PyQtGraph, which doesn't support selection OOTB (+is poorly
        ##   contrived and under-documented);
        ## * Orange.widgets.visualize.ScatterPlotGraph, which comes without
        ##   any documentation or comprehensible examples of use whatsoever;
        ## * QGraphicsView, which would work, but lacks graphing features,
        ##   namely labels, axes, ticks.
        ##
        ## I don't feel like pursuing any of those right now, so I am letting
        ## it to be figured out at a later date.
        #~ button = self.scatter_button = gui.button(vbox, self, 'Scatter plot',
                                                  #~ callback=self.show_scatter,
                                                  #~ autoDefault=False)
        #~ button.setDisabled(True)
        #~ self.scatter = self.ScatterPlotWindow(self)

        box = gui.widgetBox(vbox, 'Antecedent')
        gui.lineEdit(box, self, 'filterKeywordsAntecedent', 'Contains:',
                     callback=self.filter_change, orientation='horizontal',
                     tooltip='A comma or space-separated list of regular '
                             'expressions.')
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.spin(hbox, self, 'filterAntecedentMin', 1, 998, label='Min. items:',
                 callback=self.filter_change)
        gui.spin(hbox, self, 'filterAntecedentMax', 2, 999, label='Max. items:',
                 callback=self.filter_change)
        gui.rubber(hbox)

        box = gui.widgetBox(vbox, 'Consequent')
        gui.lineEdit(box, self, 'filterKeywordsConsequent', 'Contains:',
                     callback=self.filter_change, orientation='horizontal',
                     tooltip='A comma or space-separated list of regular '
                             'expressions.')
        hbox = gui.widgetBox(box, orientation='horizontal')
        gui.spin(hbox, self, 'filterConsequentMin', 1, 998, label='Min. items:',
                 callback=self.filter_change)
        gui.spin(hbox, self, 'filterConsequentMax', 2, 999, label='Max. items:',
                 callback=self.filter_change)
        gui.checkBox(box, self, 'filterSearch',
                     label='Apply these filters in search',
                     tooltip='If checked, the rules are filtered according '
                             'to these filter conditions already in the search '
                             'phase. \nIf unchecked, the only filters applied '
                             'during search are the ones above, '
                             'and the generated rules \nare filtered afterwards '
                             'only for display, i.e. only the matching association '
                             'rules are shown.')
        gui.rubber(hbox)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, 'autoSend', 'Send selection')

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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, sortRole=OWAssociate.ITEM_DATA_ROLE, **kwargs)

        def filterAcceptsRow(self, row, parent):
            widget = self.parent()
            antecedent = self.sourceModel().index(row, 6, parent)
            consequent = self.sourceModel().index(row, 8, parent)
            return bool(widget.isSizeMatch(antecedent.data(OWAssociate.ITEM_DATA_ROLE),
                                           consequent.data(OWAssociate.ITEM_DATA_ROLE)) and
                        widget.isRegexMatch(antecedent.data(),
                                            consequent.data()))

    def find_rules(self):
        if self.data is None or not len(self.data):
            return
        if self._is_running:
            self._is_running = False
            return

        self.button.button.setText('Cancel')

        self._is_running = True
        data = self.data
        self.table.model().clear()

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

        model = QStandardItemModel(self.table)
        for col, (label, tooltip) in enumerate([("Supp", "Support"),
                                                ("Conf", "Confidence (support / antecedent support)"),
                                                ("Covr", "Coverage (antecedent support / number of examples)"),
                                                ("Strg", "Strength (consequent support / antecedent support)"),
                                                ("Lift", "Lift (number of examples * confidence / consequent support)"),
                                                ("Levr", "Leverage ((support * number of examples - antecedent support * consequent support) / (number of examples)²)"),
                                                ("Antecedent", None),
                                                ("", None),
                                                ("Consequent", None)]):
            item = QStandardItem(label)
            item.setToolTip(tooltip)
            model.setHorizontalHeaderItem(col, item)

        #~ # Aggregate rules by common (support,confidence) for scatterplot
        #~ scatter_agg = defaultdict(list)

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
                    #~ scatter_agg[(round(support / n_examples, 2), round(confidence, 2))].append((left, right))
                    nRules += 1
                    progress.advance()

                    if not self._is_running or nRules >= self.maxRules:
                        break

                qApp.processEvents()

                if not self._is_running or nRules >= self.maxRules:
                    break

        # Populate the TableView
        table = self.table
        table.setHidden(True)
        table.setSortingEnabled(False)
        proxy_model = self.proxy_model
        proxy_model.setSourceModel(model)
        table.setModel(proxy_model)
        for i in range(model.columnCount()):
            table.resizeColumnToContents(i)
        table.setSortingEnabled(True)
        table.setHidden(False)

        self.button.button.setText('Find Rules')

        self.nRules = nRules
        self.nFilteredRules = proxy_model.rowCount()  # TODO: continue; also add in owitemsets
        self.nSelectedRules = 0
        self.nSelectedExamples = 0
        self._is_running = False

        #~ self.scatter_agg = scatter_agg
        #~ self.scatter_button.setDisabled(not nRules)
        #~ if self.scatter.isVisible():
            #~ self.show_scatter()

    class ScatterPlotWindow(QMainWindow):

        class PlotWidget(pg.PlotWidget):
            _selection = []
            selectionChanged = pyqtSignal()

            def setScatter(self, scatter):
                self.scatter = scatter

            def mousePressEvent(self, event):
                self._start_pos = event.pos()
                if event.button() == Qt.LeftButton:
                    # Save the current selection and restore it on mouse{Move,Release}
                    if not event.modifiers() & Qt.ShiftModifier:
                        self._selection = []
                QGraphicsView.mousePressEvent(self, event)
            def mouseMoveEvent(self, event):
                QGraphicsView.mouseMoveEvent(self, event)

            SELECTED_PEN = QPen(QBrush(QColor('#ee3300')), .3)

            def mouseReleaseEvent(self, event):
                QGraphicsView.mouseReleaseEvent(self, event)
                if event.button() != Qt.LeftButton:
                    return
                _start_pos = self.plotItem.items[0].mapFromScene(self._start_pos)
                _end_pos = self.plotItem.items[0].mapFromScene(event.pos())
                sx, sy = _start_pos.x(), _start_pos.y()
                ex, ey = _end_pos.x(), _end_pos.y()
                if sx > ex: sx, ex = ex, sx
                if sy < ey: sy, ey = ey, sy
                data = self.scatter.data
                selected_indices = ((sx <= data['x']) &
                                    (data['x'] <= ex) &
                                    (ey <= data['y']) &
                                    (data['y'] <= sy)).nonzero()[0]
                self._selection.extend(selected_indices)

                data['pen'][selected_indices] = self.SELECTED_PEN
                self.scatter.points()
                for item in data['item'][selected_indices]:
                    item.updateItem()

                print(data['data'][selected_indices])
                # TODO: emit selectionChanged

        def __init__(self, parent):
            super().__init__(parent)

            self.setWindowTitle('Association rules scatter plot')
            scatter = self.scatter = pg.ScatterPlotItem(size=5, symbol='s', pen=pg.mkPen(None))
            plot = self.plot = self.PlotWidget(background='w',
                                               labels=dict(left='Confidence',
                                                           bottom='Support'))
            plot.setScatter(scatter)
            plot.setLimits(xMin=0, xMax=1.02, yMin=0, yMax=1.02,
                           minXRange=0.5, minYRange=0.5)
            self.setCentralWidget(plot)
            plot.addItem(scatter)
            plot.hideButtons()
            self.lastClicked = []
            scatter.sigClicked.connect(self.clicked)

        def clicked(self, plot, points):
            for p in self.lastClicked:
                p.resetPen()
            for p in points:
                p.setPen(self.plot.SELECTED_PEN)
            lastClicked = points

        def sizeHint(self, hint=QSize(400, 400)):
            return hint


    def show_scatter(self):
        spots = [dict(pos=pos,
                      data=rules,
                      brush=pg.mkColor(np.clip(1/len(rules), 0, .8)),
                      pen=pg.mkColor(np.clip(1/len(rules), .1, .8) - .1))
                 for pos, rules in self.scatter_agg.items()]
        self.scatter.scatter.setData(spots=spots)
        self.scatter.plot.setRange(xRange=(0, 1), yRange=(0, 1))
        self.scatter.show()

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
            self.commit()
        if self.autoFind and not is_error:
            self.find_rules()


if __name__ == "__main__":
    a = QApplication([])
    ow = OWAssociate()

    data = Table("zoo")
    ow.set_data(data)

    ow.show()
    a.exec()
