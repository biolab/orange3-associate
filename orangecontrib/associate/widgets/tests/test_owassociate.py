import os
import unittest

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.associate.widgets.owassociate import OWAssociate


class TestOWAssociate(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWAssociate)
        self.market = Table(os.path.join(os.path.dirname(__file__),
                                         'data', 'market-basket'))

    def test_rules_output(self):
        self.send_signal(self.widget.Inputs.data, self.market)
        self.widget.button.click()
        output = self.get_output(self.widget.Outputs.rules)
        self.assertTrue(output)

    def test_filter(self):
        match_filter = "milk"
        no_match_filter = "orange"

        self.send_signal(self.widget.Inputs.data, self.market)

        # no filtering
        self.widget.find_rules()
        output = self.get_output(self.widget.Outputs.rules)
        self.assertEqual(38, len(output))

        # filter by existing column
        self.widget.filterKeywordsAntecedent = match_filter
        self.widget.filter_change()
        self.widget.find_rules()
        output = self.get_output(self.widget.Outputs.rules)
        self.assertEqual(5, len(output))

        # filter by non-existing word
        self.widget.filterKeywordsAntecedent = no_match_filter
        self.widget.filter_change()
        self.widget.find_rules()
        self.assertTrue(self.widget.Warning.filter_no_match.is_shown())
        output = self.get_output(self.widget.Outputs.rules)
        self.assertIsNone(output)

        # run again with defaults
        self.widget.filterKeywordsAntecedent = ""
        self.widget.filter_change()
        self.widget.find_rules()
        self.assertFalse(self.widget.Warning.filter_no_match.is_shown())


if __name__ == "__main__":
    unittest.main()
