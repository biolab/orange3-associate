import os

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.associate.widgets.owassociate import OWAssociate


class TestOWAssociate(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWAssociate)

    def test_rules_output(self):
        market = Table(os.path.join(os.path.dirname(__file__), 'data', 'market-basket'))
        self.send_signal(self.widget.Inputs.data, market)
        self.widget.controls.autoFind.click()
        output = self.get_output(self.widget.Outputs.rules)
        self.assertTrue(output)
