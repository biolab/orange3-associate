import unittest
from unittest.mock import patch

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.associate.widgets.owitemsets import OWItemsets


class TestOWItemsets(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWItemsets)

    @patch("orangecontrib.associate.widgets.owitemsets.OWItemsets.support_options",
           [0.1, 0.2, 0.5, 1, 5, 20, 30, 50])
    def test_migrate_supp_conf(self):
        settings = {'minSupport': 0.3}
        OWItemsets.migrate_settings(settings, 1)
        self.assertEqual(settings["minSupport"], 0.2)


if __name__ == "__main__":
    unittest.main()

