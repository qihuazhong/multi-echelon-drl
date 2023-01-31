import os
import sys
from unittest import TestCase

sys.path.insert(0, os.path.abspath('../'))

import unittest
from utils.graph import parse_order_sequence, read_yaml


class TestParseOrderSequence(unittest.TestCase):
    def test_classic_beer_game(self):
        nodes = ['manufacturer', 'distributor', 'wholesaler', 'retailer']
        customers_dict = {'is_external_supplier': ['manufacturer'],
                          'manufacturer': ['distributor'],
                          'distributor': ['wholesaler'],
                          'wholesaler': ['retailer'],
                          'retailer': []}

        self.assertEqual(parse_order_sequence(nodes, customers_dict),
                         ['retailer', 'wholesaler', 'distributor', 'manufacturer'])

    def test_arbitrary_network_1(self):
        nodes = ['1', '2', '3', '4', '5', '6', '7']
        customers_dict = {'6': ['4'],
                          '5': ['2', '7'],
                          '4': ['3', '1'],
                          '2': ['1'],
                          '3': [],
                          '1': ['7'],
                          '7': []}

        self.assertEqual(parse_order_sequence(nodes, customers_dict),
                         ['7', '1', '2', '3', '4', '5', '6'])

    def test_arbitrary_network_2(self):
        nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        customers_dict = {'9': ['3'],
                          '7': ['6', '2'],
                          '3': ['5', '1'],
                          '6': ['1'],
                          '5': [],
                          '1': ['2'],
                          '2': ['4'],
                          '4': ['8'],
                          '8': []}

        self.assertEqual(parse_order_sequence(nodes, customers_dict),
                         ['8', '4', '2', '1', '5', '3', '6', '7', '9'])


class TestReadYaml(TestCase):
    def test_read_yaml(self):
        network = read_yaml('network_configs/beergame.yaml')

        self.assertIn('nodes', network.keys())
        self.assertIn('arcs', network.keys())

        self.assertEqual(len(network['nodes']), 5)
        self.assertEqual(len(network['arcs']), 4)

        self.assertEqual(network['arcs'][0]['info_leadtime'], 2)
        self.assertEqual(network['arcs'][3]['info_leadtime'], 1)


if __name__ == '__main__':
    unittest.main()


