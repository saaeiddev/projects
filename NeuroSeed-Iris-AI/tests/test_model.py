"""Run with: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import IrisNet, analyze, load_model
import torch


class TestNeuroSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.mean, cls.std, cls.classes = load_model()

    def test_network_dimensions(self):
        self.assertEqual(tuple(self.model(torch.zeros(1, 4)).shape), (1, 3))
        self.assertEqual(self.model.layer1.out_features, 12)
        self.assertEqual(self.model.layer2.out_features, 8)

    def test_prediction_probability(self):
        r = analyze([5.1, 3.5, 1.4, 0.2], self.model, self.mean, self.std, self.classes)
        self.assertEqual(r['class'], 'Iris setosa')
        self.assertAlmostEqual(sum(r['probabilities'].values()), 1.0, places=5)
        self.assertEqual(len(r['hidden_layer_1']), 12)
        self.assertEqual(len(r['hidden_layer_2']), 8)

    def test_neuron_arithmetic(self):
        r = analyze([6.0, 2.7, 4.5, 1.5], self.model, self.mean, self.std, self.classes)
        total = sum(r['neuron_1_weighted_terms']) + r['neuron_1_bias']
        self.assertAlmostEqual(total, r['neuron_1_z'], places=5)
        self.assertAlmostEqual(max(0.0, total), r['neuron_1_activation'], places=5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            analyze([1.0, 2.0], self.model, self.mean, self.std, self.classes)


if __name__ == '__main__':
    unittest.main()
