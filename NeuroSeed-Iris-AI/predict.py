"""Run a prediction and inspect the math performed by the first hidden neuron."""
import argparse
from model import analyze

parser = argparse.ArgumentParser(description='Run NeuroSeed: four Iris flower measurements in centimeters.')
parser.add_argument('--sepal-length', type=float, default=5.1)
parser.add_argument('--sepal-width', type=float, default=3.5)
parser.add_argument('--petal-length', type=float, default=1.4)
parser.add_argument('--petal-width', type=float, default=0.2)
args = parser.parse_args()

result = analyze([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width])
print('\n--- NeuroSeed: real neural network prediction ---')
print('Predicted species:', result['class'])
for label, prob in result['probabilities'].items():
    print(f'  {label}: {prob:.1%}')
print('\nInside hidden neuron #1 (using standardized inputs):')
for index, (x, term) in enumerate(zip(result['normalized_inputs'], result['neuron_1_weighted_terms']), 1):
    print(f'  x{index} = {x:+.4f}; x{index} * w{index} = {term:+.4f}')
print(f'  + bias = {result["neuron_1_bias"]:+.4f}')
print(f'  sum (z) = {result["neuron_1_z"]:+.4f}')
print(f'  activation ReLU(z) = {result["neuron_1_activation"]:.4f}')
print('\nThe weights were learned during training; they are not hard-coded rules.')
