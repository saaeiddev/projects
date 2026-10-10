"""NeuroSeed: a real, small, from-scratch-trained neural network (no pretrained model)."""
from pathlib import Path
import io
import zlib

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'iris_model.pt'
FEATURES = (
    'Sepal length (cm)',
    'Sepal width (cm)',
    'Petal length (cm)',
    'Petal width (cm)',
)


class IrisNet(nn.Module):
    """4 inputs -> 12 hidden neurons -> 8 hidden neurons -> 3 classes."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 12)
        self.layer2 = nn.Linear(12, 8)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)  # raw logits; CrossEntropyLoss applies softmax internally


def load_model(path=MODEL_FILE):
    """Load the trained network, its standardization statistics and label names."""
    if Path(path).exists():
        artifact = torch.load(path, map_location='cpu', weights_only=True)
    else:
        # The published repository stores trained weights in a lossless zlib archive.
        compressed = Path(str(path) + '.z').read_bytes()
        artifact = torch.load(io.BytesIO(zlib.decompress(compressed)), map_location='cpu', weights_only=True)
    model = IrisNet()
    model.load_state_dict(artifact['state_dict'])
    model.eval()
    return model, artifact['mean'].float(), artifact['std'].float(), artifact['classes']


def analyze(inputs, model=None, mean=None, std=None, classes=None):
    """Predict and expose first-neuron math so the network isn't a black box."""
    if model is None:
        model, mean, std, classes = load_model()
    values = np.asarray(inputs, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Provide exactly four finite numeric inputs.')

    x = (torch.tensor(values) - mean) / std
    with torch.no_grad():
        first_z = model.layer1(x)
        first_a = torch.relu(first_z)
        second_z = model.layer2(first_a)
        second_a = torch.relu(second_z)
        logits = model.output(second_a)
        prob = torch.softmax(logits, dim=0).numpy()
        weights = model.layer1.weight[0].numpy()
        bias = float(model.layer1.bias[0])

    return {
        'class': classes[int(prob.argmax())],
        'probabilities': {name: float(prob[i]) for i, name in enumerate(classes)},
        'normalized_inputs': x.numpy().tolist(),
        'neuron_1_weighted_terms': (x.numpy() * weights).tolist(),
        'neuron_1_bias': bias,
        'neuron_1_z': float(first_z[0]),
        'neuron_1_activation': float(first_a[0]),
        'hidden_layer_1': first_a.numpy().tolist(),
        'hidden_layer_2': second_a.numpy().tolist(),
        'logits': logits.numpy().tolist(),
    }
