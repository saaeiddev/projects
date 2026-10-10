"""Train NeuroSeed on the public Iris dataset (all weights learned here)."""
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn

from model import IrisNet, ROOT

SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    iris = load_iris()
    x = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    class_names = [f'Iris {name}' for name in iris.target_names]

    # 60% train, 20% validation, 20% test; keep test data unseen.
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
    )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    def tensor_features(features):
        return torch.tensor((features - mean) / std, dtype=torch.float32)

    train_x, val_x, test_x = map(tensor_features, [x_train, x_val, x_test])
    train_y = torch.tensor(y_train, dtype=torch.long)
    val_y = torch.tensor(y_val, dtype=torch.long)

    net = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    history = {'train_loss': [], 'validation_loss': []}
    best_loss, patience_count, best_state = float('inf'), 0, None

    for epoch in range(1000):
        net.train()
        optimizer.zero_grad()
        train_loss = criterion(net(train_x), train_y)
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(val_x), val_y))
            logged_train_loss = float(criterion(net(train_x), train_y))
        history['train_loss'].append(logged_train_loss)
        history['validation_loss'].append(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            patience_count = 0
            best_state = deepcopy(net.state_dict())
            best_epoch = epoch + 1
        else:
            patience_count += 1
        if patience_count >= 100:
            break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        predicted = net(test_x).argmax(dim=1).numpy()
    accuracy = accuracy_score(y_test, predicted)

    artifact = {
        'state_dict': best_state,
        'mean': torch.tensor(mean, dtype=torch.float32),
        'std': torch.tensor(std, dtype=torch.float32),
        'classes': class_names,
    }
    torch.save(artifact, ROOT / 'iris_model.pt')

    report = {
        'model': 'NeuroSeed IrisNet (4 -> 12 -> 8 -> 3)',
        'seed': SEED,
        'samples': {'train': len(y_train), 'validation': len(y_val), 'test': len(y_test)},
        'best_epoch': best_epoch,
        'epochs_ran': epoch + 1,
        'test_accuracy': round(float(accuracy), 6),
        'test_correct': int(np.sum(predicted == y_test)),
        'test_total': len(y_test),
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(y_test, predicted).tolist(),
        'classification_report': classification_report(y_test, predicted, target_names=class_names, output_dict=True),
    }
    (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    plt.figure(figsize=(9, 5))
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['validation_loss'], label='Validation loss')
    plt.axvline(best_epoch - 1, linestyle='--', alpha=0.5, label=f'Best epoch: {best_epoch}')
    plt.xlabel('Training epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title('NeuroSeed - Learning Curve')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(ROOT / 'training_curve.png', dpi=160)
    plt.close()

    print(f'Trained from scratch. Epochs: {epoch + 1}; best checkpoint: epoch {best_epoch}')
    print(f'Unseen test accuracy: {accuracy:.1%} ({int(np.sum(predicted == y_test))}/{len(y_test)})')
    print('Saved: iris_model.pt, metrics.json, training_curve.png')
    return report


if __name__ == '__main__':
    main()
