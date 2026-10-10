# 🧠 NeuroSeed — My First Trainable AI Model

A **neural network built and trained from scratch** with Python and PyTorch, using the classic Iris dataset. No pretrained AI service is used.

**Created by Amir Saeid Dehghan (2026).** · MIT License

## What it does

From four measurements in centimeters (sepal length and width, petal length and width), the model predicts one of three Iris species: Setosa, Versicolor, Virginica. It also explains the **actual weighted-input calculations of the first hidden neuron**, so you can see what a neural network really does.

**Architecture:** 4 inputs → 12 ReLU units → 8 ReLU units → 3 output scores → Softmax probabilities.

**Evaluation:** 28 of 30 unseen Iris samples classified correctly (**93.3%**). The train/validation/test split is 90/30/30. This is a small educational dataset, not a guarantee for real-world flower classification. Detailed results are in `metrics.json`.

## Run it yourself

Install Python 3.10+ and open the `NeuroSeed-Iris-AI` folder in VS Code. Run:

```bash
python -m pip install -r requirements.txt
python predict.py
python app.py
```

`python predict.py` prints a prediction, confidence scores, and weights/bias math. `python app.py` opens an interactive Gradio web interface (usually at http://127.0.0.1:7860).

To try your own measurements:

```bash
python predict.py --sepal-length 6.0 --sepal-width 2.7 --petal-length 4.5 --petal-width 1.5
```

To **train the weights again from scratch**:

```bash
python train.py
python -m unittest discover -s tests -v
```

Training recreates `iris_model.pt`, `metrics.json`, and `training_curve.png`. The pretrained weights for *this specific scratch-trained network* are included so inference can run immediately; it is not a third-party pretrained model.

## Use Google Colab (no local setup)

Open `colab_quickstart.ipynb` in Google Colab. It clones this public repository and runs the included prediction, training, and interactive UI without needing to upload a ZIP.

## Files

- `model.py`: model structure, inference and per-neuron calculations.
- `iris_model.pt.z`: losslessly compressed learned weights and feature normalization values (automatically loaded by `model.py`). Running `train.py` produces the uncompressed `iris_model.pt`.
- `train.py`: complete reproducible training pipeline.
- `predict.py`: interactive command-line prediction.
- `app.py`: Gradio UI.
- `metrics.json`: held-out test results. Recreate the learning curve image by running `python train.py`.
- `colab_quickstart.ipynb`: Colab notebook.
- `tests/test_model.py`: four tests.
- `requirements.txt`: Python dependencies.

## Share or deploy

GitHub hosts source files, not a running Python service. To let visitors execute this app in a web browser, deploy the *folder's content* in a Hugging Face Space with Gradio SDK and a CPU plan.

**Scope:** Educational Iris classification only; not a photo classifier or general chatbot. Output scores may not be calibrated probabilities. Out-of-range values may give unreliable predictions.
