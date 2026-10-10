"""Deploy this file as a free-CPU Hugging Face Space (SDK: Gradio)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

from model import analyze, load_model

model, mean, std, classes = load_model()


def predict(sepal_length, sepal_width, petal_length, petal_width):
    r = analyze([sepal_length, sepal_width, petal_length, petal_width], model, mean, std, classes)
    fig, ax = plt.subplots(figsize=(7.3, 2.9))
    labels = [s.removeprefix('Iris ') for s in classes]
    percentages = [100 * r['probabilities'][c] for c in classes]
    bars = ax.barh(labels, percentages, color=['#7d93ed', '#6bd9bc', '#efb46b'])
    ax.set_xlim(0, 105)
    ax.set_xlabel('Model probability (%)')
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in percentages], padding=3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    breakdown = (
        '**Hidden neuron #1 calculation** (inputs standardized before multiplying):\n\n'
        + ' + '.join(f'({v:+.3f})' for v in r['neuron_1_weighted_terms'])
        + f' + bias ({r["neuron_1_bias"]:+.3f}) = **{r["neuron_1_z"]:+.3f}**'
        + f'\n\nReLU activation = **{r["neuron_1_activation"]:.3f}**'
        + '\n\n**Architecture:** 4 inputs → 12 ReLU neurons → 8 ReLU neurons → 3 output scores → Softmax probabilities.'
    )
    return f'### Predicted: {r["class"]}', fig, breakdown, {
        'first_hidden_layer_activations': [round(n, 4) for n in r['hidden_layer_1']],
        'second_hidden_layer_activations': [round(n, 4) for n in r['hidden_layer_2']],
        'raw_output_scores_logits': [round(n, 4) for n in r['logits']],
    }


CSS = '''
.gradio-container {max-width: 1060px !important; margin:auto !important;}
#intro {text-align: center; padding: 15px 0 0;}
#intro h1 {font-size: clamp(28px,4vw,42px) !important;}
'''
with gr.Blocks(title='NeuroSeed | An AI model trained from scratch') as demo:
    gr.Markdown('''
# 🧠 NeuroSeed · Neural Network Lab
### A real, original neural network — trained from scratch, not a pretrained AI API.
Give the model **four flower measurements (cm)**. It predicts one of three Iris species and shows how a hidden neuron computes its output.
''', elem_id='intro')
    with gr.Row():
        with gr.Column(scale=1):
            sepal_l = gr.Slider(4.3, 7.9, value=5.1, step=0.1, label='Sepal length (cm)')
            sepal_w = gr.Slider(2.0, 4.4, value=3.5, step=0.1, label='Sepal width (cm)')
            petal_l = gr.Slider(1.0, 6.9, value=1.4, step=0.1, label='Petal length (cm)')
            petal_w = gr.Slider(0.1, 2.5, value=0.2, step=0.1, label='Petal width (cm)')
            run = gr.Button('Run neural network', variant='primary')
        with gr.Column(scale=1):
            result = gr.Markdown()
            chart = gr.Plot(label='Prediction scores')
    gr.Markdown('### 🔬 See what happens inside the network')
    explanation = gr.Markdown()
    with gr.Accordion('Neuron activations and raw scores', open=False):
        internals = gr.JSON()
    gr.Markdown('**Important:** This model classifies the Iris dataset only. Its scores are not guaranteed to be well-calibrated probabilities, and it cannot recognize arbitrary flowers or photos.')
    run.click(predict, [sepal_l, sepal_w, petal_l, petal_w], [result, chart, explanation, internals])
    gr.Examples(
        examples=[[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 4.5, 1.5], [6.7, 3.0, 5.2, 2.3]],
        inputs=[sepal_l, sepal_w, petal_l, petal_w],
        label='Try example measurements',
    )
    demo.load(predict, [sepal_l, sepal_w, petal_l, petal_w], [result, chart, explanation, internals])

if __name__ == '__main__':
    demo.launch(css=CSS)
