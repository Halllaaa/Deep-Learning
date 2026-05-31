# Deep Learning Project Roadmap

Source assignment: `Projet_Deep-Learning_EMSI.pdf`

## What The Project Is Asking For

You must produce an individual deep learning project with three independent but connected parts:

1. Tabular data classification with an MLP.
2. Image classification with a CNN.
3. Sequence/text modeling with RNN, LSTM, GRU, and a small Seq2Seq system.

For each part, the teacher expects:

- theory explanation;
- PyTorch implementation;
- experiments and comparisons;
- metrics, plots, and visualizations;
- critical analysis;
- one synthesis answer based on a real dataset.

The final report must connect the three parts and explain why different data structures need different neural architectures.

## Recommended Dataset Choices

To keep the project realistic and finishable, use datasets that are standard, accepted, and easy to explain.

### Part I: Tabular MLP

Recommended dataset: Breast Cancer Wisconsin.

Why:

- binary classification, so metrics are easy to interpret;
- real tabular medical dataset;
- small enough to train quickly;
- good for accuracy, precision, recall, F1-score, and confusion matrix.

Alternative: Wine Quality, but it needs more preprocessing and is slightly more annoying for a first full project.

### Part II: CNN

Recommended dataset: Fashion-MNIST.

Why:

- real image classification task;
- available through `torchvision`;
- more meaningful than plain MNIST;
- small enough for experiments with CNN variants.

Use it to compare:

- simple MLP on flattened images;
- LeNet-style CNN;
- CNN variants with different padding, stride, pooling, filter count, and optional `1x1` convolution.

### Part III: Sequences

Recommended approach: start with character-level or word-level language modeling, then add a small Seq2Seq toy translation task.

Why:

- RNN, LSTM, and GRU can be compared cleanly using perplexity;
- Seq2Seq can be demonstrated on a small parallel dataset;
- easier to finish than a large production translation system.

If the teacher requires a real public corpus, use a small `fra-eng` Tatoeba-style dataset or another simple parallel corpus.

## Project Structure To Build

Suggested folder layout:

```text
deep_learning_project/
  README.md
  requirements.txt
  data/
  notebooks/
    01_mlp_tabular.ipynb
    02_cnn_images.ipynb
    03_rnn_seq2seq.ipynb
  src/
    utils.py
    metrics.py
    part1_mlp.py
    part2_cnn.py
    part3_sequences.py
  outputs/
    figures/
    models/
    tables/
  report/
    report.md
```

## Part I: MLP Explanation

An MLP is a feed-forward neural network. It receives a vector of features, passes it through fully connected layers, applies nonlinear activations, and outputs class probabilities or logits.

Key concepts to explain:

- `nn.Module`: base class for PyTorch models.
- `forward`: defines the computation from input to output.
- Parameters: trainable tensors such as weights and biases.
- Gradient: derivative used to update parameters.
- Backpropagation: computes gradients from the loss.
- `state_dict`: dictionary containing model weights.
- Device: CPU or GPU location where tensors and model live.

Experiments to run:

- MLP with `nn.Sequential`;
- MLP with a custom class;
- Gaussian initialization;
- constant initialization;
- Xavier initialization;
- save and reload the best model;
- evaluate with accuracy, precision, recall, F1-score, and confusion matrix.

Main critical question:

Is an MLP suitable for tabular classification? Usually yes, if features are normalized and the dataset has nonlinear patterns, but it can be limited when data is small, noisy, imbalanced, or when tree-based models would capture feature interactions more naturally.

## Part II: CNN Explanation

A CNN is designed for images because it preserves spatial structure. Instead of flattening an image immediately, it scans local regions using filters.

Key concepts to explain:

- locality: nearby pixels are related;
- weight sharing: the same filter is reused across the image;
- feature hierarchy: early layers detect edges, deeper layers detect shapes;
- padding: controls border handling and output size;
- stride: controls how far the filter moves;
- pooling: reduces spatial size;
- channels: RGB or feature maps;
- `1x1` convolution: mixes channels without changing spatial size.

Experiments to run:

- manual 2D cross-correlation;
- manual max-pooling and average-pooling;
- compare manual operations with PyTorch layers;
- LeNet-style CNN;
- compare padding, stride, pooling type, number of filters, and `1x1` convolution;
- visualize feature maps;
- compare MLP vs CNN on the same image dataset.

Main critical question:

A CNN is usually better than an MLP for images because it exploits spatial structure and uses far fewer parameters. Padding, stride, pooling, and depth affect how much spatial detail is preserved, how fast dimensions shrink, and how expressive the model becomes.

## Part III: RNN/LSTM/GRU/Seq2Seq Explanation

Sequence models handle ordered data. The output depends not only on the current input, but also on previous elements.

Key concepts to explain:

- language model: predicts the next token;
- chain rule: decomposes sequence probability into conditional probabilities;
- perplexity: measures how uncertain the model is;
- BPTT: backpropagation through time;
- gradient clipping: prevents exploding gradients;
- RNN: basic recurrent model;
- LSTM: uses gates and memory cell to keep long-term information;
- GRU: simpler gated model, often faster than LSTM;
- Seq2Seq: encoder compresses input sequence, decoder generates output sequence;
- teacher forcing: feeds the true previous token during training;
- greedy decoding: always chooses the best next token;
- beam search: keeps several candidate sequences.

Experiments to run:

- train RNN, LSTM, and GRU on the same corpus;
- compare loss, perplexity, stability, and runtime;
- show effect of gradient clipping;
- implement mini Seq2Seq;
- compare greedy decoding and beam search;
- evaluate using perplexity or BLEU depending on the task.

Main critical question:

RNNs model sequences naturally, but simple RNNs struggle with long dependencies. LSTM and GRU improve memory through gating. Seq2Seq is needed when the input and output are both sequences, especially when their lengths differ, as in translation.

## Final Transversal Discussion

The final answer should compare the three data types:

- tabular data: independent feature vector, so MLP is reasonable;
- image data: spatial grid with local patterns, so CNN is better;
- sequence data: ordered elements with temporal/contextual dependency, so recurrent or encoder-decoder models are better.

The central idea:

Deep learning uses the same supervised learning principle, but the architecture must match the structure of the data.

## Recommended Work Order

1. Create environment and folder structure.
2. Finish Part I completely.
3. Finish Part II completely.
4. Finish Part III in a smaller but clean version.
5. Generate result tables and figures.
6. Write the report progressively, not at the end.
7. Add the final transversal discussion.

## Immediate Next Step

Start with Part I because it is the foundation for PyTorch:

1. Create the project folders.
2. Prepare the Breast Cancer Wisconsin dataset.
3. Build the `nn.Sequential` MLP.
4. Train and evaluate it.
5. Then rewrite it as a custom `nn.Module`.

