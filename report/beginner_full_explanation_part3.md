# Beginner Full Explanation - Part III IMDb Sequences

This document explains Part III from the beginning.

Part III is harder than Part I and Part II because text is not just a table or an image. Text is a sequence. The order of words matters.

---

# 1. What Dataset Did We Use?

We used the IMDb movie review dataset.

The original IMDb dataset contains:

```text
25,000 training reviews
25,000 test reviews
50,000 reviews total
```

For a laptop-friendly experiment, our script used a smaller subset:

```text
600 training reviews
100 validation reviews
200 test reviews
```

So Part III used:

```text
900 IMDb reviews total
```

Why not use all 50,000?

Because RNN, LSTM, GRU and Seq2Seq training on text can be slow on CPU. The smaller subset allows you to run and understand the project on your laptop.

---

# 2. What Is A Sequence?

A sequence is data where order matters.

Examples:

```text
text
speech
music
time series
DNA
```

For text:

```text
this movie was good
```

is different from:

```text
good was movie this
```

So the model must read tokens in order.

---

# 3. What Is Tokenization?

Computers do not understand raw text directly.

Tokenization transforms text into tokens.

Example:

```text
This movie was good!
```

becomes:

```text
["this", "movie", "was", "good", "!"]
```

In our code, tokenization is simple:

- lowercase the text;
- remove HTML breaks;
- keep words, numbers, and punctuation such as `.`, `!`, `?`.

---

# 4. What Is A Vocabulary?

A vocabulary maps tokens to numbers.

Example:

```text
"movie" -> 34
"good" -> 81
"bad" -> 92
```

Neural networks work with numbers, not words.

Our vocabulary size was:

```text
2,500 tokens
```

This means the model only knows the 2,500 most useful tokens from the training subset.

---

# 5. Special Tokens

The script uses special tokens:

```text
<pad>
<unk>
<sos>
<eos>
```

`<pad>` is used to fill short sequences.

`<unk>` means unknown word. If a word is not in the vocabulary, it becomes `<unk>`.

`<sos>` means start of sequence.

`<eos>` means end of sequence.

---

# 6. What Is A Language Model?

A language model predicts the next token.

Example:

```text
Input: this movie was
Prediction: good
```

It learns patterns in text.

The mathematical idea:

```text
P(w1, w2, w3, ..., wn)
```

means the probability of a full sequence.

Using the chain rule:

```text
P(w1, w2, ..., wn)
= P(w1) * P(w2|w1) * P(w3|w1,w2) * ...
```

In simple words:

```text
predict each next word using the previous words
```

---

# 7. What Did We Train?

We trained four language-model experiments:

```text
RNN without gradient clipping
RNN with gradient clipping
LSTM with gradient clipping
GRU with gradient clipping
```

Why?

The assignment asks us to compare:

```text
RNN
LSTM
GRU
gradient clipping
perplexity
```

---

# 8. What Is An RNN?

RNN means Recurrent Neural Network.

It reads a sequence step by step.

At each step, it keeps a hidden state.

The hidden state is like memory.

Example:

```text
Token 1 -> hidden state
Token 2 -> updated hidden state
Token 3 -> updated hidden state
```

Problem:

Simple RNNs can struggle with long context.

---

# 9. What Is BPTT?

BPTT means Backpropagation Through Time.

It is backpropagation for sequences.

The RNN is unfolded across time steps, and gradients are calculated through the whole sequence.

Problem:

Gradients can become too large or too small.

---

# 10. What Is Gradient Clipping?

Gradient clipping limits gradients when they become too large.

In code:

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This helps stabilize training.

In our results, RNN with clipping performed better than RNN without clipping.

---

# 11. What Is LSTM?

LSTM means Long Short-Term Memory.

It is a more advanced recurrent model.

It uses gates to decide:

```text
what to remember
what to forget
what to output
```

LSTM is usually better than a simple RNN for long sequences.

---

# 12. What Is GRU?

GRU means Gated Recurrent Unit.

It is similar to LSTM but simpler.

It also uses gates, but with fewer components.

GRU is often faster than LSTM.

---

# 13. What Is Perplexity?

Perplexity measures how uncertain a language model is.

Lower perplexity is better.

Example:

```text
perplexity = 100
```

is better than:

```text
perplexity = 1000
```

In our laptop-sized experiment, the perplexities are still high because:

- the dataset subset is small;
- training uses only a few epochs;
- vocabulary is limited;
- IMDb reviews contain many different writing styles.

---

# 14. Language Model Results

Final results:

```text
RNN clipped:
test perplexity = 1397.93

GRU clipped:
test perplexity = 1614.65

LSTM clipped:
test perplexity = 1875.72

RNN without clipping:
test perplexity = 1959.10
```

Best model in this run:

```text
RNN with gradient clipping
```

Important interpretation:

This does not mean RNN is always better than LSTM or GRU.

It means that in our small, short, CPU-friendly experiment, the clipped RNN reached the lowest perplexity.

With more data and more training, LSTM or GRU often become stronger because they handle longer context better.

---

# 15. What Is Seq2Seq?

Seq2Seq means sequence-to-sequence.

It maps one sequence to another sequence.

Examples:

```text
English sentence -> French sentence
long text -> summary
question -> answer
incorrect sentence -> corrected sentence
```

Because you asked to use IMDb, we used IMDb for a mini reconstruction task.

The task:

```text
input review snippet -> reconstruct same snippet
```

This is not translation, but it is still a real sequence-to-sequence task.

---

# 16. Encoder And Decoder

The Seq2Seq model has two parts:

```text
encoder
decoder
```

The encoder reads the input sequence.

The decoder generates the output sequence.

Structure:

```text
IMDb snippet -> encoder -> hidden state -> decoder -> reconstructed snippet
```

---

# 17. Teacher Forcing

During training, the decoder receives the correct previous token.

Example:

If the target is:

```text
this movie was good
```

the decoder input is:

```text
<sos> this movie was
```

and the expected output is:

```text
this movie was good
```

This helps the model learn faster.

---

# 18. Greedy Decoding

Greedy decoding chooses the most probable token at each step.

It is fast and simple.

Problem:

It can make an early mistake and then continue badly.

---

# 19. Beam Search

Beam search keeps several possible sequences instead of only one.

Example:

```text
beam width = 3
```

means it keeps the 3 best candidate sequences while decoding.

Beam search is slower but explores more possibilities.

---

# 20. Seq2Seq Results

Seq2Seq reconstruction results:

```text
greedy BLEU-like score = 0.0972
beam search BLEU-like score = 0.0960
```

These scores are low.

This means the model learned the training loss but still struggles to generate accurate sequences freely.

This is normal for a small Seq2Seq model without attention, trained on a small IMDb subset.

---

# 21. Why Are Seq2Seq Predictions Weak?

The model is small.

The dataset subset is small.

The model has no attention mechanism.

IMDb text is diverse and difficult.

The decoder can collapse into frequent words.

This is a good limitation to discuss in the report.

Do not pretend the Seq2Seq model is perfect. A good scientific report explains both success and limits.

---

# 22. What Files Were Created?

Script:

```text
src/part3_imdb_sequences.py
```

Report draft:

```text
report/part3_imdb_sequences.md
```

Language model results:

```text
outputs/tables/part3_lm_results.csv
```

Seq2Seq results:

```text
outputs/tables/part3_seq2seq_results.csv
```

Seq2Seq predictions:

```text
outputs/tables/part3_seq2seq_predictions.txt
```

Generated text examples:

```text
outputs/tables/part3_language_generations.txt
```

Figures:

```text
outputs/figures/part3_language_model_curves.png
outputs/figures/part3_seq2seq_training_curves.png
```

---

# 23. Command To Run Part III

In VS Code terminal:

```powershell
cd "C:\Users\halas\OneDrive\Bureau\DL"
python -u .\src\part3_imdb_sequences.py
```

The `-u` option shows output immediately while the script is running.

---

# 24. What To Say In The Final Report

You can write:

```text
In this part, IMDb reviews were used as a real sequential text corpus. The text was tokenized, converted into a vocabulary, and used to train recurrent language models. RNN, LSTM and GRU were compared using test loss and perplexity. Gradient clipping was also tested to illustrate its role in stabilizing recurrent training.
```

You can also write:

```text
The clipped RNN obtained the lowest perplexity in this limited experiment, but this result should be interpreted carefully. The training setup used a reduced IMDb subset and few epochs for CPU feasibility. In larger experiments, LSTM and GRU may outperform simple RNNs because their gating mechanisms are better suited to long dependencies.
```

For Seq2Seq:

```text
A mini Seq2Seq model was implemented with a recurrent encoder and decoder. The task was reconstruction of short IMDb review snippets. Greedy decoding and beam search were compared using a simplified BLEU-like score. The low score shows the difficulty of free sequence generation with a small model and limited training data, especially without attention.
```

---

# 25. Main Idea To Remember

Part III shows that text needs models that understand order.

For tabular data:

```text
MLP
```

For image data:

```text
CNN
```

For text sequences:

```text
RNN, LSTM, GRU, Seq2Seq
```

The main project idea is:

```text
The architecture must match the structure of the data.
```

