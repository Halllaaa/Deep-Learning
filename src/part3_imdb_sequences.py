from collections import Counter
from pathlib import Path
from time import perf_counter
import math
import random
import re
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMDB_DIR = DATA_DIR / "aclImdb"
IMDB_ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"
FIGURE_DIR = OUTPUTS / "figures"
TABLE_DIR = OUTPUTS / "tables"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_and_extract_imdb() -> None:
    if IMDB_DIR.exists():
        print(f"IMDb dataset already available at: {IMDB_DIR}")
        return

    if not IMDB_ARCHIVE.exists():
        print("Downloading IMDb dataset. This happens only once.")
        urllib.request.urlretrieve(IMDB_URL, IMDB_ARCHIVE)
        print(f"Downloaded archive to: {IMDB_ARCHIVE}")

    print("Extracting IMDb dataset...")
    with tarfile.open(IMDB_ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"Extracted IMDb dataset to: {IMDB_DIR}")


def clean_html(text: str) -> str:
    return text.replace("<br />", " ").replace("<br/>", " ")


def tokenize(text: str) -> list[str]:
    text = clean_html(text.lower())
    return re.findall(r"[a-z]+|[0-9]+|[.!?]", text)


def load_reviews(split: str, limit_per_class: int) -> list[str]:
    reviews = []
    for label in ["pos", "neg"]:
        folder = IMDB_DIR / split / label
        files = sorted(folder.glob("*.txt"))[:limit_per_class]
        for path in files:
            reviews.append(path.read_text(encoding="utf-8", errors="ignore"))
    random.shuffle(reviews)
    return reviews


def build_vocab(texts: list[str], max_vocab_size: int = 5000, min_freq: int = 2) -> tuple[dict[str, int], list[str]]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    words = [
        word
        for word, count in counter.most_common(max_vocab_size - len(SPECIAL_TOKENS))
        if count >= min_freq
    ]
    id_to_token = SPECIAL_TOKENS + words
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_tokens(tokens: list[str], token_to_id: dict[str, int], add_eos: bool = True) -> list[int]:
    ids = [token_to_id.get(token, UNK_ID) for token in tokens]
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode_ids(ids: list[int], id_to_token: list[str]) -> str:
    words = []
    for idx in ids:
        if idx in [PAD_ID, SOS_ID]:
            continue
        if idx == EOS_ID:
            break
        words.append(id_to_token[idx] if idx < len(id_to_token) else UNK)
    return " ".join(words)


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int = 30, max_sequences: int = 2500):
        self.inputs = []
        self.targets = []
        max_start = max(0, len(token_ids) - seq_len - 1)
        step = seq_len

        for start in range(0, max_start, step):
            if len(self.inputs) >= max_sequences:
                break
            chunk = token_ids[start : start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.inputs.append(torch.tensor(chunk[:-1], dtype=torch.long))
                self.targets.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, cell_type: str, embed_dim: int = 48, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        recurrent_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.recurrent = recurrent_cls(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        sequence_output, _ = self.recurrent(embedded)
        sequence_output = self.dropout(sequence_output)
        return self.output(sequence_output)


def make_lm_token_stream(texts: list[str], token_to_id: dict[str, int]) -> list[int]:
    stream = []
    for text in texts:
        stream.extend(encode_tokens(tokenize(text), token_to_id, add_eos=True))
    return stream


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total)


def train_language_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 2,
    lr: float = 1e-3,
    clip_value: float | None = 1.0,
):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_perplexity": [], "max_grad_norm": []}
    best_state = None
    best_val_loss = float("inf")
    model.to(device)
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        max_grad = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            max_grad = max(max_grad, gradient_norm(model.parameters()))
            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_perplexity = evaluate_language_model(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_perplexity)
        history["max_grad_norm"].append(max_grad)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_perplexity={val_perplexity:.2f} "
            f"max_grad_norm={max_grad:.2f}",
            flush=True,
        )

    elapsed = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, elapsed


def evaluate_language_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def generate_language_sample(model, prompt: str, token_to_id, id_to_token, device, max_new_tokens: int = 20) -> str:
    model.eval()
    ids = [token_to_id.get(token, UNK_ID) for token in tokenize(prompt)]
    if not ids:
        ids = [SOS_ID]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([ids[-30:]], dtype=torch.long).to(device)
            logits = model(input_tensor)
            logits[:, :, [PAD_ID, UNK_ID, SOS_ID]] = -1e9
            next_id = int(torch.argmax(logits[0, -1]).item())
            ids.append(next_id)
            if next_id == EOS_ID:
                break
    return decode_ids(ids, id_to_token)


def plot_lm_history(histories: dict[str, dict]) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history["val_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Language model validation loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history["val_perplexity"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Language model perplexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "part3_language_model_curves.png", dpi=160)
    plt.close()


class Seq2SeqDataset(Dataset):
    def __init__(self, texts: list[str], token_to_id: dict[str, int], seq_len: int = 12, max_samples: int = 800):
        self.samples = []
        for text in texts:
            tokens = tokenize(text)
            if len(tokens) < 4:
                continue
            core = encode_tokens(tokens[: seq_len - 1], token_to_id, add_eos=True)
            core = core[:seq_len]
            if UNK_ID in core:
                continue
            core = core + [PAD_ID] * (seq_len - len(core))
            decoder_input = [SOS_ID] + core[:-1]
            self.samples.append(
                (
                    torch.tensor(core, dtype=torch.long),
                    torch.tensor(decoder_input, dtype=torch.long),
                    torch.tensor(core, dtype=torch.long),
                )
            )
            if len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Seq2SeqGRU(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, decoder_input):
        src_embedded = self.embedding(src)
        _, hidden = self.encoder(src_embedded)
        dec_embedded = self.embedding(decoder_input)
        dec_output, _ = self.decoder(dec_embedded, hidden)
        return self.output(dec_output)

    def encode(self, src):
        src_embedded = self.embedding(src)
        _, hidden = self.encoder(src_embedded)
        return hidden

    def decode_step(self, token, hidden):
        embedded = self.embedding(token)
        output, hidden = self.decoder(embedded, hidden)
        logits = self.output(output[:, -1, :])
        return logits, hidden


def train_seq2seq(model, train_loader, val_loader, device, epochs: int = 3, lr: float = 1e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    model.to(device)
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, decoder_input, target in train_loader:
            src = src.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(src, decoder_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * src.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_seq2seq_loss(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

    elapsed = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, elapsed


def evaluate_seq2seq_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, decoder_input, target in loader:
            src = src.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)
            logits = model(src, decoder_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total_loss += loss.item() * src.size(0)
    return total_loss / len(loader.dataset)


def greedy_decode(model, src, max_len: int, device) -> list[int]:
    model.eval()
    src = src.unsqueeze(0).to(device)
    hidden = model.encode(src)
    token = torch.tensor([[SOS_ID]], dtype=torch.long).to(device)
    output_ids = []

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model.decode_step(token, hidden)
            logits[:, [PAD_ID, UNK_ID, SOS_ID]] = -1e9
            next_id = int(torch.argmax(logits, dim=1).item())
            output_ids.append(next_id)
            token = torch.tensor([[next_id]], dtype=torch.long).to(device)
            if next_id == EOS_ID:
                break
    return output_ids


def beam_search_decode(model, src, max_len: int, device, beam_width: int = 3) -> list[int]:
    model.eval()
    src = src.unsqueeze(0).to(device)
    hidden = model.encode(src)
    beams = [([SOS_ID], 0.0, hidden)]

    with torch.no_grad():
        for _ in range(max_len):
            candidates = []
            for sequence, score, beam_hidden in beams:
                last_token = sequence[-1]
                if last_token == EOS_ID:
                    candidates.append((sequence, score, beam_hidden))
                    continue
                token = torch.tensor([[last_token]], dtype=torch.long).to(device)
                logits, new_hidden = model.decode_step(token, beam_hidden)
                logits[:, [PAD_ID, UNK_ID, SOS_ID]] = -1e9
                log_probs = torch.log_softmax(logits, dim=1)
                top_scores, top_ids = torch.topk(log_probs, beam_width, dim=1)
                for idx in range(beam_width):
                    next_id = int(top_ids[0, idx].item())
                    next_score = score + float(top_scores[0, idx].item())
                    candidates.append((sequence + [next_id], next_score, new_hidden.clone()))
            beams = sorted(candidates, key=lambda item: item[1] / len(item[0]), reverse=True)[:beam_width]
            if all(sequence[-1] == EOS_ID for sequence, _, _ in beams):
                break

    return beams[0][0][1:]


def simple_bleu(candidate: list[int], reference: list[int], max_n: int = 2) -> float:
    candidate = [token for token in candidate if token not in [PAD_ID, SOS_ID, EOS_ID]]
    reference = [token for token in reference if token not in [PAD_ID, SOS_ID, EOS_ID]]
    if not candidate or not reference:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1))
        ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        precisions.append((overlap + 1) / (sum(cand_ngrams.values()) + 1))

    brevity_penalty = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(1, len(candidate)))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def evaluate_seq2seq_decoding(model, dataset, id_to_token, device, max_items: int = 30):
    greedy_scores = []
    beam_scores = []
    prediction_lines = []

    for idx in range(min(max_items, len(dataset))):
        src, _, target = dataset[idx]
        greedy_ids = greedy_decode(model, src, max_len=len(target), device=device)
        beam_ids = beam_search_decode(model, src, max_len=len(target), device=device)
        target_ids = target.tolist()
        greedy_scores.append(simple_bleu(greedy_ids, target_ids))
        beam_scores.append(simple_bleu(beam_ids, target_ids))

        if idx < 8:
            prediction_lines.extend(
                [
                    f"Example {idx + 1}",
                    f"Source:    {decode_ids(src.tolist(), id_to_token)}",
                    f"Reference: {decode_ids(target_ids, id_to_token)}",
                    f"Greedy:    {decode_ids(greedy_ids, id_to_token)}",
                    f"Beam:      {decode_ids(beam_ids, id_to_token)}",
                    "",
                ]
            )

    predictions_path = TABLE_DIR / "part3_seq2seq_predictions.txt"
    predictions_path.write_text("\n".join(prediction_lines), encoding="utf-8")
    return float(np.mean(greedy_scores)), float(np.mean(beam_scores)), predictions_path


def plot_seq2seq_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Seq2Seq reconstruction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "part3_seq2seq_training_curves.png", dpi=160)
    plt.close()


def run_part3():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    download_and_extract_imdb()
    train_texts = load_reviews("train", limit_per_class=350)
    test_texts = load_reviews("test", limit_per_class=100)
    random.shuffle(train_texts)
    val_texts = train_texts[-100:]
    train_texts = train_texts[:-100]

    print(f"IMDb texts used: train={len(train_texts)}, validation={len(val_texts)}, test={len(test_texts)}", flush=True)
    token_to_id, id_to_token = build_vocab(train_texts, max_vocab_size=2500, min_freq=2)
    vocab_size = len(id_to_token)
    print(f"Vocabulary size: {vocab_size}", flush=True)

    train_stream = make_lm_token_stream(train_texts, token_to_id)
    val_stream = make_lm_token_stream(val_texts, token_to_id)
    test_stream = make_lm_token_stream(test_texts, token_to_id)

    lm_train = LanguageModelDataset(train_stream, seq_len=20, max_sequences=600)
    lm_val = LanguageModelDataset(val_stream, seq_len=20, max_sequences=120)
    lm_test = LanguageModelDataset(test_stream, seq_len=20, max_sequences=120)
    lm_train_loader = DataLoader(lm_train, batch_size=64, shuffle=True)
    lm_val_loader = DataLoader(lm_val, batch_size=64, shuffle=False)
    lm_test_loader = DataLoader(lm_test, batch_size=64, shuffle=False)

    lm_experiments = {
        "rnn_no_clipping": ("rnn", None),
        "rnn_clipped": ("rnn", 1.0),
        "lstm_clipped": ("lstm", 1.0),
        "gru_clipped": ("gru", 1.0),
    }
    lm_results = []
    lm_histories = {}
    generations = []

    for name, (cell_type, clip_value) in lm_experiments.items():
        print(f"\n=== Training language model: {name} ===", flush=True)
        model = RNNLanguageModel(vocab_size=vocab_size, cell_type=cell_type)
        epochs = 1 if name == "rnn_no_clipping" else 2
        history, elapsed = train_language_model(
            model,
            lm_train_loader,
            lm_val_loader,
            device,
            epochs=epochs,
            clip_value=clip_value,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        test_loss, test_perplexity = evaluate_language_model(model, lm_test_loader, device, criterion)
        sample = generate_language_sample(model, "this movie was", token_to_id, id_to_token, device)
        lm_histories[name] = history
        generations.extend([f"{name}: {sample}", ""])

        model_path = MODEL_DIR / f"part3_{name}.pt"
        torch.save(
            {
                "model_name": name,
                "cell_type": cell_type,
                "state_dict": model.state_dict(),
                "vocab_size": vocab_size,
            },
            model_path,
        )
        lm_results.append(
            {
                "model": name,
                "clip_value": "None" if clip_value is None else clip_value,
                "test_loss": test_loss,
                "test_perplexity": test_perplexity,
                "max_gradient_norm": max(history["max_grad_norm"]),
                "train_time_seconds": elapsed,
                "model_path": str(model_path),
            }
        )

    plot_lm_history(lm_histories)
    (TABLE_DIR / "part3_language_generations.txt").write_text("\n".join(generations), encoding="utf-8")
    lm_results_df = pd.DataFrame(lm_results).sort_values("test_perplexity", ascending=True)
    lm_results_path = TABLE_DIR / "part3_lm_results.csv"
    lm_results_df.to_csv(lm_results_path, index=False)

    print("\n=== Training Seq2Seq reconstruction model ===", flush=True)
    seq_train = Seq2SeqDataset(train_texts, token_to_id, seq_len=8, max_samples=350)
    seq_val = Seq2SeqDataset(val_texts, token_to_id, seq_len=8, max_samples=60)
    seq_test = Seq2SeqDataset(test_texts, token_to_id, seq_len=8, max_samples=60)
    seq_train_loader = DataLoader(seq_train, batch_size=64, shuffle=True)
    seq_val_loader = DataLoader(seq_val, batch_size=64, shuffle=False)
    seq_model = Seq2SeqGRU(vocab_size=vocab_size)
    seq_history, seq_elapsed = train_seq2seq(seq_model, seq_train_loader, seq_val_loader, device, epochs=10)
    plot_seq2seq_history(seq_history)
    greedy_bleu, beam_bleu, predictions_path = evaluate_seq2seq_decoding(seq_model, seq_test, id_to_token, device)

    seq_model_path = MODEL_DIR / "part3_seq2seq_gru.pt"
    torch.save({"model_name": "seq2seq_gru_reconstruction", "state_dict": seq_model.state_dict()}, seq_model_path)
    seq_results_df = pd.DataFrame(
        [
            {
                "model": "seq2seq_gru_reconstruction",
                "task": "IMDb review snippet reconstruction",
                "greedy_bleu_like_score": greedy_bleu,
                "beam_search_bleu_like_score": beam_bleu,
                "train_time_seconds": seq_elapsed,
                "model_path": str(seq_model_path),
            }
        ]
    )
    seq_results_path = TABLE_DIR / "part3_seq2seq_results.csv"
    seq_results_df.to_csv(seq_results_path, index=False)

    print("\n=== Part III Results ===")
    print("\nLanguage model results:")
    print(lm_results_df)
    print("\nSeq2Seq results:")
    print(seq_results_df)
    print(f"\nLanguage model results saved at: {lm_results_path}")
    print(f"Seq2Seq results saved at: {seq_results_path}")
    print(f"Seq2Seq example predictions saved at: {predictions_path}")
    print(f"Figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    run_part3()
