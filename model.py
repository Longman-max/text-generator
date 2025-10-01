import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def load_dataset(path, n_chars=40, step=3):
    """Load encoded data and create input-output sequences."""
    with open(path, "rb") as f:
        encoded, char_to_int, int_to_char = pickle.load(f)

    vocab_size = len(char_to_int)
    print(f"Loaded dataset: {len(encoded)} characters, vocab size {vocab_size}")

    sequences = []
    next_chars = []

    for i in range(0, len(encoded) - n_chars, step):
        sequences.append(encoded[i: i + n_chars])
        next_chars.append(encoded[i + n_chars])

    X = np.array(sequences)
    y = np.array(next_chars)

    print(f"Prepared {X.shape[0]} sequences")
    return X, y, vocab_size, int_to_char

def build_model(vocab_size, seq_length):
    """Build LSTM text generation model with embedding."""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=seq_length),
        LSTM(256, return_sequences=False),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model

def sample(preds, temperature=1.0):
    """Sample an index from a probability array."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(model, seed, int_to_char, length=200, temperature=1.0):
    """Generate text from a seed sequence."""
    result = []
    for _ in range(length):
        x_pred = np.array(seed).reshape(1, -1)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = int_to_char[next_index]

        result.append(next_char)
        seed = seed[1:] + [next_index]
    return "".join(result)

if __name__ == "__main__":
    # Load dataset
    X, y, vocab_size, int_to_char = load_dataset("data/data.pkl", n_chars=40)

    # Build model
    model = build_model(vocab_size, X.shape[1])

    # Train model
    model.fit(X, y, batch_size=128, epochs=5)

    # Generate text
    seed = X[100].tolist()
    print("\nGenerated text:\n")
    print(generate_text(model, seed, int_to_char, length=500, temperature=0.7))
