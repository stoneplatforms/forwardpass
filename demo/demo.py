#!/usr/bin/env python3
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ----------------------------
# Helpers
# ----------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0.0, x)

def pad_or_truncate(seq, max_len):
    return seq[:max_len]

def add_special_tokens(tokens):
    return ["<bos>"] + tokens + ["<eos>"]

def simple_tokenize(s: str):
    # Tiny whitespace tokenizer. Real LLMs use BPE; this keeps it readable.
    # Lowercase to reduce vocab noise.
    parts = s.strip().lower().split()
    return parts

def build_vocab(user_tokens):
    # Build a tiny vocab from user tokens + a small set of common continuations
    base = ["<bos>", "<eos>", ".", ",", "?", "!", "the", "a", "an", "on", "in", "is", "are", "of", "and", "to"]
    vocab = []
    seen = set()
    for t in base + user_tokens:
        if t not in seen:
            seen.add(t); vocab.append(t)
    # Ensure we have at least some tokens even if user gives empty input
    if len(vocab) < 6:
        for t in ["cat", "dog", "blue", "sky"]:
            if t not in seen:
                vocab.append(t); seen.add(t)
    return vocab

def pretty(name, arr):
    print(f"\n{name}: shape={arr.shape}")
    print(arr)

def pretty_named_vectors(name, tokens, vecs):
    print(f"\n{name}: shape={vecs.shape}")
    for i, (tok, vec) in enumerate(zip(tokens, vecs)):
        print(f"[{i:02d}] {tok:>6s} -> {vec}")

def attention_weights(q, K, dk):
    # q: (d,), K: (T,d)
    scores = (K @ q) / np.sqrt(dk)  # (T,)
    w = softmax(scores, axis=0)
    return w, scores

# ----------------------------
# Tiny Transformer (1 layer, 1 head)
# ----------------------------
class TinyTransformer:
    def __init__(self, vocab, d_model=16, d_ff=32, seed=0):
        self.vocab = vocab
        self.v2i = {v:i for i,v in enumerate(vocab)}
        self.i2v = {i:v for v,i in self.v2i.items()}
        self.V = len(vocab)
        self.d = d_model
        self.d_ff = d_ff
        self.rng = np.random.default_rng(seed)

        # Parameters
        self.E = self.rng.normal(0, 0.2, size=(self.V, self.d))     # token embeddings
        self.P = self.positional_table(256, self.d)                 # sinusoidal-ish positions

        # Self-attention (single head for clarity)
        self.W_Q = self.rng.normal(0, 0.2, size=(self.d, self.d))
        self.W_K = self.rng.normal(0, 0.2, size=(self.d, self.d))
        self.W_V = self.rng.normal(0, 0.2, size=(self.d, self.d))
        self.W_O = self.rng.normal(0, 0.2, size=(self.d, self.d))   # output proj for the head

        # Feed-forward
        self.W1 = self.rng.normal(0, 0.2, size=(self.d, self.d_ff))
        self.b1 = np.zeros((self.d_ff,))
        self.W2 = self.rng.normal(0, 0.2, size=(self.d_ff, self.d))
        self.b2 = np.zeros((self.d,))

        # Final vocab projection
        self.W_vocab = self.rng.normal(0, 0.2, size=(self.d, self.V))
        self.b_vocab = np.zeros((self.V,))

    def positional_table(self, max_len, d):
        # Simple deterministic positions (sinusoidal-like but compact)
        P = np.zeros((max_len, d))
        pos = np.arange(max_len).reshape(-1, 1)
        div = np.exp(np.linspace(0, np.log(10000), d//2))
        P[:, 0::2] = np.sin(pos / div)
        P[:, 1::2] = np.cos(pos / div)
        return P

    def encode(self, tokens):
        ids = [self.v2i.get(t, None) for t in tokens]
        if any(i is None for i in ids):
            missing = [tokens[k] for k,i in enumerate(ids) if i is None]
            raise ValueError(f"Unknown tokens: {missing}")
        return np.array(ids, dtype=np.int32)

    def forward(self, ids, trace=True):
        T = len(ids)

        # Embeddings + Positions
        X_tok = self.E[ids]                     # (T, d)
        X_pos = self.P[:T]                      # (T, d)
        X = X_tok + X_pos                       # (T, d)
        if trace:
            pretty_named_vectors("Token embeddings E[token]", [self.i2v[i] for i in ids], X_tok)
            pretty("Positional encodings P[:T]", X_pos)
            pretty("Input to attention X = E + P", X)

        # Projections for a single head
        Q = X @ self.W_Q                        # (T, d)
        K = X @ self.W_K                        # (T, d)
        V = X @ self.W_V                        # (T, d)
        if trace:
            pretty("Q", Q); pretty("K", K); pretty("V", V)

        # Scaled dot-product attention per token (causal mask omitted for demo)
        attn_weights = np.zeros((T, T))
        context = np.zeros_like(X)
        for i in range(T):
            w, scores = attention_weights(Q[i], K, self.d)
            attn_weights[i] = w
            context[i] = w @ V  # weighted sum over time

            if trace:
                print(f"\n--- Attention at position {i} (token='{self.i2v[ids[i]]}') ---")
                print("scores:", scores)
                print("weights (softmax):", w)
                print("context vector:", context[i])

        # Head output projection
        H_attn = context @ self.W_O             # (T, d)
        if trace:
            pretty("Head output H_attn = context @ W_O", H_attn)

        # Feed-forward
        H_ff = relu(H_attn @ self.W1 + self.b1) @ self.W2 + self.b2  # (T, d)
        if trace:
            pretty("Feed-forward output H_ff", H_ff)

        # Residual-style combine (simple)
        H = X + H_attn + H_ff
        if trace:
            pretty("Final hidden H (X + attn + ff)", H)

        # Vocab projection (next-token distribution for each position)
        logits = H @ self.W_vocab + self.b_vocab   # (T, V)
        probs = softmax(logits, axis=-1)           # (T, V)
        if trace:
            pretty("Logits (per position)", logits)
            pretty("Probs  (per position)", probs)

        return {
            "X_tok": X_tok, "X_pos": X_pos, "X": X,
            "Q": Q, "K": K, "V": V,
            "attn": attn_weights, "context": context,
            "H_attn": H_attn, "H_ff": H_ff, "H": H,
            "logits": logits, "probs": probs
        }

    def topk_next(self, probs_row, k=10):
        idx = np.argsort(-probs_row)[:k]
        return [(self.i2v[i], float(probs_row[i])) for i in idx]

# ----------------------------
# CLI / Main
# ----------------------------
def main():
    # 1) Read input
    if len(sys.argv) > 1:
        phrase = " ".join(sys.argv[1:]).strip()
    else:
        phrase = input("Enter a word or phrase: ").strip()
    if not phrase:
        phrase = "the cat sat"

    # 2) Tokenize and build vocab
    user_tokens = simple_tokenize(phrase)
    tokens = add_special_tokens(user_tokens)
    vocab = build_vocab(user_tokens)

    print("\n=== INPUT ===")
    print("Raw phrase :", phrase)
    print("Tokens     :", tokens)
    print("Vocab      :", vocab)

    # 3) Initialize model
    model = TinyTransformer(vocab=vocab, d_model=16, d_ff=32, seed=123)

    # 4) Encode
    ids = model.encode(tokens)
    print("\nToken IDs  :", ids)

    # 5) Forward pass with tracing
    out = model.forward(ids, trace=True)

    # 6) Show next-token prediction from final position
    print("\n=== NEXT-TOKEN PREDICTION (from last token) ===")
    last_probs = out["probs"][-1]  # distribution after final token (usually <eos>)
    top = model.topk_next(last_probs, k=min(10, model.V))
    for tok, p in top:
        print(f"{tok:>8s} : {p:.4f}")

    # 7) Also show from the last *content* token if longer than 2
    if len(ids) > 2:
        print("\n=== NEXT-TOKEN PREDICTION (from last content token) ===")
        last_content_idx = len(ids) - 2
        last_content_probs = out["probs"][last_content_idx]
        top2 = model.topk_next(last_content_probs, k=min(10, model.V))
        print(f"(Conditioning on token '{vocab[ids[last_content_idx]]}')")
        for tok, p in top2:
            print(f"{tok:>8s} : {p:.4f}")

if __name__ == "__main__":
    main()
