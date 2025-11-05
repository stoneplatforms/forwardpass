#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy mini language model:
- whitespace tokenizer
- sinusoidal positions
- 1-head self-attention, 1 layer
- causal mask
- trains ONLY embeddings and output head (keeps attention/FFN fixed) for simplicity
- prints losses and top-k predictions before/after
"""
import sys, math, numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---------- tiny data ----------
DEFAULT_CORPUS = [
    "the cat sat on the mat .",
    "the dog sat on the rug .",
    "kevin is a developer .",
    "kevin is a menace .",
    "language models predict tokens .",
    "menace is a strong word .",
]

# ---------- helpers ----------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def relu(x): return np.maximum(0.0, x)

def tok(s): return ["<bos>"] + s.strip().lower().split() + ["<eos>"]

def build_vocab(texts):
    base = ["<bos>","<eos>",".",",","?","!","the","a","an","on","in","is","are","of","and","to"]
    vocab, seen = [], set()
    for t in base + sum([tok(x) for x in texts], []):
        if t not in seen:
            seen.add(t); vocab.append(t)
    v2i = {v:i for i,v in enumerate(vocab)}
    i2v = {i:v for v,i in v2i.items()}
    return vocab, v2i, i2v

def encode(line, v2i): return np.array([v2i[w] for w in tok(line)], dtype=np.int32)

def positional_table(max_len, d):
    P = np.zeros((max_len, d), dtype=np.float32)
    pos = np.arange(max_len, dtype=np.float32).reshape(-1,1)
    div = np.exp(np.linspace(0, np.log(10000), d//2)).reshape(1,-1)
    P[:, 0::2] = np.sin(pos / div)
    P[:, 1::2] = np.cos(pos / div)
    return P

# ---------- model ----------
class TinyNumPyLM:
    def __init__(self, V, d=32, d_ff=64, max_len=128, seed=123):
        self.V, self.d, self.d_ff = V, d, d_ff
        rng = np.random.default_rng(seed)
        # trainable
        self.E = rng.normal(0, 0.2, size=(V, d)).astype(np.float32)        # token embeddings (train)
        self.W_vocab = rng.normal(0,0.2,size=(d,V)).astype(np.float32)     # vocab head (train)
        self.b_vocab = np.zeros((V,), dtype=np.float32)

        # fixed (for simplicity)
        self.P = positional_table(max_len, d).astype(np.float32)
        self.W_Q = rng.normal(0, 0.2, size=(d,d)).astype(np.float32)
        self.W_K = rng.normal(0, 0.2, size=(d,d)).astype(np.float32)
        self.W_V = rng.normal(0, 0.2, size=(d,d)).astype(np.float32)
        self.W_O = rng.normal(0, 0.2, size=(d,d)).astype(np.float32)
        self.W1  = rng.normal(0, 0.2, size=(d,d_ff)).astype(np.float32)
        self.b1  = np.zeros((d_ff,), dtype=np.float32)
        self.W2  = rng.normal(0, 0.2, size=(d_ff,d)).astype(np.float32)
        self.b2  = np.zeros((d,), dtype=np.float32)

    def forward(self, ids, trace=False):
        T = len(ids)
        X_tok = self.E[ids]                  # (T,d)
        X = X_tok + self.P[:T]               # (T,d)

        Q = X @ self.W_Q                     # (T,d)
        K = X @ self.W_K                     # (T,d)
        Vv = X @ self.W_V                    # (T,d)

        # causal attention
        context = np.zeros_like(X)
        attn = np.zeros((T,T), dtype=np.float32)
        scale = 1.0 / math.sqrt(self.d)
        for i in range(T):
            scores = (K @ Q[i]) * scale      # (T,)
            # mask future
            scores[i+1:] = -1e9
            w = softmax(scores, axis=0)
            attn[i] = w
            context[i] = w @ Vv

        H_attn = context @ self.W_O          # (T,d)
        H_ff = relu(H_attn @ self.W1 + self.b1) @ self.W2 + self.b2
        H = X + H_attn + H_ff                # (T,d)

        logits = H @ self.W_vocab + self.b_vocab   # (T,V)
        probs  = softmax(logits, axis=-1)          # (T,V)
        return {
            "X_tok":X_tok, "X":X, "Q":Q, "K":K, "V":Vv, "attn":attn,
            "H_attn":H_attn, "H_ff":H_ff, "H":H, "logits":logits, "probs":probs
        }

    def loss_and_grads(self, ids):
        """
        Cross-entropy next-token loss.
        Backprop ONLY into E and (W_vocab, b_vocab). Keep attention/FFN fixed to stay compact.
        """
        out = self.forward(ids, trace=False)
        probs = out["probs"]                     # (T,V)
        T = len(ids)
        # next-token targets
        x_idx = ids[:-1]; y_idx = ids[1:]
        p = probs[:-1, :]                        # (T-1, V)

        # loss
        rows = np.arange(T-1)
        loss = -np.log(p[rows, y_idx] + 1e-12).mean()

        # grads wrt logits via (p - y)
        g_logits = p.copy()
        g_logits[rows, y_idx] -= 1.0
        g_logits /= (T-1)

        # backprop into W_vocab, b_vocab, and E via H
        H = out["H"][:-1, :]                     # (T-1, d)
        g_W_vocab = H.T @ g_logits               # (d,V)
        g_b_vocab = g_logits.sum(axis=0)         # (V,)

        g_H = g_logits @ self.W_vocab.T          # (T-1, d)

        # We do a very small step to propagate into embeddings via the residual path (X contributes to H)
        # H ≈ X + (fixed terms), so dL/dE[ids[:-1]] += g_H
        g_E = np.zeros_like(self.E)
        for t, idx in enumerate(x_idx):
            g_E[idx] += g_H[t]

        return loss, {"g_E":g_E, "g_Wv":g_W_vocab, "g_bv":g_b_vocab}

    def step(self, grads, lr=1e-2, clip=1.0, wd=0.0):
        # clip
        gE, gWv, gbv = grads["g_E"], grads["g_Wv"], grads["g_bv"]
        # (simple global clip)
        def clip_(g):
            n = np.linalg.norm(g)
            if n > clip: g *= (clip / (n + 1e-12))
            return g
        gE = clip_(gE); gWv = clip_(gWv); gbv = clip_(gbv)

        # weight decay on weights (not biases)
        self.E      -= lr * (gE + wd * self.E)
        self.W_vocab-= lr * (gWv + wd * self.W_vocab)
        self.b_vocab-= lr * gbv

    def topk(self, probs_row, i2v, k=8):
        idx = np.argsort(-probs_row)[:k]
        return [(i2v[i], float(probs_row[i])) for i in idx]

# ---------- training loop ----------
def train_numpy(corpus, epochs=50, lr=5e-2, wd=1e-4, d=32, seed=123):
    vocab, v2i, i2v = build_vocab(corpus)
    model = TinyNumPyLM(V=len(vocab), d=d, d_ff=64, max_len=256, seed=seed)

    # Encode lines to id arrays
    seqs = [encode(line, v2i) for line in corpus]

    # quick eval before
    probe = seqs[0]
    out0 = model.forward(probe)
    print("Before training top-k (last content pos):")
    print(model.topk(out0["probs"][-2], i2v, k=min(8,len(vocab))))

    for ep in range(1, epochs+1):
        losses = []
        # simple SGD over lines
        for ids in seqs:
            loss, grads = model.loss_and_grads(ids)
            model.step(grads, lr=lr, wd=wd)
            losses.append(loss)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:03d}  loss {np.mean(losses):.4f}")

    out = model.forward(probe)
    print("After training top-k (last content pos):")
    print(model.topk(out["probs"][-2], i2v, k=min(8,len(vocab))))
    return model, vocab, v2i, i2v

def main():
    user = " ".join(sys.argv[1:]).strip()
    corpus = DEFAULT_CORPUS if not user else DEFAULT_CORPUS + [user]
    print("Using corpus:")
    for ln in corpus: print("  -", ln)
    train_numpy(corpus, epochs=60, lr=4e-2, wd=1e-4, d=32)

if __name__ == "__main__":
    main()
