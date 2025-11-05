#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy mini language model:
- whitespace tokenizer
- sinusoidal positions
- 1-head self-attention, 1 layer
- causal mask
- Photographic Memory Attention (PMA) integration
- trains embeddings, vocab head, attention, and FFN layers
- uses gradient scaling for attention and FFN layers
- prints losses and top-k predictions before/after
"""
import sys, math, numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---------- tiny data ----------
DEFAULT_CORPUS = [
    # Basic syntax & animals
    "the cat sat on the mat .",
    "the dog sat on the rug .",
    "the bird sang on the branch .",
    "the fish swims in the pond .",
    "the fox jumped over the fence .",
    "the rabbit hid under the bush .",

    # People and roles
    "kevin is a developer .",
    "kevin is a menace .",
    "kevin writes code for models .",
    "kevin drinks coffee while coding .",
    "kevin fixes bugs in python .",
    "kevin likes neural networks .",

    # Repetitive patterns (for PMA to catch)
    "language models predict tokens .",
    "language models learn from data .",
    "language models forget slowly .",
    "language models remember patterns .",
    "attention is all you need .",
    "memory helps prediction accuracy .",
    "strong memory leads to fast learning .",
    "fast learners adapt to new tasks .",

    # Factual/semantic variety
    "menace is a strong word .",
    "developer builds software .",
    "software runs on computers .",
    "computers store information .",
    "memory stores information for later .",
    "photographic memory improves learning .",
    "deep networks use attention .",
    "attention focuses on relevant tokens .",
    "context determines meaning .",
    "meaning changes with context .",

    # Creative structure (longer dependencies)
    "the cat that chased the mouse sat on the mat .",
    "the dog that barked loudly sat on the rug .",
    "kevin who codes in rust loves efficiency .",
    "kevin remembers every bug he fixed .",
    "models that remember perform better .",
    "forgetting reduces prediction quality .",
    "memory compression saves compute .",
    "bias correction improves recall .",
    "recall without overfitting is ideal .",
    "attention and memory work together .",
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

        # Photographic Memory Attention (PMA) components
        self.mem_size = 512
        self.MK = np.zeros((self.mem_size, self.d), dtype=np.float32)
        self.MV = np.zeros((self.mem_size, self.d), dtype=np.float32)
        self.mem_ptr = 0
        # Trainable memory projections (separate from attention W_K/W_V)
        self.W_mem_k = rng.normal(0, 0.2, size=(d, d)).astype(np.float32)  # Memory key projection (trainable)
        self.W_mem_v = rng.normal(0, 0.2, size=(d, d)).astype(np.float32)  # Memory value projection (trainable)
        self.W_m  = rng.normal(0, 0.2, size=(self.d, self.d)).astype(np.float32)  # Memory transformation (trainable)
        self.W_g  = rng.normal(0, 0.2, size=(2*self.d, 1)).astype(np.float32)
        # Learned memory compression (CLS-like token)
        self.W_pool = rng.normal(0, 0.2, size=(d, 1)).astype(np.float32)  # Learned pooling weights
        self.alpha = 0.5
        self.use_gate = False
        
        # Gradient scaling factors for attention and FFN
        self.attn_scale = 0.1  # Scale attention gradients
        self.ffn_scale = 0.1   # Scale FFN gradients
        self.mem_scale = 0.1   # Scale memory gradients

    def mem_write(self, H):
        # Improved compression: learned weighted pooling instead of simple mean
        # Compute attention-like weights for pooling
        T = H.shape[0]
        pool_scores = H @ self.W_pool  # (T, 1)
        pool_weights = softmax(pool_scores, axis=0)  # (T, 1)
        s = (pool_weights.T @ H)  # (1, d) - weighted combination
        
        # Use memory-specific projections
        k = (s @ self.W_mem_k).astype(np.float32)  # (1, d)
        v = (s @ self.W_mem_v).astype(np.float32)  # (1, d)
        i = self.mem_ptr % self.mem_size
        self.MK[i] = k.squeeze(0)  # Store as (d,)
        self.MV[i] = v.squeeze(0)  # Store as (d,)
        self.mem_ptr += 1

    def forward(self, ids, trace=False):
        T = len(ids)
        X_tok = self.E[ids]                  # (T,d)
        X = X_tok + self.P[:T]               # (T,d)

        Q = X @ self.W_Q                     # (T,d)
        K = X @ self.W_K                     # (T,d)
        Vv = X @ self.W_V                    # (T,d)

        # --- Memory read (PMA) ---
        valid_m = min(self.mem_ptr, self.mem_size)
        if valid_m > 0:
            MK = self.MK[:valid_m]
            MV = self.MV[:valid_m]
            mem_scores = (Q @ MK.T) / math.sqrt(self.d)   # (T, valid_m)
            mem_w = softmax(mem_scores, axis=1)           # (T, valid_m)
            R = mem_w @ MV                                 # (T, d)
        else:
            R = np.zeros_like(Q)
            mem_w = np.zeros((T, 1))  # Dummy for diagnostics
        Mtilde = R @ self.W_m                               # (T, d)

        # causal attention
        context = np.zeros_like(X)
        attn = np.zeros((T,T), dtype=np.float32)
        scale = 1.0 / math.sqrt(self.d)
        for i in range(T):
            scores = (K @ Q[i]) * scale      # (T,)
            # add memory-derived bias
            bias_i = (K @ Mtilde[i]) * self.alpha
            scores += bias_i
            # mask future
            scores[i+1:] = -1e9
            w = softmax(scores, axis=0)
            attn[i] = w
            context[i] = w @ Vv

        H_attn = context @ self.W_O          # (T,d)
        # FFN forward with intermediate values for backprop
        H_ff_pre = H_attn @ self.W1 + self.b1  # (T, d_ff)
        H_ff_relu = relu(H_ff_pre)             # (T, d_ff)
        H_ff = H_ff_relu @ self.W2 + self.b2   # (T, d)
        # Optional gating between local path and memory read
        if self.use_gate:
            QR = np.concatenate([Q, R], axis=1)            # (T, 2d)
            gamma = 1.0 / (1.0 + np.exp(-(QR @ self.W_g).squeeze(-1)))
            gamma = gamma[:, None]
            H_local = X + H_attn + H_ff
            H = (1.0 - gamma) * H_local + gamma * R
        else:
            H = X + H_attn + H_ff                # (T,d)

        logits = H @ self.W_vocab + self.b_vocab   # (T,V)
        probs  = softmax(logits, axis=-1)          # (T,V)
        
        # Compute memory bias statistics for diagnostics
        mem_bias_norms = np.linalg.norm(Mtilde, axis=1) if valid_m > 0 else np.array([0.0])
        avg_bias_norm = float(np.mean(mem_bias_norms))
        
        # Compute memory retrieval quality (how much memory is being used)
        # Higher values mean memory is more confidently retrieved
        if valid_m > 0:
            max_mem_weights = float(np.max(mem_w))
            avg_mem_weights = float(np.mean(np.max(mem_w, axis=1)))  # Average of max weights per position
            # Find top-k memory retrievals for diagnostics
            # For each position, find which memory slot is most retrieved
            top_mem_indices = np.argmax(mem_w, axis=1)  # (T,) - which memory slot per position
            top_mem_values = np.max(mem_w, axis=1)  # (T,) - confidence per position
            # Overall: which memory slots are most frequently retrieved
            unique, counts = np.unique(top_mem_indices, return_counts=True)
            top_retrieved_mems = list(zip(unique.tolist(), counts.tolist()))
            top_retrieved_mems.sort(key=lambda x: x[1], reverse=True)
            top_5_mems = top_retrieved_mems[:5] if len(top_retrieved_mems) >= 5 else top_retrieved_mems
        else:
            max_mem_weights = 0.0
            avg_mem_weights = 0.0
            top_5_mems = []
        
        out = {
            "X_tok":X_tok, "X":X, "Q":Q, "K":K, "V":Vv, "attn":attn,
            "H_attn":H_attn, "H_ff":H_ff, "H":H, "logits":logits, "probs":probs,
            "R":R, "context":context, "H_ff_pre":H_ff_pre, "H_ff_relu":H_ff_relu,
            "mem_size":valid_m, "avg_bias_norm":avg_bias_norm, "Mtilde":Mtilde,
            "max_mem_weight":max_mem_weights, "avg_mem_weight":avg_mem_weights,
            "top_retrieved_mems":top_5_mems, "mem_w":mem_w if valid_m > 0 else None,
            "H_for_mem":H  # Store H for memory write backprop
        }
        # Write to memory after forward (store intermediate values for backprop)
        T_mem = H.shape[0]
        pool_scores = H @ self.W_pool  # (T_mem, 1)
        pool_weights = softmax(pool_scores, axis=0)  # (T_mem, 1)
        s = (pool_weights.T @ H)  # (1, d)
        k = (s @ self.W_mem_k)  # (1, d)
        v = (s @ self.W_mem_v)  # (1, d)
        i = self.mem_ptr % self.mem_size
        self.MK[i] = k.squeeze(0).astype(np.float32)
        self.MV[i] = v.squeeze(0).astype(np.float32)
        self.mem_ptr += 1
        
        # Store intermediates for backprop
        out["pool_scores"] = pool_scores
        out["pool_weights"] = pool_weights
        out["s"] = s
        out["mem_write_idx"] = i
        
        return out

    def loss_and_grads(self, ids):
        """
        Cross-entropy next-token loss.
        Backprop through embeddings, vocab head, attention, and FFN layers.
        Uses gradient scaling for attention and FFN.
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

        # backprop into W_vocab, b_vocab
        H = out["H"][:-1, :]                     # (T-1, d)
        g_W_vocab = H.T @ g_logits               # (d,V)
        g_b_vocab = g_logits.sum(axis=0)         # (V,)

        g_H = g_logits @ self.W_vocab.T          # (T-1, d)

        # Backprop through residual: H = X + H_attn + H_ff
        # Note: we only use positions 0 to T-2 for loss, so slice accordingly
        H_full = out["H"]                        # (T, d)
        H_attn_full = out["H_attn"]              # (T, d)
        H_ff_full = out["H_ff"]                  # (T, d)
        X_full = out["X"]                        # (T, d)
        context_full = out["context"]             # (T, d)
        
        # Pad g_H to match full sequence length (no gradient at last position)
        g_H_full = np.zeros_like(H_full)
        g_H_full[:-1] = g_H
        
        # Residual connections: g_H flows to X, H_attn, H_ff equally
        g_X = g_H_full.copy()                    # (T, d)
        g_H_attn = g_H_full.copy()               # (T, d)
        g_H_ff = g_H_full.copy()                 # (T, d)

        # --- FFN backward pass ---
        H_ff_pre = out["H_ff_pre"][:-1, :]      # (T-1, d_ff)
        H_ff_relu = out["H_ff_relu"][:-1, :]    # (T-1, d_ff)
        H_ff = H_ff_full[:-1, :]                # (T-1, d)
        g_H_ff = g_H_ff[:-1, :]                 # (T-1, d)
        
        # H_ff = H_ff_relu @ W2 + b2
        g_b2 = g_H_ff.sum(axis=0)                # (d,)
        g_W2 = H_ff_relu.T @ g_H_ff              # (d_ff, d)
        g_H_ff_relu = g_H_ff @ self.W2.T         # (T-1, d_ff)
        
        # ReLU: g_H_ff_pre = g_H_ff_relu * (H_ff_pre > 0)
        g_H_ff_pre = g_H_ff_relu * (H_ff_pre > 0)  # (T-1, d_ff)
        
        # H_ff_pre = H_attn @ W1 + b1
        H_attn = H_attn_full[:-1, :]             # (T-1, d)
        g_b1 = g_H_ff_pre.sum(axis=0)             # (d_ff,)
        g_W1 = H_attn.T @ g_H_ff_pre             # (d, d_ff)
        g_H_attn_from_ffn = g_H_ff_pre @ self.W1.T  # (T-1, d)
        
        # Apply FFN gradient scaling
        g_W1 *= self.ffn_scale
        g_b1 *= self.ffn_scale
        g_W2 *= self.ffn_scale
        g_b2 *= self.ffn_scale

        # --- Attention backward pass ---
        # Combine gradients: H_attn gets gradient from both residual and FFN
        g_H_attn_full = np.zeros_like(H_attn_full)
        g_H_attn_full[:-1] = g_H_attn[:-1] + g_H_attn_from_ffn  # (T, d)
        
        # H_attn = context @ W_O
        context = context_full[:-1, :]           # (T-1, d)
        g_context = g_H_attn_full[:-1, :] @ self.W_O.T  # (T-1, d)
        g_W_O = context.T @ g_H_attn_full[:-1, :]  # (d, d)
        
        # Backprop through attention mechanism (causal, loop-based)
        # For each position i: context[i] = w @ Vv where w = softmax(scores)
        Q_full = out["Q"]                        # (T, d)
        K_full = out["K"]                        # (T, d)
        V_full = out["V"]                        # (T, d)
        attn_full = out["attn"]                   # (T, T)
        Mtilde_full = out["Mtilde"]              # (T, d)
        
        g_Q = np.zeros_like(Q_full)
        g_K = np.zeros_like(K_full)
        g_V = np.zeros_like(V_full)
        g_Mtilde = np.zeros_like(Mtilde_full)    # For memory backprop
        
        scale = 1.0 / math.sqrt(self.d)
        
        for i in range(T-1):  # Only up to T-1 since we don't compute loss at last position
            w = attn_full[i, :i+1]              # (i+1,) attention weights (causal)
            g_context_i = g_context[i]          # (d,)
            
            # g_V: context[i] = w @ Vv, so g_Vv += w[j] * g_context_i
            for j in range(i+1):  # Causal: only up to position i
                g_V[j] += w[j] * g_context_i
            
            # Backprop through softmax: g_scores[j] = w[j] * (g_context_i @ V[j] - sum_k w[k] * (g_context_i @ V[k]))
            V_contribs = np.array([g_context_i @ V_full[j] for j in range(i+1)])  # (i+1,)
            sum_term = w @ V_contribs
            scores_grad = np.zeros(i+1)
            for j in range(i+1):
                scores_grad[j] = w[j] * (V_contribs[j] - sum_term)
            
            # scores = (K @ Q[i]) * scale + bias
            # bias = (K @ Mtilde[i]) * alpha
            # g_K: accumulate from scores = K @ Q[i]
            for j in range(i+1):
                g_K[j] += scores_grad[j] * Q_full[i] * scale
            g_Q[i] += K_full[:i+1].T @ (scores_grad * scale)
            
            # Backprop through memory bias: bias_i = (K @ Mtilde[i]) * alpha
            # g_Mtilde[i] accumulates from all positions j where K[j] is used
            for j in range(i+1):
                g_Mtilde[i] += scores_grad[j] * K_full[j] * self.alpha
        
        # Backprop through memory transformation: Mtilde = R @ W_m
        R_full = out["R"]                        # (T, d)
        g_R = g_Mtilde @ self.W_m.T              # (T, d)
        g_W_m = R_full[:-1, :].T @ g_Mtilde[:-1, :]  # (d, d) - only use positions with loss
        
        # Apply memory gradient scaling
        g_W_m *= self.mem_scale
        
        # Q = X @ W_Q, K = X @ W_K, V = X @ W_V
        X_for_attn = X_full[:-1, :]              # (T-1, d)
        g_Q_for_attn = g_Q[:-1, :]                # (T-1, d)
        g_K_for_attn = g_K[:-1, :]                # (T-1, d)
        g_V_for_attn = g_V[:-1, :]                # (T-1, d)
        
        g_W_Q = X_for_attn.T @ g_Q_for_attn      # (d, d)
        g_W_K = X_for_attn.T @ g_K_for_attn      # (d, d)
        g_W_V = X_for_attn.T @ g_V_for_attn      # (d, d)
        
        # Apply attention gradient scaling
        g_W_Q *= self.attn_scale
        g_W_K *= self.attn_scale
        g_W_V *= self.attn_scale
        g_W_O *= self.attn_scale

        # --- Memory write weights backward (approximate) ---
        # Memory write affects future sequences, so we approximate by using current retrieval quality
        # Compute gradients for W_pool, W_mem_k, W_mem_v through memory write operation
        H_for_mem = out.get("H_for_mem")
        pool_scores = out.get("pool_scores")
        pool_weights = out.get("pool_weights")
        s = out.get("s")
        R_full = out.get("R")
        valid_m = out.get("mem_size", 0)
        
        g_W_pool = np.zeros_like(self.W_pool)
        g_W_mem_k = np.zeros_like(self.W_mem_k)
        g_W_mem_v = np.zeros_like(self.W_mem_v)
        
        if H_for_mem is not None and pool_scores is not None and s is not None and valid_m > 0 and R_full is not None:
            # Use memory retrieval quality as signal: want to improve retrieval
            # Approximate: gradients through memory write to improve future retrievals
            # Use R (memory read) as target - want written memories to be retrievable
            
            # Compute gradients through memory write assuming it affects future retrievals
            # s = pool_weights.T @ H, k = s @ W_mem_k, v = s @ W_mem_v
            # Use R as proxy for what we want: gradients encourage better alignment
            
            # Approximate: gradients through written memory to improve retrieval
            # Use current R as target for what we want written memories to match
            R_target = R_full[-1:, :]  # Use last position as target (1, d)
            
            # Backprop through k = s @ W_mem_k (assuming k should match Q queries)
            # Use Q_full as proxy for what k should align with
            Q_target = Q_full[-1:, :]  # (1, d)
            g_k = (Q_target - s @ self.W_mem_k) * 0.01  # Small gradient to align keys with queries
            g_W_mem_k = (s.T @ g_k) * self.mem_scale  # (d, d)
            
            # Backprop through v = s @ W_mem_v (assuming v should match retrieved values)
            g_v = (R_target - s @ self.W_mem_v) * 0.01
            g_W_mem_v = (s.T @ g_v) * self.mem_scale  # (d, d)
            
            # Backprop through s = pool_weights.T @ H
            g_s = (g_k @ self.W_mem_k.T + g_v @ self.W_mem_v.T)  # (1, d)
            g_pool_weights = g_s @ H_for_mem.T  # (1, T)
            
            # Backprop through softmax in pool_weights
            T_mem = H_for_mem.shape[0]
            g_pool_scores = pool_weights * (g_pool_weights.T - (pool_weights * g_pool_weights.T).sum())
            
            # Backprop through pool_scores = H @ W_pool
            g_W_pool = H_for_mem.T @ g_pool_scores * self.mem_scale  # (d, 1)

        # --- Embeddings backward ---
        # X = X_tok + P, and X contributes to H via residual
        # Also X contributes via Q, K, V
        g_X_total = g_X[:-1, :] + g_Q_for_attn @ self.W_Q.T + g_K_for_attn @ self.W_K.T + g_V_for_attn @ self.W_V.T
        g_E = np.zeros_like(self.E)
        for t, idx in enumerate(x_idx):
            g_E[idx] += g_X_total[t]

        return loss, {
            "g_E": g_E,
            "g_Wv": g_W_vocab,
            "g_bv": g_b_vocab,
            "g_W_Q": g_W_Q,
            "g_W_K": g_W_K,
            "g_W_V": g_W_V,
            "g_W_O": g_W_O,
            "g_W1": g_W1,
            "g_b1": g_b1,
            "g_W2": g_W2,
            "g_b2": g_b2,
            "g_W_m": g_W_m,
            "g_W_pool": g_W_pool,
            "g_W_mem_k": g_W_mem_k,
            "g_W_mem_v": g_W_mem_v,
        }, {
            "mem_size": out.get("mem_size", 0),
            "avg_bias_norm": out.get("avg_bias_norm", 0.0),
            "avg_mem_weight": out.get("avg_mem_weight", 0.0),
            "max_mem_weight": out.get("max_mem_weight", 0.0),
            "top_retrieved_mems": out.get("top_retrieved_mems", [])
        }

    def step(self, grads, lr=1e-2, clip=1.0, wd=0.0):
        # Clip gradients (simple global clip)
        def clip_(g):
            n = np.linalg.norm(g)
            if n > clip: g *= (clip / (n + 1e-12))
            return g
        
        # Extract and clip all gradients
        gE = clip_(grads["g_E"])
        gWv = clip_(grads["g_Wv"])
        gbv = clip_(grads["g_bv"])
        g_W_Q = clip_(grads["g_W_Q"])
        g_W_K = clip_(grads["g_W_K"])
        g_W_V = clip_(grads["g_W_V"])
        g_W_O = clip_(grads["g_W_O"])
        g_W1 = clip_(grads["g_W1"])
        g_b1 = clip_(grads["g_b1"])
        g_W2 = clip_(grads["g_W2"])
        g_b2 = clip_(grads["g_b2"])
        g_W_m = clip_(grads["g_W_m"])
        g_W_pool = clip_(grads["g_W_pool"])
        g_W_mem_k = clip_(grads["g_W_mem_k"])
        g_W_mem_v = clip_(grads["g_W_mem_v"])

        # Update embeddings and vocab head
        self.E      -= lr * (gE + wd * self.E)
        self.W_vocab-= lr * (gWv + wd * self.W_vocab)
        self.b_vocab-= lr * gbv

        # Update attention weights (with weight decay)
        self.W_Q -= lr * (g_W_Q + wd * self.W_Q)
        self.W_K -= lr * (g_W_K + wd * self.W_K)
        self.W_V -= lr * (g_W_V + wd * self.W_V)
        self.W_O -= lr * (g_W_O + wd * self.W_O)

        # Update FFN weights (with weight decay)
        self.W1 -= lr * (g_W1 + wd * self.W1)
        self.b1 -= lr * g_b1
        self.W2 -= lr * (g_W2 + wd * self.W2)
        self.b2 -= lr * g_b2
        
        # Update memory weights (with weight decay)
        self.W_m -= lr * (g_W_m + wd * self.W_m)
        self.W_pool -= lr * (g_W_pool + wd * self.W_pool)
        self.W_mem_k -= lr * (g_W_mem_k + wd * self.W_mem_k)
        self.W_mem_v -= lr * (g_W_mem_v + wd * self.W_mem_v)

    def topk(self, probs_row, i2v, k=8):
        idx = np.argsort(-probs_row)[:k]
        return [(i2v[i], float(probs_row[i])) for i in idx]

    def grad_stats(self, grads):
        """Print gradient statistics for diagnostics."""
        stats = {}
        for name, g in grads.items():
            if g is not None:
                stats[name] = {
                    "norm": float(np.linalg.norm(g)),
                    "mean": float(np.mean(g)),
                    "std": float(np.std(g)),
                    "max": float(np.max(np.abs(g))),
                    "is_zero": bool(np.allclose(g, 0, atol=1e-8))
                }
        return stats

# ---------- training loop ----------
def train_numpy(corpus, epochs=50, lr=5e-2, wd=1e-4, d=32, seed=123, test_input=None):
    # Build vocab including test input if provided (so all tokens are known)
    vocab_corpus = corpus + ([test_input] if test_input else [])
    vocab, v2i, i2v = build_vocab(vocab_corpus)
    model = TinyNumPyLM(V=len(vocab), d=d, d_ff=64, max_len=256, seed=seed)

    # Encode lines to id arrays
    seqs = [encode(line, v2i) for line in corpus]

    # Forward pass on test input before training (if provided)
    if test_input:
        test_ids = encode(test_input, v2i)
        print(f"\n=== Forward pass BEFORE training (input: '{test_input}') ===")
        test_out = model.forward(test_ids)
        print(f"Input length: {len(test_ids)} tokens")
        print("Top-k predictions at each position:")
        for i in range(len(test_ids) - 1):  # -1 to avoid predicting after <eos>
            token = i2v.get(test_ids[i], f"<token_{test_ids[i]}>")
            topk = model.topk(test_out["probs"][i], i2v, k=min(5, len(vocab)))
            print(f"  Position {i} ('{token}'): {topk}")
        print()

    # quick eval before training (using first corpus sequence)
    probe = seqs[0]
    out0 = model.forward(probe)
    print("Before training top-k (last content pos):")
    print(model.topk(out0["probs"][-2], i2v, k=min(8,len(vocab))))
    
    # Store initial weights for change tracking
    W_Q_init = model.W_Q.copy()
    W1_init = model.W1.copy()
    E_init = model.E.copy()
    W_vocab_init = model.W_vocab.copy()

    for ep in range(1, epochs+1):
        losses = []
        grad_norms = {"attn": [], "ffn": [], "emb": [], "vocab": []}
        pma_stats = {"mem_size": [], "bias_norm": [], "mem_weight": []}
        # simple SGD over lines
        for ids in seqs:
            loss, grads, pma_info = model.loss_and_grads(ids)
            # Store gradient norms for diagnostics
            if ep == 1 or ep % 10 == 0:
                grad_norms["attn"].append(np.linalg.norm(grads["g_W_Q"]) + np.linalg.norm(grads["g_W_K"]) + 
                                         np.linalg.norm(grads["g_W_V"]) + np.linalg.norm(grads["g_W_O"]))
                grad_norms["ffn"].append(np.linalg.norm(grads["g_W1"]) + np.linalg.norm(grads["g_W2"]))
                grad_norms["emb"].append(np.linalg.norm(grads["g_E"]))
                grad_norms["vocab"].append(np.linalg.norm(grads["g_Wv"]))
                # Track PMA stats from forward pass
                pma_stats["mem_size"].append(pma_info["mem_size"])
                pma_stats["bias_norm"].append(pma_info["avg_bias_norm"])
                pma_stats["mem_weight"].append(pma_info["avg_mem_weight"])
                # Store top retrieved memories (keep only from first sequence each epoch for clarity)
                if len(pma_stats["mem_size"]) == 1:
                    pma_stats["top_mems"] = pma_info.get("top_retrieved_mems", [])
            model.step(grads, lr=lr, wd=wd)
            losses.append(loss)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:03d}  loss {np.mean(losses):.4f}")
            if ep == 1 or ep % 10 == 0:
                print(f"  Grad norms: attn={np.mean(grad_norms['attn']):.6f}, "
                      f"ffn={np.mean(grad_norms['ffn']):.6f}, "
                      f"emb={np.mean(grad_norms['emb']):.6f}, "
                      f"vocab={np.mean(grad_norms['vocab']):.6f}")
                # Show weight changes from initial
                W_Q_change = np.linalg.norm(model.W_Q - W_Q_init)
                W1_change = np.linalg.norm(model.W1 - W1_init)
                E_change = np.linalg.norm(model.E - E_init)
                W_vocab_change = np.linalg.norm(model.W_vocab - W_vocab_init)
                print(f"  Weight changes: attn={W_Q_change:.6f}, "
                      f"ffn={W1_change:.6f}, "
                      f"emb={E_change:.6f}, "
                      f"vocab={W_vocab_change:.6f}")
                # Show PMA statistics
                if pma_stats["mem_size"]:
                    avg_mem_size = np.mean(pma_stats["mem_size"])
                    avg_bias = np.mean(pma_stats["bias_norm"])
                    avg_mem_weight = np.mean(pma_stats["mem_weight"])
                    print(f"  PMA: mem_size={avg_mem_size:.0f}, bias_norm={avg_bias:.3f}, "
                          f"mem_weight={avg_mem_weight:.4f}, alpha={model.alpha:.2f}")
                    # Show top retrieved memory slots
                    if "top_mems" in pma_stats and pma_stats["top_mems"]:
                        top_str = ", ".join([f"slot#{m[0]}({m[1]}x)" for m in pma_stats["top_mems"][:3]])
                        print(f"    Top retrieved: {top_str}")

    out = model.forward(probe)
    print("After training top-k (last content pos):")
    print(model.topk(out["probs"][-2], i2v, k=min(8,len(vocab))))
    return model, vocab, v2i, i2v

def main():
    # Parse command line arguments
    test_input = None
    corpus_additions = []
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--test-input" or arg == "-t":
            if i + 1 < len(sys.argv):
                test_input = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --test-input requires a value")
                sys.exit(1)
        else:
            corpus_additions.append(arg)
            i += 1
    
    user = " ".join(corpus_additions).strip()
    corpus = DEFAULT_CORPUS if not user else DEFAULT_CORPUS + [user]
    print("Using corpus:")
    for ln in corpus: print("  -", ln)
    if test_input:
        print(f"\nTest input for forward pass: '{test_input}'")
    train_numpy(corpus, epochs=60, lr=4e-2, wd=1e-4, d=32, test_input=test_input)

if __name__ == "__main__":
    main()
