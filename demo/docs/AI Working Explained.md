# Photographic Memory Attention (PMA1) — NumPy Implementation Guide

This file describes how to integrate **Photographic Memory Attention (PMA)** into your existing NumPy Transformer demo. It adds long-term recall and avoids redundant probability updates by biasing token-to-token attention with memory-derived priors.

---

## 0. Overview

**Goal:** Reduce repeated probability learning by allowing the model to "remember" patterns through a persistent key/value memory.  
Each forward pass:
1. Performs local causal attention (as before).  
2. Reads from an external memory (past contexts).  
3. Adds a bias term to attention logits using retrieved memory.  
4. Optionally gates between the local and memory outputs.  

---

## 1. Add Memory Structures

In your model `__init__`:

```python
self.mem_size = 512
self.MK = np.zeros((self.mem_size, self.d), dtype=np.float32)
self.MV = np.zeros((self.mem_size, self.d), dtype=np.float32)
self.mem_ptr = 0

rng = np.random.default_rng(seed)
self.W_m  = rng.normal(0, 0.2, size=(self.d, self.d)).astype(np.float32)
self.W_g  = rng.normal(0, 0.2, size=(2*self.d, 1)).astype(np.float32)
self.alpha = 0.5
```

---

## 2. Memory Write Function

```python
def mem_write(self, H):
    s = H.mean(axis=0, keepdims=True)
    k = (s @ self.W_K).astype(np.float32)
    v = (s @ self.W_V).astype(np.float32)
    i = self.mem_ptr % self.mem_size
    self.MK[i] = k
    self.MV[i] = v
    self.mem_ptr += 1
```

---

## 3. Read From Memory (in `forward`)

After computing `Q, K, V`:

```python
valid_m = min(self.mem_ptr, self.mem_size)
if valid_m > 0:
    MK = self.MK[:valid_m]
    MV = self.MV[:valid_m]
    mem_scores = (Q @ MK.T) / np.sqrt(self.d)
    mem_w = softmax(mem_scores, axis=1)
    R = mem_w @ MV
else:
    R = np.zeros_like(Q)

Mtilde = R @ self.W_m
```

---

## 4. Modify the Attention Loop

Replace:

```python
for i in range(T):
    scores = (K @ Q[i]) * scale
    scores[i+1:] = -1e9
    w = softmax(scores, axis=0)
    attn[i] = w
    context[i] = w @ Vv
```

With:

```python
for i in range(T):
    scores = (K @ Q[i]) / np.sqrt(self.d)
    bias_i = (K @ Mtilde[i]) * self.alpha
    scores += bias_i
    scores[i+1:] = -1e9
    w = softmax(scores, axis=0)
    attn[i] = w
    context[i] = w @ Vv
```

---

## 5. Optional Gating

After combining local outputs:

```python
QR = np.concatenate([Q, R], axis=1)
gamma = 1.0 / (1.0 + np.exp(-(QR @ self.W_g).squeeze(-1)))
gamma = gamma[:, None]
H_local = X + H_attn + H_ff
H = (1 - gamma) * H_local + gamma * R
```

---

## 6. Write Memory After Forward

At the end of `forward` or after each training step:

```python
self.mem_write(H)
```

---

## 7. Hyperparameters

| Parameter | Meaning | Start Value |
|------------|----------|--------------|
| `alpha` | strength of memory bias | 0.5 |
| `mem_size` | total stored slots | 512 |
| `W_m`, `W_g` | fixed projections (train later) | random normal |
| `gate` | enable/disable | start disabled |

---

## 8. Debugging Tips

- Print `bias_i` for a few steps; if values dwarf attention scores, reduce `alpha`.
- Visualize `attn[i]` heatmaps; repeated spans should now show extra weight.
- Loss should remain stable; if not, clamp or scale your bias.

---

## 9. Next Steps

1. Replace mean pooling with learned compression (k-means or PCA).  
2. Train `W_m` and `W_g`.  
3. Add auxiliary memory alignment loss (InfoNCE or reconstruction).  
4. Migrate the memory read/write into a batched, vectorized path.

---

**Author:** Generated for integration with your NumPy Transformer demo.  
**Version:** PMA1 — Photographic Memory Attention v0.1
