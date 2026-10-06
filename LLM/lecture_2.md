# CME 295 Lecture 2 — Interview Revision Questions

**Scope:** This note is grounded only in the supplied Lecture 2 transcript. It includes the brief Lecture 1 recap only where Lecture 2 explicitly revisits it. It does not add later-course material as though it were taught here.

**Lecture spine:**

```text
Attention heads and attention maps
        ↓
How Transformer components evolved
        ├── position information
        │     ├── learned absolute embeddings
        │     ├── sinusoidal encodings
        │     ├── T5 relative position bias
        │     ├── ALiBi
        │     └── RoPE
        ├── normalization
        │     ├── residual Add & Norm
        │     ├── Post-Norm → Pre-Norm
        │     └── LayerNorm → RMSNorm
        └── attention efficiency
              ├── full → sliding-window attention
              └── MHA → GQA / MQA
        ↓
Transformer model families
        ├── encoder–decoder: Transformer / T5
        ├── encoder-only: BERT
        └── decoder-only: modern LLM direction
        ↓
BERT
        ├── bidirectional encoder representations
        ├── token + position + segment embeddings
        ├── MLM + NSP pre-training
        ├── downstream fine-tuning
        ├── DistilBERT
        └── RoBERTa
```

---

## Notation used in this note

```text
B     batch size
T     sequence length
D     model / embedding dimension
H     number of query or attention heads
Dh    per-head dimension, usually D / H
G     number of key/value groups in GQA
V     vocabulary size
C     number of downstream classes
L     number of stacked Transformer layers
M     number of token positions selected for MLM in a batch
```

The BERT-paper notation mentioned in the lecture is:

```text
L = number of encoder layers
H = hidden / embedding dimension
A = number of attention heads
```

This differs from the earlier lecture notation, where the number of layers was called `N`, model width was `d_model`, and head count was often written as lowercase `h`.

### Priority legend

- **🔥 Must remember:** answer immediately and derive shapes.
- **○ Should know:** explain clearly without needing every detail.
- **△ Lecture boundary:** the lecture mentions it but does not fully derive or specify it.

---

# 1. Attention heads and attention maps

## 1. 🔥 What is an attention map?

Given one attention head,

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}\right)
$$

with shapes:

```text
Q: [B, H, T, Dh]
K: [B, H, T, Dh]

QKᵀ: [B, H, T, T]
A:   [B, H, T, T]
```

For a particular head and query position `i`, row `A[i, :]` says how strongly that query retrieves from every key position.

**Memory liner:**

> An attention map is the query-to-key interaction matrix for one or more heads.

The lecture visually interprets high query–key interactions as tokens that matter for the current token.

---

## 2. 🔥 What does one row of an attention map mean?

For query token `i`:

$$
A_{i,j}
$$

is the normalized attention weight assigned to key token `j`.

```text
one query token
      ↓
[A_i1, A_i2, ..., A_iT]
      ↓
importance over all key positions
```

After softmax:

$$
\sum_{j=1}^{T} A_{i,j}=1
$$

**Memory liner:**

> A row answers: “Where does this query token look?”

---

## 3. 🔥 Why can different attention heads produce different maps?

Each head has its own learned query, key, and value projections:

$$
Q_h=XW_Q^{(h)},\qquad
K_h=XW_K^{(h)},\qquad
V_h=XW_V^{(h)}
$$

Therefore each head gets a separate opportunity to learn a different matching space.

```text
same input X
   ├── head 1 projections → attention map 1
   ├── head 2 projections → attention map 2
   └── ...
```

**Memory liner:**

> Different heads see the same sequence through different learned projections.

The lecture does not claim that every head is guaranteed to learn a clean human-interpretable linguistic function; it uses attention maps as one way to inspect what heads may have learned.

---

## 4. 🔥 How are the outputs of multiple heads recombined?

For head outputs:

$$
O_h\in\mathbb{R}^{B\times T\times D_h}
$$

concatenate over the feature dimension:

$$
O_{cat}=\operatorname{Concat}(O_1,\ldots,O_H)
\in\mathbb{R}^{B\times T\times D}
$$

then apply an output projection:

$$
O=O_{cat}W_O
$$

with:

```text
O_cat: [B, T, H × Dh] = [B, T, D]
W_O:   [D, D]
O:     [B, T, D]
```

**Memory liner:**

> Run heads in parallel, concatenate, then mix them with `W_O`.

---

# 2. Why Transformers need position information

## 5. 🔥 Why is position information necessary in self-attention?

Self-attention creates direct token-to-token links. Unlike an RNN, it does not inherently process token 1, then token 2, then token 3.

Without injected position information, the mechanism has no built-in notion that one token came before another.

**Memory liner:**

> Attention knows content interactions; position encoding tells it where the content occurred.

---

## 6. 🔥 What is the original additive position-input formulation?

For token position `t`:

$$
x_t=e_{token,t}+p_t
$$

Batched shapes:

```text
Token embeddings:    [B, T, D]
Position embeddings: [T, D]      or [B, T, D] after broadcast
Input to model:      [B, T, D]
```

The token and position vectors must have the same dimension because they are added elementwise.

**Memory liner:**

> Transformer input = token identity + position identity.

---

# 3. Learned absolute position embeddings

## 7. 🔥 What is a learned absolute position embedding?

Allocate a trainable lookup table:

$$
P\in\mathbb{R}^{T_{max}\times D}
$$

Position `t` selects row `P[t]`:

```text
position IDs: [B, T]
P table:      [Tmax, D]
lookup:       [B, T, D]
```

The values are updated through ordinary gradient descent with the rest of the model.

**Memory liner:**

> Learn one trainable vector for every supported absolute position.

---

## 8. 🔥 What limitations of learned position embeddings does the lecture identify?

### Limitation 1: training-data dependence

The learned representation of a position can absorb biases associated with what commonly appeared there in the training data.

### Limitation 2: fixed learned range

If the position table is trained only through position `Tmax`, positions beyond `Tmax` have no learned row.

```text
training positions: 0 ... 511
inference position: 700
                   ↑
       no directly learned embedding
```

**Memory liner:**

> Learned positions are flexible inside the trained range but do not naturally define unseen positions.

---

## 9. ○ What is the benefit of learned positions?

The model is free to adapt the position vectors to the training objective rather than obeying a hand-designed formula.

**Memory liner:**

> The benefit is flexibility; the cost is dependence on the learned position table and training distribution.

---

# 4. Sinusoidal positional encoding

## 10. 🔥 What equations define sinusoidal positional encoding in the lecture?

For position `m` and frequency index `i`, write:

$$
\omega_i=10000^{-2i/D}
$$

Then paired dimensions use:

$$
PE(m,2i)=\sin(\omega_i m)
$$

$$
PE(m,2i+1)=\cos(\omega_i m)
$$

Equivalent common notation is:

$$
PE(m,2i)=\sin\left(\frac{m}{10000^{2i/D}}\right)
$$

$$
PE(m,2i+1)=\cos\left(\frac{m}{10000^{2i/D}}\right)
$$

Shape:

```text
PE: [T, D]
```

**Memory liner:**

> Every position is encoded by sine/cosine pairs spanning many frequencies.

---

## 11. 🔥 Why must the sinusoidal vector have dimension `D`?

Because it is added to the token embedding:

$$
[B,T,D]+[T,D]\rightarrow[B,T,D]
$$

**Memory liner:**

> Additive position encoding must match the token-embedding width.

---

## 12. 🔥 What do low- and high-index dimensions represent?

The lecture explains:

- low `i` gives larger `\omega_i` and faster oscillation;
- high `i` gives smaller `\omega_i` and slower oscillation.

```text
small dimension index → high frequency
large dimension index → low frequency
```

**Memory liner:**

> Different feature pairs act like position clocks running at different speeds.

---

## 13. 🔥 Why does the dot product between two sinusoidal position vectors reveal relative distance?

For one sine/cosine pair:

$$
\sin(\omega_i m)\sin(\omega_i n)
+
\cos(\omega_i m)\cos(\omega_i n)
=
\cos(\omega_i(m-n))
$$

Therefore, across all pairs:

$$
PE(m)^\top PE(n)
=
\sum_i \cos(\omega_i(m-n))
$$

The interaction depends on the relative displacement `m-n`, not only on the two positions independently.

**Memory liner:**

> Paired sine/cosine features turn a position dot product into a function of relative offset.

---

## 14. 🔥 Why is similarity maximal when the two positions are identical?

If `m=n`, then:

$$
m-n=0
$$

and:

$$
\cos(0)=1
$$

so each paired term is at its maximum.

**Memory liner:**

> A position is maximally similar to itself because every relative phase is zero.

---

## 15. ○ Does sinusoidal similarity decrease monotonically with distance?

No such strict monotonic guarantee is established in the lecture. The lecturer explicitly notes the periodic nature of sine and cosine.

The intended intuition is that the multi-frequency representation exposes relative displacement, while periodic oscillation means individual cosine terms can rise again.

**Memory liner:**

> Sinusoidal position similarity is relative and periodic, not a simple monotonic distance function.

---

## 16. 🔥 Why can fixed sinusoidal encodings represent positions beyond those seen in training?

They are generated by a formula rather than selected from a finite learned table.

```text
new position m
   ↓
compute sin(ω_i m), cos(ω_i m)
   ↓
position vector exists without learning a new row
```

**Memory liner:**

> A formula can be evaluated at an unseen position; a finite learned lookup cannot.

The lecture says the original authors observed performance comparable to learned positions while gaining this ability to evaluate the formula beyond the training range.

---

# 5. Injecting relative position directly into attention

## 17. 🔥 Why move position information from the input into the attention computation?

The lecture's reasoning is:

1. Position matters when determining which token should attend to which other token.
2. Adding a position vector only at the model input affects attention indirectly.
3. A relative-position term inside the attention score changes query–key compatibility directly.

General score view:

$$
s_{m,n}
=
\frac{q_m^\top k_n}{\sqrt{D_h}}
+
\text{position term}(m,n)
$$

Then:

$$
A=\operatorname{softmax}(S)
$$

Shape-compatible view:

```text
content scores:  [B, H, T, T]
position term:   broadcastable to [B, H, T, T]
final scores:    [B, H, T, T]
```

**Memory liner:**

> Put relative position where similarity is decided: inside the attention logits.

---

# 6. T5 relative position bias

## 18. 🔥 What is T5-style relative position bias as presented in the lecture?

Compute relative displacement:

$$
\Delta=m-n
$$

Bucketize displacements and learn a bias for each bucket:

$$
b_{m,n}=b(\operatorname{bucket}(m-n))
$$

Attention score:

$$
s_{m,n}
=
\frac{q_m^\top k_n}{\sqrt{D_h}}
+b_{m,n}
$$

**Memory liner:**

> T5 learns attention-logit biases for buckets of relative distances.

---

## 19. ○ Why can a bias be added before softmax even though attention rows must sum to one?

Softmax performs the final normalization:

$$
A_{m,:}=\operatorname{softmax}(s_{m,:})
$$

Any real-valued bias may modify the logits; softmax still converts the resulting row into a normalized distribution.

**Memory liner:**

> Bias changes relative preference; softmax restores normalization.

---

# 7. ALiBi

## 20. 🔥 What is ALiBi at the level taught in this lecture?

The lecture presents ALiBi as a deterministic attention bias that is a function of relative position difference, rather than a learned bias table.

```text
relative distance
      ↓
deterministic bias
      ↓
add to attention logits
```

**Memory liner:**

> T5 learns relative biases; ALiBi uses a predetermined distance-based bias.

### △ Lecture boundary

The transcript does not provide or derive the exact ALiBi formula or per-head slopes. Do not add those details to this lecture's revision note.

---

# 8. Rotary Position Embeddings — RoPE

## 21. 🔥 What is the main idea of RoPE?

RoPE rotates query and key vectors by angles determined by their positions.

For position `m` and `n`:

$$
q'_m=R_m q_m
$$

$$
k'_n=R_n k_n
$$

The attention score uses:

$$
(q'_m)^\top k'_n
$$

**Memory liner:**

> RoPE encodes position by rotating Q and K before their dot product.

---

## 22. 🔥 What is the 2D rotation matrix?

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
$$

For a 2D vector:

$$
v=
\begin{bmatrix}
x\\y
\end{bmatrix}
$$

then:

$$
v'=R(\theta)v
$$

is the original vector rotated by angle `\theta`.

**Memory liner:**

> Rotation is a matrix multiplication that changes angle while preserving the basic 2D structure.

---

## 23. 🔥 How is 2D rotation extended to a high-dimensional query or key?

The lecture explains that dimensions are handled in 2D blocks.

Reshape conceptually:

```text
Q: [B, H, T, Dh]
        ↓ pair adjacent features
Q_pairs: [B, H, T, Dh/2, 2]
        ↓ apply one 2×2 rotation per pair
Q_rot:   [B, H, T, Dh]
```

The same is done to `K`.

**Memory liner:**

> Split the head dimension into feature pairs and rotate each pair at its own frequency.

---

## 24. 🔥 Why does the RoPE dot product depend on relative position?

Let query position be `m` and key position be `n`:

$$
q'_m=R(m\theta)q_m,
\qquad
k'_n=R(n\theta)k_n
$$

Then:

$$
(q'_m)^\top k'_n
=
q_m^\top R(m\theta)^\top R(n\theta)k_n
$$

Using the rotation identity:

$$
R(m\theta)^\top R(n\theta)
=
R((n-m)\theta)
$$

therefore:

$$
\boxed{
(q'_m)^\top k'_n
=
q_m^\top R((n-m)\theta)k_n
}
$$

The interaction is now a function of the relative displacement `n-m`.

**Memory liner:**

> Absolute rotations cancel into a relative rotation inside `QKᵀ`.

---

## 25. 🔥 Which tensors are rotated in the lecture's RoPE explanation?

```text
Q → rotate
K → rotate
V → not described as rotated
```

Shapes remain unchanged:

```text
Q before/after: [B, H, T, Dh]
K before/after: [B, H, T, Dh]
```

**Memory liner:**

> RoPE modifies the query–key geometry without changing their tensor shape.

---

## 26. ○ What two benefits of RoPE does the lecture emphasize?

1. Query–key interactions become functions of relative position.
2. The lecture reports a long-range decay property for an upper bound on attention interaction, with oscillations.

### △ Lecture boundary

The exact long-term-decay bound is not derived. The lecturer points to the RoFormer appendix for the mathematical result.

---

## 27. 🔥 Compare the position methods covered in this lecture.

| Method | Where position enters | Learned? | Key lecture idea |
|---|---|---:|---|
| Learned absolute embedding | Added to token input | Yes | Flexible, but finite learned range and data-dependent |
| Sinusoidal encoding | Added to token input | No | Multi-frequency formula exposes relative displacement |
| T5 relative bias | Added to attention logits | Yes | Learn bucketed relative-distance preferences |
| ALiBi | Added to attention logits | No | Deterministic distance-based bias |
| RoPE | Rotates Q and K | Fixed position rule in lecture | Relative position appears directly in `QKᵀ` |

**Memory liner:**

> Absolute methods modify inputs; relative-bias methods modify logits; RoPE modifies Q/K geometry.

---

# 9. Residual connections and LayerNorm

## 28. 🔥 What does “Add & Norm” mean in the original Transformer diagram?

For input `x` and sublayer `F`:

$$
y=\operatorname{LayerNorm}(x+F(x))
$$

This is now called **Post-Norm**, because normalization occurs after the residual addition.

Shapes:

```text
x:       [B, T, D]
F(x):    [B, T, D]
x + F(x):[B, T, D]
y:       [B, T, D]
```

**Memory liner:**

> Original Add & Norm = residual addition first, normalization second.

---

## 29. 🔥 What equation defines LayerNorm at the lecture's level?

For one activation vector `x\in\mathbb{R}^{D}`:

$$
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j
$$

$$
\sigma^2=\frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2
$$

$$
\boxed{
\operatorname{LN}(x)
=
\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
}
$$

Shapes:

```text
x:     [B, T, D]
μ, σ²: [B, T, 1]
γ, β:  [D]
output:[B, T, D]
```

**Memory liner:**

> Center and scale each token's feature vector, then learn a feature-wise scale and shift.

---

## 30. 🔥 Why normalize activations?

The lecture's intuition is that activation components can vary greatly across layers, making weight learning harder. Normalization keeps activations in a more controlled numerical range and is associated with improved training stability and convergence speed.

**Memory liner:**

> Keep layer activations numerically controlled so optimization is more stable.

The lecture mentions **internal covariate shift** as a keyword connected to this intuition but does not develop a formal treatment.

---

## 31. 🔥 LayerNorm vs BatchNorm — what distinction does the lecture make?

### LayerNorm

Normalize across the components of one activation vector:

```text
for each batch item and token:
normalize across D features
```

### BatchNorm

Normalize a feature using values from other examples in the batch.

```text
for each feature:
normalize using the batch dimension
```

**Memory liner:**

> LayerNorm uses features within an example; BatchNorm depends on batch statistics.

The lecture notes that Transformer models usually use layer-based normalization and that batch dependence can produce differences between training and inference behavior.

---

# 10. Post-Norm vs Pre-Norm

## 32. 🔥 What is Post-Norm?

$$
\boxed{
y=\operatorname{LN}(x+F(x))}
$$

```text
x ───────────────┐
                 + → LayerNorm → y
x → Sublayer F ──┘
```

**Memory liner:**

> Post-Norm normalizes after the residual merge.

---

## 33. 🔥 What is Pre-Norm?

$$
\boxed{
y=x+F(\operatorname{LN}(x))}
$$

```text
x ─────────────────────┐
                       + → y
x → LayerNorm → F ─────┘
```

The lecture states that modern architectures commonly place normalization before the attention or FFN sublayer.

**Memory liner:**

> Pre-Norm normalizes the sublayer input while leaving the residual path direct.

---

## 34. 🔥 What changes between Post-Norm and Pre-Norm, and what stays the same?

### Changes

The location of normalization relative to the sublayer and residual addition.

### Stays the same

```text
input shape:  [B, T, D]
output shape: [B, T, D]
residual addition remains
```

**Memory liner:**

> Same block dimensions and residual idea; different normalization placement.

---

# 11. RMSNorm

## 35. 🔥 What is RMSNorm?

For `x\in\mathbb{R}^{D}`:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}
$$

$$
\boxed{
\operatorname{RMSNorm}(x)
=
\gamma\odot\frac{x}{\operatorname{RMS}(x)}
}
$$

Shapes:

```text
x:      [B, T, D]
RMS:    [B, T, 1]
γ:      [D]
output: [B, T, D]
```

**Memory liner:**

> RMSNorm rescales by root mean square without subtracting the mean.

---

## 36. 🔥 LayerNorm vs RMSNorm?

| | LayerNorm | RMSNorm |
|---|---|---|
| Subtract mean? | Yes | No |
| Scale denominator | Standard deviation | Root mean square |
| Learned parameters in lecture | `γ` and `β` | `γ` only |
| Output shape | Same as input | Same as input |

The lecture says RMSNorm was found to have comparable convergence behavior while using fewer learned normalization parameters and less computation.

**Memory liner:**

> RMSNorm keeps the scaling part of LayerNorm and drops explicit centering and `β`.

---

# 12. Full attention complexity

## 37. 🔥 Why is full self-attention quadratic in sequence length?

Every query position interacts with every key position:

$$
T\times T=T^2
$$

Attention-score tensor:

```text
[B, H, T, T]
```

The lecture therefore describes full attention as having:

$$
\boxed{O(T^2)}
$$

pairwise interaction complexity with respect to sequence length.

**Memory liner:**

> `T` queries × `T` keys creates a `T × T` interaction map.

---

## 38. 🔥 Why does long context become expensive?

If sequence length doubles:

$$
T\rightarrow2T
$$

then the number of query–key pairs becomes:

$$
T^2\rightarrow4T^2
$$

**Memory liner:**

> Double the tokens, roughly quadruple the pairwise attention entries.

---

# 13. Sliding-window attention

## 39. 🔥 What is sliding-window attention?

Each token attends only to a local neighborhood rather than all `T` positions.

```text
Full attention:
query i → all keys 1 ... T

Sliding window:
query i → nearby keys around i
```

The resulting attention pattern is banded rather than dense.

**Memory liner:**

> Replace global token-to-token interaction with a moving local neighborhood.

---

## 40. 🔥 What is the interaction-count benefit of a window of width `W`?

Instead of roughly:

$$
T^2
$$

query–key pairs, local attention uses roughly:

$$
T\times W
$$

pairs.

```text
full interaction map:  T × T
local interaction map: T × W useful entries
```

**Memory liner:**

> Local attention scales with sequence length times window size, rather than all position pairs.

This is the direct counting consequence of the window restriction described in the lecture; the transcript does not give a detailed kernel-level complexity proof.

---

## 41. ○ How are local and global attention layers combined?

The lecture says some architectures interleave:

```text
local-attention layer
        ↓
global-attention layer
        ↓
local-attention layer
        ↓
...
```

There is no single recipe stated; combinations are model-dependent.

**Memory liner:**

> Use local layers for efficiency and occasional global layers for wider interaction.

---

## 42. 🔥 How is sliding-window attention analogous to convolutional receptive fields?

A token sees only a local neighborhood in one layer. Across successive layers, information can propagate through overlapping neighborhoods, increasing the token's effective receptive field.

```text
layer 1: token sees nearby tokens
layer 2: it receives information from their neighborhoods
layer 3: effective context grows further
```

**Memory liner:**

> Stacked local-attention layers expand effective context like stacked convolutions expand receptive field.

---

## 43. △ What does the lecture say about implementation tiling?

It notes that efficient implementations need not first materialize the entire dense attention matrix and then discard most entries; they can use clever tiled computation.

### Lecture boundary

The transcript does not name or derive a specific exact-attention kernel, memory hierarchy, or algorithm. Do not turn this into a FlashAttention note for this lecture.

---

# 14. Sharing key/value projections: MHA, GQA, MQA

## 44. 🔥 What is standard multi-head attention in terms of Q/K/V heads?

For `H` heads:

```text
H query heads
H key heads
H value heads
```

Shapes:

```text
Q: [B, H, T, Dh]
K: [B, H, T, Dh]
V: [B, H, T, Dh]
```

Each query head has its own corresponding key/value projections.

**Memory liner:**

> MHA gives every head its own Q, K, and V projections.

---

## 45. 🔥 What is Multi-Query Attention (MQA)?

All query heads share one set of keys and values:

```text
H query heads
1 key head
1 value head
```

Shapes:

```text
Q: [B, H, T, Dh]
K: [B, 1, T, Dh]
V: [B, 1, T, Dh]
```

The single K/V head is logically shared across all query heads.

**Memory liner:**

> MQA keeps many ways to ask but only one shared memory representation.

---

## 46. 🔥 What is Grouped-Query Attention (GQA)?

Use `H` query heads but only `G` key/value groups, where usually:

$$
1<G<H
$$

and each K/V group serves:

$$
\frac{H}{G}
$$

query heads.

Shapes:

```text
Q: [B, H, T, Dh]
K: [B, G, T, Dh]
V: [B, G, T, Dh]
```

**Memory liner:**

> GQA is the middle ground: many query heads share a smaller number of K/V heads.

---

## 47. 🔥 Give the unifying MHA–GQA–MQA mental map.

```text
MHA: G = H
GQA: 1 < G < H
MQA: G = 1
```

**Memory liner:**

> The variants differ mainly in how many distinct key/value groups remain.

---

## 48. 🔥 Why does the lecture keep query heads diverse but share K/V heads?

Two lecture-level motivations:

1. Multiple query heads preserve different ways of asking what information is relevant.
2. During autoregressive decoding, previous keys and values are repeatedly reused and stored, so reducing the number of K/V heads reduces memory requirements.

**Memory liner:**

> Preserve query diversity; compress the repeatedly stored key/value memory.

---

## 49. 🔥 What is the per-layer KV-cache shape implication?

The exact cache algorithm is deferred, but the head-sharing consequence is immediate.

### MHA

```text
cached K: [B, H, T, Dh]
cached V: [B, H, T, Dh]
```

### GQA

```text
cached K: [B, G, T, Dh]
cached V: [B, G, T, Dh]
```

### MQA

```text
cached K: [B, 1, T, Dh]
cached V: [B, 1, T, Dh]
```

**Memory liner:**

> KV-cache size is proportional to the number of K/V heads, not the number of query heads.

### △ Lecture boundary

The lecture explicitly postpones the full KV-cache mechanism to a later lecture. Do not add the complete decoding algorithm here.

---

## 50. ○ How does one choose MHA, GQA, or MQA according to the lecture?

The choice depends on:

```text
quality / model performance
latency constraints
memory and compute cost
model size
input length
```

The lecture says GQA is common in recent models, but not universal.

**Memory liner:**

> Head sharing is an empirical quality–latency–memory trade-off.

---

# 15. Transformer architecture families

## 51. 🔥 What are the three Transformer families introduced?

| Family | Components retained | Attention pattern | Lecture use case |
|---|---|---|---|
| Encoder–decoder | Encoder + decoder + cross-attention | Bidirectional source; causal target | Sequence-to-sequence, e.g. translation/T5 |
| Encoder-only | Encoder only | Fully bidirectional self-attention | Classification and representation tasks, e.g. BERT |
| Decoder-only | Decoder stack without encoder/cross-attention | Masked causal self-attention | Next-token generation / modern LLM direction |

**Memory liner:**

> Encoder–decoder transforms one sequence into another; encoder-only understands; decoder-only generates causally.

---

## 52. 🔥 What disappears in a decoder-only Transformer?

There is no encoder output, so cross-attention disappears.

A decoder-only block, at the level stated in the lecture, contains:

```text
masked self-attention
        ↓
FFN
```

with residual and normalization machinery around the sublayers.

**Memory liner:**

> No encoder means no encoder memory and no cross-attention.

---

## 53. ○ Why does the lecture say decoder-only models became dominant?

The lecture's framing is:

- compute could be concentrated in the decoder;
- next-token prediction is simple to construct and scale;
- the objective aligns naturally with free-form generation and chatbot behavior.

**Memory liner:**

> A simple scalable causal objective made decoder-only models attractive for general generation.

### △ Lecture boundary

The lecture postpones a deep treatment of decoder-only LLMs to later lectures.

---

# 16. T5 and encoder–decoder pre-training

## 54. 🔥 What architectural family does T5 belong to?

T5 retains both:

```text
encoder
   +
decoder
```

and therefore uses encoded source representations through decoder cross-attention.

**Memory liner:**

> T5 is an encoder–decoder Transformer family.

---

## 55. ○ What variants of T5 are mentioned?

- **T5:** the base text-to-text family.
- **mT5:** multilingual variation, involving multilingual data/vocabulary changes.
- **ByT5:** byte-level input representation with a much smaller basic byte vocabulary.

The lecture contrasts ordinary vocabularies of roughly tens of thousands with the 256 possible byte values used by a byte-level representation.

**Memory liner:**

> mT5 changes language coverage; ByT5 changes the basic input unit to bytes.

---

## 56. 🔥 What is span corruption?

Replace one or more contiguous spans in the encoder input with sentinel tokens.

Example:

```text
Original:
my teddy bear is cute and reading

Corrupted encoder input:
my teddy bear <S1> is reading
```

The decoder is trained to reconstruct the missing span(s).

**Memory liner:**

> Corrupt spans in the encoder input; make the decoder serialize the missing text.

---

## 57. 🔥 What are sentinel tokens in T5 span corruption?

A sentinel token marks the location and identity/order of a removed span.

With multiple spans:

```text
Encoder input:
A <S1> B <S2> C

Decoder target:
<S1> missing_span_1 <S2> missing_span_2 <S3>
```

Tokens produced between consecutive sentinels reconstruct the corresponding missing span.

**Memory liner:**

> Sentinel tokens delimit which missing span the decoder is currently reconstructing.

---

## 58. 🔥 What are the key tensor shapes for T5-style span reconstruction?

```text
Corrupted encoder IDs: [B, T_enc]
Encoder states:        [B, T_enc, D]
Decoder input IDs:     [B, T_dec]
Decoder states:        [B, T_dec, D]
Vocabulary logits:     [B, T_dec, V]
Target IDs:            [B, T_dec]
```

Cross-attention scores have shape:

```text
[B, H, T_dec, T_enc]
```

**Memory liner:**

> Encoder length and reconstruction-target length can differ; cross-attention bridges them.

---

## 59. 🔥 How is the T5 decoder trained according to the lecture?

The lecture explicitly mentions **teacher forcing**: provide the target sequence to the decoder during training and train the target positions together under causal masking.

**Memory liner:**

> During training, the decoder receives the known shifted reconstruction target rather than waiting for sampled tokens one by one.

The transcript does not develop the exact shift convention or loss masking.

---

# 17. Encoder-only models and BERT

## 60. 🔥 What does BERT stand for?

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers.

Break it down:

```text
Bidirectional:
each token can attend left and right

Encoder:
retain the Transformer encoder stack

Representations:
produce contextual vectors for tokens / sequence

Transformers:
self-attention + FFN encoder blocks
```

**Memory liner:**

> BERT is a bidirectional Transformer encoder trained to produce reusable contextual representations.

---

## 61. 🔥 Why is BERT called bidirectional?

Encoder self-attention has no causal future mask. Every input token can attend to tokens on both its left and right.

```text
BERT encoder token i:
can attend to positions 1 ... T

Causal decoder token i:
can attend only to positions ≤ i
```

**Memory liner:**

> BERT sees both sides of a token; causal models see only the prefix.

---

## 62. 🔥 What is the structural difference between BERT and the original encoder–decoder Transformer?

BERT removes the decoder and keeps the encoder stack.

```text
Original Transformer:
encoder → cross-attentive decoder

BERT:
encoder only → task-specific prediction head
```

There is no autoregressive decoder or cross-attention module.

**Memory liner:**

> BERT is the original Transformer's encoder repurposed for representation learning and prediction heads.

---

# 18. ELMo vs BERT

## 63. ○ What is ELMo at the level discussed in the lecture?

ELMo constructs contextual bidirectional word representations using stacked bidirectional LSTMs.

```text
forward LSTM  →
                 contextual word representation
backward LSTM ←
```

The lecture contrasts ELMo's recurrence with BERT's Transformer encoder and says recurrence makes ELMo harder to scale.

**Memory liner:**

> ELMo achieved bidirectional context with recurrent networks; BERT did it with parallel self-attention.

---

# 19. BERT special tokens

## 64. 🔥 What is `[CLS]`?

A learned placeholder token placed at the beginning of the sequence.

```text
[CLS] token_1 token_2 ... token_T
```

It participates in self-attention like every other token. Its final encoder representation is used as a sequence-level representation for classification.

Shape:

```text
all encoder outputs: [B, T, D]
CLS output:          [B, D]
```

**Memory liner:**

> `[CLS]` is a normal learned token whose final contextual vector is reserved for sequence-level decisions.

---

## 65. 🔥 Why can `[CLS]` summarize the whole sequence?

Because every encoder layer lets the `[CLS]` query interact with all token keys/values, and subsequent layers repeatedly transform the mixed representation.

```text
[CLS] initial embedding
        ↓ self-attention over all tokens
contextual [CLS]
        ↓ more encoder layers
sequence-level representation
```

**Memory liner:**

> `[CLS]` becomes global because it attends to the same bidirectional context as every other token.

---

## 66. 🔥 What is `[SEP]`?

A separator token used to mark sentence boundaries, particularly when two sentences are presented for next-sentence prediction.

Typical lecture-level structure:

```text
[CLS] sentence A [SEP] sentence B [SEP]
```

**Memory liner:**

> `[SEP]` marks where one input segment ends and another begins.

---

## 67. ○ What roles do `[MASK]` and `[PAD]` play?

### `[MASK]`

Used for selected masked-language-model positions.

### `[PAD]`

Used to extend shorter examples so sequences in a batch can be stored in a fixed-size tensor.

```text
batch token IDs: [B, T_fixed]
```

### △ Lecture boundary

The lecture mentions padding but does not explain the padding attention mask. Keep that for a later note.

---

# 20. BERT input representation

## 68. 🔥 What three embeddings are added to form a BERT input token representation?

$$
\boxed{
x_t=e^{token}_t+e^{position}_t+e^{segment}_t}
$$

Shapes:

```text
Token embedding:   [B, T, D]
Position embedding:[B, T, D] after lookup/broadcast
Segment embedding: [B, T, D]
Final input:       [B, T, D]
```

**Memory liner:**

> BERT input = what token + where it is + which sentence segment it belongs to.

---

## 69. 🔥 What is segment encoding?

Each token receives one of two learned segment embeddings:

```text
segment A embedding → all tokens in sentence A
segment B embedding → all tokens in sentence B
```

Segment IDs:

```text
[0, 0, 0, ..., 1, 1, 1, ...]
```

Embedding table:

```text
E_segment: [2, D]
```

**Memory liner:**

> Segment embeddings tell BERT which of the two input sentences each token came from.

---

## 70. △ What position encoding does BERT use according to this lecture?

The lecturer says position information is added but explicitly expresses uncertainty about whether the BERT authors used the hard-coded or learned version.

**Do not memorize a definitive answer from this transcript.**

**Memory liner:**

> This lecture establishes additive BERT position information but does not reliably settle its exact implementation.

---

## 71. 🔥 What tokenizer does the lecture associate with BERT?

**WordPiece**, described as learning subword merge decisions from the training corpus to create a target vocabulary.

The lecture gives approximately:

```text
V ≈ 30,000
```

as the BERT-scale vocabulary order in the example.

**Memory liner:**

> BERT maps text to WordPiece IDs, then learns an embedding lookup over that subword vocabulary.

---

## 72. ○ What do cased and uncased BERT mean?

```text
cased:
capitalization is retained as input information

uncased:
text is lowercased during preprocessing
```

**Memory liner:**

> Cased vs uncased describes whether capitalization survives preprocessing/tokenization.

---

# 21. BERT multi-stage training

## 73. 🔥 What are BERT's two training stages in the lecture?

### Stage 1: pre-training

Learn general contextual representations from unlabeled text using:

```text
MLM + NSP
```

### Stage 2: fine-tuning

Attach a task-specific prediction head and train for the downstream task, optionally updating some or all pre-trained weights.

**Memory liner:**

> Pre-train general representations first; specialize them with a small supervised task head later.

---

## 74. 🔥 Why is BERT pre-training described as self-supervised?

The text itself supplies the targets:

- MLM target = original token before corruption.
- NSP target = whether two corpus sentences were actually consecutive.

No separate human annotation is needed for these proxy targets.

**Memory liner:**

> Construct labels from raw text structure instead of paying humans to annotate them.

---

# 22. Masked Language Modeling — MLM

## 75. 🔥 What is BERT's MLM task?

Select some token positions, alter their inputs, and predict the original token from the bidirectional context.

```text
original sequence
      ↓ select positions
corrupted sequence
      ↓ bidirectional encoder
predict original IDs at selected positions
```

**Memory liner:**

> Hide or perturb selected tokens, then recover them using both left and right context.

---

## 76. 🔥 What 80/10/10 replacement rule is given in the lecture?

For tokens already selected for the MLM objective:

```text
80% → replace with [MASK]
10% → keep the original token unchanged
10% → replace with a random token
```

**Memory liner:**

> Selected MLM positions are mostly masked, sometimes left visible, and sometimes corrupted randomly.

### △ Lecture boundary

The transcript does not state what percentage of all sequence positions are selected for MLM. Do not insert the commonly remembered selection rate into this lecture note.

---

## 77. 🔥 What are the MLM tensor shapes?

Encoder output:

```text
H: [B, T, D]
```

If `M` selected positions are gathered:

```text
Selected states: [M, D]
MLM logits:      [M, V]
Target token IDs:[M]
```

An implementation may also compute:

```text
full logits: [B, T, V]
```

and include loss only at selected positions. The lecture's conceptual requirement is prediction on the selected subset.

**Memory liner:**

> Every selected token state is projected back to a vocabulary distribution.

---

## 78. 🔥 Why does MLM encourage bidirectional contextualization?

To recover a corrupted token, the encoder can use evidence on both sides because there is no causal mask.

```text
left context ← missing token → right context
```

**Memory liner:**

> MLM makes both left and right neighbors useful for predicting the centre.

---

## 79. △ Why exactly use the 80/10/10 mixture?

The lecture states the procedure but does not provide a detailed motivation or ablation. Keep the rule; do not invent its rationale in this lecture note.

---

# 23. Next Sentence Prediction — NSP

## 80. 🔥 What is the NSP task?

Present sentence A and sentence B together, then classify whether B truly follows A in the corpus.

The lecture gives the data construction as approximately:

```text
50%: A and B are consecutive
50%: B is selected in a non-consecutive/random way
```

Input:

```text
[CLS] A [SEP] B [SEP]
```

**Memory liner:**

> NSP is binary classification of whether two presented sentences are consecutive.

---

## 81. 🔥 How is the NSP prediction made?

Use the final `[CLS]` representation:

```text
CLS state: [B, D]
      ↓ binary classifier
NSP logits:[B, 2]
```

**Memory liner:**

> Sentence-pair structure enters through `[SEP]` and segment IDs; the decision comes from `[CLS]`.

---

## 82. 🔥 Is NSP a decoding task?

No. In this lecture, NSP is an encoder-only binary classification task.

It does not generate the next sentence. It only predicts whether sentence B follows sentence A.

**Memory liner:**

> “Next sentence prediction” predicts a relationship label, not the sentence text.

---

# 24. Fine-tuning BERT

## 83. 🔥 How is BERT fine-tuned for sequence classification?

```text
input IDs:        [B, T]
encoder outputs:  [B, T, D]
CLS state:        [B, D]
classifier weight:[D, C]
class logits:     [B, C]
```

The classifier can be a linear layer or a small FFN.

**Memory liner:**

> Sequence classification reads the final `[CLS]` vector and maps it to class logits.

---

## 84. 🔥 How is BERT used for token-level classification?

Retain every contextual token representation:

```text
encoder outputs: [B, T, D]
classifier:      D → C
logits:          [B, T, C]
```

**Memory liner:**

> Sequence labels use `[CLS]`; token labels use the corresponding token outputs.

---

## 85. 🔥 How is extractive question answering formulated in the lecture?

Predict which input token begins and ends the answer span.

```text
encoder outputs: [B, T, D]
start head:      D → 1
end head:        D → 1

start logits:    [B, T]
end logits:      [B, T]
```

**Memory liner:**

> Extractive QA turns each token into a candidate answer start and answer end.

---

## 86. ○ Must the BERT encoder be frozen during downstream training?

No single rule is prescribed. The lecture mentions both options:

```text
freeze encoder → train only small task head

or

update encoder + task head → full fine-tuning
```

The choice depends on compute and how different the target task is from pre-training.

**Memory liner:**

> Fine-tuning can mean head-only training or updating the entire pre-trained model.

---

# 25. BERT strengths and limitations in the lecture

## 87. 🔥 What are the main strengths attributed to BERT?

- Contextual representations.
- True bidirectional self-attention over the input.
- Self-supervised pre-training on large unlabeled text.
- Small labeled datasets can then adapt the representations to many classification-style tasks.

**Memory liner:**

> Learn general bidirectional language features once, then reuse them across tasks.

---

## 88. 🔥 Why is BERT not the lecture's model for free-form generation?

BERT retains only the encoder. It does not include a causal autoregressive decoder that repeatedly predicts the next token.

**Memory liner:**

> Bidirectional encoding produces representations; it does not define the lecture's autoregressive generation loop.

---

## 89. 🔥 What BERT limitations are listed?

1. Limited context in the original setting, discussed as about 512 tokens.
2. Quadratic full self-attention cost.
3. BERT-base-scale parameter count and resulting latency/cost concerns, discussed around 110 million parameters.
4. No decoder for ordinary text generation.
5. Multi-stage pre-training and downstream adaptation.
6. Open question: are both MLM and NSP necessary?

**Memory liner:**

> BERT is powerful for representations, but context cost, size, task pipeline, and objective design leave room for improvement.

---

# 26. Knowledge distillation and DistilBERT

## 90. 🔥 What is knowledge distillation?

Train a smaller **student** model to match the output distribution of a larger **teacher** model.

```text
input
 ├── teacher → soft distribution p_T
 └── student → distribution p_S

train student so p_S ≈ p_T
```

**Memory liner:**

> Transfer the teacher's probability structure, not only its top hard label.

---

## 91. 🔥 Why can soft targets carry more information than hard labels?

A hard label says only which class is correct:

```text
[0, 1, 0, 0]
```

A teacher distribution also reveals relative beliefs about alternatives:

```text
[0.05, 0.75, 0.15, 0.05]
```

The lecture's key idea is that this distribution reflects knowledge learned by the teacher.

**Memory liner:**

> Soft targets reveal which wrong answers the teacher considers more or less plausible.

---

## 92. 🔥 What KL-divergence objective is introduced?

For teacher distribution `p_T` and student distribution `p_S`:

$$
\boxed{
D_{KL}(p_T\|p_S)
=
\sum_i p_T(i)
\log\frac{p_T(i)}{p_S(i)}
}
$$

The student is trained to reduce this discrepancy.

Shape example:

```text
Teacher probabilities: [B, C] or another shared output support
Student probabilities: [B, C]
KL reduced over C
```

**Memory liner:**

> KL asks how costly it is to model the teacher's world using the student's distribution.

---

## 93. 🔥 How does hard-label cross-entropy appear as a special case?

If the teacher target is one-hot at class `y`:

$$
p_T(y)=1
$$

then:

$$
D_{KL}(p_T\|p_S)
=
-\log p_S(y)
$$

because the one-hot teacher has zero entropy.

This is the standard hard-target cross-entropy term.

**Memory liner:**

> A one-hot teacher collapses distillation back to ordinary negative log-likelihood.

---

## 94. ○ What is DistilBERT's lecture-level strategy?

Use a smaller student with fewer layers and distillation so it retains much of the larger BERT model's performance while reducing inference cost.

The lecture's takeaway is qualitative: substantial depth reduction plus teacher guidance can preserve performance surprisingly well.

**Memory liner:**

> Shrink the encoder, then recover quality by imitating the larger teacher.

### △ Lecture boundary

Temperature scaling and the full DistilBERT multi-term objective are not covered here.

---

# 27. RoBERTa

## 95. 🔥 What did RoBERTa change according to the lecture?

1. Removed the NSP objective with little or no performance loss.
2. Used dynamic masking so the same text can receive different masks across epochs.
3. Increased data size/diversity and trained more thoroughly, arguing that the original BERT setup was undertrained.

**Memory liner:**

> RoBERTa simplified the objective and strengthened the data/training recipe.

---

## 96. 🔥 What is dynamic masking?

Instead of fixing one corruption pattern permanently:

```text
same text, epoch 1 → mask pattern A
same text, epoch 2 → mask pattern B
same text, epoch 3 → mask pattern C
```

**Memory liner:**

> Re-sample which tokens are masked each time the text is seen.

---

## 97. 🔥 What does removing NSP test scientifically?

It tests whether NSP actually contributes useful representation learning beyond MLM and the rest of the training recipe.

The lecture's reported conclusion is that dropping NSP did not meaningfully hurt performance in RoBERTa's setup.

**Memory liner:**

> Ablate a pre-training objective to see whether its assumed benefit is real.

---

# 28. End-to-end BERT dataflow to remember

## 98. 🔥 Walk through BERT sequence classification from raw text to logits.

```text
raw text
   ↓ optional lowercase preprocessing
WordPiece tokenization
   ↓
[CLS] + tokens + [SEP] + [PAD]
   ↓
token IDs       [B,T]
position IDs    [B,T]
segment IDs     [B,T]
   ↓ lookup and add
X = token + position + segment
X: [B,T,D]
   ↓
L bidirectional Transformer encoder blocks
   ↓
contextual outputs H: [B,T,D]
   ↓ select CLS
H_CLS: [B,D]
   ↓ task classifier
logits: [B,C]
```

**Memory liner:**

> Tokenize, add three embedding types, encode bidirectionally, classify from `[CLS]`.

---

## 99. 🔥 Walk through BERT MLM pre-training.

```text
original token IDs [B,T]
        ↓ select MLM positions
80% [MASK] / 10% unchanged / 10% random
        ↓
corrupted IDs [B,T]
        ↓
token + position + segment embeddings
        ↓
bidirectional encoder [B,T,D]
        ↓ gather selected positions
selected states [M,D]
        ↓ vocabulary projection
MLM logits [M,V]
        ↓
compare with original token IDs [M]
```

**Memory liner:**

> Corrupt the input, but supervise against the uncorrupted token identities.

---

## 100. 🔥 Walk through BERT NSP pre-training.

```text
sentence A + sentence B
        ↓
[CLS] A [SEP] B [SEP]
        ↓
segment IDs A/B
        ↓
bidirectional encoder
        ↓
CLS state [B,D]
        ↓
binary head
        ↓
IsNext / NotNext logits [B,2]
```

**Memory liner:**

> Encode the pair jointly, then classify the sentence relation from `[CLS]`.

---

# 29. Equations to memorize cold from this lecture

## Attention recap

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{D_h}}
\right)V
}
$$

## Additive input position

$$
\boxed{x_t=e_{token,t}+p_t}
$$

## Sinusoidal frequency

$$
\boxed{\omega_i=10000^{-2i/D}}
$$

## Sinusoidal pair

$$
\boxed{PE(m,2i)=\sin(\omega_i m)}
$$

$$
\boxed{PE(m,2i+1)=\cos(\omega_i m)}
$$

## Relative-distance identity

$$
\boxed{
\sin a\sin b+\cos a\cos b=\cos(a-b)
}
$$

## Learned relative attention bias

$$
\boxed{
s_{m,n}
=
\frac{q_m^\top k_n}{\sqrt{D_h}}
+b(\operatorname{bucket}(m-n))
}
$$

## Rotation matrix

$$
\boxed{
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
}
$$

## RoPE relative-position identity

$$
\boxed{
(R_mq)^\top(R_nk)
=
q^\top R_{n-m}k
}
$$

## LayerNorm

$$
\boxed{
\operatorname{LN}(x)
=
\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
}
$$

## Post-Norm

$$
\boxed{y=\operatorname{LN}(x+F(x))}
$$

## Pre-Norm

$$
\boxed{y=x+F(\operatorname{LN}(x))}
$$

## RMSNorm

$$
\boxed{
\operatorname{RMSNorm}(x)
=
\gamma\odot
\frac{x}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}
}
$$

## BERT input embedding

$$
\boxed{
x_t=e_t^{token}+e_t^{position}+e_t^{segment}}
$$

## KL distillation loss

$$
\boxed{
D_{KL}(p_T\|p_S)
=
\sum_i p_T(i)\log\frac{p_T(i)}{p_S(i)}
}
$$

---

# 30. Tensor shapes to memorize cold

```text
Input token IDs
[B, T]

Token / position / segment embeddings
[B, T, D]

Multi-head Q
[B, H, T, Dh]

MHA K/V
[B, H, T, Dh]

GQA K/V
[B, G, T, Dh]

MQA K/V
[B, 1, T, Dh]

Full self-attention scores
[B, H, T, T]

Encoder–decoder cross-attention scores
[B, H, T_dec, T_enc]

LayerNorm or RMSNorm input/output
[B, T, D]

BERT encoder outputs
[B, T, D]

BERT CLS state
[B, D]

Sequence-classification logits
[B, C]

Token-classification logits
[B, T, C]

Extractive-QA start logits
[B, T]

Extractive-QA end logits
[B, T]

MLM selected states
[M, D]

MLM selected-position logits
[M, V]

NSP logits
[B, 2]

T5 decoder vocabulary logits
[B, T_dec, V]
```

The two shape relations that matter most are:

$$
\boxed{
[B,H,T,D_h]
\times
[B,H,D_h,T]
=
[B,H,T,T]
}
$$

and for encoder–decoder cross-attention:

$$
\boxed{
[B,H,T_{dec},D_h]
\times
[B,H,D_h,T_{enc}]
=
[B,H,T_{dec},T_{enc}]
}
$$

---

# 31. Ten high-value comparisons from this lecture

## Position methods

```text
Learned absolute:
lookup position → add to input

Sinusoidal:
formula position → add to input

T5 relative bias:
learn distance-bucket bias → add to logits

ALiBi:
deterministic distance bias → add to logits

RoPE:
rotate Q/K → relative offset appears in dot product
```

## Normalization

```text
Post-Norm:
LN(x + F(x))

Pre-Norm:
x + F(LN(x))
```

## LayerNorm vs RMSNorm

```text
LayerNorm:
centre + rescale, learn γ and β

RMSNorm:
rescale without centring, learn γ
```

## Attention range

```text
Full attention:
all T × T pairs

Sliding window:
local T × W useful pairs
```

## K/V sharing

```text
MHA: H KV heads
GQA: G KV heads
MQA: 1 KV head
```

## Architecture family

```text
Encoder–decoder:
source encoding + causal target decoding

Encoder-only:
bidirectional contextual representations

Decoder-only:
causal next-token prediction
```

## T5 vs BERT objective

```text
T5:
span corruption → decoder reconstructs spans

BERT:
MLM + NSP → encoder learns representations
```

## BERT MLM vs NSP

```text
MLM:
predict original token identity

NSP:
predict whether sentence B follows sentence A
```

## Sequence vs token downstream task

```text
Sequence classification:
use [CLS]

Token classification / QA:
use each token's output
```

## BERT vs DistilBERT vs RoBERTa

```text
BERT:
MLM + NSP, full base encoder

DistilBERT:
smaller student imitates teacher distribution

RoBERTa:
remove NSP, dynamic masks, more/better training data
```

---

# 32. Interview questions to practise aloud

1. Why does self-attention require explicit position information?
2. What is the parameter and extrapolation trade-off of learned absolute positions?
3. Write the sinusoidal positional equations from memory.
4. Show why a sine/cosine pair produces `cos(ω(m-n))` in a dot product.
5. Why is sinusoidal similarity not strictly monotonic with distance?
6. Why inject relative position directly into attention logits?
7. How does T5 relative position bias work at a high level?
8. How does ALiBi differ from learned relative position bias in this lecture?
9. Write the 2D rotation matrix used to explain RoPE.
10. Derive why rotated Q/K dot products depend on `n-m`.
11. What is the tensor shape before and after applying RoPE?
12. What does Add & Norm mean in the original Transformer?
13. Write LayerNorm with `μ`, `σ²`, `γ`, and `β`.
14. What is the difference between LayerNorm and BatchNorm?
15. Write Post-Norm and Pre-Norm equations.
16. What does RMSNorm remove relative to LayerNorm?
17. Why is full attention `O(T²)` in sequence length?
18. What is sliding-window attention?
19. How can stacking local-attention layers grow the effective receptive field?
20. Why should efficient local attention avoid computing the full dense matrix first?
21. Compare MHA, GQA, and MQA using head counts and tensor shapes.
22. Why does K/V sharing reduce the future KV-cache footprint?
23. What are the three Transformer architecture families?
24. What disappears when moving from encoder–decoder to decoder-only?
25. What is T5 span corruption?
26. What are sentinel tokens and what does the T5 decoder emit?
27. Why is T5 training still causal in the decoder despite seeing the full target during teacher forcing?
28. What does BERT stand for, component by component?
29. Why is BERT truly bidirectional at the encoder level?
30. What are `[CLS]`, `[SEP]`, `[MASK]`, and `[PAD]` used for?
31. Why can the final `[CLS]` representation support sequence classification?
32. Write the three-term BERT input embedding equation.
33. What does a BERT segment embedding encode?
34. What is the 80/10/10 MLM corruption rule?
35. Which MLM positions contribute to the prediction task?
36. Is NSP generative next-sentence modelling or binary classification?
37. Draw the NSP input and output tensor shapes.
38. Compare sequence classification, token classification, and extractive QA heads on BERT.
39. What are the benefits of pre-training on self-supervised text before fine-tuning?
40. What are the lecture's main limitations of BERT?
41. What information does a teacher's soft distribution provide beyond a hard label?
42. Write the KL divergence used for teacher–student matching.
43. Show how one-hot teacher targets reduce to ordinary cross-entropy.
44. What is DistilBERT's high-level efficiency strategy?
45. Which BERT training assumptions did RoBERTa challenge?
46. What is dynamic masking?
47. Why is removing NSP an informative ablation?

---

# 33. Fifteen memory liners for interview day

```text
1. Attention map:
   One row tells where one query token retrieves information.

2. Position need:
   Attention knows interactions, not inherent sequence order.

3. Learned positions:
   Flexible lookup, but finite trained range.

4. Sinusoidal positions:
   Multi-frequency sine/cosine clocks encode relative offset.

5. Relative bias:
   Put distance preference directly inside attention logits.

6. RoPE:
   Rotate Q and K; absolute angles collapse to relative rotation.

7. LayerNorm:
   Normalize each token across its feature dimension.

8. Pre-Norm:
   Normalize before the sublayer and keep a direct residual route.

9. RMSNorm:
   Rescale by root mean square without mean subtraction.

10. Sliding window:
    Local attention replaces all-pairs interaction with neighborhoods.

11. GQA:
    Keep many query heads but share fewer K/V heads.

12. Model families:
    Encoder–decoder transforms; encoder-only represents; decoder-only generates.

13. BERT input:
    Token + position + segment embedding.

14. BERT pre-training:
    MLM recovers tokens; NSP classifies sentence order.

15. BERT variants:
    DistilBERT compresses with a teacher; RoBERTa improves the training recipe.
```

---

# 34. Lecture boundaries — do not silently add these to this note

The transcript mentions or motivates some topics but does **not** fully teach the following:

- Exact ALiBi formula and head-specific slopes.
- Full proof of RoPE's long-range attention upper-bound decay.
- Exact KV-cache decoding algorithm and implementation.
- FlashAttention or a named tiled-attention kernel.
- Full decoder-only LLM training pipeline.
- Exact teacher-forcing target shift convention in T5.
- The fraction of all BERT tokens selected for MLM.
- Why the MLM replacement rule uses precisely 80/10/10.
- Padding attention masks.
- A reliable definitive statement in this transcript about whether BERT's positions are learned or hard-coded; the lecturer expresses uncertainty.
- Distillation temperature and DistilBERT's complete loss.
- Detailed BERT, T5, DistilBERT, or RoBERTa optimizer/hyperparameter recipes.

Keep these for separate notes when their lectures introduce them.

---

# Final 60-second lecture mental map

```text
The original Transformer survives, but key components evolved:

POSITION
absolute input embeddings
    → relative logit biases
    → RoPE rotations of Q/K

NORMALIZATION
Post-Norm LayerNorm
    → Pre-Norm
    → often RMSNorm

ATTENTION COST
full T² attention
    → local/sliding windows

ATTENTION MEMORY
MHA
    → GQA
    → MQA

MODEL FAMILY
encoder–decoder → T5 span corruption
encoder-only   → BERT MLM + NSP
                    ├── DistilBERT: compress by teacher imitation
                    └── RoBERTa: remove NSP, dynamic masks, more training

decoder-only  → introduced as the modern LLM direction,
                but deferred to later lectures
```
