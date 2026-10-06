# CME 295 — Lecture 1 Interview Revision Questions

> **Scope:** This sheet is restricted to concepts actually introduced or discussed in the uploaded lecture transcript. It intentionally excludes later-topic additions such as BPE algorithms, full LSTM gate equations, residual connections, LayerNorm, teacher forcing, decoder-only LLMs, RoPE, KV cache, FlashAttention, and modern inference techniques.
>
> **Goal:** Revise this lecture non-sequentially through interview questions, memory liners, equations, mechanisms, and tensor shapes.

---

## 0. Lecture mental map

```text
NLP task types and evaluation
            ↓
        Tokenization
            ↓
 One-hot token representation
            ↓
 Learned token embeddings / Word2Vec
            ↓
 RNNs and sequential representations
            ↓
 Long-range dependency problem
            ↓
          Attention
            ↓
 Encoder–decoder Transformer
            ↓
 Autoregressive prediction until EOS
```

**Master memory liner:**

> Turn text into tokens, tokens into representations, representations into contextual representations, and contextual representations into predictions.

---

## Notation used in this sheet

The lecture sometimes presents matrices as `D_model × sequence length`. This sheet uses the common row-per-token convention:

| Symbol | Meaning |
|---|---|
| `V` | vocabulary size |
| `N` | sequence length |
| `N_s` | source-sequence length |
| `N_t` | target-sequence length |
| `D` | token-embedding / model dimension |
| `D_hid` | hidden dimension in the Word2Vec-like network or RNN context |
| `d_k` | query/key projection dimension |
| `d_v` | value projection dimension |
| `H` | number of attention heads |

**Batch dimension is omitted** because the lecture explains one sequence at a time.

---

# 1. NLP task formulations and evaluation

_Lecture region: approximately 00:10:53–00:21:03._

## Q1. What is NLP in the framing of this lecture?

**Answer:** Natural language processing is the field concerned with manipulating and computing with text.

**Memory liner:**

> NLP makes text usable by computational models.

---

## Q2. What are the three high-level NLP task buckets introduced in the lecture?

1. **Classification:** one input text → one prediction.
2. **Multi-classification / token-level prediction:** one input text → multiple predictions over parts of the text.
3. **Generation:** input text → variable-length output text.

**Memory liner:**

> One label, many local labels, or a generated sequence.

---

## Q3. What are examples of text classification?

- Sentiment classification.
- Intent detection.
- Language detection.
- Topic modelling.

**Mental model:**

```text
input text
    ↓
one sequence-level prediction
```

---

## Q4. What does the lecture call “multi-classification,” and what are its examples?

It means predicting multiple labels associated with different parts of the input text.

Examples:

- Named entity recognition.
- Part-of-speech tagging.
- Dependency parsing.
- Constituency parsing.

**Memory liner:**

> Instead of labelling the whole sentence, label its components.

---

## Q5. What distinguishes generation from classification?

Generation produces text whose output length is not known beforehand.

Examples:

- Machine translation.
- Question answering.
- Summarisation.
- Code generation.
- Poem or general text generation.

**Memory liner:**

> Classification chooses from fixed labels; generation builds a variable-length sequence.

---

## Q6. What is accuracy?

Accuracy is the percentage of observations predicted correctly.

**Memory liner:**

> Accuracy asks: how often was the prediction correct overall?

---

## Q7. What is precision?

Precision asks: among all examples predicted as positive, how many were actually positive?

### Equation to remember

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

**Memory liner:**

> When I said “positive,” how often was I right?

---

## Q8. What is recall?

Recall asks: among all examples that were truly positive, how many did the model identify as positive?

### Equation to remember

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

**Memory liner:**

> Of all real positives, how many did I find?

---

## Q9. What is the F1 score?

F1 is the harmonic mean of precision and recall.

### Equation to remember

$$
F_1=2\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

**Memory liner:**

> F1 compresses precision and recall into one balanced score.

---

## Q10. Why can accuracy be misleading on an imbalanced dataset?

If 99% of examples belong to one class, a model can predict that majority class for every example and obtain about 99% accuracy while failing to identify the minority class.

**Memory liner:**

> High accuracy can hide complete minority-class failure.

---

## Q11. At what level should named entity recognition be evaluated?

The lecture says classification-style metrics can be aggregated at:

- Token level.
- Entity-type level, such as performance on locations.

**Memory liner:**

> Match the metric aggregation level to the prediction unit.

---

## Q12. Why is generation evaluation harder than classification evaluation?

There can be multiple valid ways to express or translate the same meaning.

**Memory liner:**

> In generation, one input may have many acceptable outputs.

---

## Q13. Why do machine-translation datasets require paired text?

The source-language sequence must be aligned with a target-language sequence. The lecture mentions WMT as a well-known collection of paired translation data.

**Memory liner:**

> Translation supervision requires source–target pairs.

---

## Q14. What is BLEU at the level covered in the lecture?

BLEU measures how well a generated translation agrees with a reference text.

**Direction:** Higher is better.

**Memory liner:**

> BLEU compares a candidate translation against reference wording.

---

## Q15. What is ROUGE at the level covered in the lecture?

ROUGE is a family of reference-based metrics that compares generated output with reference text in a different way from BLEU.

**Direction:** Higher is better.

**Memory liner:**

> ROUGE is another reference-overlap family, not one single metric.

---

## Q16. What is the limitation of reference-based generation metrics?

They require labelled reference text, which is expensive and time-consuming to create.

The lecture also motivates later discussion of reference-free metrics but does not explain one in this session.

**Memory liner:**

> Reference metrics need expensive answers written in advance.

---

## Q17. What is perplexity at the level covered in this lecture?

Perplexity uses probabilities output by the model and measures how surprised the model is.

**Direction:** Lower is better.

**Memory liner:**

> Lower perplexity means the observed text was less surprising to the model.

**Lecture boundary:** The exact perplexity equation is not derived in this lecture.

---

## Q18. What historical progression does the lecture emphasise?

```text
RNN ideas:          1980s
LSTMs:              1990s
Word2Vec:           2010s / 2013
Transformer:        2017
Large-scale LLMs:   2020s through more data and compute
```

**Memory liner:**

> Modern LLMs combine older modelling ideas with much larger data and compute.

---

# 2. Tokenization

_Lecture region: approximately 00:23:00–00:30:08._

## Q19. Why is tokenization needed?

Models operate on numerical representations rather than raw text. Tokenization first divides text into discrete units that can later be represented numerically.

**Memory liner:**

> Before representing text, decide what the units of text are.

---

## Q20. What is a token?

A token is one unit produced by the tokenizer.

Depending on the tokenizer, it may be:

- A word.
- A subword fragment.
- A character.

---

## Q21. What is word-level tokenization?

The sentence is split into complete words.

**Advantage:** Simple and often creates shorter sequences.

**Problems:**

- Similar forms such as `bear` and `bears` become separate vocabulary entries.
- Similar forms such as `run` and `runs` become separate entries.
- Higher risk of unseen words at inference.

**Memory liner:**

> Word tokenization is simple, but it duplicates morphology and increases OOV risk.

---

## Q22. Why is morphology a problem for word-level tokenization?

Related forms are treated as unrelated token IDs unless the model independently learns similar representations for them.

```text
bear   → token A
bears  → token B

run    → token C
runs   → token D
```

**Memory liner:**

> Similar spellings and meanings do not imply shared token identity.

---

## Q23. What is subword tokenization in this lecture’s framing?

Subword tokenization attempts to reuse common parts or roots across related words.

Example:

```text
bear
bears
  ↓
shared “bear” component
```

**Memory liner:**

> Subwords reuse pieces shared by multiple words.

---

## Q24. What is the main advantage of subword tokenization?

It can exploit shared word structure and reduce the risk of out-of-vocabulary inputs compared with word-level tokenization.

---

## Q25. What is the main cost of subword tokenization?

It usually produces longer sequences than word-level tokenization.

**Memory liner:**

> Better vocabulary reuse comes at the cost of more tokens.

---

## Q26. What is character-level tokenization?

Each character becomes a token.

**Advantages:**

- More robust to misspellings.
- More robust to casing variation.
- Very low risk that a complete word is unrepresentable.

**Disadvantages:**

- Much longer sequences.
- Slower processing and inference.
- A single character may have weak standalone semantic meaning.

**Memory liner:**

> Characters are robust but create long, semantically weak token sequences.

---

## Q27. What is OOV?

OOV means **out of vocabulary**: a token appears at inference time that the vocabulary learned at training time cannot identify.

**Memory liner:**

> OOV means the tokenizer cannot map the observed unit to a known vocabulary entry.

---

## Q28. What happens to an OOV item in the lecture’s example?

It is mapped to a reserved unknown-token representation.

```text
unseen token
    ↓
  <UNK>
```

**Important consequence:** Different unseen items can collapse to the same unknown representation.

---

## Q29. Compare the three tokenization levels.

| Tokenization | Main advantage | Main disadvantage | OOV risk |
|---|---|---|---|
| Word | Short/simple sequence | Duplicates related word forms | High |
| Subword | Shares components; balances trade-offs | Longer than word sequences | Lower |
| Character | Robust to spelling/casing variation | Very long sequences; weak character semantics | Very low |

**Master memory liner:**

> Larger units shorten sequences; smaller units improve coverage.

---

## Q30. Why does longer tokenization increase computation?

The model must process more token positions. The lecture states that model complexity and inference time depend on sequence length.

**Memory liner:**

> More tokens mean more model operations.

**Lecture boundary:** The lecture does not derive a formal sequence-complexity equation here.

---

## Q31. What vocabulary sizes does the lecture mention?

The lecture gives rough orders of magnitude:

- Tens of thousands for a language such as English.
- Potentially hundreds of thousands for multilingual and code-oriented models.

**Memory liner:**

> Vocabulary size depends on language coverage, task, and represented domains.

---

## Q32. Why is subword tokenization described as a useful compromise?

It balances:

```text
word-level efficiency
        +
character-level coverage
```

**Memory liner:**

> Subwords trade some sequence length for much better vocabulary reuse.

---

# 3. One-hot vectors and token similarity

_Lecture region: approximately 00:30:15–00:37:41._

## Q33. What is one-hot encoding?

For a vocabulary of size `V`, every token is represented by a vector of length `V` with one entry equal to 1 and all other entries equal to 0.

### Shape

```text
one-hot token vector: [V]
```

Example for `V = 5`:

```text
[0, 1, 0, 0, 0]
```

**Memory liner:**

> One-hot says which token it is, not what the token means.

---

## Q34. Why are different one-hot token vectors orthogonal?

Each token has its 1 in a different coordinate, so two distinct one-hot vectors have zero dot product.

### Equation

For two different vocabulary entries `i ≠ j`:

$$
e_i^\top e_j=0
$$

**Memory liner:**

> All distinct one-hot tokens look equally unrelated geometrically.

---

## Q35. Why is orthogonality undesirable for semantic representation?

The representation cannot naturally express that:

- `teddy bear` and `soft` may be related in a sentence.
- `teddy bear` and `book` may be less related.

All distinct one-hot vectors have the same zero similarity.

---

## Q36. What is cosine similarity?

Cosine similarity measures the angle between vectors.

### Equation to remember

$$
\cos(x,y)=\frac{x^\top y}{\|x\|\,\|y\|}
$$

### Interpretation

```text
same direction      → high similarity
orthogonal          → near zero similarity
opposite direction  → negative similarity
```

**Memory liner:**

> Cosine compares vector direction after normalising magnitude.

---

## Q37. What question about vector norms is raised but not resolved fully?

A student asks why cosine similarity ignores the norm. The lecturer says cosine is one useful measure, not a perfect measure, and whether vector magnitude carries meaning depends on how the vectors are trained.

**Revision note:**

> The lecture does not claim that norms are always unimportant.

---

## Q38. What properties should a useful learned representation have?

Tokens with similar or related meanings should obtain more similar vectors than unrelated tokens.

**Memory liner:**

> Useful embeddings turn linguistic relationships into geometric relationships.

---

# 4. Word2Vec, proxy objectives, and learned embeddings

_Lecture region: approximately 00:37:49–00:53:06._

## Q39. What problem does Word2Vec address in this lecture?

It learns meaningful token representations from raw text instead of using fixed one-hot vectors as semantic representations.

**Memory liner:**

> Learn token geometry from patterns of language use.

---

## Q40. What two Word2Vec approaches are introduced?

1. Continuous Bag of Words, or CBOW.
2. Skip-gram.

---

## Q41. What is CBOW?

CBOW uses surrounding words to predict a target word.

```text
context words
     ↓
 target word
```

**Memory liner:**

> CBOW predicts the centre from its surroundings.

---

## Q42. What is Skip-gram?

Skip-gram uses a target word to predict surrounding words.

```text
 target word
     ↓
context words
```

**Memory liner:**

> Skip-gram predicts the surroundings from the centre.

---

## Q43. What is a proxy task?

A proxy task is not necessarily the final objective. It is chosen because solving it should force the model to learn useful representations.

**Lecture example:** Predicting neighbouring or next words in order to learn language-aware token embeddings.

**Memory liner:**

> Train on prediction; keep the representation learned while solving it.

---

## Q44. Why should next-word or context prediction produce useful embeddings?

A model that predicts linguistic context must capture some regularities of language and relationships between words.

**Memory liner:**

> To predict context well, the model must encode how words behave together.

---

## Q45. What is the simple neural-network flow used to explain learned embeddings?

```text
one-hot input
    [V]
     ↓
learned hidden representation
    [D]
     ↓
vocabulary prediction
    [V]
```

Usually:

$$
D \ll V
$$

The lecture gives `D = 768` as one example while vocabulary size can be tens or hundreds of thousands.

---

## Q46. What are the tensor shapes in the simple Word2Vec-like network?

Using row-vector notation:

```text
input one-hot x:      [V]
input projection W1: [V, D]
hidden embedding h:  [D]
output projection W2:[D, V]
output logits/probs: [V]
```

### Shape flow

$$
[V]\,[V,D]\rightarrow[D]
$$

then

$$
[D]\,[D,V]\rightarrow[V]
$$

**Memory liner:**

> Vocabulary space compresses into embedding space, then expands back to vocabulary space.

---

## Q47. Why does one-hot multiplication behave like embedding lookup?

Only one element of the one-hot vector is non-zero, so multiplying by the first weight matrix selects the row corresponding to that token.

### Equation

$$
h=e_i^\top W_1
$$

where:

```text
e_i: [V]
W1:  [V, D]
h:   [D]
```

**Memory liner:**

> One-hot × embedding matrix selects one learned row.

---

## Q48. What happens after the vocabulary prediction is produced?

The predicted vocabulary distribution is compared with the true next-word target, a loss such as cross-entropy is computed, and backpropagation updates the network weights.

**Memory liner:**

> Predict → compare → backpropagate → improve the embedding weights.

**Lecture boundary:** Cross-entropy is named, but its mathematical formula is not derived.

---

## Q49. Which learned quantity is retained as the token representation?

The hidden vector produced by the first learned projection is used as the word/token representation.

**Memory liner:**

> The useful object is the hidden embedding, not only the final vocabulary prediction.

---

## Q50. How does the lecture suggest deciding when proxy-task training is done?

Track the proxy-task loss over epochs and stop when it converges, while remembering that good proxy loss does not automatically guarantee usefulness for every downstream task.

**Memory liner:**

> Proxy convergence is a stopping signal, not proof of downstream success.

---

## Q51. What controls the choice of embedding or hidden dimension?

The lecture frames it as a trade-off:

- Richer or more specialised tasks may benefit from a larger representation.
- Larger vectors increase computation, latency, and cost.
- The choice is often empirical and informed by prior work.

**Memory liner:**

> Bigger embeddings offer capacity but cost more to compute.

---

## Q52. What limitation do static token embeddings have?

A token receives the same base representation regardless of the sentence context.

Example:

```text
river bank
robbing a bank
```

A static token embedding alone cannot distinguish the meaning of `bank` in these contexts.

**Memory liner:**

> Static embeddings represent token identity, not token meaning in a particular sentence.

---

## Q53. Why is averaging word embeddings a weak sentence representation?

The lecture says averaging loses:

- Word order.
- Context-dependent meaning.
- Important sequential structure.

**Memory liner:**

> Averaging keeps a bag of meanings but discards the sentence structure.

---

# 5. RNNs, LSTMs, and sequential limitations

_Lecture region: approximately 00:53:23–01:04:36._

## Q54. Why introduce an RNN after static token embeddings?

An RNN attempts to represent a sequence while respecting token order and accumulating information from earlier positions.

**Memory liner:**

> RNNs add order and evolving context to token representations.

---

## Q55. What is the hidden state of an RNN?

The hidden state represents the sequence processed so far.

```text
previous hidden state
        +
current token representation
        ↓
new hidden state
```

---

## Q56. What is the lecture-level RNN recurrence?

The lecture does not derive an exact activation equation, but its mechanism can be written as:

$$
h_t=f(h_{t-1},x_t)
$$

### Conceptual shapes

```text
current token x_t:       [D]
previous hidden h_{t-1}: [D_hid]
new hidden h_t:          [D_hid]
```

**Memory liner:**

> Current token plus previous context produces updated context.

---

## Q57. How does an RNN capture word order?

Each hidden state depends on the previous hidden state, so processing `A then B` is not equivalent to processing `B then A`.

**Memory liner:**

> Order matters because recurrence is directional and sequential.

---

## Q58. How can an RNN support sequence classification?

Use the hidden state after processing the final token and project it into the label space.

```text
x1 → h1 → x2 → h2 → ... → xN → hN
                                  ↓
                              class label
```

**Memory liner:**

> Final hidden state acts as the sequence summary.

---

## Q59. How can an RNN support token-level prediction?

Use the hidden state corresponding to the token of interest and project it into a token-label space.

```text
h1 → token label 1
h2 → token label 2
...
hN → token label N
```

---

## Q60. How can an RNN support sequence generation?

Process the complete source sequence, take the final context/hidden state, and use it to begin decoding the output sequence.

**Memory liner:**

> Encode the source into a context state, then decode from that state.

---

## Q61. What is the fixed-context bottleneck in early RNN encoder–decoder models?

The entire meaning of the source sequence is compressed into a single hidden/context state.

**Memory liner:**

> One vector must carry everything the decoder may later need.

---

## Q62. What is a long-range dependency problem?

Information introduced early in the sequence may need to influence a prediction many steps later, but it must pass through many recurrent transformations.

**Memory liner:**

> Distant evidence has a long path through recurrent states.

---

## Q63. What does an LSTM add at the level covered in the lecture?

It adds a cell state `C_t` alongside the hidden/activation state, aiming to preserve information judged important over longer spans.

```text
hidden state h_t
      +
 cell state C_t
```

**Memory liner:**

> LSTM adds a dedicated memory pathway on top of the recurrent hidden state.

**Lecture boundary:** The input, forget, and output gates are not derived in this session.

---

## Q64. Why can gradients vanish or explode in an RNN?

Backpropagation through time repeatedly multiplies quantities associated with each recurrent step.

```text
many factors below 1  → product approaches 0
many factors above 1  → product grows very large
```

**Memory liner:**

> Repeated multiplication can erase or amplify the learning signal.

---

## Q65. What is the connection between vanishing gradients and memory?

If the gradient reaching an early time step becomes nearly zero, the model cannot effectively update parameters based on distant later errors.

**Memory liner:**

> No gradient signal means no effective learning from distant dependencies.

---

## Q66. Why are RNN computations slow on long sequences?

To compute the hidden state at position `t`, the model must first compute all earlier hidden states.

```text
h1 → h2 → h3 → ... → hN
```

These steps are sequentially dependent.

**Memory liner:**

> Recurrence prevents processing all sequence positions at once.

---

## Q67. What is the lecture’s transition from Word2Vec to RNNs to attention?

```text
Word2Vec:
learn token meaning, but not sentence-specific context or order

RNN:
add order and context, but struggle with long paths and sequential compute

Attention:
create direct links to relevant information
```

**Memory liner:**

> Static meaning → sequential context → direct contextual retrieval.

---

# 6. Attention and self-attention

_Lecture region: approximately 01:04:46–01:13:39._

## Q68. Why was attention introduced in the lecture’s narrative?

The decoder should be able to access the relevant source information directly rather than relying only on one final context vector.

**Memory liner:**

> Let the current prediction look directly at the useful source positions.

---

## Q69. How does attention shorten the information path?

RNN-style path:

```text
source token → many recurrent states → decoder
```

Attention-style path:

```text
current prediction query ─────────→ relevant source token
```

**Memory liner:**

> Attention replaces long recurrent routing with direct matching.

---

## Q70. What is self-attention?

Self-attention computes each token’s representation as a function of other tokens in the same sequence.

**Memory liner:**

> Each token directly consults the sequence to build its contextual representation.

---

## Q71. How does self-attention solve the `bank` ambiguity example conceptually?

The representation of `bank` is recomputed from its surrounding tokens, so it can differ between contexts such as:

```text
river bank
robbing a bank
```

**Memory liner:**

> The same token ID can produce different contextual representations.

---

## Q72. What do Query, Key, and Value mean?

- **Query:** what information is the current token looking for?
- **Key:** what representation is used to determine whether another token matches the query?
- **Value:** what vector is retrieved and combined if that token is relevant?

**Memory liner:**

> Q asks, K matches, V supplies.

---

## Q73. Are Q, K, and V manually assigned meanings?

No. They are obtained through learned projection matrices.

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

**Memory liner:**

> Their interpretation is intuitive, but the projections are learned.

---

## Q74. Why is the matrix formulation useful?

The self-attention computation over an entire sequence can be expressed using matrix operations, which map well to GPU hardware.

**Memory liner:**

> Attention turns many token interactions into parallel matrix multiplication.

---

# 7. Scaled dot-product attention and tensor shapes

_Lecture region: approximately 01:30:25–01:35:03._

## Q75. What is the scaled dot-product attention equation?

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
$$

**Master memory liner:**

> Compare with `QKᵀ`, normalise with softmax, and retrieve from `V`.

---

## Q76. What are the input embedding-matrix shapes?

Using one row per token:

```text
sequence representation X: [N, D]
```

The lecture describes the same matrix using sequence length and `D_model`; matrix orientation is a convention.

---

## Q77. What are the Q/K/V projection shapes for one attention computation?

```text
X:   [N, D]
W_Q: [D, d_k]
W_K: [D, d_k]
W_V: [D, d_v]

Q:   [N, d_k]
K:   [N, d_k]
V:   [N, d_v]
```

### Projection equations

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

---

## Q78. What is the shape of the query–key score matrix?

$$
QK^\top:
[N,d_k]\,[d_k,N]\rightarrow[N,N]
$$

```text
attention scores: [N, N]
```

**Memory liner:**

> Every query token receives one score for every key token.

---

## Q79. What does one row of `QKᵀ` represent?

One row contains the similarity scores between one query token and every key token in the sequence.

```text
one query
   ↓
score against key 1, key 2, ..., key N
```

---

## Q80. What does softmax do in attention?

It turns the query–key scores in each row into a normalised weighting over keys.

```text
raw similarities
      ↓
softmax weights
      ↓
which values matter more?
```

**Memory liner:**

> Softmax converts matching scores into relative retrieval weights.

---

## Q81. Why divide by `sqrt(d_k)`?

The lecture says dot products tend to grow as the query/key dimension grows. Scaling normalises their magnitude before softmax.

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

**Memory liner:**

> Larger query/key vectors produce larger dot products, so scale them before softmax.

---

## Q82. What happens when the attention weights multiply `V`?

```text
attention weights: [N, N]
V:                 [N, d_v]
output:            [N, d_v]
```

### Shape equation

$$
[N,N]\,[N,d_v]\rightarrow[N,d_v]
$$

Each output row is a weighted sum of value vectors for one query.

**Memory liner:**

> Matching decides the weights; values provide the information being mixed.

---

## Q83. What is the full one-head shape flow?

```text
X                     [N, D]
 ├─ XW_Q → Q           [N, d_k]
 ├─ XW_K → K           [N, d_k]
 └─ XW_V → V           [N, d_v]

QKᵀ                    [N, N]
softmax(QKᵀ/√d_k)      [N, N]
weights @ V            [N, d_v]
```

---

# 8. Encoder–decoder Transformer architecture

_Lecture region: approximately 01:14:06–01:25:26 and 01:29:34–01:41:17._

## Q84. What are the two major parts of the original Transformer architecture introduced here?

1. Encoder.
2. Decoder.

The lecture uses machine translation as the running application.

**Memory liner:**

> The encoder represents the source; the decoder generates the target.

---

## Q85. What does the encoder receive?

The source-language token sequence.

```text
source text
    ↓
tokenization
    ↓
source token representations
    ↓
encoder
```

---

## Q86. What is the encoder’s main role?

It creates rich, context-aware representations of the source tokens, allowing source tokens to attend to one another.

**Memory liner:**

> The encoder contextualises every source token using the source sequence.

---

## Q87. What does the decoder receive?

It receives the beginning-of-sequence token initially and then the target tokens generated so far.

```text
BOS → generated target prefix → next-token prediction
```

---

## Q88. What are the two attention uses inside the decoder?

1. **Masked self-attention:** attend to the target tokens decoded so far.
2. **Cross-attention:** attend to the encoder’s source representations.

**Memory liner:**

> The decoder consults its own past and the encoded source.

---

## Q89. What is masked self-attention?

The decoder can only attend to target tokens already available; it cannot look at future target tokens.

```text
position t can use:
1, 2, ..., t

position t cannot use:
t+1, t+2, ...
```

**Memory liner:**

> Decode from the past, never from the unseen future.

**Lecture boundary:** The explicit numerical mask matrix is not derived in this session.

---

## Q90. In decoder cross-attention, where do Q, K, and V come from?

```text
Q ← decoder representation
K ← encoder output
V ← encoder output
```

**Memory liner:**

> The decoder asks the question; the encoder stores the searchable source information.

---

## Q91. What are the cross-attention tensor shapes?

Using the same row-per-token convention:

```text
decoder queries Q: [N_t, d_k]
encoder keys K:    [N_s, d_k]
encoder values V:  [N_s, d_v]
```

Therefore:

$$
QK^\top:
[N_t,d_k]\,[d_k,N_s]\rightarrow[N_t,N_s]
$$

and:

$$
[N_t,N_s]\,[N_s,d_v]\rightarrow[N_t,d_v]
$$

**Memory liner:**

> Each target query scores every source position.

---

## Q92. What is the difference between encoder self-attention, decoder self-attention, and cross-attention?

| Attention type | Query source | Key/value source | What it expresses |
|---|---|---|---|
| Encoder self-attention | Source | Source | A source token as a function of the source sequence |
| Decoder masked self-attention | Target prefix | Target prefix | A target token as a function of decoded target history |
| Cross-attention | Decoder | Encoder | A target representation as a function of source information |

**Master memory liner:**

> Source↔source, past target↔past target, and target→source.

---

## Q93. Why are positional encodings required?

RNNs process tokens one at a time and therefore contain order in their recurrence. Self-attention creates direct links and does not by itself carry that same sequential ordering signal.

**Memory liner:**

> Attention tells the model what relates; position encoding tells it where tokens occur.

---

## Q94. How does the original Transformer combine token and position information?

The lecture says sinusoidal position information is added elementwise to the learned token embedding.

### Shape

```text
token embeddings:    [N, D]
position encodings:  [N, D]
combined input:      [N, D]
```

### Equation

$$
X_{input}=E_{token}+E_{position}
$$

**Lecture boundary:** The sine/cosine formula itself is not derived in this session.

---

## Q95. What is the purpose of BOS?

BOS marks the beginning of the output sequence and provides the first decoder input.

**Memory liner:**

> BOS tells the decoder: begin generating now.

---

## Q96. What is the purpose of EOS?

EOS marks the end of a sequence. Autoregressive generation stops when EOS is generated.

**Memory liner:**

> EOS is the learned stop signal.

---

# 9. Multi-head attention

_Lecture region: approximately 01:23:09–01:25:26 and 01:35:12–01:37:08._

## Q97. What does “head” mean in multi-head attention?

Each head uses its own learned query, key, and value projections and performs an attention computation.

**Memory liner:**

> A head is one learned attention projection pathway.

---

## Q98. Why use several heads?

Multiple heads provide additional degrees of freedom for learning different associations and representations.

The lecture compares this loosely with using multiple filters in a convolutional network.

**Memory liner:**

> Several heads let the model inspect relationships through several learned projections.

---

## Q99. Are attention heads explicitly forced to be different?

No. The lecture says there is normally no explicit constraint forcing head diversity. Different behaviours can emerge through learning and the objective.

**Memory liner:**

> Diversity is permitted by separate parameters, not guaranteed by an explicit rule.

---

## Q100. How are multi-head outputs combined?

The outputs of the heads are concatenated and then passed through a learned output projection `W_O` to return to the original embedding dimension.

### Shape template

If every head returns `[N, d_v]`:

```text
head 1 output: [N, d_v]
head 2 output: [N, d_v]
...
head H output: [N, d_v]

concatenation: [N, H·d_v]
W_O:           [H·d_v, D]
final output:  [N, D]
```

### Shape equation

$$
[N,Hd_v]\,[Hd_v,D]\rightarrow[N,D]
$$

**Memory liner:**

> Run several attentions, concatenate their outputs, project back to model dimension.

---

# 10. Feed-forward layer and stacked blocks

_Lecture region: approximately 01:15:52–01:16:16 and 01:37:25–01:38:46._

## Q101. What is the feed-forward layer’s role in this lecture?

It provides another learned transformation and more degrees of freedom for building useful representations after attention.

**Memory liner:**

> Attention mixes contextual information; the feed-forward layer enriches the resulting representation.

---

## Q102. How does the feed-forward hidden dimension compare with the input/output dimension?

The lecture states that its hidden dimension is larger than its input and output dimensions.

### Lecture-level shape

```text
input:          [N, D]
expanded hidden:[N, D_ff]    where D_ff > D
output:         [N, D]
```

**Lecture boundary:** The exact activation function and FFN equation are not given here.

---

## Q103. Is there only one encoder and one decoder block?

No. The architecture stacks `N` encoder blocks and `N` decoder blocks.

```text
encoder block × N
        ↓
final encoder representation
        ↓
used by cross-attention in decoder blocks
```

**Memory liner:**

> Contextual representations are refined through repeated layers.

---

## Q104. Which encoder output is supplied to the decoder?

The final representation produced by the stack of encoder blocks is supplied as the keys and values for decoder cross-attention.

---

# 11. Label smoothing

_Lecture region: approximately 01:25:34–01:29:25._

## Q105. Why introduce label smoothing for language prediction?

For a prefix such as “what a great ...”, several continuations may be plausible. A hard one-hot label declares one observed continuation to be the only possible answer.

**Memory liner:**

> Language often has several plausible next tokens, even when training shows one target.

---

## Q106. What target does label smoothing use?

For the correct vocabulary entry:

$$
1-\epsilon
$$

For every incorrect entry:

$$
\frac{\epsilon}{V-1}
$$

### Target shape

```text
smoothed target distribution: [V]
```

**Memory liner:**

> Keep most probability on the target, spread a small amount over the rest.

---

## Q107. What behaviour does label smoothing encourage?

It discourages the model from becoming completely certain that the observed target is the only possible continuation.

The lecture notes that it improved BLEU in the original Transformer context.

---

## Q108. How is label smoothing different from softmax?

- **Softmax** acts on the model’s output scores to produce a predicted distribution.
- **Label smoothing** changes the target distribution used for training.

**Memory liner:**

> Softmax changes predictions; label smoothing changes labels.

---

# 12. Full end-to-end Transformer dataflow

_Lecture region: approximately 01:29:34–01:41:17._

## Q109. What is the complete source-side dataflow?

```text
source text
   ↓
tokenization
   ↓
add BOS/EOS where appropriate
   ↓
learned token embeddings
   +
position encodings
   ↓
encoder self-attention
   ↓
feed-forward transformation
   ↓
repeat for N encoder blocks
   ↓
context-aware source representations
```

---

## Q110. What is the complete target-side dataflow?

```text
BOS / generated target prefix
   ↓
learned token embeddings
   +
position encodings
   ↓
masked target self-attention
   ↓
cross-attention to encoder output
   ↓
feed-forward transformation
   ↓
repeat for N decoder blocks
   ↓
linear projection
   ↓
softmax over vocabulary
   ↓
next-token prediction
```

---

## Q111. What is the final output dimension before choosing the next token?

The decoder is projected to a vector whose size equals the vocabulary size.

```text
next-token scores/probabilities: [V]
```

**Memory liner:**

> One scalar score or probability per vocabulary entry.

---

## Q112. How does autoregressive decoding proceed?

```text
1. Feed BOS.
2. Predict the next token.
3. Convert that token to its embedding.
4. Feed the generated prefix back into the decoder.
5. Predict another token.
6. Repeat until EOS.
```

**Memory liner:**

> Generate one token, append it, and use it to generate the next.

---

## Q113. What stops the generation loop?

The decoder stops after generating the EOS token.

---

## Q114. What is the single most useful high-level explanation of the original encoder–decoder Transformer?

> The encoder builds contextual source-token representations. The decoder repeatedly uses its generated history and cross-attends to those source representations to predict the next target token.

---

# 13. Essential tensor-shape sheet

These are the shapes directly supported by the lecture’s described dimensions and matrix operations.

## Token representation

```text
one-hot token:          [V]
embedding matrix:       [V, D]
learned token vector:   [D]
```

$$
[V]\,[V,D]\rightarrow[D]
$$

---

## Word2Vec-like proxy network

```text
input:                  [V]
hidden representation:  [D]
vocabulary output:      [V]
```

```text
[V] → [D] → [V]
```

---

## Sequence embedding matrix

```text
N tokens, each D-dimensional:
X = [N, D]
```

---

## Position-aware embeddings

```text
token embedding matrix: [N, D]
position matrix:        [N, D]
combined matrix:        [N, D]
```

---

## One-head self-attention

```text
X:                       [N, D]
W_Q:                     [D, d_k]
W_K:                     [D, d_k]
W_V:                     [D, d_v]
Q:                       [N, d_k]
K:                       [N, d_k]
V:                       [N, d_v]
QKᵀ:                     [N, N]
attention weights:       [N, N]
attention output:        [N, d_v]
```

Key derivations:

$$
[N,d_k]\,[d_k,N]=[N,N]
$$

$$
[N,N]\,[N,d_v]=[N,d_v]
$$

---

## Cross-attention

```text
decoder Q:               [N_t, d_k]
encoder K:               [N_s, d_k]
encoder V:               [N_s, d_v]
attention scores:        [N_t, N_s]
attention output:        [N_t, d_v]
```

Key derivations:

$$
[N_t,d_k]\,[d_k,N_s]=[N_t,N_s]
$$

$$
[N_t,N_s]\,[N_s,d_v]=[N_t,d_v]
$$

---

## Multi-head combination

```text
one head:                [N, d_v]
H concatenated heads:    [N, H·d_v]
output projection W_O:   [H·d_v, D]
final MHA output:        [N, D]
```

---

## Feed-forward transformation

```text
input:                   [N, D]
expanded hidden:         [N, D_ff]
output:                  [N, D]

D_ff > D
```

---

## Final vocabulary prediction

```text
decoder representation for current position: [D]
vocabulary output:                            [V]
```

---

# 14. Equations to remember from this lecture

## Classification metrics

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
F_1=2\frac{PR}{P+R}
$$

---

## Cosine similarity

$$
\cos(x,y)=\frac{x^\top y}{\|x\|\,\|y\|}
$$

---

## One-hot embedding lookup

$$
h=e_i^\top W_1
$$

---

## Lecture-level RNN recurrence

$$
h_t=f(h_{t-1},x_t)
$$

---

## Q/K/V projections

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

---

## Scaled dot-product attention

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

---

## Token plus position

$$
X_{input}=E_{token}+E_{position}
$$

---

## Label smoothing

$$
q_y=1-\epsilon
$$

$$
q_i=\frac{\epsilon}{V-1},\qquad i\neq y
$$

---

# 15. Twenty interview memory liners

1. **Task types:** One label, many local labels, or a generated sequence.
2. **Accuracy:** High accuracy can hide minority-class failure.
3. **Precision:** When I predicted positive, how often was I right?
4. **Recall:** Of all real positives, how many did I find?
5. **Generation evaluation:** One input can have many acceptable outputs.
6. **Tokenization:** Decide the text units before learning their representations.
7. **Tokenizer trade-off:** Larger units shorten sequences; smaller units improve coverage.
8. **OOV:** Unknown inputs collapse to an unknown-token representation.
9. **One-hot:** It stores identity, not meaning.
10. **Embedding:** Learn linguistic relationships as geometric relationships.
11. **Proxy task:** Train on prediction to learn a useful representation.
12. **CBOW vs Skip-gram:** Context predicts centre; centre predicts context.
13. **RNN:** Current token plus previous context produces new context.
14. **RNN weakness:** Long recurrent paths hurt memory, gradients, and parallelism.
15. **Attention:** Directly retrieve the information relevant to the current prediction.
16. **Q/K/V:** Q asks, K matches, V supplies.
17. **Attention equation:** Compare, scale, softmax, retrieve.
18. **Cross-attention:** The decoder asks; the encoder supplies source information.
19. **Multi-head:** Several learned attention projections run in parallel.
20. **Autoregressive decoding:** Predict, append, repeat until EOS.

---

# 16. Final interview-day checklist

You should be able to answer these without notes:

- [ ] What are the three NLP task categories introduced in the lecture?
- [ ] Why can accuracy be misleading?
- [ ] Define precision, recall, and F1.
- [ ] Why is generation evaluation difficult?
- [ ] What do BLEU, ROUGE, and perplexity measure at a high level?
- [ ] Compare word, subword, and character tokenization.
- [ ] Explain OOV and the unknown token.
- [ ] Why are one-hot vectors poor semantic representations?
- [ ] Explain cosine similarity and its limitation discussed in class.
- [ ] Explain CBOW, Skip-gram, and proxy-task representation learning.
- [ ] Derive the `[V] → [D] → [V]` Word2Vec-like shape flow.
- [ ] Explain why a one-hot multiplication is an embedding lookup.
- [ ] Explain what an RNN hidden state represents.
- [ ] Map RNNs to classification, token labelling, and generation.
- [ ] Explain long-range dependencies and vanishing/exploding gradients.
- [ ] Explain why RNN processing is sequential and slow.
- [ ] Explain why attention was introduced.
- [ ] Define Query, Key, and Value intuitively.
- [ ] Write the scaled dot-product attention equation.
- [ ] Derive `[N,d_k] × [d_k,N] → [N,N]`.
- [ ] Derive `[N,N] × [N,d_v] → [N,d_v]`.
- [ ] Explain why attention divides by `sqrt(d_k)`.
- [ ] Distinguish encoder self-attention, masked decoder self-attention, and cross-attention.
- [ ] Derive the cross-attention score shape `[N_t,N_s]`.
- [ ] Explain why position information is added.
- [ ] Explain what one attention head means.
- [ ] Explain how multi-head outputs are concatenated and projected by `W_O`.
- [ ] Explain the role and dimensional expansion of the feed-forward layer.
- [ ] Explain BOS and EOS.
- [ ] Explain label smoothing and distinguish it from softmax.
- [ ] Walk through the complete encoder–decoder translation pipeline.
- [ ] Walk through the autoregressive decoding loop until EOS.

---

# 17. One-minute recap

```text
Text tasks require different outputs and evaluation metrics.

Tokenization converts text into word, subword, or character units.
Smaller units improve coverage but make sequences longer.

One-hot vectors identify tokens but cannot express semantic similarity.
Word2Vec learns embeddings through a proxy prediction task.

RNNs add order and an evolving hidden state, but long recurrent paths
create memory, gradient, and computational problems.

Attention creates direct query–key matches and retrieves weighted values:

    softmax(QKᵀ / √d_k)V

The Transformer encoder contextualises source tokens.
The decoder uses masked target self-attention and cross-attention to the source.
Multiple heads learn different projection pathways.
The final decoder vector is projected over the vocabulary.
Generation starts with BOS and repeats until EOS.
```
