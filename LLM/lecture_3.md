# CME 295 Lecture 3 — LLMs, Decoding, Prompting, and Efficient Inference

> **Scope:** This note is grounded in the supplied **Lecture 3 transcript**. It uses clearer interview-oriented explanations and shape derivations where they directly clarify material taught in the lecture. It does **not** silently import later-course topics.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, intuitive answers, memory liners, equations, tensor shapes, mechanisms, trade-offs, and lecture boundaries.

**Source transcript:** CME 295 Lecture 3 — `https://www.youtube.com/watch/Q5baLehv5So`

**Lecture spine:**

```text
Transformer model families recap
        ↓
What makes a model an LLM?
        ↓
Decoder-only causal language modelling
        ↓
Mixture of Experts (MoE)
        ├── dense vs sparse routing
        ├── token-level top-k experts
        ├── active vs total parameters
        └── routing collapse + load balancing
        ↓
How the next token is selected
        ├── greedy decoding
        ├── beam search
        ├── sampling
        ├── top-k / top-p
        └── temperature
        ↓
Constrained generation
        └── guided decoding
        ↓
Prompting and in-context learning
        ├── context length and context rot
        ├── context / instruction / input / constraints
        ├── zero-shot / few-shot
        ├── explicit reasoning rationales
        └── self-consistency
        ↓
Efficient autoregressive inference
        ├── KV cache
        ├── GQA / MQA cache reduction
        ├── PagedAttention
        ├── Multi-Latent Attention (MLA)
        ├── speculative decoding
        └── multi-token prediction
```

---

## Notation used in this note

```text
B       batch size
T       current sequence / context length
D       model width, d_model
Dff     FFN hidden width
V       vocabulary size
L       number of decoder layers
Hq      number of query heads
Hkv     number of key/value heads
Dh      per-head dimension
E       number of experts in one MoE layer
k       number of selected experts per token
Dc      compressed latent dimension in the lecture-level MLA view
M       number of independent samples in self-consistency
Kbeam   beam width
```

### Priority legend

- **🔥 Must remember:** answer immediately; derive the equation or shape.
- **○ Should know:** explain the intuition and trade-off clearly.
- **△ Lecture boundary:** mentioned, but not fully specified or derived in this transcript.

---

# 1. Transformer families and the lecture’s definition of an LLM

## 1. 🔥 What three Transformer families does the lecture recap?

| Family | Retained components | Main lecture role |
|---|---|---|
| Encoder–decoder | Encoder + causal decoder + cross-attention | Text-to-text transformation, e.g. T5 |
| Encoder-only | Bidirectional encoder | Representations and classification, e.g. BERT |
| Decoder-only | Causal decoder stack; no cross-attention | Autoregressive text generation, e.g. GPT-like models |

**Memory liner:**

> Encoder–decoder transforms, encoder-only represents, decoder-only generates causally.

---

## 2. 🔥 What disappears when the Transformer becomes decoder-only?

There is no encoder output, so **cross-attention disappears**.

A lecture-level decoder-only block is:

```text
input token representations
          ↓
causal / masked self-attention
          ↓
feed-forward network
          ↓
residual + normalization machinery
```

**Memory liner:**

> No encoder memory means no cross-attention; only causal self-attention remains.

---

## 3. 🔥 What is a language model?

The lecture defines a language model as a model that assigns probabilities to token sequences by repeatedly predicting the next token:

$$
p(x_{1:T})
=
\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

Equivalently, at position `t` it produces:

$$
p(x_{t+1}\mid x_{\le t})
$$

**Memory liner:**

> A causal language model turns a prefix into a distribution over the next token.

---

## 4. 🔥 In what three senses is an LLM “large” in this lecture?

1. **Model size:** many parameters.
2. **Data size:** very large numbers of pre-training tokens.
3. **Compute:** substantial hardware and floating-point work for training and inference.

**Memory liner:**

> LLM scale is model capacity + training data + compute.

---

## 5. ○ Does “large” have one exact parameter threshold?

No exact mathematical threshold is established. The lecture gives rough orders of magnitude and emphasizes that terminology evolved historically.

**Memory liner:**

> “LLM” is an operational category, not a theorem with a universal parameter cutoff.

---

## 6. △ Is BERT an LLM?

**Lecture framing:** the lecturers reserve “LLM” here for very large text-generating language models and therefore do not include encoder-only BERT.

**Boundary:** terminology is not mathematically universal. For this lecture, remember the intended distinction:

```text
BERT → encoder representation model
modern LLM in this lecture → large decoder-only generator
```

---

## 7. 🔥 What is the complete decoder-only next-token dataflow?

```text
prefix token IDs                    [B,T]
        ↓ embedding + position
hidden sequence                     [B,T,D]
        ↓ L causal decoder blocks
final hidden sequence               [B,T,D]
        ↓ select final position
current hidden state                [B,D]
        ↓ vocabulary projection
logits                              [B,V]
        ↓ softmax / decoding rule
next token                          [B]
        ↓ append and repeat
```

**Memory liner:**

> Encode the prefix causally, project the last state to vocabulary logits, choose one token, append, repeat.

---

## 8. 🔥 What is the difference between model probabilities and a decoding algorithm?

The model produces a distribution:

$$
p_t\in\mathbb{R}^{V}
$$

The decoding algorithm decides how to turn that distribution into a token:

```text
model:       what probabilities should each token receive?
decoder:     how should one token be selected from them?
```

**Memory liner:**

> The LM scores possibilities; the decoding policy chooses among them.

---

# 2. Mixture of Experts: the central idea

## 9. 🔥 What problem is Mixture of Experts trying to solve?

A dense model activates the same large parameter set for every token. MoE asks:

> Can we increase total model capacity without activating every parameter on every forward pass?

**Memory liner:**

> More total capacity, fewer active parameters per token.

---

## 10. 🔥 What are the router and the experts?

- **Router / gate `G`:** scores which experts should process a token.
- **Expert `E_i`:** a trainable subnetwork that transforms that token representation.

The generic mixture is:

$$
\hat y
=
\sum_{i=1}^{E}G_i(x)E_i(x)
$$

where `G_i(x)` is the routing weight assigned to expert `i`.

**Memory liner:**

> The router decides who works; experts perform the transformation.

---

## 11. 🔥 What is the router tensor flow?

For token representations:

```text
X:              [B,T,D]
router weight:  [D,E]
router logits:  [B,T,E]
router probs:   [B,T,E]
```

A simple formulation is:

$$
R=XW_G+b_G
$$

$$
P=\operatorname{softmax}(R,\text{expert dimension})
$$

Each row `P[b,t,:]` scores all experts for one token.

**Memory liner:**

> One contextual token vector becomes one probability distribution over experts.

---

## 12. 🔥 Dense MoE vs sparse MoE?

### Dense MoE

All experts may contribute:

$$
y=\sum_{i=1}^{E}P_i(x)E_i(x)
$$

### Sparse MoE

Keep only the top `k` experts:

$$
S(x)=\operatorname{TopK}(P(x),k)
$$

$$
y=\sum_{i\in S(x)}\tilde P_i(x)E_i(x)
$$

where `\tilde P` denotes the selected routing weights, commonly renormalized over the chosen experts.

**Memory liner:**

> Dense mixes everyone; sparse dispatches each token to only its top-k experts.

---

## 13. 🔥 What does `top-k` mean in MoE?

For every token, the router chooses the `k` highest-scoring experts.

```text
router probabilities [E]
          ↓ TopK(k)
expert indices         [k]
expert weights         [k]
```

Typical lecture examples are `k=1` or `k=2`.

**Memory liner:**

> MoE `k` counts active experts per token, not attention heads.

---

## 14. 🔥 Is routing done per sequence or per token?

The lecture describes **token-level routing**.

```text
token 1 → expert 3
token 2 → expert 1
token 3 → expert 3
token 4 → expert 7
```

Two tokens in the same prompt may use different experts in the same layer.

**Memory liner:**

> Each token makes its own expert choice at every MoE layer.

---

## 15. 🔥 When is the routing decision made inside a decoder block?

At the MoE replacement for the FFN, after the token has already been contextualized by self-attention:

```text
causal self-attention
        ↓
contextual token x
        ↓
layer-specific router G(x)
        ↓
selected FFN expert(s)
```

**Memory liner:**

> Attention contextualizes the token first; the router then dispatches that contextual token through the FFN stage.

---

## 16. 🔥 Where is MoE inserted in the Transformer?

The lecture places MoE in the **feed-forward network**, replacing one dense FFN with several candidate FFNs.

```text
dense block:
attention → one FFN

MoE block:
attention → router → one/few FFN experts
```

**Memory liner:**

> In this lecture, the experts are alternative FFNs, not alternative attention heads.

---

## 17. 🔥 Why is the FFN a logical place for MoE?

A two-layer FFN roughly has:

$$
W_1\in\mathbb{R}^{D\times D_{ff}},
\qquad
W_2\in\mathbb{R}^{D_{ff}\times D}
$$

Ignoring biases, its parameter count is approximately:

$$
\boxed{2DD_{ff}}
$$

Since `Dff` is normally larger than `D`, the FFN contains substantial capacity and computation.

**Memory liner:**

> The FFN is a large per-token parameter bank, so conditional activation saves meaningful work.

---

## 18. 🔥 What is one expert’s tensor shape?

For the tokens assigned to expert `i`, let their count be `N_i`:

```text
expert input:       [N_i,D]
first projection:   [D,Dff]
expanded features:  [N_i,Dff]
second projection:  [Dff,D]
expert output:      [N_i,D]
```

The outputs are scattered back to their original `[B,T,D]` token positions.

**Memory liner:**

> Gather routed tokens, run an FFN, then scatter the outputs back.

---

## 19. 🔥 Is the number of experts tied to the number of attention heads?

No. They are independent architectural choices.

```text
attention heads → parallel Q/K/V matching spaces
MoE experts     → alternative FFN transformations
```

**Memory liner:**

> Heads decide how tokens communicate; experts decide how token features are transformed.

---

## 20. 🔥 Does each Transformer layer have the same router and experts?

The lecture says the routers and expert weights are generally **layer-specific** and not shared.

A token may therefore take different routes across depth:

```text
layer 1 → expert 3
layer 2 → expert 1
layer 3 → expert 6
```

**Memory liner:**

> Routing is both token-specific and layer-specific.

---

# 3. MoE capacity, compute, and training pathologies

## 21. 🔥 Total parameters vs active parameters?

- **Total parameters:** every expert parameter stored in the model.
- **Active parameters:** only the parameters used for the selected expert paths in one forward pass.

If an MoE layer has `E` experts but activates `k`, then total expert capacity grows roughly with `E`, while token-level FFN compute grows mainly with `k`.

**Memory liner:**

> Storage follows all experts; per-token compute follows selected experts.

---

## 22. 🔥 Does MoE reduce total parameter count?

No. It normally **increases total parameter count** because many experts are stored.

Its benefit is conditional computation:

```text
large total model
      +
small active subset per token
```

**Memory liner:**

> MoE is parameter-rich but activation-sparse.

---

## 23. 🔥 What are FLOPs?

FLOPs means floating-point operations: an approximate count of additions, multiplications, and related numerical operations.

**Memory liner:**

> FLOPs estimate arithmetic work, not parameter storage or wall-clock latency by themselves.

---

## 24. ○ Why can sparse MoE have fewer FLOPs than dense MoE?

Dense MoE evaluates every expert. Sparse MoE evaluates only `k` selected experts:

```text
Dense MoE compute  ∝ E expert evaluations
Sparse MoE compute ∝ k expert evaluations, k << E
```

**Memory liner:**

> Sparse routing avoids computing outputs that receive no useful weight.

---

## 25. ○ What does the lecture mean by MoE being sample-efficient?

The lecture reports that an MoE model may reach a target quality with less training progress than a smaller-capacity dense model.

**Memory liner:**

> Conditional capacity can improve quality per observed training token, even though systems costs remain a trade-off.

---

## 26. 🔥 What is routing collapse?

Routing collapse occurs when the router repeatedly sends most tokens to only a small subset of experts, leaving other experts underused.

```text
bad routing:
expert 1 ← almost everything
expert 2 ← almost everything else
experts 3...E ← nearly no tokens
```

Consequences:

- wasted parameters;
- overloaded popular experts;
- poorly trained unused experts;
- loss of the intended conditional capacity.

**Memory liner:**

> Routing collapse means a few experts monopolize the tokens.

---

## 27. 🔥 What quantities does the lecture use for load balancing?

For expert `i`:

- `f_i`: fraction of tokens actually routed to expert `i`;
- `P_i`: average router probability assigned to expert `i`.

A lecture-level auxiliary loss is:

$$
\boxed{
\mathcal{L}_{aux}
=
\alpha E\sum_{i=1}^{E}f_iP_i
}
$$

The intent is to encourage more balanced expert use.

**Memory liner:**

> Balance both actual traffic `f_i` and router preference `P_i` across experts.

---

## 28. ○ What is the intuition behind the auxiliary load-balancing loss?

If one expert receives both a high routing probability and a large token fraction, the product `f_iP_i` is large. The extra objective discourages excessive concentration.

The desired qualitative state is closer to:

$$
f_i\approx \frac{1}{E},
\qquad
P_i\approx \frac{1}{E}
$$

**Memory liner:**

> Make “who the router prefers” and “where tokens actually go” less concentrated.

---

## 29. ○ What is noisy gating?

Noise is added to routing scores so that non-dominant experts sometimes receive tokens, increasing exploration during training.

```text
router logits
      + noise
      ↓
TopK experts
```

**Memory liner:**

> Perturb routing so early winners do not permanently monopolize traffic.

---

## 30. △ How is hard top-k routing differentiated through?

The lecture raises this issue but does not give a complete answer for the hard token-fraction term or implementation details.

Remember only what the lecture establishes:

- router probabilities arise from trainable projections and softmax;
- router and experts are trained jointly by backpropagation;
- the exact treatment of discrete selection is outside this transcript.

---

# 4. From logits to next-token probabilities

## 31. 🔥 How are next-token logits produced?

For the final hidden state:

$$
h_t\in\mathbb{R}^{B\times D}
$$

apply a vocabulary projection:

$$
z_t=h_tW_{vocab}+b
$$

with:

```text
h_t:       [B,D]
W_vocab:   [D,V]
logits z:  [B,V]
```

**Memory liner:**

> The vocabulary head converts one hidden vector into one score per token.

---

## 32. 🔥 What does softmax do to logits?

$$
p_i
=
\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

It creates a categorical distribution:

$$
p_i>0,
\qquad
\sum_i p_i=1
$$

**Memory liner:**

> Logits are unconstrained scores; softmax converts relative score gaps into probabilities.

---

## 33. 🔥 What is the probability of an autoregressively generated sequence?

$$
\boxed{
p(y_{1:T}\mid x)
=
\prod_{t=1}^{T}p(y_t\mid x,y_{<t})}
$$

Taking logs turns products into sums:

$$
\boxed{
\log p(y_{1:T}\mid x)
=
\sum_{t=1}^{T}\log p(y_t\mid x,y_{<t})
}
$$

**Memory liner:**

> Sequence score is the sum of conditional token log-probabilities.

---

# 5. Greedy decoding and beam search

## 34. 🔥 What is greedy decoding?

At every step choose:

$$
\boxed{y_t=\arg\max_i p(i\mid y_{<t},x)}
$$

**Memory liner:**

> Greedy chooses the best token now and never revisits the decision.

---

## 35. 🔥 Why is greedy decoding deterministic in the mathematical model?

For fixed parameters and the same prefix, the forward pass gives the same logits. `argmax` then chooses the same token.

```text
same prefix
   → same logits
   → same argmax
   → same continuation
```

**Memory liner:**

> Deterministic scores plus deterministic selection give deterministic decoding.

---

## 36. 🔥 Why is greedy decoding only locally optimal?

It maximizes the next-token probability, not the product of probabilities for the complete sequence.

A lower-probability first token can lead to much stronger later probabilities and therefore a higher-probability overall sequence.

**Memory liner:**

> Best next step does not imply best complete path.

---

## 37. 🔥 What is beam search?

Beam search keeps the `Kbeam` highest-scoring partial sequences after each expansion.

```text
start with BOS
      ↓ expand all next tokens
keep best Kbeam partial paths
      ↓ expand each retained path
keep best Kbeam again
      ↓ repeat until completion
```

**Memory liner:**

> Greedy keeps one path; beam search keeps several competing prefixes.

---

## 38. 🔥 What is beam width?

`Kbeam` is the number of active partial sequences retained at each step.

```text
Kbeam = 1  → greedy-like search
larger Kbeam → broader search, more compute and memory
```

---

## 39. ○ Is beam search globally optimal?

No. It is a truncated search: paths pruned early can never return.

**Memory liner:**

> Beam search is less myopic than greedy, but it is still approximate.

---

## 40. 🔥 Why does raw sequence probability favor short outputs?

Every token probability lies in `[0,1]`. Multiplying another such probability cannot increase the product:

$$
\prod_{t=1}^{T+1}p_t
\le
\prod_{t=1}^{T}p_t
$$

Equivalently, token log-probabilities are non-positive, so longer sums become more negative.

**Memory liner:**

> Adding tokens adds probability factors below one, creating a short-sequence bias.

---

## 41. ○ How does beam search address length bias?

The lecture says practical beam search adds a length-dependent correction or normalization, conceptually of the form:

$$
\operatorname{score}(y)
=
\frac{\sum_t\log p(y_t\mid y_{<t},x)}{\operatorname{lengthPenalty}(|y|)}
$$

### △ Lecture boundary

The precise normalization formula and exponent are not specified in the transcript.

---

## 42. 🔥 Why is beam search not the default for open-ended chat generation in the lecture?

- maintaining several paths is expensive;
- it still favors highly likely, conservative text;
- it provides less diversity and creativity than sampling.

The lecture associates beam search more naturally with tasks such as machine translation.

**Memory liner:**

> Beam search optimizes likelihood-oriented paths; sampling is better suited to diverse open-ended generation.

---

# 6. Sampling, top-k, and top-p

## 43. 🔥 What is ancestral token sampling?

Draw the next token from the model distribution:

$$
y_t\sim p(\cdot\mid y_{<t},x)
$$

High-probability tokens are selected more often, but lower-probability tokens remain possible.

**Memory liner:**

> Sampling converts model uncertainty into variation across generations.

---

## 44. 🔥 Why can unrestricted sampling be risky?

Even a very unlikely token has non-zero probability under softmax and may occasionally be sampled, potentially derailing the continuation.

**Memory liner:**

> The long probability tail contains rare but potentially poor choices.

---

## 45. 🔥 What is top-k sampling?

Let `S_k` be the `k` highest-probability vocabulary items. Mask all others, renormalize, and sample from `S_k`:

$$
\tilde p_i
=
\begin{cases}
\frac{p_i}{\sum_{j\in S_k}p_j}, & i\in S_k\\[4pt]
0, & i\notin S_k
\end{cases}
$$

**Memory liner:**

> Keep a fixed number of likely tokens, discard the rest, then sample.

---

## 46. 🔥 What is top-p / nucleus sampling?

Sort tokens by descending probability and choose the smallest set `S_p` such that:

$$
\sum_{i\in S_p}p_i\ge p
$$

Then renormalize and sample inside that set.

**Memory liner:**

> Top-k fixes the number of candidates; top-p fixes the retained probability mass.

---

## 47. 🔥 Why can top-p adapt better than top-k?

```text
confident distribution:
small set already contains mass p

uncertain distribution:
larger set is needed to contain mass p
```

**Memory liner:**

> Nucleus size expands or contracts with model uncertainty.

---

# 7. Temperature scaling

## 48. 🔥 What is temperature-scaled softmax?

For logits `z_i` and temperature `\tau>0`:

$$
\boxed{
p_i(\tau)
=
\frac{\exp(z_i/\tau)}
{\sum_{j=1}^{V}\exp(z_j/\tau)}}
$$

Shapes do not change:

```text
logits:        [B,V]
temperature:   scalar
probabilities: [B,V]
```

**Memory liner:**

> Temperature rescales logit gaps before softmax.

---

## 49. 🔥 What does low temperature do?

For `0<\tau<1`, dividing by a small number enlarges logit differences. Softmax becomes sharper and concentrates probability on high-logit tokens.

```text
low temperature
      ↓
larger relative logit gaps
      ↓
spikier distribution
      ↓
more conservative / repeatable samples
```

**Memory liner:**

> Low temperature amplifies preferences already present in the logits.

---

## 50. 🔥 What does high temperature do?

For `\tau>1`, logit differences shrink. The distribution becomes flatter, giving lower-ranked tokens more probability.

**Memory liner:**

> High temperature compresses score differences and increases diversity.

---

## 51. 🔥 What happens as temperature tends to zero?

Let `k=argmax_i z_i`, assuming a unique maximum. Rewrite:

$$
p_i(\tau)
=
\frac{\exp((z_i-z_k)/\tau)}
{\sum_j\exp((z_j-z_k)/\tau)}
$$

For `i\ne k`, `z_i-z_k<0`, so:

$$
\exp((z_i-z_k)/\tau)\rightarrow0
\quad\text{as}\quad\tau\rightarrow0^+
$$

Therefore:

$$
p_k\rightarrow1,
\qquad
p_i\rightarrow0\;(i\ne k)
$$

**Memory liner:**

> Zero-temperature limit approaches argmax selection.

---

## 52. 🔥 What happens as temperature tends to infinity?

For finite logits:

$$
\frac{z_i}{\tau}\rightarrow0
$$

and therefore:

$$
\exp(z_i/\tau)\rightarrow1
$$

so:

$$
\boxed{p_i\rightarrow\frac{1}{V}}
$$

**Memory liner:**

> Infinite temperature erases logit differences and approaches uniform sampling.

---

## 53. 🔥 Does temperature change token ranking?

For positive `\tau`, division by `\tau` is monotonic, so it does not change the ordering of logits:

$$
z_i>z_j
\Longleftrightarrow
z_i/\tau>z_j/\tau
$$

It changes the **probability gaps**, not which logit is largest.

**Memory liner:**

> Temperature changes confidence, not rank order.

---

## 54. 🔥 Where does generation randomness come from?

In the lecture’s mathematical picture:

- Transformer forward computation is deterministic for fixed inputs and parameters.
- Randomness enters when a token is **sampled** from the output distribution.

```text
forward pass → deterministic logits
sampling     → stochastic token choice
```

**Memory liner:**

> The network computes the distribution; sampling injects randomness.

---

## 55. ○ Does `temperature = 0` literally go into the softmax equation?

No: division by zero is undefined. In practice, “temperature zero” is used as shorthand for deterministic greedy/argmax decoding.

**Memory liner:**

> `T=0` is an API convention for argmax, not a legal softmax denominator.

---

## 56. △ Why might nominally deterministic inference still vary in practice?

The lecture briefly attributes this to hardware and numerical effects, including different orders of floating-point reductions.

**Boundary:** exact reproducibility settings, kernels, distributed execution, and numerical-analysis details are not developed in this transcript.

---

# 8. Guided decoding and constrained outputs

## 57. 🔥 What is guided decoding?

Guided decoding filters the next-token vocabulary so only tokens compatible with a required structure remain valid.

For a validity mask `m_i`:

$$
m_i=
\begin{cases}
0,&\text{token }i\text{ is valid}\\
-\infty,&\text{token }i\text{ is invalid}
\end{cases}
$$

Conceptually:

$$
p=\operatorname{softmax}(z+m)
$$

Invalid tokens receive probability zero.

**Memory liner:**

> Constrain the token distribution during generation instead of checking the format only afterward.

---

## 58. 🔥 Why is guided decoding better than repeatedly asking for valid JSON?

Naive approach:

```text
generate freely
   ↓
parse
   ↓ invalid?
retry entire generation
```

Guided approach:

```text
track valid parser state
   ↓
mask invalid next tokens
   ↓
generate only legal continuations
```

**Memory liner:**

> Prevention at each token is more reliable than retrying after a malformed sequence.

---

## 59. ○ What determines which tokens are currently valid?

The lecture points to mechanisms such as:

- finite-state machines;
- grammar-based constraints.

At each prefix, the constraint state determines permitted next tokens.

---

## 60. △ What does the lecture not teach about guided decoding?

It does not derive:

- how a JSON schema is compiled into a state machine;
- tokenization/grammar interactions;
- parser-state algorithms;
- performance overhead.

Keep those for a dedicated structured-generation note.

---

# 9. Context length and context quality

## 61. 🔥 What is context length?

The lecture uses **context length**, **context size**, and **window size** for the number of tokens the model can accommodate in its input context for a pass.

```text
prompt / context token IDs: [B,T]
T ≤ model-supported context limit
```

**Memory liner:**

> Context length is a token budget, not a character or word budget.

---

## 62. 🔥 Is a larger advertised context window automatically better?

No. A model may technically accept a long context while becoming worse at locating and using the relevant information inside it.

**Memory liner:**

> Capacity to ingest context is not the same as ability to exploit every token reliably.

---

## 63. 🔥 What is the “needle-in-a-haystack” test described in the lecture?

A target fact—the needle—is inserted into a much larger context—the haystack. The model is queried for that fact while context length and distractors vary.

```text
large document with hidden fact
              ↓
question about hidden fact
              ↓
measure retrieval success
```

**Memory liner:**

> Bury one answer in a long prompt and test whether the model can retrieve it.

---

## 64. 🔥 What does the lecture call “context rot”?

The lecture uses the term for degradation in a model’s ability to retrieve and ground the correct information as the context becomes longer or more distracting.

**Memory liner:**

> More context can dilute access to the context that actually matters.

---

## 65. 🔥 What is the role of distractors?

Distractors are irrelevant or competing pieces of context that make retrieval harder.

**Memory liner:**

> Context quality matters alongside context quantity.

---

## 66. ○ What practical lesson does the lecture draw for retrieval problems?

Supply the most relevant context rather than indiscriminately filling the entire window.

```text
retrieve / select relevant evidence
              ↓
place concise evidence in prompt
              ↓
ask model to answer from it
```

**Memory liner:**

> A targeted context can be more useful than a maximal context.

---

## 67. △ What is outside this lecture’s context discussion?

The transcript does not teach:

- a full retrieval-augmented generation pipeline;
- chunking or embedding retrieval;
- reranking;
- position-dependent long-context failure curves;
- formal long-context benchmarks.

---

# 10. Prompt anatomy

## 68. 🔥 What four prompt components does the lecture propose as a mental model?

1. **Context:** background or setting.
2. **Instruction:** the task to perform.
3. **Input:** the specific object/data to process.
4. **Constraints:** restrictions on output, behavior, or format.

```text
PROMPT
├── context
├── instruction
├── input
└── constraints
```

**Memory liner:**

> Tell the model the situation, the task, the data, and the boundaries.

---

## 69. ○ Is this four-part prompt structure a formal theory?

No. The lecture presents it as a useful decomposition, not a mathematically unique prompt grammar.

---

## 70. 🔥 Why separate instruction from input?

It clarifies:

```text
instruction = operation to perform
input       = object on which to perform it
```

Example:

```text
Instruction: classify sentiment.
Input:       "This teddy bear is wonderful."
```

**Memory liner:**

> Instruction is the function; input is its argument.

---

## 71. 🔥 Why make constraints explicit?

Constraints can specify:

- output format;
- length;
- permitted behavior;
- safety boundaries;
- required fields.

**Memory liner:**

> A model cannot reliably honor a boundary that the prompt never communicates.

---

# 11. In-context learning: zero-shot and few-shot

## 72. 🔥 What is in-context learning?

The model changes its behavior using information or examples supplied in the prompt, **without updating model weights**.

```text
parameters θ: fixed
prompt:       changed
behavior:     adapted
```

**Memory liner:**

> Learn the task from the context, not through gradient descent.

---

## 73. 🔥 Why is “learning” overloaded in in-context learning?

No optimizer changes `θ`. The adaptation exists only for the current context.

**Memory liner:**

> In-context learning is temporary conditioning, not parameter training.

---

## 74. 🔥 Zero-shot vs few-shot prompting?

### Zero-shot

Provide the task instruction and target input without demonstrations.

### Few-shot

Provide example input–output pairs before the target query.

```text
example input 1 → example output 1
example input 2 → example output 2
...
target input    → model completes target output
```

**Memory liner:**

> Zero-shot explains; few-shot demonstrates.

---

## 75. 🔥 Why can few-shot examples help?

They communicate:

- intended task interpretation;
- output structure;
- style;
- label semantics;
- pattern of the desired mapping.

**Memory liner:**

> Demonstrations reduce ambiguity by showing the mapping rather than only describing it.

---

## 76. 🔥 What are the costs of few-shot prompting?

- collecting good demonstrations;
- consuming context tokens;
- increasing attention work and prompt-processing cost;
- potentially anchoring the model too strongly to a narrow set of examples.

**Memory liner:**

> Examples buy guidance with context budget and possible demonstration bias.

---

## 77. ○ Why might a strong instruction outperform examples in some cases?

The lecture argues that a finite demonstration set may constrain the model toward those particular examples, while a clear natural-language procedure can sometimes generalize better to new inputs.

**Memory liner:**

> Examples show instances; instructions can express the underlying rule.

---

## 78. △ Does the lecture establish that zero-shot is generally better than few-shot?

No. It explicitly treats this as task- and model-dependent and says the literature is evolving.

---

# 12. Explicit reasoning rationales and self-consistency

## 79. 🔥 What does the lecture mean by chain-of-thought prompting?

It asks the model to produce explicit intermediate reasoning or a rationale before the final answer, rather than immediately returning only the answer.

```text
problem
  ↓
intermediate rationale
  ↓
final answer
```

**Memory liner:**

> Give the model output space for intermediate steps before committing to the answer.

---

## 80. ○ Why can explicit intermediate steps improve task performance?

The lecture’s intuition is that decomposing the problem allows useful intermediate relations to be represented in generated tokens before the final decision.

**Memory liner:**

> Intermediate tokens act as a scratch space for multi-step computation.

---

## 81. 🔥 What is the latency trade-off of generating rationales?

More generated tokens require more autoregressive decoding steps:

```text
answer only:          few output tokens
rationale + answer:   many more output tokens
```

**Memory liner:**

> Better decomposition may cost more output tokens and therefore more latency.

---

## 82. △ Does a generated rationale prove what happened inside the model?

The transcript treats visible rationales as useful outputs and debugging aids, but it does not establish that they are faithful records of the model’s hidden computation.

For this lecture, remember the prompting mechanism and token-cost trade-off; keep rationale-faithfulness questions separate.

---

## 83. 🔥 What is self-consistency?

Run the model multiple times, obtain several reasoning paths and answers, extract the final answers, and select the majority result.

```text
same problem
  ├── sample 1 → rationale → answer A
  ├── sample 2 → rationale → answer B
  ├── sample 3 → rationale → answer A
  └── sample 4 → rationale → answer A

majority vote → answer A
```

**Memory liner:**

> Sample many solution paths and vote on the final answer.

---

## 84. 🔥 Why can self-consistency be more robust than one sample?

Individual stochastic generations may make different mistakes. If correct reasoning paths dominate, majority voting can suppress occasional failures.

**Memory liner:**

> Use diversity across runs as an ensemble.

---

## 85. 🔥 What must be extracted for self-consistency voting?

The final answer must be identifiable independently of the rationale. The lecture suggests:

- instructing the model to put the answer last;
- regex or deterministic parsing;
- another model to extract the answer.

**Memory liner:**

> Voting requires a canonical answer representation.

---

## 86. ○ Can self-consistency samples run in parallel?

Yes. The branches are independent and do not enter one another’s contexts.

With sufficient parallel hardware, wall-clock latency can approach the slowest branch, although total compute grows with the number of samples.

**Memory liner:**

> Parallelism can reduce wall time, not total work.

---

## 87. △ When does self-consistency help?

The lecture discusses benchmarked arithmetic/reasoning-style tasks and emphasizes that improvement must be measured against ground-truth labels.

It does not claim universal benefit for all generative tasks.

---

# 13. Exact versus approximate inference improvements

## 88. 🔥 What two efficiency categories does the lecture introduce?

### Exact techniques

Avoid redundant work or manage memory better while preserving the intended model computation/distribution.

### Approximate techniques

Change or simplify the computation and accept a quality approximation for lower cost.

**Memory liner:**

> Exact methods reorganize the same job; approximate methods trade fidelity for efficiency.

---

## 89. ○ What exact-efficiency opportunities does the lecture name?

- remove repeated computation;
- manage memory allocation better;
- algebraically reformulate representations;
- verify several candidate tokens efficiently.

---

## 90. ○ Why separate attention-level and output-level optimization?

```text
attention level:
manage historical token states and attention memory

output level:
reduce how many expensive sequential target-model steps are needed
```

**Memory liner:**

> Optimize both what the model stores and how many decoding steps it performs.

---

# 14. KV cache

## 91. 🔥 What redundant computation appears during naive autoregressive decoding?

Suppose the prefix grows:

```text
step 1: [x1]
step 2: [x1,x2]
step 3: [x1,x2,x3]
...
```

Without caching, the model would repeatedly recompute key and value projections for `x1`, then for `x1,x2`, and so on.

**Memory liner:**

> The past does not change, so its K/V projections should not be rebuilt at every step.

---

## 92. 🔥 What does KV caching store?

At every decoder layer, cache the keys and values of all previously processed tokens.

Per layer:

```text
K_cache: [B,Hkv,T,Dh]
V_cache: [B,Hkv,T,Dh]
```

For `L` layers, a conceptual stacked view is:

```text
K_cache: [L,B,Hkv,T,Dh]
V_cache: [L,B,Hkv,T,Dh]
```

**Memory liner:**

> Cache the searchable memory: past keys and past values.

---

## 93. 🔥 Why are past queries not cached for future attention?

At the new decoding step, only the **new token’s query** asks what past information it needs. Old queries were used for old outputs and are not required to compute the new token’s attention.

```text
new step needs:
Q_new
K_past + K_new
V_past + V_new

new step does not need:
Q_past
```

**Memory liner:**

> Cache what future queries will search, not queries whose jobs are already finished.

---

## 94. 🔥 What is the per-layer cached decoding dataflow?

At step `t`:

```text
new hidden token x_t                [B,1,D]
       ├── Q projection → Q_t       [B,Hq,1,Dh]
       ├── K projection → K_t       [B,Hkv,1,Dh]
       └── V projection → V_t       [B,Hkv,1,Dh]

append K_t,V_t to cache
K_1:t, V_1:t                        [B,Hkv,t,Dh]

Q_t attends to all cached K/V
attention scores                    [B,Hq,1,t]
attention output                    [B,Hq,1,Dh]
```

**Memory liner:**

> Compute Q/K/V only for the new token, append K/V, then query the accumulated cache.

---

## 95. 🔥 Why does the KV cache grow linearly with generated context length?

Ignoring metadata, the number of cached scalar values is approximately:

$$
\boxed{2LBH_{kv}TD_h}
$$

The factor `2` is for keys and values.

For element size `s` bytes:

$$
\boxed{
\text{KV bytes}
\approx
2LBH_{kv}TD_hs
}
$$

**Memory liner:**

> KV memory scales with layers × batch × KV heads × tokens × head width × two tensors.

---

## 96. 🔥 Why is KV cache mainly an inference concept in this lecture?

During teacher-forced training, the entire shifted sequence is processed together; there is no need to grow the prefix one sampled token at a time.

```text
training:
all target positions available → parallel causal pass

inference:
future tokens unknown → sequential decoding → reuse cache
```

**Memory liner:**

> Teacher forcing parallelizes known targets; KV caching accelerates unknown-token autoregression.

---

## 97. 🔥 Does KV caching remove sequential generation?

No. Token `t+1` still depends on token `t`. Caching avoids recomputing past projections but does not reveal future tokens.

**Memory liner:**

> KV cache removes redundant history computation, not the autoregressive dependency.

---

# 15. MHA, GQA, and MQA as cache-size controls

## 98. 🔥 How do attention variants change the KV cache?

### Multi-head attention

```text
Hq query heads
Hq key heads
Hq value heads
Hkv = Hq
```

### Grouped-query attention

```text
Hq query heads
G  key heads
G  value heads
Hkv = G, where 1 < G < Hq
```

### Multi-query attention

```text
Hq query heads
1 key head
1 value head
Hkv = 1
```

**Memory liner:**

> Keep many ways to query while reducing how many memories must be cached.

---

## 99. 🔥 What are the K/V cache shapes under MHA, GQA, and MQA?

```text
MHA:
K,V [B,Hq,T,Dh]

GQA:
K,V [B,G,T,Dh]

MQA:
K,V [B,1,T,Dh]
```

**Memory liner:**

> Cache size follows the number of KV heads, not the number of query heads.

---

## 100. 🔥 Why keep query heads diverse while sharing K/V?

Lecture-level intuition:

- different queries preserve multiple ways of asking what information is relevant;
- past keys and values are stored and reused at every step, so sharing K/V directly reduces cache memory.

**Memory liner:**

> Preserve query expressiveness; compress the repeatedly stored memory.

---

## 101. ○ Are GQA/MQA and sparse attention the same optimization?

No.

```text
GQA / MQA:
reduce distinct K/V head representations

sparse / local attention:
reduce which token positions interact
```

**Memory liner:**

> One compresses the head axis; the other sparsifies the sequence axis.

---

# 16. Naive KV allocation and memory fragmentation

## 102. 🔥 Why is preallocating the maximum context length wasteful?

An inference server does not know when each request will emit EOS. Reserving the full maximum context for every request creates unused space when generations end early.

```text
reserved request capacity: 2048 tokens
actual request usage:        230 tokens
unused reservation:         1818 token slots
```

**Memory liner:**

> Maximum-length reservation turns uncertain generation length into wasted memory.

---

## 103. 🔥 What is internal fragmentation in the lecture’s memory-management picture?

Space is allocated inside a request’s reserved region but never used by actual generated tokens.

**Memory liner:**

> Internal fragmentation is empty space trapped inside an allocation.

---

## 104. 🔥 What is external fragmentation?

Free memory exists, but it is split into disconnected gaps that are inconvenient for a new large contiguous allocation.

```text
used | free | used | free | used
```

**Memory liner:**

> External fragmentation is enough total free space in the wrong physical arrangement.

---

## 105. ○ Why does fragmentation reduce serving throughput?

KV cache memory limits how many requests can coexist. Wasted or unusable gaps mean fewer active sequences can be served concurrently.

**Memory liner:**

> Poor memory packing converts available bytes into lower request capacity.

---

# 17. PagedAttention

## 106. 🔥 What is the core idea of PagedAttention?

Store a sequence’s KV cache in fixed-size blocks rather than one large contiguous region reserved for its maximum possible length.

```text
logical sequence KV
block 0 → block 1 → block 2 → ...

physical memory
blocks may live in different available locations
```

**Memory liner:**

> Allocate KV memory a page at a time as the sequence grows.

---

## 107. 🔥 What is the analogy behind the name “PagedAttention”?

It resembles virtual-memory paging:

- logical token positions form a contiguous sequence;
- a block table maps them to non-contiguous physical memory blocks.

**Memory liner:**

> Logical continuity does not require physical contiguity.

---

## 108. 🔥 What does the block table do?

It maps logical KV blocks for a request to physical blocks in memory.

```text
logical block 0 → physical block 17
logical block 1 → physical block  4
logical block 2 → physical block 29
```

The attention implementation follows the mapping when retrieving cached K/V.

---

## 109. 🔥 How many blocks does a sequence need?

If block size is `P` tokens and current sequence length is `T`:

$$
\boxed{N_{blocks}=\left\lceil\frac{T}{P}\right\rceil}
$$

Only the final block can be partially unused.

**Memory liner:**

> Paging confines most internal waste to at most the tail of the final block.

---

## 110. 🔥 How does PagedAttention improve concurrency?

By reducing over-reservation and fragmentation, the server can fit more live KV caches in the same device memory.

**Memory liner:**

> Better KV packing means more simultaneous requests.

---

## 111. ○ Does PagedAttention change the mathematical attention result?

At the lecture level, no. It changes **where and how K/V are stored and retrieved**, not which cached values attention conceptually uses.

**Memory liner:**

> PagedAttention is memory virtualization for the KV cache, not a new semantic attention rule.

---

## 112. △ What PagedAttention details are outside this transcript?

The lecture does not derive:

- allocator algorithms;
- copy-on-write or prefix sharing;
- exact kernels;
- scheduler design;
- throughput equations.

It establishes fixed-size blocks, logical-to-physical mapping, and fragmentation reduction.

---

# 18. Multi-Latent Attention (MLA): lecture-level mental model

## 113. 🔥 What problem does MLA target in this lecture?

Even with KV caching, every layer must retain substantial key/value representations for every token. MLA asks whether those representations can be cached in a lower-dimensional latent form.

**Memory liner:**

> Instead of caching expanded K and V, cache a compact latent from which they can be reconstructed.

---

## 114. 🔥 What is the lecture-level low-rank factorization idea?

For token representation `x\in\mathbb{R}^{D}`, compress it:

$$
\boxed{c=xW_D}
$$

where:

```text
x:   [B,T,D]
W_D: [D,Dc]
c:   [B,T,Dc]
Dc < expanded KV representation size
```

Then reconstruct key/value forms through learned up-projections:

$$
K=cW_U^K,
\qquad
V=cW_U^V
$$

**Memory liner:**

> Down-project once, cache the bottleneck, up-project when K/V are needed.

---

## 115. 🔥 What is shared in the lecture’s MLA explanation?

The lecture says the compressed latent can be shared:

- across key and value reconstruction;
- across attention heads.

Different decompression matrices can still produce different K and V representations.

```text
                 ┌→ key decompression → K
x → shared latent c
                 └→ value decompression → V
```

**Memory liner:**

> One cached code can feed multiple learned reconstructions.

---

## 116. 🔥 What is cached under this simplified MLA view?

Instead of storing separate K/V vectors for every head, cache one compact latent representation per token per Transformer layer.

Conceptual cache:

```text
latent cache: [L,B,T,Dc]
```

rather than:

```text
K cache: [L,B,Hkv,T,Dh]
V cache: [L,B,Hkv,T,Dh]
```

**Memory liner:**

> Cache one low-dimensional code rather than many expanded head-wise K/V vectors.

---

## 117. 🔥 Why can this reduce memory?

The standard KV cache stores approximately:

$$
2H_{kv}D_h
$$

scalars per token per layer.

The simplified latent cache stores approximately:

$$
D_c
$$

scalars per token per layer.

The compression is useful when:

$$
D_c\ll2H_{kv}D_h
$$

**Memory liner:**

> Cache reduction comes from replacing expanded head-wise state with a smaller shared bottleneck.

---

## 118. ○ Is the low-rank dimension learned or chosen?

The lecture says the **size** `Dc` is an architectural design choice; the projection weights themselves are learned.

**Memory liner:**

> Choose the bottleneck width; learn the compression and decompression matrices.

---

## 119. ○ What non-memory benefit does the lecture mention?

It reports that shared compressed representations may also improve model performance, possibly through a regularizing effect.

**Boundary:** the transcript does not establish a detailed causal explanation or ablation.

---

## 120. △ What should not be inferred from this simplified MLA note?

The lecture does not give the complete production MLA formulation, including every interaction with positional encoding, decoupled components, or exact implementation algebra.

For this lecture, memorize:

```text
KV projection
    → low-dimensional shared latent
    → K/V reconstruction
    → smaller cache
```

---

# 19. Speculative decoding

## 121. 🔥 What bottleneck does speculative decoding attack?

Autoregressive target-model decoding normally advances only one token per expensive sequential step.

Speculative decoding tries to validate several likely future tokens in one target-model pass.

**Memory liner:**

> Let a cheap model propose several tokens; let the expensive model verify them together.

---

## 122. 🔥 What are the draft and target models?

- **Draft model:** smaller/faster model that proposes a short continuation.
- **Target model:** larger model whose output distribution must be preserved.

```text
prefix
  ↓
draft model generates k candidate tokens sequentially
  ↓
target model scores the candidate block in one pass
  ↓
accept / reject according to verification rule
```

---

## 123. 🔥 What is the basic speculative-decoding flow?

```text
1. Start from current accepted prefix.
2. Draft model proposes d1,...,dk.
3. Target model evaluates the prefix plus all draft tokens together.
4. Compare draft and target probabilities token by token.
5. Accept a valid prefix of proposals.
6. At the first rejection, sample a corrected token from an adjusted distribution.
7. Continue from the accepted/corrected prefix.
```

**Memory liner:**

> Propose cheaply, verify in parallel, keep the longest accepted prefix.

---

## 124. 🔥 Why can the target model score several proposed tokens in one forward pass?

Given the proposed block, causal masking permits the target model to compute the next-token distribution at every position of that block simultaneously.

```text
input positions: prefix, d1, d2, ..., dk
output distributions:
P(d1 | prefix)
P(d2 | prefix,d1)
...
P(next | prefix,d1,...,dk)
```

**Memory liner:**

> A known candidate block turns several would-be decoding steps into parallel position evaluations.

---

## 125. 🔥 What happens if all draft tokens are accepted?

The generation advances by several tokens after one target-model verification pass. The target pass also provides a distribution for the token following the accepted draft block, allowing one additional target-consistent token to be sampled.

**Memory liner:**

> Full acceptance converts one target pass into multiple accepted output tokens plus the next distribution.

---

## 126. 🔥 What happens at the first rejected token?

The accepted prefix before that position is retained. Generation resumes from the rejection point using a corrected distribution designed to preserve the target model’s distribution.

**Memory liner:**

> Keep the valid prefix; repair at the first disagreement.

---

## 127. 🔥 Why is rejection sampling central?

The draft proposals come from a different distribution. The acceptance/rejection correction ensures that the final samples follow the target model rather than the draft model.

**Memory liner:**

> The draft accelerates proposal; the correction preserves target-model sampling.

---

## 128. △ What exact speculative-decoding equations should be memorized from this transcript?

The lecture explains the distribution-preserving acceptance/rejection mechanism and references a proof using the law of total probability, but the exact acceptance and correction formulas are not reliably verbalized in the transcript.

Do **not** reconstruct a formula from the garbled sentence. For this lecture, memorize the mechanism and invariant:

> The output distribution should match the target model despite using draft proposals.

---

## 129. 🔥 Why can speculative decoding be faster even though the large model still checks tokens?

The lecture’s systems intuition is that autoregressive inference is often limited by moving model weights and memory rather than by using every arithmetic unit. Scoring several positions in one large-model pass can use hardware more efficiently than launching one large-model pass per token.

**Memory liner:**

> Trade many memory-bound target launches for one wider verification pass.

---

## 130. ○ When will speculative decoding provide little speedup?

From the mechanism, speedup weakens when:

- the draft model is not much cheaper;
- proposed tokens are frequently rejected;
- verification overhead is high.

These are direct consequences of the lecture’s propose-and-verify design, though no speedup formula is derived.

---

# 20. Multi-token prediction

## 131. 🔥 What is multi-token prediction in this lecture?

Attach several prediction heads to the final decoder representation so training predicts multiple future tokens rather than only the immediate next token.

Conceptually, for one position:

```text
shared hidden state h_t [B,D]
   ├── head 1 → predict x_{t+1} [B,V]
   ├── head 2 → predict x_{t+2} [B,V]
   ├── head 3 → predict x_{t+3} [B,V]
   └── ...
```

**Memory liner:**

> One hidden state supervises several future-token prediction heads.

---

## 132. 🔥 How does the training objective differ from ordinary next-token prediction?

Ordinary objective:

$$
\mathcal{L}_{NTP}
=
-\log p_1(x_{t+1}\mid x_{\le t})
$$

Lecture-level multi-token objective:

$$
\boxed{
\mathcal{L}_{MTP}
=
\sum_{j=1}^{m}
\lambda_j
\left[-\log p_j(x_{t+j}\mid x_{\le t})\right]
}
$$

The transcript establishes multiple future targets; exact weights `\lambda_j` are not specified.

**Memory liner:**

> Change supervision from one future offset to several future offsets.

---

## 133. 🔥 How can multi-token heads act like an internal draft model?

At inference:

- auxiliary future-token heads propose several tokens;
- the primary first-token head/model verifies them;
- accepted tokens advance decoding.

**Memory liner:**

> Speculative proposals come from extra heads inside the same model instead of a separate draft network.

---

## 134. 🔥 Speculative decoding vs multi-token prediction?

| | Speculative decoding | Multi-token prediction in this lecture |
|---|---|---|
| Draft source | Separate smaller model | Auxiliary heads in same model |
| Training change required? | Not necessarily | Yes: multi-future-token objective |
| Verification | Target model | Primary/main prediction path |
| Main idea | External cheap proposer | Embedded proposer |

**Memory liner:**

> Speculation uses a separate helper; MTP builds the helper into the model and training objective.

---

## 135. △ Is lecture-level MTP guaranteed to reproduce the exact target sampling distribution?

The lecture says the described paper uses a greedy acceptance variation and does not retain the same clean distribution-preservation property as the earlier speculative-decoding formulation.

Remember the distinction:

```text
classical speculative decoding → target-distribution correction
lecture MTP variant           → internal drafts + greedy verification variation
```

---

# 21. High-level implementation mental maps

## 136. 🔥 Sparse MoE forward pass — PyTorch-like mental model

```python
# x: [B, T, D]
router_logits = router(x)                 # [B, T, E]
router_prob = softmax(router_logits, -1)  # [B, T, E]

weights, expert_ids = topk(
    router_prob,
    k=k,
    dim=-1,
)                                          # both [B, T, k]

# Conceptually:
# 1. gather tokens assigned to each expert
# 2. run that expert FFN
# 3. multiply by routing weight
# 4. scatter-add back into original token positions

y = zeros_like(x)                          # [B, T, D]
for expert_id in range(E):
    token_positions = assignments_to(expert_id)
    x_i = gather(x, token_positions)        # [N_i, D]
    y_i = expert[expert_id](x_i)            # [N_i, D]
    y_i = y_i * selected_weight(...)        # [N_i, D]
    scatter_add(y, token_positions, y_i)
```

**Interview warning:** real implementations batch and dispatch tokens efficiently; they do not normally use a Python loop like this.

---

## 137. 🔥 Sampling dataflow

```python
# hidden: [B, D]
logits = lm_head(hidden)                    # [B, V]
scaled = logits / temperature

# Optional truncation
scaled = apply_top_k_or_top_p_mask(scaled)

prob = softmax(scaled, dim=-1)              # [B, V]
next_token = categorical_sample(prob)       # [B]
```

**Memory liner:**

> Temperature reshapes, top-k/top-p truncate, categorical sampling chooses.

---

## 138. 🔥 KV-cache generation loop

```python
cache = empty_cache()
token = initial_token

while token != EOS:
    # Only the newest token enters each decoder layer.
    logits, cache = model.decode_one_token(
        token,                  # [B, 1]
        cache=cache,
    )

    token = select_next_token(logits[:, -1])
```

Inside one attention layer:

```python
q_new = q_proj(x_new)                       # [B, Hq, 1, Dh]
k_new = k_proj(x_new)                       # [B, Hkv, 1, Dh]
v_new = v_proj(x_new)                       # [B, Hkv, 1, Dh]

K_cache = append(K_cache, k_new, token_dim)
V_cache = append(V_cache, v_new, token_dim)

scores = q_new @ K_cache.transpose(-2, -1)  # [B, Hq, 1, T]
out = softmax(scores, -1) @ V_cache         # [B, Hq, 1, Dh]
```

---

## 139. 🔥 Paged KV-cache mental model

```python
# Logical block IDs for one request
logical_blocks = [0, 1, 2, ...]

# Block table maps them to physical GPU blocks
block_table = {
    0: 17,
    1: 4,
    2: 29,
}

# When the current block fills, allocate any available block.
if current_block_is_full:
    block_table[next_logical_id] = allocator.get_free_block()
```

**Memory liner:**

> Append logical pages; place physical pages wherever memory is available.

---

## 140. 🔥 Speculative-decoding mental loop

```python
while not finished:
    draft_tokens = draft.generate(prefix, num_tokens=k)

    # One target pass scores all proposed positions.
    target_distributions = target.score_block(prefix, draft_tokens)

    accepted_prefix, correction = verify_with_rejection_sampling(
        draft_tokens,
        target_distributions,
    )

    prefix.extend(accepted_prefix)
    if correction is not None:
        prefix.append(correction)
```

**Boundary:** the transcript does not provide a cleanly transcribed formula for `verify_with_rejection_sampling`.

---

# 22. Equations to memorize cold

## Causal language-model factorization

$$
\boxed{
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})}
$$

## Sequence log-probability

$$
\boxed{
\log p(x_{1:T})
=
\sum_{t=1}^{T}\log p(x_t\mid x_{<t})
}
$$

## Vocabulary projection

$$
\boxed{z_t=h_tW_{vocab}+b}
$$

## Softmax

$$
\boxed{
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}}
$$

## Temperature-scaled softmax

$$
\boxed{
p_i(\tau)=
\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}}
$$

## Generic MoE

$$
\boxed{
y=\sum_{i=1}^{E}G_i(x)E_i(x)}
$$

## Sparse top-k MoE

$$
\boxed{
y=\sum_{i\in\operatorname{TopK}(G(x),k)}
\tilde G_i(x)E_i(x)}
$$

## Approximate FFN parameter count

$$
\boxed{N_{FFN}\approx2DD_{ff}}
$$

## MoE auxiliary load-balancing loss

$$
\boxed{
\mathcal{L}_{aux}=\alpha E\sum_{i=1}^{E}f_iP_i
}
$$

## Top-p candidate set

$$
\boxed{
S_p=\text{smallest highest-probability set such that}
\sum_{i\in S_p}p_i\ge p
}
$$

## KV-cache scalar count

$$
\boxed{N_{KV}\approx2LBH_{kv}TD_h}
$$

## KV-cache memory

$$
\boxed{
M_{KV}\approx2LBH_{kv}TD_h\times\text{bytes per element}
}
$$

## Number of cache pages

$$
\boxed{
N_{blocks}=\left\lceil\frac{T}{P}\right\rceil
}
$$

## Lecture-level MLA compression

$$
\boxed{c=xW_D}
$$

$$
\boxed{K=cW_U^K,\qquad V=cW_U^V}
$$

## Lecture-level multi-token objective

$$
\boxed{
\mathcal{L}_{MTP}
=
\sum_{j=1}^{m}\lambda_j
\left[-\log p_j(x_{t+j}\mid x_{\le t})\right]
}
$$

---

# 23. Tensor shapes to memorize cold

```text
Input token IDs
[B, T]

Decoder hidden sequence
[B, T, D]

Current final hidden state
[B, D]

Vocabulary projection
W_vocab: [D, V]

Next-token logits / probabilities
[B, V]

MoE router weight
[D, E]

MoE router logits / probabilities
[B, T, E]

Top-k expert IDs / weights
[B, T, k]

Tokens gathered for expert i
[N_i, D]

Expert hidden expansion
[N_i, Dff]

Expert output
[N_i, D]

MoE layer output
[B, T, D]

New query during cached decoding
[B, Hq, 1, Dh]

New K/V
[B, Hkv, 1, Dh]

Per-layer K/V cache after T tokens
[B, Hkv, T, Dh]

All-layer conceptual K/V cache
[L, B, Hkv, T, Dh]

Cached-attention scores for one new token
[B, Hq, 1, T]

MLA compressed latent per layer
[B, T, Dc]

Conceptual all-layer MLA cache
[L, B, T, Dc]

MTP output with m prediction heads
m tensors of [B, V]
# or conceptually [B, m, V]
```

The most important cached-attention multiplication is:

$$
\boxed{
[B,H_q,1,D_h]
\times
[B,H_q,D_h,T]
=
[B,H_q,1,T]
}
$$

followed by:

$$
\boxed{
[B,H_q,1,T]
\times
[B,H_q,T,D_h]
=
[B,H_q,1,D_h]
}
$$

When `Hkv < Hq`, K/V groups are logically shared or broadcast to their associated query heads.

---

# 24. Twelve high-value comparisons

## 1. Dense model vs sparse MoE

```text
Dense model:
every token activates the same FFN parameters

Sparse MoE:
every token activates only top-k FFN experts
```

## 2. Total vs active parameters

```text
total parameters  = everything stored
active parameters = parameters touched for this token/pass
```

## 3. Attention heads vs MoE experts

```text
heads   = alternative attention projection spaces
experts = alternative FFN transformations
```

## 4. Greedy vs beam vs sampling

```text
greedy:
one locally best token

beam:
several high-likelihood partial sequences

sampling:
stochastic draw from a token distribution
```

## 5. Top-k vs top-p

```text
top-k:
fixed candidate count

top-p:
adaptive candidate count, fixed cumulative mass
```

## 6. Temperature vs top-k/top-p

```text
temperature:
changes relative probability sharpness across logits

top-k/top-p:
sets some token probabilities to zero by truncating support
```

## 7. Prompt retry vs guided decoding

```text
retry:
detect malformed result after generation

guided decoding:
prevent malformed next-token choices during generation
```

## 8. Parameter learning vs in-context learning

```text
training / fine-tuning:
change weights

in-context learning:
change prompt, keep weights fixed
```

## 9. Few-shot vs self-consistency

```text
few-shot:
multiple demonstrations inside one prompt

self-consistency:
multiple independent output samples for one problem
```

## 10. KV cache vs PagedAttention

```text
KV cache:
what historical tensors should be reused?

PagedAttention:
how should those cached tensors be allocated in memory?
```

## 11. GQA vs MLA

```text
GQA:
reduce the number of distinct KV heads

MLA:
compress cached K/V information through a lower-dimensional latent
```

## 12. Speculative decoding vs MTP

```text
speculative decoding:
separate cheap draft model proposes tokens

MTP:
extra heads inside the target model propose future tokens
```

---

# 25. Interview questions to practise aloud

## LLM foundations

1. What is the autoregressive factorization of a token sequence?
2. Why does a decoder-only model not contain cross-attention?
3. Distinguish model probabilities from decoding policy.
4. In what senses does this lecture call an LLM “large”?
5. Walk from prefix token IDs to next-token probabilities with tensor shapes.

## Mixture of Experts

6. What problem does sparse MoE solve?
7. What are the router and expert functions?
8. Write the dense and top-k sparse MoE equations.
9. Derive the router tensor shape `[B,T,E]`.
10. Why is routing performed at token level?
11. At which point in a decoder block is an MoE router applied?
12. Why replace the FFN rather than the attention heads in this lecture?
13. Derive the approximate `2DDff` FFN parameter count.
14. Distinguish total and active parameters.
15. Why can total capacity grow without proportional token-level FLOPs?
16. Why are attention-head count and expert count independent?
17. What is routing collapse?
18. Define `f_i` and `P_i` in the load-balancing objective.
19. What behavior does `αEΣ_i f_iP_i` encourage?
20. What is noisy gating trying to prevent?
21. Which aspects of hard top-k differentiability are left unresolved here?

## Decoding

22. Write the sequence probability and log-probability equations.
23. Why is greedy decoding locally but not globally optimal?
24. What does beam width mean?
25. Why is beam search still approximate?
26. Why does raw sequence probability favor shorter sequences?
27. Why can beam search be suitable for translation but less attractive for open-ended chat?
28. What is ancestral sampling?
29. Why truncate the low-probability tail?
30. Define top-k sampling mathematically.
31. Define top-p sampling mathematically.
32. Why is top-p’s candidate count adaptive?
33. How does sampling differ from model training?

## Temperature and constrained generation

34. Write temperature-scaled softmax.
35. Prove the `T→0+` argmax limit.
36. Prove the `T→∞` uniform limit.
37. Does positive temperature change logit ranking?
38. Where does mathematical randomness enter generation?
39. Why is literal division by temperature zero invalid?
40. What is guided decoding?
41. How does a validity mask enforce output syntax?
42. Why can guided generation outperform generate-parse-retry?

## Context and prompting

43. What does context length count?
44. Why does long accepted context not guarantee successful retrieval?
45. Explain needle-in-a-haystack evaluation.
46. What does the lecture mean by context rot?
47. Why can distractors hurt grounding?
48. What practical context-selection lesson follows?
49. Decompose a prompt into context, instruction, input, and constraints.
50. Why should instruction and input be separated?
51. What is in-context learning, and what does not change during it?
52. Compare zero-shot and few-shot prompting.
53. What are the benefits and costs of demonstrations?
54. Why might a procedural instruction generalize beyond finite examples?
55. What is explicit rationale prompting at the level of this lecture?
56. What latency cost follows from longer rationale outputs?
57. What is self-consistency?
58. Why does voting require canonical answer extraction?
59. How can self-consistency branches be parallelized?
60. Why must self-consistency gains be benchmarked rather than assumed?

## Efficient inference

61. What computation does the KV cache avoid?
62. Why cache K/V but not old Q?
63. Give the per-layer KV-cache shape.
64. Derive the approximate KV-cache memory formula.
65. Why does KV cache matter at inference but not ordinary teacher-forced training?
66. Does KV cache remove sequential token dependence?
67. Compare MHA, GQA, and MQA in cache shape.
68. Why do fewer KV heads reduce memory?
69. Distinguish GQA from sequence-sparse attention.
70. Why is maximum-context preallocation wasteful?
71. Distinguish internal and external fragmentation.
72. Explain PagedAttention with the virtual-memory analogy.
73. What does a block table map?
74. Derive `ceil(T/P)` allocated blocks.
75. Does PagedAttention change the model’s mathematical attention rule?
76. What does MLA compress?
77. Write the lecture-level down-projection and K/V reconstruction equations.
78. Why can one shared latent be smaller than an expanded KV cache?
79. Which parts of full MLA are outside this transcript?
80. Explain speculative decoding in seven steps.
81. Why can a target model score a draft block in one pass?
82. What invariant should rejection sampling preserve?
83. Why does high draft acceptance improve speed?
84. When would speculative decoding provide little benefit?
85. What training change defines multi-token prediction?
86. How can auxiliary heads become an internal draft mechanism?
87. Contrast classical speculative decoding with the lecture’s greedy MTP variation.

---

# 26. Twenty-five interview-day memory liners

```text
1. Language model:
   A prefix maps to a probability distribution over the next token.

2. Decoder-only:
   No encoder means no cross-attention.

3. LLM scale:
   Large model, large token corpus, large compute.

4. MoE:
   More total capacity, fewer active parameters per token.

5. Router:
   One contextual token becomes one distribution over experts.

6. Sparse MoE:
   Select top-k experts, run only those FFNs, combine their outputs.

7. MoE placement:
   Experts replace the parameter-heavy FFN, not the attention heads.

8. Active parameters:
   Storage follows all experts; per-token compute follows selected experts.

9. Routing collapse:
   A few experts monopolize the traffic.

10. Load balancing:
    Balance router preference and actual token assignments.

11. Greedy decoding:
    Best next token is not necessarily the best full sequence.

12. Beam search:
    Keep several prefixes, but pruned paths never return.

13. Sequence score:
    Product of probabilities becomes a sum of token log-probabilities.

14. Top-k vs top-p:
    Fixed candidate count versus adaptive cumulative probability mass.

15. Temperature:
    Low sharpens; high flattens; positive temperature preserves ranking.

16. Guided decoding:
    Mask invalid next tokens before sampling.

17. Context quality:
    More context is not automatically more usable context.

18. Prompt anatomy:
    Context, instruction, input, constraints.

19. In-context learning:
    Adapt behavior through the prompt without changing weights.

20. Self-consistency:
    Sample independent solution paths and vote on final answers.

21. KV cache:
    Cache past K/V because the past is unchanged and future queries reuse it.

22. PagedAttention:
    Logical sequence continuity does not require contiguous physical KV memory.

23. MLA:
    Cache a compact latent and reconstruct K/V when needed.

24. Speculative decoding:
    Propose cheaply, verify together, preserve the target distribution.

25. Multi-token prediction:
    Train several future-token heads and reuse them as internal drafts.
```

---

# 27. Previous-day priority checklist

## Tier 1 — must answer immediately

- [ ] Write autoregressive sequence factorization.
- [ ] Draw decoder-only next-token dataflow with `[B,T,D] → [B,V]`.
- [ ] Explain router, expert, dense MoE, and sparse top-k MoE.
- [ ] Derive router shape `[B,T,E]`.
- [ ] Explain why MoE replaces the FFN.
- [ ] Distinguish total parameters, active parameters, and FLOPs.
- [ ] Explain routing collapse and the auxiliary balancing loss.
- [ ] Compare greedy, beam, sampling, top-k, and top-p.
- [ ] Write temperature softmax and its low/high-temperature limits.
- [ ] Explain guided decoding.
- [ ] Define context length and context rot.
- [ ] Distinguish zero-shot, few-shot, and self-consistency.
- [ ] Explain exactly what K/V caching reuses.
- [ ] Derive `[L,B,Hkv,T,Dh]` cache shape and memory scaling.
- [ ] Compare MHA, GQA, and MQA.
- [ ] Explain internal vs external fragmentation.
- [ ] Explain PagedAttention and its block table.
- [ ] Explain the simplified MLA compression/decompression path.
- [ ] Walk through speculative decoding.
- [ ] Explain multi-token prediction and how it differs from a separate draft model.

## Tier 2 — should explain clearly

- [ ] Why greedy is locally optimal only.
- [ ] Why beam scores have short-output bias.
- [ ] Why top-p adapts to model uncertainty.
- [ ] Why temperature does not alter rank order.
- [ ] Why a large context window does not guarantee retrieval.
- [ ] Prompt anatomy: context/instruction/input/constraints.
- [ ] Rationale-token latency trade-off.
- [ ] Why parallel self-consistency reduces wall time but not total compute.
- [ ] Why KV caching does not make generation non-autoregressive.
- [ ] Why paging improves serving concurrency.
- [ ] Why speculative speedup depends on draft acceptance.

## Tier 3 — preserve as lecture boundaries

- [ ] Exact differentiation through hard top-k routing.
- [ ] Exact beam length-penalty formula.
- [ ] Grammar/FSM compilation for guided decoding.
- [ ] Formal long-context benchmark details.
- [ ] Faithfulness of generated rationales.
- [ ] Full PagedAttention allocator/kernel design.
- [ ] Complete production MLA equations.
- [ ] Exact speculative acceptance/correction equation from the slides.
- [ ] Exact MTP loss weights and acceptance algorithm.

---

# 28. Lecture boundaries — do not silently add these to this note

The transcript mentions, motivates, or gestures toward several subjects without teaching them fully:

- A universal formal definition or parameter threshold for “LLM.”
- Expert-capacity limits, token dropping, expert parallelism, or communication costs in distributed MoE.
- Exact gradients/estimators through hard top-k routing and the discrete `f_i` statistic.
- Exact beam-search length-penalty equations.
- Repetition penalties, typical sampling, min-p, contrastive decoding, or diverse beam search.
- Detailed hardware reproducibility controls for deterministic inference.
- Construction of finite-state machines or grammars for guided decoding.
- A full RAG architecture, retrieval embeddings, chunking, or reranking.
- Formal analysis of whether visible generated rationales faithfully represent hidden model computation.
- FlashAttention or other named exact-attention kernels.
- Prefix caching, cache sharing, copy-on-write, continuous batching, or scheduler design.
- Full vLLM/PagedAttention implementation internals.
- Complete DeepSeek MLA, including every positional and projection detail.
- The exact speculative-decoding acceptance and correction equations; the transcript’s spoken version is incomplete/garbled.
- Detailed speculative-decoding speedup models.
- Full multi-token-prediction architecture, loss weighting, or greedy verification algorithm.

Keep these for dedicated future notes or paper-level study.

---

# 29. Final 60-second lecture mental map

```text
MODERN LLM IN THIS LECTURE
large decoder-only causal language model
prefix → next-token probabilities → decoding rule → append token

CAPACITY
Dense FFN: activate the same parameters for every token
MoE FFN: router selects top-k experts per contextual token
          → more total parameters
          → controlled active parameters
          → must prevent routing collapse

GENERATION QUALITY
Greedy: locally best, deterministic, low diversity
Beam: several likely paths, expensive, length bias
Sampling: stochastic
  ├── top-k: fixed candidate count
  ├── top-p: adaptive probability mass
  └── temperature: sharpen or flatten probabilities
Guided decoding: mask structurally invalid next tokens

PROMPTING
context quality matters; very long context can hurt retrieval
prompt = context + instruction + input + constraints
zero-shot = instruction only
few-shot = demonstrations
explicit rationale = more intermediate tokens
self-consistency = sample many paths and vote

INFERENCE EFFICIENCY
KV cache:
reuse past K/V; compute only new token projections

GQA / MQA:
reduce number of cached KV heads

PagedAttention:
store KV in mapped fixed-size blocks to reduce fragmentation

MLA:
cache one compact latent and reconstruct K/V

Speculative decoding:
small draft proposes several tokens
large target verifies them together
accept/reject so output follows target distribution

Multi-token prediction:
multiple future-token heads create an internal draft mechanism
```

---

# 30. One-sentence master takeaway

> Lecture 3 connects **what a modern decoder-only LLM is** to **how it scales capacity with sparse experts, controls output through decoding and prompting, and serves tokens efficiently by compressing, paging, caching, and speculatively verifying the autoregressive computation**.
