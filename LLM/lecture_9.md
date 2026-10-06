# CME 295 Lecture 9 - Whole-Course Recap, Multimodal Transformers, Diffusion LLMs, and Current Trends

> **Scope:** This note is grounded in the supplied **CME 295 Lecture 9 transcript: Recap & Current Trends**. Lecture 9 has two technical roles: it reconnects the major ideas from Lectures 1-8, and it introduces forward-looking topics discussed in the course's Autumn 2025 context. The recap below includes only concepts that this lecture explicitly revisits; it does not silently replace the separate detailed notes for earlier lectures.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: high-value questions, intuitive answers, equations, tensor shapes, end-to-end dataflows, comparisons, implementation sketches, failure modes, and memory liners.
>
> **Interview clarifications:** Standard equations or implementation details added to make a lecture topic interview-ready are marked **INTERVIEW CLARIFICATION**. They clarify the lecture rather than claiming that the equation was derived on the board.
>
> **Transcript normalization:** The automatic transcript occasionally distorts technical names. This note uses the standard names **Word2Vec**, **BERT**, **RoPE**, **GQA**, **MoE**, **FlashAttention**, **Bradley-Terry**, **PPO**, **GRPO**, **RAG**, **LLM-as-a-judge**, **Vision Transformer (ViT)**, **vision-language model (VLM)**, **LLaVA**, **Diffusion Transformer (DiT)**, **masked diffusion model (MDM)**, **diffusion LLM (dLLM)**, **LLaDA**, **DeepSeek-OCR**, **2D RoPE**, **Adam**, **Muon**, **model collapse**, **small language model (SLM)**, and **Pareto frontier** while preserving the lecture's intended framing.

**Source transcript:** Stanford CME 295, Lecture 9 - Recap & Current Trends  
`https://www.youtube.com/watch/Q86qzJ1K1Ss`

---

# Lecture spine

```text
PART I - THE COURSE AS ONE SYSTEM

raw text
   -> tokenization
   -> token / position representations
   -> contextualization
        RNN -> self-attention -> Transformer
   -> architecture family
        encoder-only / encoder-decoder / decoder-only
   -> scalable LLM architecture
        RoPE / GQA / Pre-Norm / MoE
   -> generation
        next-token distribution / sampling / temperature
   -> training
        pre-training -> SFT -> preference tuning
   -> advanced capabilities
        reasoning with GRPO
        fresh knowledge with RAG
        actions with tools and agents
        quality measurement with evaluation

PART II - DIRECTIONS DISCUSSED AS CURRENT TRENDS IN AUTUMN 2025

transformers leave text
   -> ViT
   -> VLMs
   -> diffusion transformers and other modalities

ideas also move back into text
   -> masked-diffusion / diffusion LLMs
   -> iterative parallel refinement

open research fronts
   -> optimizer / normalization / attention / activation / MoE design
   -> data curation, mid-training, synthetic-data model collapse
   -> smaller models and quality-cost Pareto frontiers
   -> hardware-software co-design
   -> browser / OS agents, continuous learning, factuality,
      personalization, interpretability, and safety
```

**Master memory liner:**

> The course moves from representing tokens, to modelling context, to training and aligning a generator, to connecting it with knowledge and actions, and finally to asking whether the Transformer and autoregressive generation are the final forms at all.

---

# Notation

```text
B              batch size
T              text sequence length
V              text vocabulary size
D              model / token dimension
H              number of attention heads
Dh             per-head dimension, usually D / H
G              number of key/value groups in GQA
E              number of MoE experts
K              number of selected experts, documents, or candidates
N              generic number of examples or tokens

C_img          number of image channels, usually 3 for RGB
H_img, W_img   image height and width
P              image patch side length
N_img          number of image patches = (H_img/P)(W_img/P)
D_img          image-encoder feature width
T_txt          number of text tokens

M              Boolean mask over token positions
S              number of diffusion refinement steps
G_rollout      number of GRPO completions sampled per prompt
r_i            reward for completion i
A_i            advantage for completion i
pi_theta       trainable policy
pi_old         policy used to generate the current rollout batch
pi_ref         frozen reference / SFT policy
beta           KL or reference-regularization strength
```

### Priority legend

- **MUST REMEMBER:** answer immediately and derive the key equation or shape.
- **SHOULD KNOW:** explain the mechanism, trade-off, or failure mode clearly.
- **LECTURE BOUNDARY:** mentioned in the lecture but not derived there.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture interview-ready.

---

# 1. The whole course as one mental model

## Q1. MUST REMEMBER - What is the central story of the course as reconstructed in Lecture 9?

```text
Represent text
    -> contextualize it
    -> generate with a Transformer
    -> scale and train the model
    -> teach useful behavior
    -> align preferences
    -> improve reasoning
    -> connect external knowledge and actions
    -> evaluate the resulting system
```

Each stage addresses a limitation of the previous stage.

**Memory liner:**

> Every later technique repairs a limitation exposed by the earlier model.

---

## Q2. MUST REMEMBER - What four major weaknesses of a standalone assistant does the course ultimately address?

```text
Limited multi-step reasoning
    -> reasoning traces + verifiable-reward RL / GRPO

Stale or private knowledge
    -> RAG and external search / databases

No ability to act
    -> tool calling and agent loops

Hard-to-measure free-form behavior
    -> direct verifiers, LLM judges, and benchmarks
```

**Memory liner:**

> Reason, retrieve, act, evaluate.

---

## Q3. SHOULD KNOW - Why is the system view more valuable than memorizing a catalogue of acronyms?

An interviewer often asks where a component belongs and what failure it fixes. A good answer should connect:

```text
problem -> mechanism -> objective -> inference behavior -> cost / risk
```

For example, GQA is not just a head-layout acronym; it reduces the repeatedly stored K/V memory during autoregressive inference while preserving multiple query heads.

**Memory liner:**

> Know why a component exists, not only what its acronym expands to.

---

## Q4. MUST REMEMBER - What is the difference between model capability and system capability?

A base model supplies learned prediction and reasoning. A deployed system can add:

```text
retrieval
search
calculators
code execution
tool APIs
memory
retries
verification
safety checks
```

Therefore, model quality alone does not determine product quality.

**Memory liner:**

> The model predicts tokens; the system engineers context, tools, feedback, and control around those predictions.

---

# 2. From raw text to self-attention

## Q5. MUST REMEMBER - Why is tokenization the first modelling decision?

Neural models process numerical arrays, not raw strings. Tokenization chooses the discrete units that will receive IDs and embeddings.

```text
text -> token strings -> token IDs -> vectors
```

**Memory liner:**

> Before learning a representation, decide what object is being represented.

---

## Q6. MUST REMEMBER - What is the core tokenization trade-off?

| Unit | Sequence length | Vocabulary / OOV behavior | Main intuition |
|---|---:|---|---|
| Word | Short | Large vocabulary; high OOV risk | Easy units, poor morphology sharing |
| Subword | Medium | Reuses common fragments; low OOV risk | Standard compromise |
| Character / byte | Long | Tiny reusable alphabet | Robust but computationally expensive |

**Memory liner:**

> Larger units shorten sequences; smaller units improve coverage and reuse.

---

## Q7. MUST REMEMBER - What did Word2Vec add beyond one-hot identity?

One-hot vectors only identify vocabulary entries. Word2Vec learns geometry through a proxy prediction task such as:

```text
CBOW:      context -> centre word
Skip-gram: centre word -> context
```

Tokens that behave similarly in linguistic contexts can acquire similar vectors.

**Memory liner:**

> Predict context to learn meaning.

---

## Q8. MUST REMEMBER - Why are Word2Vec embeddings called static?

A vocabulary item has one learned vector regardless of sentence context:

```text
river bank  -> same initial bank vector
bank loan   -> same initial bank vector
```

The vector does not itself resolve the sense intended in the current sentence.

**Memory liner:**

> Static embeddings encode the token type; contextual models encode the token in this sentence.

---

## Q9. MUST REMEMBER - What did the RNN add, and what did it fail to solve?

An RNN maintains an evolving sequence state:

$$
h_t = f(x_t, h_{t-1}).
$$

It adds order and context, but information and gradients must pass through every intermediate state. This creates:

- long-range dependency and vanishing/exploding-gradient problems;
- a sequential computation path that is difficult to parallelize.

**Memory liner:**

> RNNs carry context step by step; the long path is both their mechanism and their bottleneck.

---

## Q10. MUST REMEMBER - What conceptual leap does self-attention make?

A token can directly retrieve information from any relevant token rather than waiting for information to pass recurrently through every position.

```text
RNN:       token i -> state i+1 -> ... -> state j
Attention: token i ---------------------> token j
```

**Memory liner:**

> Attention creates a direct information path between tokens.

---

## Q11. MUST REMEMBER - What are Q, K, and V?

- **Query:** what information is this position seeking?
- **Key:** how should each candidate position be matched?
- **Value:** what content should be returned if that position is relevant?

They are learned projections:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

**Memory liner:**

> Q asks, K matches, V supplies.

---

## Q12. MUST REMEMBER - State and derive scaled dot-product attention.

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}\right)V
}
$$

Multi-head shapes:

```text
Q, K, V:          [B, H, T, Dh]
K transpose:      [B, H, Dh, T]
QK^T scores:      [B, H, T, T]
attention weights:[B, H, T, T]
head output:      [B, H, T, Dh]
concatenated:     [B, T, D]
```

The two key multiplications are:

$$
[B,H,T,D_h]\,[B,H,D_h,T]\rightarrow[B,H,T,T],
$$

$$
[B,H,T,T]\,[B,H,T,D_h]\rightarrow[B,H,T,D_h].
$$

**Memory liner:**

> Compare every query with every key, normalize, then retrieve a weighted mixture of values.

---

## Q13. MUST REMEMBER - Why divide the logits by `sqrt(Dh)`?

**INTERVIEW CLARIFICATION:** If query/key components have roughly unit variance, the dot product variance grows with `Dh`. Dividing by `sqrt(Dh)` keeps logits at a scale where softmax is not immediately saturated.

**Memory liner:**

> Scale away the dimensional growth of the dot product before softmax.

---

## Q14. MUST REMEMBER - What are the original Transformer's three attention patterns?

| Attention | Q source | K/V source | Visibility |
|---|---|---|---|
| Encoder self-attention | source | source | all source positions |
| Decoder causal self-attention | target prefix | target prefix | current and previous target positions |
| Cross-attention | decoder | encoder | all encoded source positions |

Cross-attention score shape:

```text
[B, H, T_target, T_source]
```

**Memory liner:**

> Source-to-source, past-target-to-past-target, and target-to-source.

---

# 3. The modern Transformer design space

## Q15. MUST REMEMBER - Why does attention need positional information?

Without a positional mechanism, content interactions do not inherently encode which token came first or how far apart two tokens are.

**Memory liner:**

> Attention models content relationships; position modelling supplies sequence geometry.

---

## Q16. MUST REMEMBER - What is RoPE's central idea?

RoPE rotates query and key feature pairs by position-dependent angles:

$$
q'_m=R_mq_m,\qquad k'_n=R_nk_n.
$$

Then:

$$
(q'_m)^\top k'_n
= q_m^\top R_m^\top R_n k_n
= q_m^\top R_{n-m}k_n.
$$

Absolute rotations combine into a relative rotation dependent on `n-m`.

**Memory liner:**

> Rotate Q and K absolutely; obtain relative position inside their dot product.

---

## Q17. MUST REMEMBER - What tensors and shapes does RoPE preserve?

```text
Q before/after: [B, H, T, Dh]
K before/after: [B, H, T, Dh]
V:              unchanged in the lecture's formulation
```

Conceptually, adjacent features are grouped into 2D pairs:

```text
[B,H,T,Dh] -> [B,H,T,Dh/2,2] -> rotate -> [B,H,T,Dh]
```

**Memory liner:**

> RoPE changes Q/K geometry, not their shape.

---

## Q18. MUST REMEMBER - MHA vs GQA vs MQA?

```text
MHA: H query heads, H K/V heads
GQA: H query heads, G K/V heads, 1 < G < H
MQA: H query heads, 1 K/V head
```

Shapes:

```text
Q: [B,H,T,Dh]
K,V in MHA: [B,H,T,Dh]
K,V in GQA: [B,G,T,Dh]
K,V in MQA: [B,1,T,Dh]
```

**Memory liner:**

> Keep many ways to ask; share some or all of the stored memory representations.

---

## Q19. MUST REMEMBER - Why does GQA matter especially during decoding?

Autoregressive inference repeatedly reuses and stores past keys and values. Per-layer cache storage is proportional to the number of K/V heads:

```text
MHA cache: [B,H,T,Dh] for K and V
GQA cache: [B,G,T,Dh] for K and V
MQA cache: [B,1,T,Dh] for K and V
```

**Memory liner:**

> GQA reduces KV-cache memory and bandwidth without collapsing query-head diversity.

---

## Q20. MUST REMEMBER - Pre-Norm vs Post-Norm?

Post-Norm:

$$
y=\operatorname{LN}(x+F(x)).
$$

Pre-Norm:

$$
y=x+F(\operatorname{LN}(x)).
$$

The lecture highlights the historical movement from the original Post-Norm block toward Pre-Norm designs in modern LLMs.

**Memory liner:**

> Post-Norm normalizes the merged output; Pre-Norm normalizes the sublayer input and preserves a direct residual path.

---

## Q21. MUST REMEMBER - What are the three Transformer architecture families?

| Family | Architecture | Typical role revisited in the lecture |
|---|---|---|
| Encoder-only | bidirectional encoder stack | representations / classification, e.g. BERT |
| Encoder-decoder | source encoder + causal decoder + cross-attention | sequence-to-sequence, e.g. T5 |
| Decoder-only | causal decoder stack without cross-attention | modern generative LLMs, e.g. GPT-style models |

**Memory liner:**

> Encoder-only understands, encoder-decoder transforms, decoder-only continues and generates.

---

## Q22. SHOULD KNOW - Why can encoder-only models classify but not directly generate autoregressively in the lecture's framing?

Their native stack produces contextual representations of the full input. A classifier can project a pooled or special-token representation into classes. They do not contain the causal decoder process used to emit a variable-length output token by token.

**Memory liner:**

> Representation is not the same mechanism as autoregressive decoding.

---

# 4. Large language models, MoE, and decoding

## Q23. MUST REMEMBER - What is an LLM in the course's practical framing?

A very large decoder-only Transformer trained as a causal language model on enormous token corpora and compute budgets.

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

**Memory liner:**

> A modern text LLM is a scaled causal Transformer that assigns probabilities to the next token.

---

## Q24. MUST REMEMBER - What problem does sparse Mixture of Experts solve?

It increases total model capacity without activating all parameters for every token.

```text
contextual token x
    -> router scores over E experts
    -> select top K
    -> run only selected FFN experts
    -> weighted combine
```

**Memory liner:**

> More total parameters, fewer active parameters per token.

---

## Q25. MUST REMEMBER - Where is MoE usually inserted in the Transformer according to the course recap?

In place of the dense feed-forward network of selected Transformer blocks. Each expert is an FFN, and routing occurs at the token level.

Router shapes:

```text
hidden states: [B,T,D]
router logits: [B,T,E]
top-k indices: [B,T,K]
expert output: [B,T,D]
```

**Memory liner:**

> Modern Transformer MoE usually sparsifies the FFN, not the attention heads.

---

## Q26. SHOULD KNOW - What is the conceptual MoE equation?

$$
y(x)=\sum_{e\in\operatorname{TopK}(g(x))}\alpha_e(x)E_e(x).
$$

Here `g` is the router, `E_e` is expert `e`, and `alpha_e` is the normalized routing weight.

**Memory liner:**

> Route, compute selected experts, then mix their outputs.

---

## Q27. INTERVIEW CLARIFICATION - What failure appears if routing is unconstrained?

The lecture's earlier MoE material calls out **routing collapse**: a small subset of experts receives most tokens while other experts receive little training. Typical mitigations include an auxiliary load-balancing objective and noisy routing.

**Memory liner:**

> Sparse capacity is useful only if the router actually uses the capacity.

---

## Q28. MUST REMEMBER - Greedy decoding vs sampling?

Greedy decoding chooses:

$$
x_t=\arg\max_i p_i.
$$

Sampling draws:

$$
x_t\sim\operatorname{Categorical}(p).
$$

Greedy output is deterministic under deterministic computation; sampling permits diverse continuations.

**Memory liner:**

> Greedy takes the mode; sampling draws from the distribution.

---

## Q29. MUST REMEMBER - What does temperature do?

Given logits `z`:

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

```text
tau < 1 -> sharper, more deterministic
tau = 1 -> original softmax
tau > 1 -> flatter, more diverse
```

**Memory liner:**

> Temperature changes entropy, not the learned logits themselves.

---

# 5. Scaling, training efficiency, and the three-stage training pipeline

## Q30. MUST REMEMBER - What do scaling laws say at a high level?

Over the ranges studied, language-model loss improves predictably as model size, dataset size, and compute increase. Under a fixed compute budget, there is a trade-off between using more parameters and training on more tokens.

**Memory liner:**

> Scale helps, but compute must be allocated between model capacity and data exposure.

---

## Q31. MUST REMEMBER - What is the lecture's Chinchilla-style token rule of thumb?

$$
N_{tokens}\approx 20N_{params}.
$$

Example:

```text
100B parameters -> roughly 2T training tokens
```

The lecture presents this as a useful order-of-magnitude rule, not a universal law for every architecture and data mixture.

**Memory liner:**

> Many early large models were parameter-rich but data-poor.

---

## Q32. INTERVIEW CLARIFICATION - What rough compute equation is commonly associated with dense Transformer pre-training?

$$
C\approx 6N_{params}N_{tokens}
$$

for training FLOPs under common simplifying assumptions.

Do not treat the constant as architecture-independent; MoE activation sparsity, sequence details, and implementation change the exact cost.

**Memory liner:**

> Pre-training compute is roughly proportional to parameters times training tokens.

---

## Q33. MUST REMEMBER - What is FlashAttention?

FlashAttention computes the same exact softmax attention while tiling work to reduce movement between large slow HBM and small fast on-chip SRAM. It avoids materializing the full attention matrix in HBM and may recompute intermediates in backward rather than store them.

**Memory liner:**

> FlashAttention is an exact IO-aware implementation, not an approximate attention rule.

---

## Q34. MUST REMEMBER - Why can recomputation be faster than storage?

Modern accelerators can be limited by memory traffic rather than arithmetic. Recomputing a cheap intermediate can cost less wall-clock time than writing it to and reading it from slow memory.

**Memory liner:**

> More FLOPs can still mean less time when they replace expensive memory movement.

---

## Q35. MUST REMEMBER - Data parallelism vs model parallelism?

```text
Data parallelism:
    split the batch across devices
    replicate or shard model state
    synchronize gradients

Model parallelism:
    split one model computation across devices
    examples: tensor, pipeline, expert parallelism
```

**Memory liner:**

> Data parallelism divides examples; model parallelism divides the network computation.

---

## Q36. MUST REMEMBER - What are the three major training stages revisited in Lecture 9?

```text
Pre-training
    huge raw corpus
    causal next-token prediction
    learns language/code structure and knowledge

Supervised fine-tuning / instruction tuning
    smaller, high-quality input-output data
    teaches useful response behavior

Preference tuning
    preferred vs rejected behavior
    injects positive and negative comparative signal
    aligns usefulness, style, friendliness, safety, etc.
```

**Memory liner:**

> Pre-training learns what text looks like; SFT teaches what to do; preference tuning teaches what to prefer and avoid.

---

## Q37. MUST REMEMBER - Why is a pre-trained model not automatically a helpful assistant?

Its objective is to continue observed text, not necessarily to interpret a user's query as an instruction and give a direct, safe, useful response.

**Memory liner:**

> A good autocompleter is not yet an aligned assistant.

---

## Q38. SHOULD KNOW - Why is preference data often easier to collect than ideal SFT outputs?

Humans generally find it easier to compare two candidate responses than to write the globally best response from scratch.

```text
hard: write the perfect answer
 easier: choose A or B
```

**Memory liner:**

> Comparison is cheaper than creation.

---

# 6. Reward models and RL-based alignment

## Q39. MUST REMEMBER - What is the Bradley-Terry preference model?

For prompt `x`, preferred completion `y_w`, and rejected completion `y_l`:

$$
P(y_w\succ y_l\mid x)
=
\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

Reward-model loss:

$$
\boxed{
\mathcal L_{RM}
=-\log\sigma(r_w-r_l)
}
$$

**Memory liner:**

> Make the preferred completion's scalar score exceed the rejected completion's score.

---

## Q40. MUST REMEMBER - Why is reward-model training pairwise but reward inference pointwise?

Training needs two outputs to establish an ordering. Once learned, the model maps one prompt-completion pair to one scalar:

```text
training:
(x, y_w), (x, y_l) -> r_w, r_l -> pairwise loss

inference:
(x, y) -> scalar reward r
```

**Memory liner:**

> Learn a scoring function from comparisons; use the learned scorer on one answer at a time.

---

## Q41. MUST REMEMBER - What is the high-level RLHF loop?

```text
prompt x
    -> current policy samples completion y
    -> reward model scores (x,y)
    -> policy update increases expected reward
    -> regularization keeps policy near a trusted reference
```

**Memory liner:**

> Generate, score, update, constrain.

---

## Q42. MUST REMEMBER - Why keep the policy close to the SFT/reference model?

The reward model is an imperfect proxy. If optimized without constraint, the policy can exploit reward-model errors rather than become genuinely better. The SFT model is already linguistically and behaviorally competent, so the reference acts as a regularizer.

A common high-level objective is:

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta(\cdot|x)}[r(x,y)]
-
\beta D_{KL}(\pi_\theta\|\pi_{ref}).
$$

**Memory liner:**

> Improve the reward without abandoning the useful distribution you started from.

---

## Q43. MUST REMEMBER - What is reward hacking?

The policy discovers a behavior that receives high proxy reward but does not satisfy the real human goal.

```text
proxy reward rises
true quality stalls or falls
```

**Memory liner:**

> Optimizing an imperfect evaluator can expose its loopholes.

---

## Q44. SHOULD KNOW - Why also limit changes from the previous RL iteration?

Large policy updates can destabilize on-policy training. PPO-style probability-ratio clipping limits how far the current policy moves from the rollout policy in one update.

**Memory liner:**

> The reference constrains destination drift; the old policy constrains step size.

---

## Q45. INTERVIEW CLARIFICATION - What is the PPO clipped policy objective?

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)},
$$

$$
L^{clip}
=
\mathbb E_t\left[
\min\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
$$

- `A_t > 0`: increase the action probability, but not without bound.
- `A_t < 0`: decrease it, but not without bound.

**Memory liner:**

> Reinforce good sampled actions and suppress bad ones without taking an excessively large policy step.

---

# 7. Reasoning models and GRPO

## Q46. MUST REMEMBER - What changes when an assistant becomes a reasoning model in the lecture's framing?

```text
ordinary assistant:
    prompt -> final answer

reasoning model:
    prompt -> reasoning trajectory -> final answer
```

The extra generated trajectory gives the model more sequential computation and supports problem decomposition.

**Memory liner:**

> Reasoning models spend generated tokens on intermediate computation before committing to the answer.

---

## Q47. MUST REMEMBER - Why are math and coding attractive domains for reasoning RL?

They provide **verifiable rewards**:

```text
math -> parse final answer and compare with ground truth
code -> compile / execute tests
```

A learned human-preference reward model is not required to determine correctness for these cases.

**Memory liner:**

> When correctness is executable or exactly checkable, the environment itself can provide the reward.

---

## Q48. MUST REMEMBER - What does a value model do in PPO-style RLHF?

It estimates expected future return from a partial generated state under the policy. The reward and value estimate are combined to form an advantage: how much better or worse the sampled action/trajectory was than a baseline expectation.

**Memory liner:**

> Reward says what happened; value estimates what was expected; advantage measures the surprise relative to expectation.

---

## Q49. MUST REMEMBER - What is GRPO's central simplification?

For each prompt, sample a group of completions and compare each completion's reward with the rewards of the same-prompt group. This removes the separately trained value model.

```text
one prompt x
    -> G_rollout completions
    -> G_rollout rewards
    -> group-relative advantages
    -> policy update
```

**Memory liner:**

> Replace a learned critic with a same-prompt reward baseline formed from multiple rollouts.

---

## Q50. MUST REMEMBER - What is the standard group-relative advantage used in the lecture?

$$
\mu_r=\frac{1}{G}\sum_{i=1}^{G}r_i,
\qquad
\sigma_r=\sqrt{\frac{1}{G}\sum_i(r_i-\mu_r)^2+\varepsilon},
$$

$$
\boxed{A_i=\frac{r_i-\mu_r}{\sigma_r}}
$$

Shapes:

```text
rewards:    [B,G]
mean/std:   [B,1]
advantages: [B,G]
```

**Memory liner:**

> Score each completion relative to its siblings for the same prompt.

---

## Q51. MUST REMEMBER - PPO vs GRPO?

| | PPO-style RLHF | GRPO-style reasoning RL |
|---|---|---|
| Completions per prompt | often one or a rollout batch | explicit same-prompt group |
| Baseline | learned value model / GAE | group reward statistics |
| Trainable models | policy + value | policy only |
| Reward in lecture use | learned preference reward | often exact/verifiable reward |
| Main course association | preference alignment | reasoning training |

**Memory liner:**

> PPO learns a critic; GRPO uses the rollout group as the baseline.

---

## Q52. MUST REMEMBER - What models are required in verifiable-reward GRPO?

```text
trainable policy
frozen reference policy
verifier / deterministic reward code
```

No learned reward model is necessary when correctness is directly verifiable; no value model is necessary in GRPO.

**Memory liner:**

> Policy + reference + verifier.

---

## Q53. MUST REMEMBER - What length bias in the original GRPO formulation does Lecture 9 recap?

Per-response averaging by response length makes the contribution of each token depend on the length of its completion. For negative-advantage samples, short bad answers can be penalized more strongly per token than long bad answers, unintentionally making long incorrect answers comparatively less costly.

**Memory liner:**

> Per-response length normalization can make verbosity a way to dilute negative credit.

---

## Q54. SHOULD KNOW - How do Dr.GRPO and DAPO-style changes address this lecture-level issue?

The lecture recalls two directions:

- remove the problematic per-response length normalization;
- use a normalization shared across tokens/completions rather than one tied to each response's own length.

**LECTURE BOUNDARY:** The exact objectives and all additional modifications are not re-derived in Lecture 9.

**Memory liner:**

> Equalize token credit so completion length does not distort reward attribution.

---

# 8. RAG, tools, and agents

## Q55. MUST REMEMBER - Why is RAG needed even for a large-context LLM?

A model's parametric knowledge is bounded by its training data and cutoff. Putting an entire changing knowledge base into every prompt is expensive, context-limited, and vulnerable to distractors. RAG tries to retrieve only the relevant evidence.

**Memory liner:**

> Do not make the model search the whole haystack at generation time; retrieve the likely needles first.

---

## Q56. MUST REMEMBER - What are the three letters in RAG?

```text
Retrieve relevant evidence
Augment the prompt with it
Generate the answer
```

**Memory liner:**

> Retrieve, augment, generate.

---

## Q57. MUST REMEMBER - Candidate retrieval vs reranking?

```text
Candidate retrieval / bi-encoder:
    encode query independently
    compare against precomputed document embeddings
    cheap; operate over huge corpus; prioritize recall

Reranking / cross-encoder:
    jointly encode query + candidate
    model their token-level interactions
    expensive; operate over a short candidate list; improve ordering
```

**Memory liner:**

> Retrieve broadly with separable embeddings; judge precisely with joint interaction.

---

## Q58. MUST REMEMBER - What are the dense-retrieval tensor shapes?

```text
query embedding:        [B,D]
document embeddings:   [N_docs,D]
similarity scores:      [B,N_docs]
top-K document indices: [B,K]
```

For normalized embeddings:

$$
\operatorname{cos}(q,d)=q^\top d.
$$

**Memory liner:**

> Retrieval is a matrix similarity problem after the corpus embeddings are precomputed.

---

## Q59. MUST REMEMBER - What is tool calling?

The LLM does not execute the backend function internally. It predicts a structured invocation:

```text
1. choose tool name and arguments
2. application runtime executes the function
3. tool result is inserted into the conversation
4. LLM interprets it and produces the final response
```

**Memory liner:**

> The model proposes the call; ordinary software executes it.

---

## Q60. MUST REMEMBER - What distinguishes an agent from a single tool call?

An agent iteratively evaluates progress toward a goal and may plan and invoke multiple tools:

```text
observe -> plan / reason -> act -> observe -> ... -> finish
```

**Memory liner:**

> A tool call is one action; an agent is a controlled loop of reasoning, actions, observations, and termination.

---

## Q61. SHOULD KNOW - Why do agent errors compound?

If each step succeeds with probability `p`, then under an independent simplification a trajectory of `S` required steps succeeds with probability:

$$
p^S.
$$

Even a strong per-step model can become unreliable on a long horizon.

**Memory liner:**

> Long-horizon autonomy multiplies small local failure probabilities.

---

# 9. Evaluating LLMs and agentic systems

## Q62. MUST REMEMBER - Why are lexical metrics such as BLEU, ROUGE, and METEOR insufficient for general LLM evaluation?

Semantically correct answers can use very different wording. Reference-overlap metrics are repeatable but often penalize paraphrases and correlate imperfectly with human judgment on open-ended tasks.

**Memory liner:**

> String overlap is not semantic equivalence.

---

## Q63. MUST REMEMBER - What is LLM-as-a-judge?

A judge model receives:

```text
original prompt
candidate response, or response A and B
explicit evaluation criterion / rubric
```

and returns:

```text
analysis / rationale
score, pass-fail verdict, or pairwise preference
```

**Memory liner:**

> Use a language model as a scalable semantic evaluator, not as an unquestioned oracle.

---

## Q64. MUST REMEMBER - What judge biases does Lecture 9 recap?

- **Position bias:** preference changes when A/B ordering changes.
- **Verbosity bias:** longer answers are preferred because they are longer.
- **Self-enhancement bias:** a model tends to favor outputs resembling or produced by itself.

**Memory liner:**

> A learned evaluator carries model-specific preferences and presentation biases.

---

## Q65. SHOULD KNOW - How do you mitigate these judge biases?

```text
position bias     -> swap A/B and check consistency
verbosity bias    -> explicit rubric + concise reference examples
self-enhancement  -> use a different / stronger judge and human calibration
all judge tasks   -> low temperature, structured output, versioned prompt
```

**Memory liner:**

> Treat the judge like another fallible model that needs evaluation.

---

## Q66. MUST REMEMBER - Why are benchmarks a capability profile rather than one universal ranking?

Different benchmarks target different slices:

```text
knowledge
math / reasoning
coding
safety
agent / tool success
```

A model can dominate one dimension and lose on another. Cost, latency, and safety may also matter more than a small average-quality gain.

**Memory liner:**

> A benchmark suite describes a shape of capability, not a single essence of intelligence.

---

## Q67. MUST REMEMBER - What is Goodhart's-law warning for benchmarks?

> When a measure becomes a target, it can stop being a good measure.

Training data, prompts, and model behavior can overfit public benchmarks without proportionally improving real user value.

**Memory liner:**

> Optimize the task, not the scoreboard proxy.

---

# 10. Trend 1 - Vision Transformers

## Q68. MUST REMEMBER - Why can a Transformer process images at all?

Self-attention operates on vectors, not intrinsically on words. If image regions are converted into vectors, they can be treated as a token sequence.

**Memory liner:**

> The Transformer only requires a sequence of vectors; the vectors need not represent text.

---

## Q69. MUST REMEMBER - What is the ViT pipeline?

```text
image [B,C_img,H_img,W_img]
    -> split into non-overlapping P x P patches
    -> flatten each patch
    -> linear patch projection to D
    -> prepend learned CLS token
    -> add position embeddings
    -> Transformer encoder
    -> take encoded CLS representation
    -> classification head
```

**Memory liner:**

> Turn patches into tokens, contextualize them, classify from a global token.

---

## Q70. MUST REMEMBER - Derive the number and size of image patches.

Assuming dimensions are divisible by patch size `P`:

$$
N_{img}=\frac{H_{img}}{P}\frac{W_{img}}{P}.
$$

Each flattened RGB patch has dimension:

$$
D_{patch}=P^2C_{img}.
$$

Example:

```text
image: 224 x 224 x 3
patch: 16 x 16
N_img = 14 x 14 = 196
D_patch = 16 x 16 x 3 = 768
```

**Memory liner:**

> Patch count comes from the spatial grid; patch width comes from pixels times channels.

---

## Q71. MUST REMEMBER - What are the ViT tensor shapes?

```text
input image:          [B,C_img,H_img,W_img]
patchified pixels:    [B,N_img,P*P*C_img]
patch projection W:   [P*P*C_img,D]
patch tokens:         [B,N_img,D]
CLS + patch tokens:   [B,N_img+1,D]
encoder output:       [B,N_img+1,D]
CLS representation:   [B,D]
class logits:         [B,N_classes]
attention map:        [B,H,N_img+1,N_img+1]
```

**Memory liner:**

> After patch projection, the image looks to the encoder like an ordinary token sequence.

---

## Q72. MUST REMEMBER - Why add image-position embeddings?

A set of patch vectors does not by itself say which patch came from the top-left, center, or bottom-right. Position information restores the 2D arrangement after the image has been serialized into a sequence.

**Memory liner:**

> Patch content says what is present; patch position says where it was present.

---

## Q73. MUST REMEMBER - What does the ViT CLS token do?

It is a learned token that participates in every encoder layer. Its final contextual representation is used as a global image summary for classification.

**Memory liner:**

> The CLS token learns to gather the image information useful for the downstream label.

---

## Q74. MUST REMEMBER - CNN inductive bias vs ViT inductive bias?

```text
CNN:
    strong locality and translation-related architectural bias
    parameter sharing through convolution
    often data efficient

ViT:
    weak built-in spatial locality
    global patch interactions through attention
    can learn powerful relationships given enough data
```

The lecture's key observation is that lower architectural bias can work extremely well when training data is sufficiently large.

**Memory liner:**

> CNNs encode how to look; ViTs ask data to teach the model how to look.

---

## Q75. SHOULD KNOW - What does it mean that ViT has lower inductive bias?

It does not hard-code convolutional locality and translation structure to the same extent. This gives flexibility but generally increases the burden on data and optimization.

**Memory liner:**

> Fewer built-in assumptions mean more behavior must be learned from examples.

---

# 11. Trend 2 - Vision-language models

## Q76. MUST REMEMBER - What additional problem must a VLM solve beyond ViT?

It must map visual features into a representation space that a language model can consume and then combine visual and textual context while generating text.

```text
image -> vision tokens
text  -> text tokens
              ↓
       multimodal fusion
              ↓
     autoregressive response
```

**Memory liner:**

> A VLM needs visual tokenization, modality alignment, fusion, and language generation.

---

## Q77. MUST REMEMBER - What is the common early-fusion / concatenation architecture described in the lecture?

```text
image
   -> vision encoder
   -> visual features [B,N_img,D_img]
   -> learned projector [D_img -> D]
   -> image tokens [B,N_img,D]

text IDs
   -> token embedding
   -> text tokens [B,T_txt,D]

concatenate image tokens and text tokens
   -> [B,N_img+T_txt,D]
   -> decoder-only LLM
   -> answer tokens
```

A model such as LLaVA is cited as an example of this broad pattern.

**Memory liner:**

> Project visual features into the LLM width and let the causal model read them as prefix tokens.

---

## Q78. MUST REMEMBER - What are the early-fusion VLM tensor shapes?

```text
vision features:       [B,N_img,D_img]
multimodal projector:  [D_img,D]
projected image tokens:[B,N_img,D]
text tokens:           [B,T_txt,D]
combined prefix:       [B,N_img+T_txt,D]
LLM logits:            [B,N_img+T_txt+T_out,V]
```

Depending on the training mask, loss is normally applied only where output text should be predicted.

**Memory liner:**

> The projector solves width mismatch; concatenation solves interface mismatch.

---

## Q79. MUST REMEMBER - What is the cross-attention alternative?

Keep visual features as a separate memory:

```text
text hidden states -> Q
image features     -> K,V
```

Cross-attention score shape:

```text
[B,H,T_txt,N_img]
```

The lecture says this pattern exists but presents concatenating projected visual tokens with text tokens as the more common route in its examples.

**Memory liner:**

> Early fusion puts modalities in one sequence; cross-attention keeps vision as an external memory.

---

## Q80. SHOULD KNOW - Early fusion vs cross-attention trade-off?

| | Early fusion | Cross-attention |
|---|---|---|
| Interface | one unified token stream | separate image memory |
| Reuse decoder-only LLM | very direct | requires added cross-attention blocks |
| Attention cost | joint sequence can be long | explicit text-to-image interaction |
| Architectural modularity | simple unified stack | image and text pathways remain clearer |

**Memory liner:**

> Early fusion simplifies architecture; cross-attention preserves modality separation.

---

## Q81. LECTURE BOUNDARY - What VLM details are not developed here?

Lecture 9 does not derive:

- contrastive vision-language pre-training;
- image-text alignment losses;
- instruction-data construction;
- frozen vs jointly trained encoders;
- visual-token compression modules;
- hallucination-specific VLM objectives.

Keep those in separate multimodal notes.

---

# 12. Transformers beyond text and classification

## Q82. MUST REMEMBER - What broader lesson does ViT establish?

The Transformer is a general vector-interaction architecture. Once a modality is converted into tokens or latent vectors, self-attention can model relationships in that modality.

The lecture points to use in:

```text
image understanding
image generation / diffusion transformers
speech
recommendation
multimodal systems
```

**Memory liner:**

> Tokenization is the bridge that lets the same attention machinery enter a new modality.

---

## Q83. SHOULD KNOW - What is a Diffusion Transformer at the level of this lecture?

A diffusion generative model in which a Transformer replaces or contributes to the neural denoiser, allowing attention-based interaction among image or latent patches.

**LECTURE BOUNDARY:** Lecture 9 does not derive the DiT training objective or latent-diffusion pipeline.

**Memory liner:**

> DiT is architectural cross-pollination: diffusion supplies the generative process; the Transformer supplies the denoising network.

---

# 13. Trend 3 - Diffusion language models

## Q84. MUST REMEMBER - What is the inference bottleneck of an autoregressive language model?

Autoregressive generation factorizes:

$$
p(x_{1:T}\mid c)
=
\prod_{t=1}^{T}p(x_t\mid x_{<t},c).
$$

At inference, token `t+1` cannot be sampled before token `t` exists. Therefore, generating `T` new tokens requires roughly `T` sequential decoding iterations.

**Memory liner:**

> Autoregressive training parallelizes with a causal mask; autoregressive inference remains sequential across generated positions.

---

## Q85. MUST REMEMBER - Why can autoregressive training be parallel even though inference is sequential?

During training, the complete target sequence is already known. The model receives shifted tokens at all positions and a causal mask prevents each position from reading future targets.

```text
training:
known target sequence + causal mask -> all position losses in one pass

inference:
unknown future target -> generate one token, append, repeat
```

**Memory liner:**

> Teacher-forced training knows the future labels; inference must create them.

---

## Q86. MUST REMEMBER - What is the image-diffusion mental model used by the lecture?

```text
forward process:
clean image -> gradually add noise -> near-Gaussian noise

learned reverse process:
noise -> repeatedly predict/remove corruption -> clean sample
```

The lecture uses a sculpture analogy: the random initial block is progressively refined by removing what does not belong.

**Memory liner:**

> Diffusion learns iterative denoising from an easy-to-sample random state toward the data distribution.

---

## Q87. MUST REMEMBER - Why can image diffusion use Gaussian noise naturally?

Image pixels or latent features are continuous. Gaussian corruption is easy to sample, mathematically tractable, and supplies randomness for diverse generation.

**Memory liner:**

> Continuous data admits continuous noise; discrete tokens do not admit ordinary additive Gaussian corruption directly.

---

## Q88. MUST REMEMBER - What is the key obstacle when transferring diffusion from images to text?

Text is a sequence of discrete vocabulary symbols. Adding a small real-valued perturbation to a token ID does not produce a meaningful nearby token.

```text
continuous image value + small noise -> still a valid real number
integer token ID + small noise        -> not a semantic token operation
```

**Memory liner:**

> The corruption process must respect the discrete state space.

---

## Q89. MUST REMEMBER - What text analogue of noise does the lecture introduce?

The **mask token**. The forward corruption process progressively replaces more original tokens with `[MASK]`; the reverse model learns to reconstruct masked content.

```text
clean sequence
    -> partially masked sequence
    -> more masked sequence
    -> all / nearly all masked

reverse:
all masked
    -> coarse predictions
    -> repeated remasking / refinement
    -> complete sequence
```

**Memory liner:**

> Noise is to images what information removal through masking is to discrete text.

---

## Q90. INTERVIEW CLARIFICATION - What is a simple absorbing-mask corruption model?

For original token `x_i^0`, define a time-dependent mask probability `alpha_t`:

$$
q(x_i^t\mid x_i^0)
=
(1-\alpha_t)\,\delta(x_i^t=x_i^0)
+
\alpha_t\,\delta(x_i^t=[MASK]).
$$

As `t` increases, `alpha_t` increases and more positions enter the absorbing mask state.

**Memory liner:**

> A discrete diffusion schedule controls the probability that each original token has been erased by time `t`.

---

## Q91. INTERVIEW CLARIFICATION - What training loss is natural for a masked-diffusion language model?

Sample a clean sequence, corruption level, and mask pattern; predict the original token at corrupted positions:

$$
\mathcal L_{MDM}
=
-\mathbb E_{x,t,M}
\left[
\sum_{i:M_i=1}
\log p_\theta(x_i\mid x_{\setminus M},M,t,c)
\right].
$$

Shapes:

```text
clean token IDs:      [B,T]
mask:                 [B,T] bool
corrupted IDs:        [B,T]
time / noise level:   [B] or [B,1]
model logits:         [B,T,V]
loss positions:       masked positions only
```

**Memory liner:**

> Corrupt many positions, predict their clean identities, and condition the denoiser on the corruption level.

---

## Q92. MUST REMEMBER - What happens at diffusion-LM inference time?

Condition on a prompt and initialize the response region with masks. At every refinement step:

1. predict token distributions at many or all masked positions;
2. commit high-confidence predictions or sample candidates;
3. optionally remask uncertain positions;
4. repeat until the response is complete or the step budget ends.

**Memory liner:**

> Generate by iterative global refinement rather than left-to-right commitment.

---

## Q93. MUST REMEMBER - What are the masked-diffusion inference tensor shapes?

```text
prompt IDs:             [B,T_prompt]
response canvas:        [B,T_out] initially MASK
combined IDs:           [B,T_prompt+T_out]
current masked state:   [B,T_prompt+T_out]
logits each step:       [B,T_prompt+T_out,V]
confidence per position:[B,T_out]
updated response:       [B,T_out]
```

**Memory liner:**

> One model pass predicts distributions across the whole response canvas.

---

## Q94. MUST REMEMBER - Why can diffusion-LM decoding reduce latency?

Autoregressive decoding needs approximately one sequential model call per generated token. A masked-diffusion model can update many positions per refinement step, so it needs approximately `S` sequential denoising steps, where `S` may be much smaller than output length `T_out`.

```text
AR sequential depth:        about T_out
Diffusion sequential depth: about S
```

**Memory liner:**

> Parallelize across output positions; remain sequential only across refinement steps.

---

## Q95. INTERVIEW CLARIFICATION - Why is speed-up not simply `T_out / S`?

Each diffusion step processes the entire response canvas, while one cached autoregressive step processes a new token against cached K/V. Actual latency depends on:

- number of refinement steps;
- full-sequence compute per step;
- sequence length and attention pattern;
- hardware utilization and batch size;
- confidence/remasking policy;
- quality target.

**Memory liner:**

> Diffusion reduces sequential depth, but each step is wider; wall-clock speed depends on both.

---

## Q96. MUST REMEMBER - Why is masked diffusion naturally suited to fill-in-the-middle?

The model can condition on tokens on both sides of a masked region and refine the missing positions jointly. A purely left-to-right model must use a special training/inference format to exploit right context.

**Memory liner:**

> Bidirectional denoising makes arbitrary missing-region completion a native task.

---

## Q97. MUST REMEMBER - What coarse-to-fine analogy does the lecture use for diffusion text generation?

Writing a speech often begins with a rough plan and then iteratively refines sections rather than producing the final document perfectly from the first word onward.

```text
masked / rough global draft
    -> broad structure
    -> local wording
    -> final polished sequence
```

**Memory liner:**

> Diffusion writes by revising a global draft; autoregression writes by extending a prefix.

---

## Q98. MUST REMEMBER - Autoregressive LM vs masked-diffusion LM?

| | Autoregressive LM | Masked-diffusion LM |
|---|---|---|
| Factorization | left-to-right conditional product | iterative denoising / reconstruction |
| Initial response state | empty prefix | masked response canvas |
| Positions updated per step | one new position | many positions |
| Sequential depth | output length | diffusion-step count |
| Native right-context use | no during ordinary generation | yes |
| KV cache | central optimization | not the same decoding structure |
| Main lecture strength | established quality and reasoning ecosystem | speed and whole-sequence refinement |
| Main lecture challenge | sequential latency | quality gap and adapting AR techniques |

**Memory liner:**

> Autoregression extends; diffusion revises.

---

## Q99. SHOULD KNOW - What terminology does the lecture mention?

- **ARM:** autoregressive model.
- **MDM:** masked diffusion model.
- **dLLM / diffusion LLM:** diffusion-based language model.
- **LLaDA:** a cited masked-diffusion language-model paper that develops the mathematics more fully.

The lecture notes that naming is still evolving.

---

## Q100. MUST REMEMBER - What open problems for diffusion LMs does the lecture emphasize?

1. Closing the quality gap with strong autoregressive frontier models.
2. Adapting techniques built around autoregressive trajectories, especially explicit reasoning chains.
3. Choosing refinement schedules and step budgets that preserve speed without losing quality.

**Memory liner:**

> The parallel generation mechanism is promising, but the surrounding training and reasoning ecosystem was built for autoregression.

---

# 14. Cross-modal design transfer

## Q101. MUST REMEMBER - What two-way transfer between modalities is the lecture highlighting?

```text
Text -> vision:
    Transformer / attention becomes ViT and DiT

Vision -> text:
    diffusion-style iterative generation becomes masked-diffusion LMs
```

**Memory liner:**

> Research advances can move in both directions once the underlying abstraction is recognized.

---

## Q102. SHOULD KNOW - What is the lecture-level DeepSeek-OCR observation?

The cited work suggests that a relatively small number of visual tokens can encode enough information to reconstruct large amounts of text. The lecture uses it to motivate the possibility that visual patches can act as a compact representation of text-heavy content.

**Memory liner:**

> A page image can sometimes compress textual information into fewer visual tokens than ordinary text tokenization would use.

---

## Q103. SHOULD KNOW - Why might visual tokens encode text compactly?

The lecture points to the representational density of 2D patches: layout, glyph shapes, emojis, and spatial organization can be captured jointly, whereas text tokenization serializes them into many discrete symbols.

**LECTURE BOUNDARY:** It does not derive the compression ratio, OCR objective, or architecture.

---

## Q104. MUST REMEMBER - Why does 1D RoPE need adaptation for images?

Text positions lie on one ordered axis. Image patches lie on a 2D grid with row and column coordinates. A multimodal positional mechanism should preserve relative displacement in both spatial directions.

```text
text position:  t
image position: (row, column)
```

**Memory liner:**

> A 2D modality needs a position geometry that represents two axes.

---

## Q105. INTERVIEW CLARIFICATION - What is a simple 2D RoPE mental model?

Split head features into subspaces and rotate one subspace according to row position and another according to column position:

$$
q'_{r,c}
=
R_{row}(r)\oplus R_{col}(c)\;q_{r,c},
$$

with an analogous operation for `k`. Dot products can then depend on relative row and column offsets.

```text
Q,K: [B,H,N_img,Dh]
Dh split into row-pairs and column-pairs
shape remains unchanged
```

**Memory liner:**

> Encode horizontal and vertical displacement through separate rotational feature pairs.

---

# 15. The Transformer is still a design space

## Q106. MUST REMEMBER - What is the lecture's main warning about the phrase “the Transformer architecture”?

It is not one frozen blueprint. Modern systems make empirical choices about:

```text
optimizer
normalization placement and type
attention head sharing / local-global patterns
activation function
MoE vs dense FFNs
number of layers
number of heads
FFN width
position mechanism
```

**Memory liner:**

> Transformer is a family of interacting design choices, not one immutable diagram from 2017.

---

## Q107. SHOULD KNOW - What optimizer trend does the lecture mention?

Adam has been a long-standing default, while the lecture cites newer work around **Muon** and **MuonClip** as examples of research challenging optimizer conventions.

**LECTURE BOUNDARY:** It does not derive the Muon update rule or establish that it has replaced Adam.

**Memory liner:**

> Even foundational training choices remain empirical research questions.

---

## Q108. MUST REMEMBER - What normalization evolution does the lecture recap?

```text
original Transformer: Post-Norm + LayerNorm
many modern LLMs:     Pre-Norm + often RMSNorm
```

This changes both where normalization is placed and what statistic it computes.

**Memory liner:**

> Modernization can change the location and the mathematical form of normalization.

---

## Q109. SHOULD KNOW - Why is attention design not settled?

Different models mix:

- MHA, GQA, or MQA;
- local/sliding-window and global attention;
- different layerwise patterns;
- different positional mechanisms.

The right configuration depends on quality, memory, latency, context, and hardware.

**Memory liner:**

> Attention architecture is a multi-objective systems trade-off.

---

## Q110. SHOULD KNOW - Why do activation functions matter inside an LLM?

The FFN is a large fraction of parameters and compute. Its nonlinearity influences optimization and expressivity. The lecture notes the shift from plain ReLU toward alternatives such as GELU and continuing experimentation.

**Memory liner:**

> A small formula inside every FFN is multiplied across every token and layer.

---

## Q111. MUST REMEMBER - Why must architecture comparisons control the training recipe?

A better score can come from architecture, data, token budget, optimizer, regularization, or compute. Without controlled ablations, one cannot attribute the gain to the claimed component.

**Memory liner:**

> Compare one design decision at a time, or admit that the result is a system-level comparison.

---

# 16. Data quality, mid-training, and model collapse

## Q112. MUST REMEMBER - What data shift does the lecture say has occurred on the public web?

Early models could scrape a web dominated by human-authored content. Increasingly, web data contains model-generated text, making provenance and diversity less reliable.

**Memory liner:**

> Future models may train on an internet partly generated by earlier models.

---

## Q113. MUST REMEMBER - What is model collapse at the level discussed in the lecture?

Repeatedly training on synthetic outputs can narrow the observed distribution because model-generated text is often less diverse than the original human distribution. Rare modes can disappear, and later models learn from an increasingly impoverished sample.

**Memory liner:**

> Recursive synthetic training can turn distribution tails into missing data.

---

## Q114. SHOULD KNOW - Why is “synthetic data is bad” too simplistic?

The lecture's conclusion is not that all generated data is unusable. It motivates stronger curation and high-quality training stages. Synthetic data can be useful when generated, filtered, verified, and mixed intentionally rather than recursively accepted as an uncontrolled replacement for human data.

**Memory liner:**

> The issue is uncontrolled distribution narrowing, not the mere fact that a model produced the sample.

---

## Q115. MUST REMEMBER - What is mid-training in this lecture's framing?

A stage between broad pre-training and task-specific fine-tuning:

```text
pre-training
    huge, broad, noisy corpus
        -> mid-training
           still large-scale, but more curated / relevant / high quality
               -> SFT and alignment
```

The objective may remain language modelling while the data distribution becomes more targeted.

**Memory liner:**

> Mid-training changes what the model studies before teaching it how to behave.

---

## Q116. SHOULD KNOW - Data curation vs more raw tokens?

More tokens can improve coverage, but low-quality, duplicated, contaminated, or low-diversity tokens can waste compute or damage the learned distribution. The lecture forecasts increased emphasis on data selection rather than indiscriminate scraping.

**Memory liner:**

> Once raw quantity is abundant, marginal progress moves toward data quality and composition.

---

## Q117. INTERVIEW CLARIFICATION - What should a serious data-ablation report contain?

```text
provenance and licenses
human vs synthetic fraction
deduplication rules
quality filters
language / domain mixture
contamination checks
sample weighting
token counts after every stage
controlled model-and-compute comparison
```

This list is a practical extension of the lecture's data-curation argument.

---

# 17. The quality-cost Pareto frontier and smaller models

## Q118. MUST REMEMBER - What is the lecture's predicted “second frontier” after benchmark quality improves?

Achieving nearly the same useful quality with much lower inference cost, latency, and energy.

```text
first frontier:  maximize capability
second frontier: preserve capability while minimizing serving cost
```

**Memory liner:**

> Once quality is adequate, efficiency becomes product capability.

---

## Q119. MUST REMEMBER - What is a small language model (SLM) in this context?

A deliberately smaller model intended to offer favorable latency, cost, deployment, or on-device properties while retaining enough capability for a target task.

**Memory liner:**

> The best model is often the smallest one that reliably satisfies the use case.

---

## Q120. MUST REMEMBER - Define Pareto dominance for model selection.

Model A dominates model B if A is no worse on every chosen dimension and strictly better on at least one.

A model is Pareto-optimal if no other model dominates it.

Example dimensions:

```text
quality
latency
cost
energy
safety
context length
```

**Memory liner:**

> Pareto-optimal means there is no free improvement left across the selected objectives.

---

## Q121. SHOULD KNOW - Why can a weaker benchmark model be the better production model?

It may be:

- far cheaper per query;
- faster enough to improve user experience;
- easier to host privately;
- safer or easier to constrain;
- more reliable on the narrow product distribution.

**Memory liner:**

> Product value is use-case utility divided by total system cost and risk, not leaderboard score alone.

---

# 18. Hardware-software co-design

## Q122. MUST REMEMBER - What hardware lesson does FlashAttention already teach?

Algorithm quality depends on the memory hierarchy and data movement of the accelerator, not only on arithmetic operation count.

**Memory liner:**

> The same equation can have radically different performance depending on where intermediate data moves.

---

## Q123. SHOULD KNOW - Why might general-purpose GPU primitives be suboptimal for future LLMs?

GPUs are exceptionally strong at matrix multiplication, but Transformer execution also stresses memory bandwidth, attention-specific dataflow, synchronization, and communication. Hardware tailored to these patterns may improve latency and energy efficiency.

**Memory liner:**

> Workloads eventually reshape the hardware that runs them.

---

## Q124. SHOULD KNOW - What analog-computing direction does the lecture mention?

A cited proof of concept embeds useful operations into physical signal behavior so that the result arises from the hardware dynamics rather than a long sequence of digital instructions. The lecture reports potential latency and energy benefits.

**LECTURE BOUNDARY:** The device physics, noise, precision, manufacturability, and exact architecture are not developed.

**Memory liner:**

> Instead of simulating every operation, design hardware whose physics performs the operation.

---

# 19. Product and research challenges ahead

## Q125. MUST REMEMBER - What does “democratizing agents” mean in the lecture?

Making tool-using workflows constructible and usable through natural language by people who do not know agent frameworks or programming.

**Memory liner:**

> Move agents from expert-built orchestration to ordinary user-defined workflows.

---

## Q126. SHOULD KNOW - Why are browser and OS agents a natural next step?

Many digital tasks consist of navigating interfaces, retrieving information, filling forms, and invoking applications. A tool-using model could coordinate these actions at a higher level than individual app-specific assistants.

**Memory liner:**

> The browser or operating system is a large tool environment with a user goal on top.

---

## Q127. MUST REMEMBER - What blocks reliable long-horizon agents?

```text
compounding planning errors
wrong tool selection or arguments
untrusted page / prompt injection
state-tracking failures
ambiguous success conditions
unsafe irreversible actions
latency and cost growth
```

**Memory liner:**

> Autonomy is limited less by one impressive step than by consistent correctness across every step.

---

## Q128. SHOULD KNOW - Why does the lecture mention customer service as a useful test?

It exposes dimensions that benchmark answers can miss: empathy, groundedness, policy understanding, recovery from misunderstanding, and knowing when to escalate to a human.

**Memory liner:**

> Useful assistance requires social and operational judgment, not only fluent text.

---

## Q129. MUST REMEMBER - What is the continuous-learning challenge?

Model weights are typically fixed after a costly training pipeline. RAG and tools supply changing information externally, but they do not by themselves let the model safely consolidate new experience into its parameters without regressions or contamination.

**Memory liner:**

> Retrieval updates context; continual learning updates competence—and doing the latter safely remains difficult.

---

## Q130. MUST REMEMBER - Why does the lecture question the word “hallucination”?

A next-token model is trained to produce probable continuations, not to map every statement to a verified fact database. Unsupported but plausible text is therefore partly an objective mismatch rather than an inexplicable anomaly.

**Memory liner:**

> A language model can optimize plausibility without being explicitly trained to optimize truth.

---

## Q131. SHOULD KNOW - How do RAG and tools mitigate but not eliminate factual errors?

They expose evidence or external computation, but the system can still:

- retrieve the wrong source;
- misread a source;
- ignore evidence;
- call the wrong tool;
- synthesize an unsupported conclusion;
- use stale or malicious data.

**Memory liner:**

> Grounding creates an evidence path; it does not guarantee the model follows it correctly.

---

## Q132. SHOULD KNOW - Why are personalization, interpretability, and safety separate research problems?

- **Personalization:** which user-specific preferences should persist, and with what privacy controls?
- **Interpretability:** why did the system choose this answer or action?
- **Safety:** which requests or actions should be blocked, transformed, or escalated?

Improving one does not automatically solve the others.

**Memory liner:**

> Helpful to whom, explainable in what sense, and safe under whose policy are distinct questions.

---

# 20. End-to-end dataflows to reproduce on a whiteboard

## Q133. MUST REMEMBER - Draw the full modern LLM training-and-use lifecycle.

```text
RAW CORPUS
    -> tokenizer + data curation
    -> causal pre-training
    -> base model
    -> optional mid-training on targeted high-quality corpus
    -> supervised / instruction tuning
    -> SFT assistant
    -> preference tuning or verifiable-reward RL
    -> aligned / reasoning model

DEPLOYMENT HARNESS
    user request
        -> context / memory
        -> retrieval and tool selection
        -> model reasoning / generation
        -> tool execution / verification
        -> final response
        -> evaluation logs and feedback
```

**Memory liner:**

> Train the predictor, align its behavior, then engineer the environment in which it operates.

---

## Q134. MUST REMEMBER - Draw a multimodal assistant pipeline.

```text
image [B,C,H,W]
    -> vision encoder
    -> visual tokens [B,N_img,D_img]
    -> projector [D_img -> D]

text prompt
    -> text embeddings [B,T_txt,D]

[visual tokens ; text tokens]
    -> causal multimodal LLM
    -> output text tokens
    -> optional tools / retrieval / judge
```

---

## Q135. MUST REMEMBER - Draw a masked-diffusion text generator.

```text
prompt tokens c
response canvas = [MASK, MASK, ..., MASK]

for step s = S ... 1:
    logits = denoiser(prompt, current_canvas, s)
    estimate tokens at all masked positions
    commit high-confidence positions
    remask uncertain positions if required

return completed response
```

---

## Q136. MUST REMEMBER - Draw an agentic evaluation loop.

```text
user goal
   -> route relevant tools / retrieve context
   -> model proposes action
   -> policy / safety validation
   -> execute tool
   -> structured observation
   -> verify progress and state
   -> repeat or stop
   -> final response

log at every stage:
   selected tool
   arguments
   execution result
   state transition
   verifier result
   final outcome
```

**Memory liner:**

> An agent is only debuggable when every decision and state transition is observable.

---

# 21. Minimal implementation mental models

## Q137. MUST REMEMBER - What is the minimal PyTorch-like ViT patchification flow?

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        # Kernel = stride = patch size creates non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, C, H, W]
        x = self.proj(image)          # [B, D, H/P, W/P]
        x = x.flatten(2)              # [B, D, N_img]
        x = x.transpose(1, 2)         # [B, N_img, D]
        return x
```

Then:

```python
patches = patch_embed(image)                        # [B,N,D]
cls = cls_token.expand(image.size(0), -1, -1)      # [B,1,D]
x = torch.cat([cls, patches], dim=1)               # [B,N+1,D]
x = x + position_embedding[:, : x.size(1)]         # [B,N+1,D]
x = transformer_encoder(x)                         # [B,N+1,D]
logits = classifier(x[:, 0])                        # [B,num_classes]
```

**Memory liner:**

> A strided convolution is an efficient learned patch-flatten-and-project operation.

---

## Q138. MUST REMEMBER - What is the minimal early-fusion VLM flow?

```python
# vision_features: [B, N_img, D_img]
vision_features = vision_encoder(images)

# Map visual width into the language-model width.
image_tokens = multimodal_projector(vision_features)  # [B,N_img,D]

text_tokens = token_embedding(input_ids)              # [B,T_txt,D]

# Special modality/boundary tokens are often inserted in real systems.
multimodal_tokens = torch.cat(
    [image_tokens, text_tokens],
    dim=1,
)                                                     # [B,N_img+T_txt,D]

logits = causal_llm(inputs_embeds=multimodal_tokens)  # [...,V]
```

Key implementation questions:

- Is the vision encoder frozen?
- Which positions receive language-model loss?
- How many visual tokens are retained?
- How are image boundaries and modality types marked?

**Memory liner:**

> Encode vision, project width, concatenate, and let the LLM contextualize the joint sequence.

---

## Q139. INTERVIEW CLARIFICATION - What is a minimal masked-diffusion training step?

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class MaskedBatch:
    clean_ids: torch.Tensor       # [B,T]
    corrupted_ids: torch.Tensor   # [B,T]
    corruption_mask: torch.Tensor # [B,T] bool
    noise_level: torch.Tensor     # [B]


def make_corrupted_batch(
    clean_ids: torch.Tensor,
    mask_token_id: int,
) -> MaskedBatch:
    batch_size = clean_ids.size(0)

    # One corruption level per sequence.
    noise_level = torch.rand(batch_size, device=clean_ids.device)
    corruption_mask = (
        torch.rand_like(clean_ids, dtype=torch.float32)
        < noise_level[:, None]
    )

    corrupted_ids = clean_ids.clone()
    corrupted_ids[corruption_mask] = mask_token_id

    return MaskedBatch(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        corruption_mask=corruption_mask,
        noise_level=noise_level,
    )


def masked_diffusion_loss(model, batch: MaskedBatch) -> torch.Tensor:
    logits = model(
        input_ids=batch.corrupted_ids,
        noise_level=batch.noise_level,
    )                                   # [B,T,V]

    token_loss = F.cross_entropy(
        logits.transpose(1, 2),         # [B,V,T]
        batch.clean_ids,                # [B,T]
        reduction="none",
    )                                   # [B,T]

    selected = token_loss[batch.corruption_mask]
    if selected.numel() == 0:
        return token_loss.mean() * 0.0
    return selected.mean()
```

The exact corruption schedule and objective vary by paper; this code captures the lecture's **mask-and-reconstruct** mental model.

---

## Q140. INTERVIEW CLARIFICATION - What is a minimal iterative unmasking loop?

```python
@torch.no_grad()
def iterative_unmask(
    model,
    prompt_ids: torch.Tensor,   # [B,T_prompt]
    output_length: int,
    mask_token_id: int,
    steps: int,
) -> torch.Tensor:
    batch_size = prompt_ids.size(0)
    canvas = torch.full(
        (batch_size, output_length),
        mask_token_id,
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    )

    for step in range(steps, 0, -1):
        input_ids = torch.cat([prompt_ids, canvas], dim=1)
        noise_level = torch.full(
            (batch_size,),
            step / steps,
            device=prompt_ids.device,
        )
        logits = model(input_ids=input_ids, noise_level=noise_level)
        response_logits = logits[:, -output_length:, :]  # [B,T_out,V]

        probs = response_logits.softmax(dim=-1)
        confidence, prediction = probs.max(dim=-1)       # [B,T_out]

        currently_masked = canvas.eq(mask_token_id)
        remaining = currently_masked.sum(dim=1)

        # Commit a growing number of the most confident masked positions.
        for b in range(batch_size):
            if remaining[b] == 0:
                continue
            commit_count = max(1, int(remaining[b].item() / step))
            score = confidence[b].masked_fill(~currently_masked[b], -1.0)
            positions = score.topk(commit_count).indices
            canvas[b, positions] = prediction[b, positions]

    return canvas
```

Real diffusion decoders may sample, remask low-confidence tokens, use classifier-free guidance, or apply a mathematically specified reverse transition.

**Memory liner:**

> Each step predicts globally and commits selectively.

---

## Q141. MUST REMEMBER - What is the minimal GRPO group-advantage computation?

```python
# rewards: [B,G]
mean = rewards.mean(dim=1, keepdim=True)                  # [B,1]
std = rewards.std(dim=1, keepdim=True, unbiased=False)   # [B,1]
advantages = (rewards - mean) / (std + 1e-6)             # [B,G]
```

A full implementation then broadcasts each completion-level advantage across that completion's generated-token positions and applies a clipped policy-ratio term plus reference-policy KL regularization.

**Memory liner:**

> Normalize within each prompt's rollout group, not across unrelated prompts.

---

## Q142. MUST REMEMBER - How would you experimentally compare autoregressive and diffusion LMs fairly?

Control:

```text
parameter count or active FLOPs
training-token budget
data and tokenizer
context and output length
decoding-quality target
hardware and batching
```

Measure:

```text
exact task quality / benchmark score
tokens or sequences per second
time to first useful output
end-to-end latency
energy and memory
number of model passes
quality vs diffusion steps
fill-in-the-middle quality
```

A fair study should report a **quality-latency curve**, not only the fastest configuration of one model against the highest-quality configuration of the other.

**Memory liner:**

> Compare frontiers, not cherry-picked operating points.

---

## Q143. SHOULD KNOW - What ViT ablations would isolate the source of improvement?

```text
patch size
training data scale
position mechanism
CLS pooling vs mean pooling
model width and depth
augmentation and regularization
CNN baseline at matched compute
pre-training vs training from scratch
```

**Memory liner:**

> Separate the effect of attention from the effect of scale, data, and training recipe.

---

## Q144. SHOULD KNOW - How would you test model collapse empirically?

1. Start from a high-diversity human reference corpus.
2. Train generation `M_0`.
3. Sample synthetic data from `M_0`, optionally mix with controlled human fractions.
4. Train `M_1`, then repeat for several generations.
5. Track:

```text
held-out human-data loss
rare-event recall
topic / lexical diversity
calibration
factuality
distribution-tail coverage
synthetic-detection rate
```

Compare unfiltered recursive synthetic data with curated, verified, and human-mixed variants.

**Memory liner:**

> Model collapse is a multi-generation distribution experiment, not one synthetic-data training run.

---

## Q145. MUST REMEMBER - How should a production team choose a model on a Pareto frontier?

1. Define minimum quality and safety constraints.
2. Measure on the actual product distribution.
3. Remove models that violate hard constraints.
4. Compare non-dominated models on cost, latency, reliability, and deployment needs.
5. Pick the operating point that maximizes product utility, not the public benchmark average.

**Memory liner:**

> First satisfy constraints; then optimize among feasible non-dominated choices.

---

# 22. Equations to memorize cold

## Core attention and architecture

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}\right)V
}
$$

$$
(q'_m)^\top k'_n
=
q_m^\top R_{n-m}k_n
$$

$$
y_{MoE}(x)=
\sum_{e\in\operatorname{TopK}(g(x))}
\alpha_e(x)E_e(x)
$$

## Causal language modelling and decoding

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

## Scaling and training

$$
N_{tokens}\approx20N_{params}
$$

$$
C\approx6N_{params}N_{tokens}
\qquad\text{(rough dense-model estimate)}
$$

## Preference learning and RL

$$
P(y_w\succ y_l\mid x)
=
\sigma(r_w-r_l)
$$

$$
\mathcal L_{RM}=-\log\sigma(r_w-r_l)
$$

$$
\max_\theta
\mathbb E[r(x,y)]
-
\beta D_{KL}(\pi_\theta\|\pi_{ref})
$$

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

$$
L^{clip}
=
\mathbb E_t\left[
\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\right]
$$

$$
A_i=\frac{r_i-\mu_r}{\sigma_r+\varepsilon}
$$

## Retrieval

$$
\operatorname{cos}(q,d)
=
\frac{q^\top d}{\|q\|\|d\|}
$$

## Vision Transformer

$$
N_{img}=\frac{H_{img}}{P}\frac{W_{img}}{P}
$$

$$
D_{patch}=P^2C_{img}
$$

$$
X_{patch}=\operatorname{reshape}(I)W_{patch}
$$

## Masked-diffusion text

$$
q(x_i^t\mid x_i^0)
=
(1-\alpha_t)\delta(x_i^t=x_i^0)
+
\alpha_t\delta(x_i^t=[MASK])
$$

$$
\mathcal L_{MDM}
=
-\mathbb E\sum_{i:M_i=1}
\log p_\theta(x_i\mid x_{\setminus M},M,t,c)
$$

---

# 23. Tensor shapes to memorize cold

## Text Transformer

```text
Token IDs                         [B,T]
Token embeddings                 [B,T,D]
Q/K/V after head split           [B,H,T,Dh]
Self-attention scores            [B,H,T,T]
Attention output per head        [B,H,T,Dh]
Concatenated attention output    [B,T,D]
Vocabulary logits                [B,T,V]
```

## GQA / KV memory

```text
Q                                [B,H,T,Dh]
K,V                              [B,G,T,Dh]
Per-layer cached K,V             [B,G,T_cache,Dh]
```

## MoE

```text
Hidden states                    [B,T,D]
Router logits                    [B,T,E]
Top-K expert IDs                 [B,T,K]
Selected expert outputs          [B,T,K,D]
Combined MoE output              [B,T,D]
```

## Reward / GRPO

```text
Preferred reward                 [B]
Rejected reward                  [B]
Prompt-group rewards             [B,G_rollout]
Group advantages                [B,G_rollout]
Completion token log-probs       [B,G_rollout,T_out]
```

## Dense retrieval

```text
Query embeddings                 [B,D]
Document embeddings              [N_docs,D]
Similarity matrix                [B,N_docs]
Top-K IDs                        [B,K]
Cross-encoder pair inputs        [B*K,T_pair]
Reranker scores                  [B,K]
```

## Vision Transformer

```text
Image                            [B,C_img,H_img,W_img]
Patch pixels                     [B,N_img,P*P*C_img]
Patch tokens                     [B,N_img,D]
CLS + patch sequence             [B,N_img+1,D]
ViT attention scores             [B,H,N_img+1,N_img+1]
CLS output                       [B,D]
Class logits                     [B,N_classes]
```

## Early-fusion VLM

```text
Vision features                  [B,N_img,D_img]
Projected image tokens           [B,N_img,D]
Text tokens                      [B,T_txt,D]
Combined multimodal sequence     [B,N_img+T_txt,D]
```

## Cross-attention VLM

```text
Text queries                     [B,H,T_txt,Dh]
Image keys / values              [B,H,N_img,Dh]
Cross-attention scores           [B,H,T_txt,N_img]
```

## Masked-diffusion LM

```text
Clean IDs                        [B,T]
Corruption mask                  [B,T]
Corrupted IDs                    [B,T]
Noise level                      [B] or [B,1]
Denoising logits                 [B,T,V]
Position confidence              [B,T]
```

**Single most important shape reminder:**

> Once image patches or other modality features are projected to `[B,N,D]`, the Transformer can process them using the same sequence machinery as text tokens.

---

# 24. High-value comparison table

| Question | Option A | Option B | Essential distinction |
|---|---|---|---|
| Sequence context | RNN | self-attention | recurrent compression vs direct retrieval |
| Position | absolute addition | RoPE | input position vector vs relative Q/K geometry |
| K/V heads | MHA | GQA / MQA | independent vs shared stored memories |
| Capacity | dense FFN | sparse MoE | all parameters active vs routed subset |
| Decoding | greedy | sampling | mode vs stochastic draw |
| Training stage | pre-training | SFT | learn language distribution vs desired behavior |
| Alignment data | SFT output | preference pair | imitate one answer vs compare accepted/rejected |
| RL baseline | PPO value model | GRPO group | learned critic vs same-prompt rollout statistics |
| Knowledge | parametric weights | RAG | training-time memory vs retrieved evidence |
| Action | tool call | agent | one structured action vs iterative goal pursuit |
| Evaluation | overlap metric | LLM judge | lexical reference match vs learned semantic rubric |
| Vision backbone | CNN | ViT | strong locality bias vs global learned interactions |
| Multimodal fusion | concatenated tokens | cross-attention | unified stream vs separate modality memory |
| Text generation | autoregressive | masked diffusion | prefix extension vs iterative global refinement |
| Data stage | pre-training | mid-training | broad scale vs curated large-scale specialization |
| Model choice | best benchmark | Pareto choice | one score vs quality-cost-risk trade-off |

---

# 25. Research Scientist follow-up questions

## Q146. Why might lower inductive bias help at scale but hurt in low-data regimes?

With abundant data, a flexible model can learn useful invariances rather than having them hard-coded. With little data, those same degrees of freedom can increase sample complexity and overfitting.

**Memory liner:**

> Inductive bias trades flexibility for data efficiency.

---

## Q147. Could a diffusion LM and autoregressive LM be combined?

**INTERVIEW CLARIFICATION:** Plausible hybrid designs include:

- autoregressive planning followed by diffusion refinement;
- diffusion drafting followed by AR verification;
- blockwise semi-autoregressive generation with diffusion inside a block;
- AR generation for high-confidence anchors and denoising for remaining positions.

A strong research answer should define which failure the hybrid targets and compare quality-latency frontiers.

---

## Q148. What is the hardest credit-assignment issue in reasoning RL?

A final verifier returns one completion-level result, while the model produced many intermediate tokens. It is unclear which reasoning steps caused success or failure. GRPO broadcasts a relative completion advantage, but it does not directly identify the decisive token or reasoning step.

**Memory liner:**

> Verifiable final answers solve outcome evaluation, not process-level credit assignment.

---

## Q149. Why can visual-token compression be useful but risky?

It can shorten context and preserve 2D structure, but may discard exact character identity, fine print, reading order, or layout details. Compression must be evaluated on downstream reconstruction and reasoning, not token count alone.

**Memory liner:**

> Fewer tokens help only if the compressed representation keeps the facts the task needs.

---

## Q150. What is the deepest Lecture 9 research message?

Successful ideas are often modality- and stack-agnostic abstractions:

```text
attention = learned interaction among vectors
diffusion = iterative refinement from corruption
routing = conditional computation
retrieval = external non-parametric memory
alignment = optimization against a behavioral objective
evaluation = explicit measurable proxy for desired behavior
```

Progress often comes from moving one abstraction into a new domain, then redesigning it for that domain's constraints.

**Memory liner:**

> Research taste is recognizing which abstraction transfers and which domain-specific assumption must change.

---

# 26. Rapid-fire oral revision questions

## Course synthesis

1. Give the entire CME 295 course in one dataflow.
2. Which four weaknesses motivate reasoning, RAG, tools, and evaluation?
3. Model capability vs system capability?
4. Why is a component's purpose more important than its acronym?

## Foundations

5. Word vs subword vs character tokenization?
6. Why are static embeddings insufficient?
7. What does an RNN hidden state represent?
8. Why do RNNs struggle with long-range dependencies?
9. Q, K, and V in one sentence each?
10. Derive `[B,H,T,T]` attention scores.
11. Why scale by `sqrt(Dh)`?
12. Encoder self-attention vs causal self-attention vs cross-attention?
13. Why is positional information necessary?
14. Why does RoPE expose relative displacement?
15. Pre-Norm vs Post-Norm?
16. Encoder-only vs encoder-decoder vs decoder-only?

## LLM architecture and generation

17. Dense FFN vs sparse MoE?
18. Total parameters vs active parameters?
19. Why route at the token level?
20. What is routing collapse?
21. Greedy decoding vs sampling?
22. What does temperature change?
23. MHA vs GQA vs MQA?
24. Why does GQA reduce decoding cost?

## Scaling and training

25. What do scaling laws claim?
26. What does the 20-tokens-per-parameter rule mean?
27. Why is it a rule of thumb rather than universal truth?
28. Explain FlashAttention without saying “it is faster attention.”
29. Why can recomputation outperform storage?
30. Data parallelism vs model parallelism?
31. Pre-training vs SFT vs preference tuning?
32. Why is a base LM not an assistant?
33. Why is pairwise preference collection attractive?

## Alignment and reasoning

34. Write the Bradley-Terry preference probability.
35. Pairwise reward training vs pointwise reward inference?
36. Describe the RLHF loop.
37. Why use a reference policy?
38. What is reward hacking?
39. `pi_old` vs `pi_ref`?
40. Reward vs value vs advantage?
41. Why do verifiable rewards matter?
42. PPO vs GRPO?
43. Derive group-relative advantage.
44. Why can GRPO remove the value model?
45. What is GRPO length bias?
46. How do normalization changes address it?

## RAG, tools, and evaluation

47. Why can long context not replace retrieval completely?
48. Retrieve, augment, generate?
49. Bi-encoder vs cross-encoder?
50. What should candidate retrieval optimize?
51. What should reranking optimize?
52. What exactly does the LLM do in tool calling?
53. Tool call vs agent?
54. Why do agent errors compound?
55. Why do lexical metrics fail on paraphrases?
56. What does an LLM judge receive and return?
57. Position, verbosity, and self-enhancement bias?
58. Why are benchmarks a profile rather than a scalar?
59. Explain Goodhart's law for LLM benchmarks.

## Vision Transformers and VLMs

60. Why can self-attention process images?
61. Derive `N_img` and `D_patch`.
62. Walk through ViT tensor shapes.
63. Why add a CLS token?
64. Why add image-position embeddings?
65. CNN inductive bias vs ViT inductive bias?
66. Why did data scale matter for ViT?
67. How are image tokens adapted to an LLM width?
68. Early fusion vs cross-attention?
69. What is the cross-attention score shape for text queries over image patches?
70. What does a VLM projector learn?

## Diffusion LMs

71. Why is autoregressive inference sequential?
72. Why is autoregressive training parallelizable?
73. Explain image diffusion in one minute.
74. Why is Gaussian noise awkward for discrete tokens?
75. Why use mask tokens as corruption?
76. Write a simple absorbing-mask transition.
77. What is the masked-token denoising loss?
78. Walk through iterative unmasking.
79. Why can it reduce sequential depth?
80. Why is wall-clock speed not simply `T/S`?
81. Why is diffusion natural for fill-in-the-middle?
82. Extend vs revise: which belongs to AR and diffusion?
83. What quality challenges remain for dLLMs?
84. How might reasoning trajectories be adapted to diffusion?

## Trends and research judgment

85. Give examples of text-to-vision and vision-to-text idea transfer.
86. What is the DeepSeek-OCR lesson at the lecture level?
87. Why does RoPE need a 2D adaptation?
88. Why is the Transformer still a design space?
89. What optimizer trend is mentioned, and what is not established?
90. Why must architectural ablations control the data recipe?
91. What changed about internet training data?
92. Define model collapse.
93. Why is synthetic data not automatically bad?
94. What is mid-training?
95. Why are SLMs strategically important?
96. Define Pareto dominance.
97. What does hardware-software co-design mean?
98. What blocks browser and OS agents?
99. Why is continuous learning different from RAG?
100. Why can hallucination be viewed as an objective mismatch?
101. What experiment would compare AR and diffusion fairly?
102. What experiment would test recursive model collapse?
103. How would you choose a production model from a quality-cost frontier?
104. What is the most transferable abstraction in this lecture?

---

# 27. Memory liners

```text
1. The course story is represent -> contextualize -> generate -> train -> align -> reason -> ground -> act -> evaluate.

2. Reason, retrieve, act, evaluate are four repairs to four standalone-LLM weaknesses.

3. One-hot identifies; embeddings represent; attention contextualizes.

4. RNNs carry information through a chain; attention creates direct retrieval paths.

5. Q asks, K matches, V supplies.

6. Attention = compare, normalize, retrieve.

7. RoPE rotates Q/K so absolute positions become relative interactions.

8. GQA preserves many queries while sharing stored K/V memories.

9. Pre-Norm leaves a direct residual path.

10. An LLM is a scaled causal decoder, not merely a large neural network.

11. MoE increases capacity without activating every expert for every token.

12. Temperature controls distribution entropy, not model knowledge.

13. Scaling is a resource-allocation problem between parameters, data, and compute.

14. FlashAttention optimizes memory traffic while preserving exact attention.

15. Pre-training learns the distribution; SFT teaches behavior; preference tuning teaches comparison.

16. Reward models are trained pairwise but score pointwise.

17. Improve reward without drifting away from a competent reference.

18. A proxy can be hacked; regularization does not make it perfect.

19. Reward says what happened; value says what was expected; advantage compares them.

20. GRPO replaces a learned critic with same-prompt rollout statistics.

21. Verifiable rewards remove the need to learn correctness from preferences.

22. Per-response length normalization can make verbosity dilute negative credit.

23. RAG retrieves evidence; it does not update model weights.

24. Bi-encoders retrieve broadly; cross-encoders rerank precisely.

25. The model proposes a tool call; software executes it.

26. A tool is an action; an agent is a loop with a stopping rule.

27. A judge is a scalable proxy, not an oracle.

28. Benchmarks describe capability profiles, not universal model quality.

29. A Transformer consumes vectors, so a patch can replace a word as a token.

30. ViT turns image patches into a sequence and classifies from a contextual global token.

31. CNNs hard-code locality; ViTs learn interaction structure from data.

32. A VLM projects image features into the language-model width.

33. Early fusion builds one token stream; cross-attention keeps vision as memory.

34. Autoregression extends a prefix; diffusion revises a canvas.

35. AR inference is sequential across tokens; diffusion is sequential across refinement steps.

36. Masking is a natural discrete analogue of information corruption.

37. Diffusion reduces sequential depth but each step processes a wider state.

38. Fill-in-the-middle is native when the model can condition on both sides.

39. Ideas transfer across modalities only after adapting the domain-specific representation and corruption process.

40. Transformer is a design family, not one fixed architecture.

41. Future progress may come as much from data curation as from adding parameters.

42. Model collapse is distribution-tail loss under recursive synthetic training.

43. Mid-training uses a large but more targeted corpus before behavioral fine-tuning.

44. The best production model is often the smallest reliable model that satisfies the task.

45. Pareto-optimal means no free improvement exists across the selected objectives.

46. Hardware performance depends on data movement, not only arithmetic count.

47. Long-horizon agent reliability multiplies local reliability problems.

48. RAG gives current evidence; continual learning would change lasting competence.

49. Next-token plausibility is not the same objective as factual truth.

50. Research taste is recognizing what abstraction transfers and what assumption must be redesigned.
```

---

# 28. Previous-day interview checklist

## Must derive without notes

- [ ] Scaled dot-product attention and `[B,H,T,T]` scores.
- [ ] RoPE's relative-rotation identity.
- [ ] MHA / GQA / MQA K/V shapes.
- [ ] Causal language-model factorization.
- [ ] Temperature-scaled softmax.
- [ ] Bradley-Terry reward probability and loss.
- [ ] KL-regularized reward objective.
- [ ] PPO probability ratio and clipped intuition.
- [ ] GRPO group-relative advantage.
- [ ] ViT patch count and flattened patch dimension.
- [ ] Early-fusion and cross-attention VLM shapes.
- [ ] Simple masked-diffusion corruption and denoising loss.

## Must explain clearly

- [ ] The full course in one coherent mental map.
- [ ] Static embedding -> RNN -> attention progression.
- [ ] Why GQA matters for decoding.
- [ ] Sparse MoE total vs active parameters.
- [ ] FlashAttention as IO-aware exact attention.
- [ ] Pre-training vs SFT vs preference tuning.
- [ ] Pairwise reward training vs pointwise inference.
- [ ] `pi_old` vs `pi_ref`.
- [ ] PPO vs GRPO.
- [ ] GRPO length bias.
- [ ] Candidate retrieval vs reranking.
- [ ] Tool call vs agent loop.
- [ ] LLM-judge biases.
- [ ] CNN vs ViT inductive bias.
- [ ] Early fusion vs cross-attention VLM.
- [ ] Autoregressive vs diffusion language generation.
- [ ] Why masking is used as discrete corruption.
- [ ] Model collapse and mid-training.
- [ ] Pareto model selection.
- [ ] Continuous learning vs external retrieval.

## Must draw on a whiteboard

- [ ] Multi-head attention tensor flow.
- [ ] Modern LLM training pipeline.
- [ ] RLHF and GRPO loops.
- [ ] RAG retrieve-rerank-augment-generate flow.
- [ ] Tool-calling runtime boundary.
- [ ] ViT patch-to-class pipeline.
- [ ] Early-fusion multimodal assistant.
- [ ] Masked-diffusion iterative refinement loop.
- [ ] Agent observe-plan-act-verification loop.
- [ ] Quality-cost Pareto frontier.

## Must be ready to critique

- [ ] Does a claimed architecture gain control data and compute?
- [ ] Does a dLLM speed claim compare at matched quality?
- [ ] Does visual compression preserve exact facts?
- [ ] Is synthetic data filtered and mixed intentionally?
- [ ] Is an agent evaluated by final prose or actual world state?
- [ ] Is a benchmark result contaminated or overoptimized?
- [ ] Does the chosen model satisfy product constraints, not merely lead a leaderboard?

---

# 29. Final 90-second lecture answer

> Lecture 9 connects the entire LLM stack and then asks which parts may change next. Text is tokenized and embedded; self-attention replaces recurrent information paths with direct query-key-value retrieval; the Transformer becomes encoder-only, encoder-decoder, or decoder-only, and modern LLMs add design choices such as RoPE, GQA, Pre-Norm, and sparse MoE. Training progresses from large-scale causal pre-training to supervised instruction tuning and preference alignment. Reward models use Bradley-Terry comparisons, while reasoning systems increasingly use verifiable rewards and GRPO, which replaces a learned value model with same-prompt rollout statistics. RAG supplies fresh evidence, tools supply actions, agents iterate, and evaluation uses direct verifiers, learned judges, and benchmark profiles.
>
> The new Lecture 9 material shows that these abstractions are moving across modalities. ViT turns image patches into tokens; VLMs project image features into an LLM or expose them through cross-attention; diffusion Transformers move attention into image generation. In the opposite direction, masked-diffusion LMs adapt image-style denoising to discrete text by replacing Gaussian noise with masking and iteratively refining many positions in parallel. That can reduce sequential decoding depth and naturally support fill-in-the-middle, but quality and reasoning methods still lag the mature autoregressive ecosystem. The lecture closes by emphasizing that architecture, optimizers, normalization, data curation, mid-training, hardware, smaller models, and agent reliability remain open. The durable lesson is to understand the abstraction, the failure it fixes, the tensor flow, and the quality-cost trade-off rather than treating today's recipe as final.

---

# 30. Lecture boundaries

The following topics are adjacent but **not developed in Lecture 9** and should remain separate notes or explicit research extensions:

- exact BPE, WordPiece, or Unigram tokenizer algorithms;
- full LayerNorm, RMSNorm, GQA, MoE, PPO, DPO, and GRPO derivations beyond the recap level;
- full ViT pre-training recipes and modern vision backbones;
- CLIP-style contrastive learning and detailed image-text alignment objectives;
- complete LLaVA or other VLM training stages;
- image/video diffusion equations, score matching, and latent diffusion;
- formal discrete-diffusion reverse processes and paper-specific dLLM samplers;
- detailed LLaDA, DeepSeek-OCR, Muon, MuonClip, or analog-hardware algorithms;
- state-space models and other non-Transformer sequence architectures not discussed here;
- a complete theory of model collapse or synthetic-data scaling;
- production browser-agent security architecture;
- continual-learning algorithms and safe online weight updates;
- personalization memory systems and formal interpretability methods;
- current product claims beyond their role as examples in the Autumn 2025 lecture.

**Final memory line:**

> The architecture, modality, generation process, data pipeline, hardware, and evaluation proxy are all design choices; none should be mistaken for a permanent law of intelligent systems.
