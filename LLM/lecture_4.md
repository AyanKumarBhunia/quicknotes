# CME 295 Lecture 4 - LLM Training, Scaling, SFT, LoRA, and QLoRA

> **Scope:** This note is grounded in the supplied **Lecture 4 transcript**. It adds standard equations, shape derivations, and implementation intuition only when they directly clarify topics taught in this lecture. Such additions are marked as **Interview clarification** when the exact detail was not derived in class.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, intuitive answers, memory liners, equations, tensor shapes, mechanisms, systems trade-offs, and explicit lecture boundaries.

**Source transcript:** CME 295 Lecture 4 - `https://www.youtube.com/watch/VlA_jt_3Qc4`

## Lecture spine

```text
Task-specific training
        -> transfer learning
        -> pre-training + task adaptation

PRE-TRAINING
        -> causal next-token prediction
        -> internet-scale text and code
        -> FLOPs, data/model scaling, Chinchilla
        -> cost, knowledge cutoff, memorization risk

LARGE-SCALE TRAINING SYSTEMS
        -> forward activations
        -> backward gradients
        -> optimizer states
        -> data parallelism
        -> ZeRO-1 / ZeRO-2 / ZeRO-3
        -> model parallelism
             - tensor parallelism
             - pipeline parallelism
             - expert parallelism

HARDWARE-AWARE EFFICIENCY
        -> FlashAttention
             - HBM vs SRAM
             - tiling and online softmax
             - recomputation in backward
        -> reduced precision
             - quantization
             - mixed-precision training

POST-TRAINING
        -> supervised fine-tuning (SFT)
        -> instruction tuning
        -> response-only loss masking
        -> helpfulness, safety, and data quality
        -> evaluation challenges
        -> optional mid-training
        -> preference tuning deferred to next lecture

PARAMETER-EFFICIENT FINE-TUNING
        -> LoRA
        -> QLoRA
             - frozen 4-bit base model
             - BF16 LoRA adapters
             - NF4 and double quantization
```

---

## Notation used in this note

```text
B          global batch size
b          local batch size per data-parallel worker
T          sequence length
V          vocabulary size
D          model width, d_model
Dff        Transformer FFN hidden width
H          number of attention heads
Dh         per-head dimension
L          number of Transformer layers
P          number of model parameters
Ntok       number of pre-training tokens
G          number of data-parallel workers / GPUs
r          LoRA rank, with r << min(d_in, d_out)
d_in       input width of a linear layer
d_out      output width of a linear layer
W0         frozen pre-trained weight matrix
A, B_lora  trainable low-rank LoRA matrices
M_t        binary mask selecting assistant/output tokens for SFT loss
```

### Priority legend

- **MUST REMEMBER:** answer immediately; derive the equation or shape.
- **SHOULD KNOW:** explain the mechanism and trade-off clearly.
- **LECTURE BOUNDARY:** mentioned, deferred, or not fully derived here.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture topic interview-ready.

---

# 1. Transfer learning and the LLM training pipeline

## Q1. MUST REMEMBER - What training paradigm does the lecture contrast with modern LLM training?

Traditional task-specific ML often trained a separate model from scratch for each task:

```text
spam data       -> spam model
sentiment data  -> sentiment model
intent data     -> intent model
```

Modern LLM training instead learns a reusable language model first, then adapts it:

```text
large general corpus
        -> pre-trained base model
        -> smaller task- or assistant-oriented training
```

**Memory liner:**

> Do not relearn language for every task; pre-train once, then specialize.

---

## Q2. MUST REMEMBER - What is transfer learning?

Transfer learning starts from parameters learned on one broad objective and reuses them for another task rather than initializing the new task from scratch.

```text
general learned knowledge
          +
small task-specific dataset
          -> specialized model
```

**Memory liner:**

> Transfer learning reuses a representation learned broadly and adapts it narrowly.

---

## Q3. MUST REMEMBER - What are the main stages of LLM training in this lecture?

```text
1. Pre-training
   Learn language, code, and broad knowledge through next-token prediction.

2. Optional mid-training
   Continue the same language-model objective on a more targeted data mixture.

3. Supervised fine-tuning / instruction tuning
   Learn how to respond helpfully to user instructions.

4. Preference tuning
   Further align behavior with preferences; deferred to the next lecture.
```

The lecture calls the post-pre-training behavioral stages part of **alignment**.

**Memory liner:**

> Pre-training builds capability; post-training shapes behavior.

---

## Q4. MUST REMEMBER - Pre-training versus fine-tuning?

| Property | Pre-training | Fine-tuning / SFT |
|---|---|---|
| Starting point | Random or newly initialized base model | Pre-trained parameters |
| Data | Very large, broad, mostly raw text/code | Smaller, curated input-output pairs |
| Main goal | Learn language and broad regularities | Adapt behavior to tasks/instructions |
| Typical loss coverage | Nearly every valid next-token position | Usually assistant/output positions only |
| Cost | Dominant training cost | Much cheaper, though still nontrivial |

**Memory liner:**

> Pre-training learns what text looks like; SFT learns what a useful answer looks like.

---

## Q5. MUST REMEMBER - Why is a next-token-pre-trained base model not automatically a helpful assistant?

The objective only asks:


a) Given this prefix, what token is likely next?

It does not directly ask:

b) What response is maximally helpful, safe, truthful, or instruction-following?

A base model may continue the apparent document pattern, ask another question, or imitate unrelated web text instead of directly helping the user.

**Memory liner:**

> Likely continuation is not the same objective as helpful assistance.

---

## Q6. SHOULD KNOW - What is mid-training in the lecture?

Mid-training is an optional stage after broad pre-training but before SFT:

```text
broad base-model checkpoint
        -> continue causal LM training
           on a targeted domain/task data mixture
        -> domain-strengthened checkpoint
```

The objective remains next-token prediction; the data distribution becomes more relevant to the desired downstream capabilities.

**Memory liner:**

> Mid-training changes the data mixture more than the objective.

---

## Q7. MUST REMEMBER - What does the lecture call alignment?

At the lecture's level:

```text
alignment = supervised fine-tuning + preference tuning
```

The purpose is to make model behavior better match human/product goals after broad pre-training.

**Memory liner:**

> Alignment turns a capable base model into a model that behaves as intended.

---

## Q8. LECTURE BOUNDARY - What is preference tuning?

The lecture names preference tuning as a later post-training stage but explicitly defers its mechanism to the next lecture.

Do not add PPO, reward models, DPO, or RLHF derivations to this Lecture 4 note as if they were covered here.

---

# 2. Causal pre-training objective and data

## Q9. MUST REMEMBER - What probability does a causal language model learn?

For a token sequence `x_1, ..., x_T`:

$$
\boxed{
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
}
$$

At each position, the model predicts the next token from the prefix.

**Memory liner:**

> A language model factorizes a sequence into one conditional next-token prediction per position.

---

## Q10. MUST REMEMBER - What is one pre-training sample after shifting?

For:

```text
[BOS, a, cute, teddy, bear, EOS]
```

construct:

```text
input:  [BOS, a,    cute,  teddy, bear]
target: [a,   cute, teddy, bear,  EOS]
```

Every target is the input shifted one position to the left.

**Memory liner:**

> Input is the prefix; label is the same sequence shifted by one token.

---

## Q11. MUST REMEMBER - What are the core pre-training tensor shapes?

```text
input_ids:  [B, T]
labels:     [B, T]
hidden:     [B, T, D]
logits:     [B, T, V]
```

At each of the `B x T` positions, the model emits one vocabulary distribution of size `V`.

---

## Q12. MUST REMEMBER - What is the causal pre-training loss?

$$
\boxed{
\mathcal{L}_{PT}
=
-\frac{1}{\sum_{b,t}m_{b,t}}
\sum_{b=1}^{B}\sum_{t=1}^{T}
 m_{b,t}
 \log p_\theta(x_{b,t+1}\mid x_{b,\le t})
}
$$

Here `m_{b,t}` ignores padding or otherwise invalid positions.

Equivalent implementation view:

```text
logits:  [B, T, V] -> flatten -> [B*T, V]
labels:  [B, T]    -> flatten -> [B*T]
loss: cross entropy over valid positions
```

**Memory liner:**

> Pre-training is token-level cross-entropy over the shifted sequence.

---

## Q13. MUST REMEMBER - If generation is sequential, how can pre-training process all positions in parallel?

During training, the complete ground-truth sequence is available. A causal mask prevents position `t` from seeing future tokens, while all positions are evaluated in one forward pass.

```text
TRAINING
full shifted sequence known
        -> causal mask
        -> losses for all positions in parallel

INFERENCE
future tokens unknown
        -> generate one new token at a time
```

**Memory liner:**

> Causal masking preserves the autoregressive objective without forcing sequential training.

---

## Q14. MUST REMEMBER - What kinds of data are used for pre-training in the lecture?

The lecture describes broad mixtures containing:

- Web pages and Common Crawl-style data.
- Encyclopedic material such as Wikipedia.
- Discussions and social content.
- Code repositories and programming forums.
- Multiple human languages and programming languages.

The purpose is to expose the model to broad language and code structure.

**Memory liner:**

> Pre-training data is broad enough that next-token prediction requires general linguistic and domain knowledge.

---

## Q15. MUST REMEMBER - Why is pre-training data size measured in tokens?

Tokens are the actual discrete units processed by the model. Document or byte counts do not directly reveal how many optimization targets the tokenizer creates.

```text
raw documents
    -> tokenize
    -> Ntok training tokens
    -> approximately Ntok next-token targets
```

**Memory liner:**

> Tokens connect data volume directly to the number of model training positions.

---

## Q16. SHOULD KNOW - Why does the pre-training data mixture matter, not only its size?

The data mixture determines which languages, domains, styles, skills, and risks are represented. Two models trained on the same number of tokens can learn different capabilities if their data mixtures differ.

```text
same token count
    + different data distribution
    -> different learned model
```

**Memory liner:**

> Token count controls scale; data mixture controls what that scale teaches.

---

## Q17. MUST REMEMBER - Why is next-token prediction a useful general proxy objective?

To predict the next token across diverse text, the model must learn regularities such as:

- Syntax and local grammar.
- Semantic compatibility.
- Long-range dependencies.
- Facts and patterns present in the data.
- Code structure and conventions.

But the objective does not guarantee truthfulness or helpfulness; it only rewards predictive likelihood.

**Memory liner:**

> Next-token prediction can induce broad capability, but its direct target remains likelihood.

---

## Q18. MUST REMEMBER - What is the knowledge cutoff?

A base model can only absorb information present in the data available before the pre-training corpus was finalized.

```text
training data ends at date d
        -> weights encode data up to roughly d
        -> later events are not learned automatically
```

**Memory liner:**

> A frozen checkpoint cannot learn events that occurred after its training data cutoff.

---

## Q19. SHOULD KNOW - Why is directly editing or injecting new knowledge into weights difficult?

A weight participates in many behaviors. Updating it to insert one fact can unintentionally alter unrelated capabilities or overwrite previous knowledge.

```text
local desired edit
        -> distributed parameter change
        -> possible regressions elsewhere
```

**Memory liner:**

> Model knowledge is distributed, so a precise factual edit need not stay local.

---

## Q20. MUST REMEMBER - What is the memorization or plagiarism risk?

The model may reproduce text or code fragments encountered in training rather than synthesizing a genuinely new response. The lecture raises this as a risk of internet-scale next-token training.

Important distinction:

```text
generalization: use learned patterns on a new example
memorization:   reproduce specific training content
```

**Memory liner:**

> Good likelihood can come from learned structure or from recalling a training sequence.

---

# 3. FLOPs, scaling laws, and compute-optimal training

## Q21. MUST REMEMBER - FLOP count versus FLOP/s?

- A **FLOP** is one floating-point operation.
- A total **FLOP count** estimates how much numerical work a training run performs.
- **FLOP/s** measures hardware throughput: how many floating-point operations are executed per second.

The literature often uses capitalization inconsistently, so infer the intended meaning from context.

**Memory liner:**

> FLOP count is work; FLOP/s is speed.

---

## Q22. MUST REMEMBER - What determines pre-training compute at the lecture's level?

The lecture's key approximation is:

$$
\text{training compute}
\propto
\text{parameter count}
\times
\text{training-token count}
$$

or:

$$
C=O(PN_{tok})
$$

The exact constant depends on architecture, sequence length, sparsity, implementation, and what operations are counted.

---

## Q23. INTERVIEW CLARIFICATION - What is the common dense-Transformer compute estimate?

A widely used back-of-the-envelope estimate is:

$$
\boxed{C_{train}\approx 6PN_{tok}}
$$

where:

```text
P      = number of model parameters
Ntok   = number of training tokens
```

Mental origin:

```text
forward pass          approximately 2 P operations/token
backward pass         approximately 4 P operations/token
--------------------------------------------------------
total                  approximately 6 P operations/token
```

This is a rough dense-model estimate, not a universal law. Attention, embeddings, MoE routing, recomputation, and hardware kernels change the realized cost.

**Memory liner:**

> Six times parameters times tokens is a planning approximation, not an architectural identity.

---

## Q24. MUST REMEMBER - How does total compute relate to training time?

Ideally:

$$
\text{time}
\approx
\frac{\text{total FLOPs}}
     {\text{sustained FLOP/s}}
$$

In practice:

$$
\text{time}
\approx
\frac{C}
     {\text{peak FLOP/s}\times \text{utilization}}
$$

Utilization is reduced by memory traffic, communication, kernel overhead, idle pipeline stages, and data-loading stalls.

**Memory liner:**

> Peak hardware throughput matters only to the extent the training system can keep it busy.

---

## Q25. MUST REMEMBER - What qualitative scaling-law observation does the lecture present?

Within the tested regimes, next-token loss generally improves as one increases:

```text
model size
training data
training compute
```

The relationship tends to be smooth enough that small-scale experiments can help predict larger runs.

**Memory liner:**

> Scale usually improves loss predictably, but the allocation of scale matters.

---

## Q26. MUST REMEMBER - What does sample efficiency mean here?

A larger model is called more sample efficient if, after processing the same number of tokens, it reaches lower loss or better performance than a smaller model.

This does **not** mean it used less compute per token.

```text
same token budget
larger model -> often lower loss
but larger model -> more operations per token
```

**Memory liner:**

> Sample efficiency measures performance per token, not performance per FLOP.

---

## Q27. MUST REMEMBER - What is the compute-allocation question behind Chinchilla-style scaling?

Given a fixed compute budget, choose both:

```text
P      model parameters
Ntok   training tokens
```

A model can be inefficiently allocated by being:

- Too large and trained on too few tokens.
- Too small and trained on unnecessarily many tokens.

**Memory liner:**

> Fixed compute creates a model-size versus data-size trade-off.

---

## Q28. MUST REMEMBER - What Chinchilla rule of thumb is taught?

The lecture presents the approximate compute-optimal relationship:

$$
\boxed{N_{tok}\approx 20P}
$$

For example, a `P`-parameter dense model would receive roughly twenty training tokens per parameter under that study's regime.

**Memory liner:**

> Chinchilla's headline heuristic is roughly twenty tokens per parameter.

---

## Q29. MUST REMEMBER - What does it mean to call a model undertrained?

Relative to a compute-optimal allocation, an undertrained model has too many parameters for the number of tokens it consumed.

```text
very large P
small Ntok / P
        -> parameters have not received enough data updates
```

It may be better, at the same compute budget, to train a smaller model on more tokens.

---

## Q30. MUST REMEMBER - Is `20 tokens per parameter` universal?

No. It is a rule of thumb from a particular study and objective regime. The lecture explicitly notes that organizations often repeat smaller scaling experiments for their own:

- Architecture.
- Data mixture.
- Optimizer and schedule.
- Hardware/software stack.
- Target evaluation and inference constraints.

**Memory liner:**

> Scaling-law constants are empirical and setup-dependent.

---

## Q31. SHOULD KNOW - How are scaling laws used before a huge run?

```text
1. Train several smaller models.
2. Vary parameter count and token count.
3. Fit a relationship between scale, compute, and loss.
4. Extrapolate to choose a larger configuration.
5. Validate assumptions during the run.
```

**Memory liner:**

> Spend small experiments to reduce the risk of an extremely expensive large experiment.

---

## Q32. MUST REMEMBER - Why is simply choosing the largest possible model not compute-optimal?

At fixed compute:

$$
C\approx 6PN_{tok}
$$

Increasing `P` forces `Ntok` down. The larger model may have more capacity but insufficient data to train that capacity well.

**Memory liner:**

> Capacity without enough training tokens is unused potential.

---

## Q33. SHOULD KNOW - What costs of pre-training does the lecture emphasize?

- Very large compute and financial cost.
- Long training time.
- Energy and ecological cost.
- Data curation and legal/memorization concerns.
- Stale knowledge after the cutoff date.
- Difficulty editing knowledge without regressions.

---

## Q34. MUST REMEMBER - Give the scaling mental map in one answer.

> Increasing parameters, tokens, and compute usually improves language-model loss. Under a fixed compute budget, however, parameters and data compete. Chinchilla-style work estimates a compute-optimal balance, roughly twenty tokens per parameter in its setting, but real labs fit their own scaling relationships because the constants depend on architecture, data, and infrastructure.

---

# 4. The training loop and why memory becomes the bottleneck

## Q35. MUST REMEMBER - What are the three main steps of one optimizer iteration?

```text
1. Forward pass
   inputs -> activations -> logits -> loss

2. Backward pass
   loss -> gradients for trainable parameters

3. Optimizer step
   gradients + optimizer state -> updated parameters
```

Usually gradients are then cleared before the next iteration.

---

## Q36. MUST REMEMBER - What are activations?

Activations are intermediate values produced by layers during the forward pass.

```text
layer input/output: [B, T, D]
attention scores:   [B, H, T, T] in a naive implementation
FFN hidden:         [B, T, Dff]
```

Many activations must remain available because backpropagation uses them to compute parameter gradients.

**Memory liner:**

> Parameters define the computation; activations record what happened for this batch.

---

## Q37. MUST REMEMBER - What are gradients?

For each trainable parameter `theta_i`:

$$
g_i=\frac{\partial\mathcal{L}}{\partial\theta_i}
$$

The gradient indicates the local direction in which the loss changes. Optimizers transform these gradients into parameter updates.

Shapes:

```text
parameter tensor: [shape_i]
gradient tensor:  [shape_i]
```

---

## Q38. MUST REMEMBER - What extra state does Adam maintain?

Adam stores running estimates of:

- The first moment: moving average of gradients.
- The second moment: moving average of squared gradients.

Therefore, in addition to parameters and gradients, two parameter-sized state tensors are often stored.

**Memory liner:**

> Adam trades extra parameter-sized memory for adaptive updates.

---

## Q39. INTERVIEW CLARIFICATION - What are the Adam moment equations?

$$
\boxed{
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
}
$$

$$
\boxed{
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
}
$$

After bias correction, the conceptual update is:

$$
\theta_t
=
\theta_{t-1}
-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

The lecture only requires the intuition that `m` and `v` are additional state that consumes memory.

---

## Q40. MUST REMEMBER - What components dominate training memory?

$$
\boxed{
M_{train}
=
M_{params}
+M_{grads}
+M_{optimizer}
+M_{activations}
+M_{temporary\ buffers}
}
$$

Different techniques attack different terms:

```text
ZeRO              -> parameter / gradient / optimizer replication
checkpointing      -> stored activations
FlashAttention     -> attention intermediates and HBM traffic
mixed precision    -> bytes per number
LoRA               -> number of trainable parameters and states
```

---

## Q41. INTERVIEW CLARIFICATION - What is a useful bytes-per-parameter estimate for mixed-precision Adam training?

A common rough accounting is:

```text
FP16/BF16 model parameter      2 bytes
FP16/BF16 gradient             2 bytes
FP32 master parameter          4 bytes
FP32 Adam first moment         4 bytes
FP32 Adam second moment        4 bytes
--------------------------------------
rough subtotal                16 bytes/parameter
```

Implementations differ, and activations are additional. For one billion parameters, this subtotal alone is roughly 16 GB before activations and buffers.

**Memory liner:**

> A billion parameters do not mean only a few GB during training; optimizer and gradient copies multiply the footprint.

---

## Q42. MUST REMEMBER - What controls activation memory?

At a high level activation memory grows with:

```text
batch size B
sequence length T
model width D
number of layers L
FFN width Dff
attention implementation
numerical precision
```

A simple non-attention activation often has shape `[B,T,D]`, while naive full attention may materialize `[B,H,T,T]`.

---

## Q43. MUST REMEMBER - Why does context length strongly affect training memory?

Full self-attention forms:

$$
S=QK^\top\in\mathbb{R}^{B\times H\times T\times T}
$$

Doubling `T` roughly quadruples the number of score entries:

$$
T^2\rightarrow(2T)^2=4T^2
$$

**Memory liner:**

> Long context enlarges both ordinary token activations and the quadratic attention interaction map.

---

## Q44. MUST REMEMBER - Why can a model fail to fit on one GPU even before a batch is processed?

The parameter, gradient, master-weight, and optimizer-state tensors alone may exceed device memory. Data parallelism does not solve this if every worker must hold a complete copy.

**Memory liner:**

> Batch sharding helps activation memory; it does not automatically make an oversized model fit.

---

## Q45. MUST REMEMBER - Compute bottleneck versus memory bottleneck?

- **Compute-bound:** arithmetic units are the limiting resource.
- **Memory-bound / IO-bound:** arithmetic units wait for data to move through the memory hierarchy.

Many Transformer operations are fast matrix multiplications, but naive attention can spend substantial time moving large intermediate tensors. FlashAttention targets this IO bottleneck.

**Memory liner:**

> Faster math does not help if the GPU is mostly waiting for tensors to arrive.


---

# 5. Data parallelism and ZeRO

## Q46. MUST REMEMBER - What is data parallelism?

Replicate the model on each worker, split the batch, and process each shard independently:

```text
global batch [B,T]
       /        |        \
GPU 0 [b,T]  GPU 1 [b,T]  ... GPU G-1 [b,T]
       \        |        /
       synchronize gradients
       update identical model copies
```

Usually:

$$
b=B/G
$$

assuming an evenly divisible batch.

**Memory liner:**

> Data parallelism divides examples, not model parameters.

---

## Q47. MUST REMEMBER - What resides on every worker in ordinary data parallelism?

Each worker typically stores:

- A full model replica.
- A local activation set for its batch shard.
- Full gradients after synchronization.
- Full optimizer state.

Only batch-dependent activation memory is naturally reduced by splitting `B`.

---

## Q48. MUST REMEMBER - How are data-parallel gradients combined?

If worker `g` computes local gradient `g^{(g)}`:

$$
\boxed{
g_{global}
=
\frac{1}{G}\sum_{j=1}^{G}g^{(j)}
}
$$

Summing rather than averaging is also possible if the learning-rate and loss-reduction convention are adjusted consistently.

**Memory liner:**

> Each GPU sees different examples; synchronization reconstructs the gradient of the global batch.

---

## Q49. SHOULD KNOW - What communication operation is commonly associated with this synchronization?

**Interview clarification:** Distributed data parallel training commonly uses an **all-reduce** operation so that every worker receives the aggregated gradient.

```text
local gradients
      -> collective reduction
      -> same global gradient on every worker
```

The lecture emphasizes the communication cost rather than the collective algorithm itself.

---

## Q50. MUST REMEMBER - What memory does data parallelism reduce?

It reduces memory proportional to the local batch, especially activations:

```text
single GPU activation batch: [B,T,...]
data-parallel worker:         [B/G,T,...]
```

It does not reduce full model, gradient, or optimizer replication in ordinary DP.

---

## Q51. MUST REMEMBER - What are the main limitations of ordinary data parallelism?

1. The complete model must fit on each GPU.
2. Gradients must be communicated every update.
3. Adding workers can produce diminishing speedups as communication dominates.
4. Full parameters, gradients, and optimizer states are redundantly stored.

**Memory liner:**

> Data parallelism buys batch capacity but pays replication and synchronization cost.

---

## Q52. MUST REMEMBER - What problem does ZeRO solve?

**ZeRO = Zero Redundancy Optimizer.**

Ordinary DP stores the same parameter-related tensors on every worker. ZeRO partitions selected states across workers so they are not permanently replicated everywhere.

```text
ordinary DP:
GPU 0: params + grads + optimizer
GPU 1: params + grads + optimizer
...

ZeRO:
partition selected states across workers
```

**Memory liner:**

> ZeRO keeps data parallelism while removing redundant model-state copies.

---

## Q53. MUST REMEMBER - What is ZeRO Stage 1?

Partition the **optimizer states** across data-parallel workers.

```text
parameters:       replicated
gradients:        replicated
optimizer states: sharded
```

For Adam, the two large moment tensors are distributed, providing substantial savings.

---

## Q54. MUST REMEMBER - What is ZeRO Stage 2?

Partition optimizer states **and gradients**:

```text
parameters:       replicated
gradients:        sharded
optimizer states: sharded
```

**Memory liner:**

> ZeRO-2 adds gradient sharding to ZeRO-1.

---

## Q55. MUST REMEMBER - What is ZeRO Stage 3?

Partition optimizer states, gradients, **and parameters**:

```text
parameters:       sharded
gradients:        sharded
optimizer states: sharded
```

Parameter shards must be gathered or communicated when a layer is computed.

**Memory liner:**

> ZeRO-3 removes the requirement that every worker permanently store the full model.

---

## Q56. MUST REMEMBER - ZeRO stage comparison?

| State | Ordinary DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| Parameters | replicated | replicated | replicated | sharded |
| Gradients | replicated | replicated | sharded | sharded |
| Optimizer states | replicated | sharded | sharded | sharded |
| Communication burden | baseline DP | higher | higher | highest of the three |

**Master memory liner:**

> Stage 1 shards optimizer state; Stage 2 also shards gradients; Stage 3 also shards parameters.

---

## Q57. MUST REMEMBER - Why does increasing the ZeRO stage increase communication?

A state that is no longer locally replicated may need to be gathered, reduced, or redistributed at the point where computation or updating requires it.

```text
less persistent memory
        <->
more state movement
```

**Memory liner:**

> ZeRO converts redundant storage into communication.

---

## Q58. SHOULD KNOW - How should one choose a ZeRO stage?

Choose the least aggressive stage that makes the run fit while preserving acceptable throughput:

- Model already fits comfortably: ordinary DP or ZeRO-1 may be enough.
- Gradient/optimizer memory is the problem: ZeRO-2.
- Full parameter copy does not fit: ZeRO-3.

The exact decision depends on network bandwidth, GPU count, model size, batch size, and implementation.

---

# 6. Model parallelism

## Q59. MUST REMEMBER - Data parallelism versus model parallelism?

```text
DATA PARALLELISM
same model on each GPU
different examples on each GPU

MODEL PARALLELISM
one model computation split across GPUs
same example may use several GPUs
```

**Memory liner:**

> Data parallelism partitions samples; model parallelism partitions model work.

---

## Q60. MUST REMEMBER - What is tensor parallelism?

Tensor parallelism splits a large tensor operation, such as a linear-layer matrix multiplication, across devices.

For:

$$
Y=XW
$$

with:

```text
X: [B,T,d_in]
W: [d_in,d_out]
```

one can split columns of `W`:

```text
GPU 0: W_0 [d_in,d_out/G]
GPU 1: W_1 [d_in,d_out/G]
...
```

Each GPU computes a slice of `Y`, then slices are concatenated or reduced depending on the partition scheme.

**Memory liner:**

> Tensor parallelism splits one large matrix operation across devices.

---

## Q61. SHOULD KNOW - What communication appears in tensor parallelism?

A later operation often needs the full activation or a reduced result, so devices exchange partial outputs.

```text
local matmul results
        -> all-gather / reduce-style communication
        -> complete logical layer output
```

The lecture requires only the high-level idea; exact row- and column-parallel algorithms are outside its scope.

---

## Q62. MUST REMEMBER - What is pipeline parallelism?

Assign consecutive layer groups to different devices:

```text
GPU 0: layers 1-8
GPU 1: layers 9-16
GPU 2: layers 17-24
GPU 3: layers 25-32
```

Activations flow from one stage to the next; backward gradients flow in reverse.

**Memory liner:**

> Pipeline parallelism partitions the model by depth.

### LECTURE BOUNDARY

Microbatch scheduling, pipeline bubbles, and 1F1B schedules are not derived in this lecture.

---

## Q63. MUST REMEMBER - What is expert parallelism?

For an MoE model, different experts can reside on different devices:

```text
token representations
        -> router
        -> send each token to selected expert device
        -> process
        -> return expert output
```

The challenge is communication and load balance because tokens may route unevenly.

**Memory liner:**

> Expert parallelism distributes MoE experts and routes tokens across devices.

---

## Q64. SHOULD KNOW - Can parallelism strategies be combined?

Yes. Large training systems often combine dimensions:

```text
data parallel groups
        x tensor parallel within each model replica
        x pipeline stages across layer groups
        x expert parallelism for MoE layers
```

This is sometimes called multidimensional or 3D/4D parallelism, although that terminology is not developed in the lecture.

---

## Q65. MUST REMEMBER - What is the universal distributed-training trade-off?

$$
\boxed{
\text{less memory per GPU}
\Longleftrightarrow
\text{more communication and coordination}
}
$$

An effective system minimizes idle time and communication relative to useful matrix multiplication.

---

# 7. FlashAttention and IO-aware exact attention

## Q66. MUST REMEMBER - What problem does FlashAttention target?

It targets the excessive movement of attention intermediates between slow, large GPU memory and fast on-chip memory.

It does **not** replace softmax attention with an approximation.

**Memory liner:**

> FlashAttention accelerates exact attention by reducing memory traffic.

---

## Q67. MUST REMEMBER - HBM versus SRAM?

| Memory | Capacity | Speed/bandwidth | Lecture role |
|---|---:|---:|---|
| HBM / GPU global memory | Large, usually GB scale | Relatively slower | Stores model tensors and large activations |
| SRAM / on-chip shared memory/registers | Much smaller, usually KB-MB per execution region | Much faster | Holds tiles while kernels compute |

The transcript sometimes pronounces HBM as "HPM"; the standard term is **HBM**, high-bandwidth memory.

**Memory liner:**

> HBM is the warehouse; SRAM is the small workbench next to the compute units.

---

## Q68. MUST REMEMBER - What does naive attention materialize?

For one layer:

$$
S=\frac{QK^\top}{\sqrt{D_h}}
$$

$$
P=\operatorname{softmax}(S)
$$

$$
O=PV
$$

Shapes:

```text
Q,K,V: [B,H,T,Dh]
S:     [B,H,T,T]
P:     [B,H,T,T]
O:     [B,H,T,Dh]
```

A naive implementation writes `S` to HBM, reads it for softmax, writes `P`, reads it again for `P@V`, and writes `O`.

**Memory liner:**

> The expensive part is often repeatedly storing and loading the `T x T` intermediates.

---

## Q69. MUST REMEMBER - Is FlashAttention approximate?

No. At the level taught, it computes the same mathematical attention output, up to normal floating-point implementation effects.

```text
same function:
softmax(QK^T / sqrt(Dh)) V

different execution schedule:
blocked, fused, IO-aware
```

**Memory liner:**

> FlashAttention changes the schedule, not the definition of attention.

---

## Q70. MUST REMEMBER - What is tiling?

Partition `Q`, `K`, and `V` into blocks small enough to fit in fast on-chip memory:

```text
load Q tile + K tile + V tile from HBM
        -> compute score tile in SRAM
        -> update running softmax/output statistics
        -> discard temporary tile
        -> move to next K/V tile
```

This avoids storing the complete score and probability matrices in HBM.

---

## Q71. MUST REMEMBER - How can softmax be computed block by block?

For one query row with scores `s_j`:

$$
\operatorname{softmax}(s_j)
=
\frac{e^{s_j}}{\sum_k e^{s_k}}
$$

One does not need all scores simultaneously if one maintains running normalization statistics while visiting blocks.

**Memory liner:**

> Softmax needs a global normalization, but that normalization can be accumulated online.

---

## Q72. INTERVIEW CLARIFICATION - What are the online-softmax statistics?

For scores processed so far, maintain:

```text
m = running maximum
l = running sum of exp(score - m)
o = running weighted value accumulator
```

When a new block with maximum `m_b` arrives:

$$
m_{new}=\max(m_{old},m_b)
$$

$$
l_{new}
=
e^{m_{old}-m_{new}}l_{old}
+
\sum_{j\in block}e^{s_j-m_{new}}
$$

The value accumulator is rescaled in the same way before adding the new block contribution. At the end, divide the accumulated numerator by `l`.

This is the mathematical reason blockwise computation remains exact and numerically stable.

---

## Q73. MUST REMEMBER - What tensor is no longer persistently materialized?

The full attention score/probability matrix:

```text
[B,H,T,T]
```

is not written as one large persistent HBM tensor. Tiles exist temporarily on chip.

The final output still has:

```text
[B,H,T,Dh]
```

---

## Q74. MUST REMEMBER - What memory complexity benefit should you remember?

At the interview level:

```text
naive attention auxiliary storage: O(T^2)
FlashAttention auxiliary storage:   avoids materialized O(T^2) map
```

The arithmetic remains quadratic for full attention; the major gain is IO and intermediate-memory behavior.

**Memory liner:**

> FlashAttention removes quadratic intermediate storage, not quadratic pairwise arithmetic.

---

## Q75. MUST REMEMBER - What does FlashAttention do during backward propagation?

Instead of storing every large forward intermediate needed for gradients, it can recompute attention quantities during backward from smaller saved statistics and the original inputs.

```text
standard approach:
store many forward activations -> reuse in backward

recomputation approach:
store less -> recompute needed values in backward
```

**Memory liner:**

> Save memory by replacing stored attention intermediates with cheap recomputation.

---

## Q76. MUST REMEMBER - How can more FLOPs produce a lower runtime?

The extra recomputation increases arithmetic, but greatly reduces expensive HBM reads and writes. If the kernel is IO-bound, this trade makes the overall operation faster.

$$
\text{more arithmetic}
+
\text{much less IO}
\Rightarrow
\text{lower wall-clock time}
$$

**Memory liner:**

> On modern GPUs, recomputing can be cheaper than reloading.

---

## Q77. MUST REMEMBER - FlashAttention versus sparse/sliding-window attention?

| Method | Mathematical interaction pattern | Exact relative to full dense attention? | Main benefit |
|---|---|---:|---|
| FlashAttention | All permitted query-key pairs | Yes | Better IO and memory schedule |
| Sliding-window/sparse attention | Only selected pairs | No, it changes the pattern | Fewer pairwise operations |

**Memory liner:**

> FlashAttention is an exact kernel optimization; sparse attention changes which tokens interact.

---

## Q78. SHOULD KNOW - Why are there FlashAttention 2/3 and later variants?

The core IO-aware idea remains, while kernels are retuned for newer GPU memory hierarchies, instruction sets, parallelism, and supported data types.

**Memory liner:**

> New FlashAttention versions adapt the same principle to evolving hardware.

---

## Q79. What is the implementation mental map?

```python
# Conceptual only: optimized libraries fuse these operations.
q = q_proj(x)  # [B,H,T,Dh]
k = k_proj(x)  # [B,H,T,Dh]
v = v_proj(x)  # [B,H,T,Dh]

# A modern backend can dispatch to an IO-aware exact kernel.
out = scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
)  # [B,H,T,Dh]
```

Do not implement FlashAttention by explicitly constructing the full `T x T` matrix in Python; the benefit comes from the fused low-level execution schedule.

---

## Q80. MUST REMEMBER - Give the 30-second FlashAttention interview answer.

> Standard attention repeatedly materializes and transfers `QK^T` and softmax probabilities through HBM, making IO a bottleneck. FlashAttention tiles Q/K/V into fast on-chip memory and uses an online softmax so each tile contributes to the exact global result without storing the full attention matrix. In backward it can recompute cheap intermediates rather than save them. It may perform more arithmetic but still run faster because it drastically reduces HBM traffic.

---

# 8. Floating-point formats, quantization, and mixed-precision training

## Q81. MUST REMEMBER - How is a floating-point number represented conceptually?

```text
sign bit      -> positive or negative
exponent bits -> dynamic range / order of magnitude
mantissa bits -> precision within that range
```

**Memory liner:**

> Exponent controls how large/small; mantissa controls how finely values are represented.

---

## Q82. MUST REMEMBER - Precision versus dynamic range?

- **Precision:** spacing between representable values near a scale; mainly influenced by mantissa bits.
- **Dynamic range:** largest and smallest magnitudes representable; mainly influenced by exponent bits.

This explains why two 16-bit formats can behave very differently.

---

## Q83. INTERVIEW CLARIFICATION - FP32 versus FP16 versus BF16?

| Format | Total bits | Exponent bits | Mantissa/fraction bits | Main intuition |
|---|---:|---:|---:|---|
| FP32 | 32 | 8 | 23 | Broad range and high precision |
| FP16 | 16 | 5 | 10 | Better precision than BF16 at similar scale, but much smaller range |
| BF16 | 16 | 8 | 7 | FP32-like range with less precision |

The lecture introduces these as different memory/throughput trade-offs; the exact bit allocation is an interview clarification.

**Memory liner:**

> FP16 spends more bits on precision; BF16 preserves FP32-like range.

---

## Q84. MUST REMEMBER - Why can lower precision save both memory and time?

Memory:

$$
\text{bytes per value}=\frac{\text{bits per value}}{8}
$$

Thus FP16/BF16 uses half the storage of FP32 for the same tensor shape.

Speed:

- More low-precision values fit in memory bandwidth and caches.
- GPUs often provide much higher low-precision matrix-multiply throughput.

**Memory liner:**

> Fewer bits mean fewer bytes moved and more arithmetic packed into each hardware operation.

---

## Q85. MUST REMEMBER - What is quantization?

Quantization maps values from a higher-precision representation to a lower-bit representation, usually using scale information and sometimes a zero point.

Generic affine form:

$$
q=\operatorname{round}\left(\frac{x}{s}\right)+z
$$

$$
\hat x=s(q-z)
$$

where:

```text
s     scale
z     zero point
q     low-bit stored value
x_hat dequantized approximation
```

The lecture gives the high-level idea and names zero-point and absmax approaches without deriving them.

---

## Q86. MUST REMEMBER - What is mixed-precision training?

Use lower precision for expensive forward/backward computation while retaining higher precision where numerical accumulation and updates are sensitive.

Lecture-level flow:

```text
FP32 master weights
       -> cast/use lower precision for forward
       -> lower-precision backward computation
       -> gradients applied to FP32 master weights
```

**Memory liner:**

> Compute cheaply, but preserve a high-precision copy for stable long-term updates.

---

## Q87. MUST REMEMBER - Why keep master weights in higher precision?

Each optimizer step may be small. If the stored weight has coarse precision, a small update can round to zero or accumulated rounding error can damage training.

```text
large weight + tiny update
        -> update may disappear in low precision
        -> FP32 master copy preserves it
```

**Memory liner:**

> Low precision can represent the direction of a batch computation while failing to preserve tiny cumulative parameter updates.

---

## Q88. MUST REMEMBER - Which quantities can have different precision?

A mixed-precision system may choose different formats for:

- Forward activations.
- Backward activations/gradients.
- Model weights used by matrix multiplication.
- Master weights.
- Optimizer moments.
- Reductions or numerically sensitive operations.

The lecture presents the broad FP32-master / FP16-compute pattern and notes that real implementations vary.

---

## Q89. INTERVIEW CLARIFICATION - Why is loss scaling associated with FP16 training?

Small FP16 gradients can underflow to zero. Multiply the loss by a scale `S` before backward:

$$
\nabla(S\mathcal{L})=S\nabla\mathcal{L}
$$

then unscale gradients before the optimizer step. Dynamic loss scaling lowers `S` if overflow is detected.

BF16's wider exponent range usually makes explicit loss scaling less necessary.

### LECTURE BOUNDARY

Loss scaling is a natural interview follow-up but is not taught in detail in the transcript.

---

## Q90. MUST REMEMBER - Quantization versus mixed precision?

| | Quantization | Mixed-precision training |
|---|---|---|
| Core action | Represent some tensors with fewer bits | Use multiple floating formats in one training process |
| Typical goal | Reduce storage, bandwidth, or inference/fine-tuning memory | Speed training while retaining stability |
| Error source | Discrete approximation / dequantization | Rounding, underflow, overflow in lower precision |
| Lecture example | Low-bit base weights in QLoRA | FP16 compute with FP32 master updates |

**Memory liner:**

> Quantization compresses values; mixed precision assigns different numerical formats to different roles.

---

## Q91. SHOULD KNOW - Why might some operations remain in higher precision?

Operations involving large reductions, small variances, exponentials, optimizer accumulation, or extreme dynamic ranges can be more numerically sensitive than large matrix multiplications.

The lecture says the best precision policy depends on the layer and setup rather than applying one format blindly everywhere.

---

## Q92. LECTURE BOUNDARY - What are zero-point and absmax quantization?

They are named as directions for handling ranges, but the transcript does not derive their exact calibration equations or compare their error properties. Keep detailed post-training quantization algorithms for a separate note.

---

## Q93. MUST REMEMBER - Give a simple memory example.

A tensor with `n` values requires approximately:

```text
FP32: 4n bytes
FP16/BF16: 2n bytes
INT8: 1n bytes
4-bit: 0.5n bytes, before metadata/packing overhead
```

For the same matrix shape, moving from FP32 to 4-bit reduces raw weight storage by about 8x.

---

## Q94. SHOULD KNOW - What can go wrong at low precision?

- Overflow: magnitude exceeds representable range.
- Underflow: small nonzero value becomes zero.
- Rounding error: nearby values collapse together.
- Outliers dominate a shared quantization scale.
- Repeated small updates disappear.

**Memory liner:**

> Lower precision works only when numerical error remains smaller than the optimization tolerance.

---

## Q95. MUST REMEMBER - Give the mixed-precision interview answer.

> Mixed-precision training exploits fast, memory-efficient low-precision matrix multiplication for most forward and backward work, while preserving higher-precision master weights and optimizer accumulation so small updates are not lost. It reduces memory traffic and increases throughput, but sensitive operations, overflow, and underflow still require care.


---

# 9. Supervised fine-tuning and instruction tuning

## Q96. MUST REMEMBER - Why is SFT needed after pre-training?

A pre-trained model is optimized to continue text, not necessarily to:

- Interpret a user's request as an instruction.
- Give a direct answer.
- Follow an output format.
- Be consistently helpful and harmless.
- Adopt an assistant dialogue style.

SFT demonstrates the desired mapping from user input to good assistant output.

**Memory liner:**

> Pre-training supplies capability; SFT demonstrates the desired interface and behavior.

---

## Q97. MUST REMEMBER - What is supervised fine-tuning?

Start from pre-trained weights and continue training on labelled input-output examples:

```text
instruction / user input x
        -> desired response y
```

The labels are the desired response tokens.

**Memory liner:**

> SFT is ordinary next-token training on curated demonstrations of desired behavior.

---

## Q98. MUST REMEMBER - What is the basic SFT data format?

Conceptually:

```text
PROMPT:
Can I put this teddy bear in a washing machine?

RESPONSE:
Check the care label; many plush toys should be hand-washed ...
```

Chat-style representation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can I wash this teddy bear?"},
    {"role": "assistant", "content": "Check the care label first ..."}
  ]
}
```

The exact serialization uses model-specific special tokens, but the semantic unit is an input paired with a desired output.

---

## Q99. MUST REMEMBER - Is the SFT objective fundamentally different from causal LM pre-training?

The token-level objective is still causal next-token cross-entropy. The key differences are:

1. The data distribution is curated around instructions and responses.
2. Prompt/input tokens usually serve as conditioning context.
3. The loss is applied mainly or only to the desired assistant/output tokens.

**Memory liner:**

> Same autoregressive machinery, different data and loss mask.

---

## Q100. MUST REMEMBER - What is the response-only SFT loss?

Let `M_{b,t}=1` for assistant response tokens and `0` for prompt/system/user tokens:

$$
\boxed{
\mathcal{L}_{SFT}
=
-\frac{1}{\sum_{b,t}M_{b,t}}
\sum_{b=1}^{B}\sum_{t=1}^{T}
M_{b,t}
\log p_\theta(z_{b,t}\mid z_{b,<t})
}
$$

The prompt is still visible through causal attention; it simply does not contribute directly to the loss.

**Memory liner:**

> Attend to the prompt; optimize the response.

---

## Q101. MUST REMEMBER - What are the SFT tensor shapes?

```text
input_ids:       [B,T]
attention_mask:  [B,T]
labels:          [B,T]
loss_mask M:     [B,T]
hidden:          [B,T,D]
logits:          [B,T,V]
```

A common implementation sets ignored label positions to `-100`:

```text
labels at prompt tokens:    -100
labels at response tokens:  next target token ID
```

Cross-entropy then ignores prompt positions.

---

## Q102. MUST REMEMBER - How does teacher forcing operate in SFT?

During training, the complete correct response is available. The model receives the prompt plus the shifted ground-truth response and predicts each response token under a causal mask.

```text
input:
[prompt tokens, BOS_response, y1, y2, ..., y_{K-1}]

target on response positions:
[y1, y2, ..., yK]
```

At inference, ground-truth response tokens are absent, so generated tokens are fed back autoregressively.

---

## Q103. MUST REMEMBER - Why not train loss on the user prompt tokens?

The prompt is given, not something the assistant should reproduce. Penalizing prompt-token prediction allocates capacity to modelling the user's text rather than learning the desired conditional response.

There are training recipes that use all tokens, but the lecture's SFT framing is response-only supervision.

**Memory liner:**

> The prompt specifies the problem; the response is the demonstrated behavior.

---

## Q104. MUST REMEMBER - Compare pre-training and SFT examples.

```text
PRE-TRAINING
raw sequence:
"The teddy bear should be hand washed ..."

loss:
early every next-token position

SFT
input:
"How should I wash my teddy bear?"

target:
"Check its care label; hand washing is often safest ..."

loss:
assistant target tokens
```

---

## Q105. MUST REMEMBER - What is instruction tuning?

Instruction tuning is SFT whose examples explicitly teach the model to execute natural-language instructions across many tasks.

```text
instruction + optional input
        -> high-quality task response
```

**Memory liner:**

> Instruction tuning is multi-task SFT for following natural-language commands.

---

## Q106. SHOULD KNOW - What task categories can appear in an instruction-tuning mixture?

The lecture mentions examples such as:

- Story and poem generation.
- List generation.
- Explanation.
- Mathematical reasoning and proof-like text.
- Code generation.
- Assistant dialogue.
- Safety-oriented behavior.

A diverse mixture teaches the common meta-behavior: infer the requested task and produce an appropriate answer.

---

## Q107. MUST REMEMBER - Human-written versus synthetic SFT data?

### Human-written

- High control and nuanced judgement.
- Expensive and slow.
- Requires detailed annotator guidelines.

### Model-generated / synthetic

- A strong model proposes instructions or responses.
- Humans or another model review/filter them.
- Scales curation more cheaply.
- Can propagate teacher-model biases and errors.

**Memory liner:**

> Synthetic data scales demonstrations; review determines whether it scales quality.

---

## Q108. MUST REMEMBER - How can safety be represented in SFT data?

The desired response may demonstrate:

- Safe refusal for harmful requests.
- A safer alternative.
- Calibrated or hedged statements rather than unjustified certainty.
- Helpful behavior on benign requests.

The lecture emphasizes that this behavior is learned from examples rather than relying only on a fragile string-matching rule.

**Memory liner:**

> Safety behavior can be part of the learned response distribution.

---

## Q109. SHOULD KNOW - Helpful versus harmless can conflict. Why?

A response that directly fulfils every request may be preferred for convenience but unsafe. A response that refuses too broadly may be safe but unhelpful.

```text
helpfulness pressure: answer the request
safety pressure:      avoid harmful assistance
```

This tension later motivates preference/alignment methods, but their algorithms are outside this lecture.

---

## Q110. MUST REMEMBER - How can a small SFT dataset create broad instruction-following behavior?

Pre-training already gives the model broad language and domain knowledge. SFT mainly teaches how to **use** that knowledge in response to instructions.

```text
pre-training:
learn concepts and language patterns

SFT:
learn that "do X" should trigger a direct, formatted response doing X
```

**Memory liner:**

> SFT redirects existing capabilities more than it teaches every domain from scratch.

---

## Q111. MUST REMEMBER - How does SFT data scale compare with pre-training data?

The lecture's key order-of-magnitude point is:

```text
pre-training: enormous token corpus
SFT:          far fewer, highly curated examples
```

SFT data is much smaller but has higher behavioural information density.

**Memory liner:**

> Pre-training wins by breadth and volume; SFT wins by precision and intent.

---

## Q112. MUST REMEMBER - Why does prompt-distribution coverage matter?

A model generalizes best when the SFT examples cover the forms, domains, difficulty, and constraints likely at deployment.

```text
training prompt distribution close to deployment
        -> easier generalization

large distribution shift
        -> more unpredictable behavior
```

**Memory liner:**

> Alignment to a task requires alignment of the training and deployment prompt distributions.

---

## Q113. SHOULD KNOW - More examples versus more diverse examples?

Repeating near-duplicate examples can teach a narrow style without covering the task space. Diverse examples can better reveal the underlying instruction-following rule.

```text
coverage > raw duplication
```

The optimal balance depends on data quality, task complexity, and model capacity.

---

## Q114. SHOULD KNOW - Will the model reproduce an SFT response word for word?

Not necessarily. At nonzero sampling temperature, generation is stochastic. Even with greedy decoding, the model may have generalized rather than stored the example exactly.

Distinguish:

```text
training memorization
model distribution after SFT
decoding randomness at inference
```

These are separate factors.

---

## Q115. MUST REMEMBER - What are the main SFT challenges in this lecture?

- Expensive high-quality data creation and review.
- Coverage mismatch between SFT prompts and real users.
- Subjective definitions of helpfulness and style.
- Safety-helpfulness trade-offs.
- Evaluation contamination and benchmark limitations.
- Full fine-tuning compute and memory cost.

LoRA and QLoRA address the final issue, not all the others.

---

## Q116. MUST REMEMBER - What does response-only SFT look like in PyTorch?

```python
# input_ids and labels are already shifted internally by the LM loss.
# Shapes: input_ids [B,T], labels [B,T]

labels = input_ids.clone()
labels[~assistant_token_mask] = -100  # ignore prompt/system/user tokens

out = model(input_ids=input_ids, attention_mask=attention_mask)
logits = out.logits                   # [B,T,V]

loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

**Memory liner:**

> The response mask changes where cross-entropy is paid, not what the decoder sees.

---

# 10. Evaluating trained and fine-tuned LLMs

## Q117. MUST REMEMBER - Why is LLM evaluation intrinsically difficult?

A useful assistant has multiple dimensions that are not perfectly correlated:

- Knowledge and factuality.
- Reasoning.
- Mathematical ability.
- Code generation.
- Instruction following.
- Helpfulness and style.
- Safety.
- Latency and cost.

A model can score highly on one dimension and be poor on another.

**Memory liner:**

> There is no single scalar definition of a good general assistant.

---

## Q118. SHOULD KNOW - What benchmark categories does the lecture discuss?

The lecture groups quantitative evaluation around areas such as:

- General language/knowledge tasks.
- Reasoning.
- Mathematical reasoning.
- Code generation.

It cites MMLU and GSM8K-style benchmarks as examples among many evolving suites.

---

## Q119. INTERVIEW CLARIFICATION - What are MMLU and GSM8K at a high level?

- **MMLU:** Massive Multitask Language Understanding; multiple-choice tasks across many academic/professional subjects.
- **GSM8K:** Grade School Math 8K; grade-school-level word problems with multi-step arithmetic reasoning.

The lecture's important point is not the acronym expansion but that a benchmark measures a particular task distribution.

---

## Q120. MUST REMEMBER - Training on the test task versus training on the test set?

### Training on the test set

The exact benchmark examples or answers appear in training. This is direct contamination.

### Training on the test task

The model is trained on similar problem forms or auxiliary data from the same capability distribution, without necessarily seeing exact held-out questions.

```text
same examples      -> test-set leakage
same task/domain   -> capability-specific training advantage
```

**Memory liner:**

> A benchmark score reflects both architecture and how closely training data matched the tested task.

---

## Q121. MUST REMEMBER - Why can benchmark comparison be unfair when data mixtures differ?

Suppose two models have similar size, but only one was heavily trained on math word problems. Comparing their GSM-like scores does not isolate intrinsic model quality; it also measures training-mixture allocation.

A fair claim should report or control, where possible:

- Exposure to the task family.
- Exact contamination checks.
- Data and compute budget.
- Prompting/evaluation protocol.

---

## Q122. MUST REMEMBER - Why do benchmark gains sometimes stop matching user-perceived gains?

Once a public benchmark becomes important, training pipelines increasingly include data resembling it. Models can improve on the benchmark distribution without an equally broad improvement on real user needs.

**Memory liner:**

> A benchmark becomes less diagnostic when it becomes a training target.

---

## Q123. MUST REMEMBER - What is the pairwise human-evaluation idea used by model arenas?

```text
same user prompt
   -> anonymous response A
   -> anonymous response B
   -> user selects preferred response
   -> aggregate many pairwise outcomes into a ranking
```

This measures perceived preference more directly than a fixed academic benchmark.

---

## Q124. SHOULD KNOW - Why can pairwise ranking be more useful than an absolute rating?

Humans often find it easier to choose the better of two concrete responses than to assign a calibrated score from 1 to 10.

Pairwise outcomes can be aggregated with rating or probabilistic comparison models, although the lecture does not derive a specific ranking formula.

---

## Q125. MUST REMEMBER - Why is user preference not equal to factual correctness?

A fluent, detailed, actionable answer can be persuasive while wrong. A general user may lack the domain expertise required to identify the error.

```text
preferred style or confidence
        !=
factual validity
```

**Memory liner:**

> Human preference captures perceived usefulness, not guaranteed truth.

---

## Q126. SHOULD KNOW - How can model identity leakage distort arena evaluation?

If a response reveals the model's identity or recognizable style, evaluators or automated adversaries may infer which model produced it and vote based on identity rather than answer quality.

This makes anonymity and anti-gaming controls important.

---

## Q127. MUST REMEMBER - What is the population-mismatch problem?

The people voting in an arena may not represent the intended user population. Preferences over verbosity, emojis, formatting, safety, or technical depth can differ by group.

```text
arena voter distribution
        !=
deployment user distribution
```

**Memory liner:**

> A preference score is conditional on who supplied the preferences.

---

## Q128. MUST REMEMBER - Why can human preference penalize safety?

Users often prefer an answer over a refusal, even when refusal is the intended safe behavior. Pure preference optimization can therefore create pressure toward over-compliance.

**Memory liner:**

> Immediate user satisfaction and product safety are not always aligned.

---

## Q129. MUST REMEMBER - Why is one leaderboard number insufficient?

A credible evaluation should examine a vector of properties:

```text
capability
factuality
robustness
safety
instruction following
latency
memory
cost
```

The right weighting depends on the use case.

---

## Q130. MUST REMEMBER - What is a practical evaluation matrix?

| Dimension | Possible method | Main caveat |
|---|---|---|
| General knowledge | fixed benchmark | contamination/task exposure |
| Math/code/reasoning | task benchmark | training-distribution matching |
| Helpfulness/style | pairwise human preference | voter population bias |
| Factuality | expert-labelled checks/tools | expensive, domain-specific |
| Safety | red-team and policy suites | evolving threats, false refusals |
| Efficiency | latency, throughput, memory | hardware and batch dependent |

**Memory liner:**

> Evaluate the deployment objective with several complementary instruments.

---

# 11. Training-stage map, mid-training, and alignment boundaries

## Q131. MUST REMEMBER - Draw the complete Lecture 4 training-stage map.

```text
RAW BROAD DATA
      -> causal pre-training
      -> base model

OPTIONAL TARGETED DATA
      -> mid-training with the same causal LM objective
      -> domain/capability-strengthened base model

CURATED INPUT-OUTPUT DEMONSTRATIONS
      -> supervised fine-tuning / instruction tuning
      -> instruction-following model

PAIRWISE OR OTHER PREFERENCE SIGNAL
      -> preference tuning
      -> further aligned assistant
```

---

## Q132. MUST REMEMBER - Mid-training versus SFT?

| | Mid-training | SFT |
|---|---|---|
| Objective | Ordinary causal LM | Causal LM with demonstration/response supervision |
| Data form | Domain-relevant text/code | Prompt-response or instruction-output pairs |
| Main effect | Strengthen knowledge/capability distribution | Shape interaction behavior |
| Loss region | Broad sequence tokens | Usually output/assistant tokens |

**Memory liner:**

> Mid-training says what domain to learn; SFT says how to answer.

---

## Q133. MUST REMEMBER - SFT versus preference tuning?

```text
SFT:
imitate a demonstrated good response

preference tuning:
learn which of multiple responses is preferred
```

Preference tuning is named but not mechanistically taught in this lecture.

---

## Q134. MUST REMEMBER - Capability versus alignment?

- **Capability:** what problems the model can solve.
- **Alignment/behavior:** when and how it applies those capabilities according to intended goals.

The two interact but are not identical.

**Memory liner:**

> A model can know how to do something without behaving as the product intends.

---

## Q135. LECTURE BOUNDARY - Which post-training topics should be left for the next note?

- Reward modelling.
- RLHF and PPO.
- DPO or other direct preference objectives.
- GRPO.
- Detailed preference-data construction.

Lecture 4 only positions preference tuning in the overall pipeline.


---

# 12. LoRA: Low-Rank Adaptation

## Q136. MUST REMEMBER - What problem does LoRA solve?

Full fine-tuning updates every parameter and therefore requires:

- Gradients for the full model.
- Optimizer state for the full model.
- Storage of a complete task-specific checkpoint.

LoRA freezes the base model and trains small low-rank update matrices instead.

**Memory liner:**

> LoRA adapts a large weight through a small trainable update.

---

## Q137. MUST REMEMBER - What is the LoRA equation?

Using the common column-vector convention:

$$
\boxed{
W=W_0+\Delta W
=W_0+B_{LoRA}A
}
$$

where:

```text
W0:      [d_out, d_in]    frozen pre-trained weight
A:       [r, d_in]        trainable down projection
B_LoRA:  [d_out, r]       trainable up projection
r << min(d_in,d_out)
```

For input `x`:

$$
\boxed{
y=W_0x+B_{LoRA}(Ax)
}
$$

A frequently used scaled form is:

$$
y=W_0x+\frac{\alpha}{r}B_{LoRA}Ax
$$

The lecture focuses on the low-rank product; the `alpha/r` scaling is a standard implementation clarification.

---

## Q138. MUST REMEMBER - Derive the LoRA tensor shapes.

For a flattened token activation:

```text
x:       [d_in]
A:       [r,d_in]
A x:     [r]
B_LoRA:  [d_out,r]
B(Ax):   [d_out]
W0 x:    [d_out]
y:       [d_out]
```

Batched Transformer view:

```text
x:             [B,T,d_in]
x @ A^T:       [B,T,r]
(x @ A^T)@B^T: [B,T,d_out]
base output:   [B,T,d_out]
final output:  [B,T,d_out]
```

**Memory liner:**

> Project down to rank `r`, project back up, and add to the frozen base output.

---

## Q139. MUST REMEMBER - How many trainable parameters does LoRA add?

Full matrix:

$$
N_{full}=d_{out}d_{in}
$$

LoRA matrices:

$$
\boxed{
N_{LoRA}=r(d_{in}+d_{out})
}
$$

The fractional trainable size is:

$$
\frac{N_{LoRA}}{N_{full}}
=
\frac{r(d_{in}+d_{out})}{d_{in}d_{out}}
$$

For `d_in=d_out=d`:

$$
\frac{N_{LoRA}}{N_{full}}=\frac{2r}{d}
$$

**Memory liner:**

> LoRA changes multiplicative parameter scaling into rank times the sum of dimensions.

---

## Q140. MUST REMEMBER - Give a numerical LoRA example.

For:

```text
d_in = d_out = 4096
r = 8
```

Full matrix:

$$
4096^2=16{,}777{,}216
$$

LoRA:

$$
8(4096+4096)=65{,}536
$$

That is about 256 times fewer trainable parameters for this matrix.

---

## Q141. MUST REMEMBER - What is computed during the LoRA forward pass?

```text
base path:
x -> frozen W0 -> y_base

adapter path:
x -> A -> rank-r latent -> B_LoRA -> delta_y

final:
y = y_base + scaled delta_y
```

Both paths contribute to the output, but gradients are retained only for the trainable adapter parameters.

---

## Q142. MUST REMEMBER - What is frozen and what is trained?

```text
W0:                frozen
A and B_LoRA:      trainable
LoRA optimizer:    stores states only for A and B_LoRA
```

This is where the major fine-tuning memory saving comes from.

**Memory liner:**

> Frozen parameters participate in forward/backward computation but receive no optimizer update.

---

## Q143. MUST REMEMBER - What does the rank `r` control?

`r` controls the dimensionality of the learned update subspace:

- Small `r`: fewer parameters and less adaptation capacity.
- Larger `r`: more expressive update and higher memory/compute.

It is a design hyperparameter. The lecture notes that common small values often work, but selection is empirical.

**Memory liner:**

> Rank is the capacity knob for the task-specific update.

---

## Q144. MUST REMEMBER - Where can LoRA be inserted?

A Transformer has many linear maps, including:

- Attention projections `W_Q`, `W_K`, `W_V`, `W_O`.
- FFN projections.

The original LoRA work emphasized attention matrices. The lecture reports that later empirical work finds FFN adaptation highly important and that current practice may place LoRA in both attention and FFN layers.

**Memory liner:**

> LoRA can adapt any linear map; target-module choice is empirical.

---

## Q145. SHOULD KNOW - Why are LoRA adapters convenient for multiple tasks?

Keep one frozen base checkpoint and a small adapter per task:

```text
base model W0
   + adapter_math
   + adapter_code
   + adapter_spam
   + adapter_sentiment
```

Only the selected adapter needs to be loaded or activated for a specialization.

**Memory liner:**

> Share the expensive base; swap the small task-specific delta.

---

## Q146. INTERVIEW CLARIFICATION - Can LoRA be merged for inference?

Yes, when the adapter is fixed:

$$
W_{merged}=W_0+\frac{\alpha}{r}B_{LoRA}A
$$

Then inference can use one ordinary matrix multiplication, removing adapter-path latency. One may instead keep adapters separate to switch tasks dynamically.

---

## Q147. SHOULD KNOW - What learning-rate observation does the lecture report?

The lecture reports empirical guidance that LoRA often uses a higher learning rate than full fine-tuning because only a small low-rank parameter set is moving.

Treat this as an empirical starting point, not a universal `10x` law. The optimal value depends on rank, target modules, optimizer, data, and model scale.

---

## Q148. SHOULD KNOW - What batch-size observation does the lecture report?

The lecture reports that very large batches may not benefit LoRA in the same way as full-matrix fine-tuning, possibly because optimization over a factorized product has different dynamics.

Again, this is an empirical observation, not a theorem. Validate it for the actual setup.

---

## Q149. MUST REMEMBER - Full fine-tuning versus LoRA?

| | Full fine-tuning | LoRA |
|---|---|---|
| Base weights | updated | frozen |
| Trainable count | approximately all `P` | small adapter subset |
| Optimizer state | full model | adapters only |
| Task checkpoint | full model copy | small adapter |
| Maximum flexibility | higher | constrained low-rank update |
| Typical memory | high | much lower |

**Memory liner:**

> Full fine-tuning changes the model; LoRA learns a compact delta around it.

---

## Q150. SHOULD KNOW - What are LoRA's limitations?

- Low-rank capacity can be insufficient for a large distribution shift.
- Quality depends on target modules and rank.
- Base model weights still require memory for forward computation.
- Training still stores activations.
- Adapter combinations can interact unpredictably.
- A small trainable count does not make data quality or evaluation easy.

---

## Q151. MUST REMEMBER - Minimal PyTorch-like LoRA layer?

```python
class LoRALinear(torch.nn.Module):
    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.rank = rank
        self.scale = alpha / rank

        # A: [r, d_in], B: [d_out, r]
        self.A = torch.nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.B = torch.nn.Parameter(
            torch.zeros(base.out_features, rank)
        )

        torch.nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_in]
        base_output = self.base(x)                 # [..., d_out]
        low_rank = torch.nn.functional.linear(x, self.A)   # [..., r]
        delta = torch.nn.functional.linear(low_rank, self.B) # [..., d_out]
        return base_output + self.scale * delta
```

The zero initialization of `B` makes the initial adapter update zero, so training starts from the original base-model function. This initialization detail is standard implementation knowledge, not explicitly developed in the lecture.

---

## Q152. LECTURE BOUNDARY - Prefix tuning and adapters?

The lecture names prefix tuning and adapter methods as alternative parameter-efficient approaches but does not explain their equations, placement, or trade-offs. Do not treat them as covered in depth here.

---

# 13. QLoRA: quantized low-rank adaptation

## Q153. MUST REMEMBER - What is QLoRA?

QLoRA combines:

```text
quantized frozen base model
          +
trainable LoRA adapters in a higher compute precision
```

The base weights consume far less memory, while task adaptation still occurs through `A` and `B_LoRA`.

**Memory liner:**

> QLoRA compresses what is frozen and preserves precision for what is learned.

---

## Q154. MUST REMEMBER - What precision pattern is taught?

At the lecture level:

```text
W0: frozen and stored in 4-bit NF4
A/B LoRA adapters: trained in BF16-like precision
matmul: quantized W0 is dequantized as needed for compute
```

Only adapter parameters receive gradients and optimizer states.

---

## Q155. MUST REMEMBER - What is NF4?

**NF4 = NormalFloat 4-bit.**

The lecture describes it as a 4-bit format designed around the approximately normal distribution of neural-network weights. Rather than using uniformly spaced real values, its representable values correspond to quantile-like regions of a normal distribution.

**Memory liner:**

> NF4 spends its 16 codes where normally distributed weights are likely to occur.

---

## Q156. SHOULD KNOW - Why use quantile-based levels instead of equal-width buckets?

If values are concentrated near zero, uniform real-number spacing wastes codes in sparse tails and provides too little resolution where most weights lie.

Quantile-oriented levels aim to place a similar probability mass in each region under the assumed weight distribution.

```text
many weights near zero
        -> more useful representational resolution near zero
```

---

## Q157. MUST REMEMBER - What is double quantization?

Quantization normally stores metadata such as scale constants for blocks of weights. Double quantization compresses those quantization constants as well.

```text
full weights
   -> 4-bit quantized weights + scales
   -> quantize the scales too
```

The second step gives an additional, smaller memory saving.

**Memory liner:**

> Quantize the weights, then also compress the metadata needed to dequantize them.

---

## Q158. MUST REMEMBER - What is the QLoRA forward dataflow?

```text
input x
  |-----------------------------|
  |                             |
  v                             v
4-bit frozen W0           BF16 trainable A
  | dequantize tile              |
  | matmul                       v
  |                         rank-r activation
  |                             |
  |                             v
  |                        BF16 trainable B
  |                             |
  v                             v
base output                 LoRA delta
  |_____________________________|
                add
                 v
               output
```

The base path is quantized for storage; the adapter path remains trainable.

---

## Q159. MUST REMEMBER - Where do gradients flow in QLoRA?

```text
loss
 -> gradients through both computational paths
 -> W0 remains frozen: no W0 update or optimizer state
 -> A and B_LoRA receive gradients and optimizer updates
```

Backpropagation still needs input/activation derivatives through the base computations, but not parameter gradients for the frozen quantized weights.

---

## Q160. MUST REMEMBER - Why does QLoRA save more memory than LoRA?

LoRA removes full-model gradients and optimizer states but usually still stores base weights in FP16/BF16.

QLoRA additionally compresses base weights to 4-bit:

```text
LoRA:
base weights roughly 2 bytes/value + small trainable adapters

QLoRA:
base weights roughly 0.5 bytes/value + metadata + small trainable adapters
```

Activations remain an important memory cost in both.

---

## Q161. MUST REMEMBER - LoRA versus QLoRA?

| | LoRA | QLoRA |
|---|---|---|
| Base model | frozen FP16/BF16 commonly | frozen 4-bit commonly |
| Adapter | trainable | trainable |
| Base-weight memory | moderate | much lower |
| Quantization error | no base quantization | yes |
| Fine-tuning accessibility | good | better on limited VRAM |

**Memory liner:**

> QLoRA is LoRA plus low-bit storage for the frozen base.

---

## Q162. MUST REMEMBER - Mixed precision versus QLoRA?

```text
mixed-precision full training:
most or all model weights are trainable;
formats differ across compute and master state

QLoRA:
base model is frozen and quantized;
only low-rank adapters are trainable
```

They solve related numerical/memory problems at different stages and can coexist in a system.

---

## Q163. SHOULD KNOW - What QLoRA topics are outside this lecture?

The transcript does not fully derive:

- NF4 codebook values.
- Block-size selection.
- Exact dequantization kernels.
- Paged optimizers from the QLoRA paper.
- Quantization-aware training.
- Detailed quality versus rank/bit-width ablations.

Keep those for a dedicated quantization/PEFT note.

---

## Q164. MUST REMEMBER - Give the 30-second QLoRA interview answer.

> QLoRA freezes the base model, stores its weights in a 4-bit NormalFloat format, and trains only BF16 low-rank LoRA adapters. The base weights are dequantized as needed for matrix multiplication, but they have no gradients or optimizer state. NF4 allocates 4-bit values according to a normal-weight assumption, and double quantization also compresses the scale constants. The result is much lower VRAM usage while retaining the ability to adapt a large model.


---

# 14. Equations to memorize cold

## 14.1 Causal language-model factorization

$$
\boxed{
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
}
$$

---

## 14.2 Pre-training cross-entropy

$$
\boxed{
\mathcal{L}_{PT}
=
-\frac{1}{N_{valid}}
\sum_{b,t}m_{b,t}
\log p_\theta(x_{b,t+1}\mid x_{b,\le t})
}
$$

---

## 14.3 Rough dense-training compute

$$
\boxed{C_{train}\approx 6PN_{tok}}
$$

This is an interview planning approximation, not an exact law.

---

## 14.4 Chinchilla-style rule of thumb taught in the lecture

$$
\boxed{N_{tok}\approx 20P}
$$

Treat the constant as empirical and setup-dependent.

---

## 14.5 Idealized wall-clock relation

$$
\boxed{
\text{time}
\approx
\frac{\text{total FLOPs}}
{\text{peak FLOP/s}\times\text{utilization}}
}
$$

---

## 14.6 Data-parallel gradient aggregation

$$
\boxed{
g_{global}=\frac{1}{G}\sum_{j=1}^{G}g^{(j)}}
$$

---

## 14.7 Adam moments

$$
\boxed{m_t=\beta_1m_{t-1}+(1-\beta_1)g_t}
$$

$$
\boxed{v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2}
$$

---

## 14.8 Scaled dot-product attention

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{D_h}}
\right)V
}
$$

FlashAttention computes this exact function with a different schedule.

---

## 14.9 Numerically stable softmax identity

For row scores `s_i` and `m=max_i s_i`:

$$
\boxed{
\operatorname{softmax}(s_i)
=
\frac{e^{s_i-m}}{\sum_j e^{s_j-m}}
}
$$

Online softmax updates `m` and the denominator block by block.

---

## 14.10 Affine quantization template

$$
\boxed{q=\operatorname{round}(x/s)+z}
$$

$$
\boxed{\hat x=s(q-z)}
$$

The lecture only introduces range-based quantization at a high level.

---

## 14.11 Response-only SFT loss

$$
\boxed{
\mathcal{L}_{SFT}
=
-\frac{1}{\sum_{b,t}M_{b,t}}
\sum_{b,t}M_{b,t}
\log p_\theta(z_{b,t}\mid z_{b,<t})
}
$$

`M=1` on assistant/output tokens and `0` on conditioning tokens.

---

## 14.12 LoRA update

$$
\boxed{W=W_0+B_{LoRA}A}
$$

or with scaling:

$$
\boxed{W=W_0+\frac{\alpha}{r}B_{LoRA}A}
$$

---

## 14.13 LoRA parameter count

$$
\boxed{N_{LoRA}=r(d_{in}+d_{out})}
$$

versus:

$$
\boxed{N_{full}=d_{in}d_{out}}
$$

For a square matrix `d x d`:

$$
\boxed{N_{LoRA}/N_{full}=2r/d}
$$

---

# 15. Tensor shapes to memorize cold

```text
PRE-TRAINING / SFT

Token IDs                 [B,T]
Attention mask            [B,T]
Labels                    [B,T]
Assistant loss mask       [B,T]
Hidden states             [B,T,D]
Vocabulary logits         [B,T,V]

TRANSFORMER ACTIVATIONS

Q, K, V                   [B,H,T,Dh]
Attention scores          [B,H,T,T]
Attention output          [B,H,T,Dh]
Merged attention output   [B,T,D]
FFN hidden                [B,T,Dff]

PARAMETER-RELATED STATE

Parameters                total P values
Gradients                 same shapes as trainable parameters
Adam first moment         same shapes as trainable parameters
Adam second moment        same shapes as trainable parameters

DATA PARALLELISM

Global token batch        [B,T]
Local token batch/GPU     [B/G,T]
Local gradient            same shape as parameter
Aggregated gradient       same shape as parameter

TENSOR PARALLEL LINEAR LAYER

X                         [B,T,d_in]
W                         [d_in,d_out]
Column shard W_j          [d_in,d_out/G]
Local output Y_j          [B,T,d_out/G]
Concatenated output       [B,T,d_out]

LORA, COLUMN-VECTOR WEIGHT CONVENTION

W0                        [d_out,d_in]
A                         [r,d_in]
B_LoRA                    [d_out,r]
B_LoRA @ A                [d_out,d_in]

LORA, BATCHED ROW-ACTIVATION VIEW

X                         [B,T,d_in]
X @ A^T                   [B,T,r]
(X @ A^T) @ B_LoRA^T     [B,T,d_out]
Base output               [B,T,d_out]
Final output              [B,T,d_out]
```

The three shape relations to remember most strongly are:

$$
\boxed{
[B,H,T,D_h]
\times
[B,H,D_h,T]
=
[B,H,T,T]
}
$$

$$
\boxed{
[B,H,T,T]
\times
[B,H,T,D_h]
=
[B,H,T,D_h]
}
$$

$$
\boxed{
[B,T,d_{in}]
\rightarrow
[B,T,r]
\rightarrow
[B,T,d_{out}]
}
$$

---

# 16. High-value comparison tables

## 16.1 Pre-training, mid-training, SFT, and preference tuning

| Stage | Data | Objective | Main role |
|---|---|---|---|
| Pre-training | Broad raw text/code | Causal next-token loss | General language and capability |
| Mid-training | Targeted domain text/code | Same causal LM loss | Strengthen selected domains/tasks |
| SFT/instruction tuning | Curated prompt-response pairs | Response-token causal loss | Teach instruction-following behavior |
| Preference tuning | Preference signal | Deferred to next lecture | Refine behavior toward preferred outputs |

---

## 16.2 Parallelism strategies

| Method | What is partitioned? | Main memory benefit | Main cost |
|---|---|---|---|
| Data parallelism | Batch | Smaller local activation batch | Gradient communication; full model replicas |
| ZeRO | Optimizer/grads/params by stage | Removes DP redundancy | Increasing state communication |
| Tensor parallelism | Matrix dimensions | Large layers split across GPUs | Frequent intra-layer communication |
| Pipeline parallelism | Layer groups | Each GPU stores fewer layers | Stage communication and pipeline bubbles |
| Expert parallelism | MoE experts | Expert parameters distributed | Token routing/all-to-all and imbalance |

---

## 16.3 ZeRO stages

```text
ZeRO-1: shard optimizer states
ZeRO-2: shard optimizer states + gradients
ZeRO-3: shard optimizer states + gradients + parameters
```

---

## 16.4 FlashAttention versus ordinary attention

| | Naive attention | FlashAttention |
|---|---|---|
| Mathematical result | Dense softmax attention | Same dense softmax attention |
| Full score matrix in HBM | Commonly materialized | Avoided |
| Execution | Separate read/write-heavy stages | Tiled and fused |
| Backward | Often saves large intermediates | Recomputes selected intermediates |
| Primary optimization | None | IO and memory hierarchy |

---

## 16.5 Numerical and adaptation methods

| Method | Frozen base? | Base precision | Trainable portion | Main purpose |
|---|---:|---|---|---|
| Full mixed-precision training | No | mixed + master state | Nearly all parameters | Faster full training |
| Quantized inference | Usually yes | low bit | none | Smaller/faster inference |
| LoRA | Yes | usually FP16/BF16 | low-rank adapters | Cheap fine-tuning |
| QLoRA | Yes | typically 4-bit NF4 | BF16 LoRA adapters | Fine-tune large models on limited VRAM |

---

## 16.6 Pre-training loss versus SFT loss

```text
PRE-TRAINING
prompt/document token positions: supervised
continuation token positions:     supervised

SFT RESPONSE-ONLY
system/user/prompt positions:     conditioning only
assistant response positions:     supervised
```

---

# 17. End-to-end implementation mental maps

## 17.1 Causal pre-training step

```python
# input_ids: [B,T]
# attention_mask: [B,T]

logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
).logits                              # [B,T,V]

shift_logits = logits[:, :-1, :]      # [B,T-1,V]
shift_labels = input_ids[:, 1:]       # [B,T-1]
shift_mask = attention_mask[:, 1:]    # [B,T-1]

labels = shift_labels.masked_fill(~shift_mask.bool(), -100)

loss = torch.nn.functional.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    labels.reshape(-1),
    ignore_index=-100,
)

loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

---

## 17.2 Data-parallel training mental map

```python
# Conceptual; DistributedDataParallel handles synchronization.
local_batch = global_batch[rank_partition]
loss = model_loss(local_batch)
loss.backward()

# DDP all-reduces parameter gradients so each replica gets
# the gradient corresponding to the global batch.
optimizer.step()
```

---

## 17.3 ZeRO mental map

```text
ordinary DP:
each rank permanently owns every model state

ZeRO:
each rank permanently owns only a shard of selected states
when computation needs a missing state:
    communicate/gather it
when updating/reducing:
    distribute result back to owners
```

---

## 17.4 FlashAttention mental map

```text
for each Q tile:
    initialize running row max, denominator, output accumulator

    for each K/V tile:
        load tile into SRAM
        compute local score tile
        update online-softmax statistics
        update weighted-value accumulator

    normalize accumulator
    write final output tile to HBM
```

---

## 17.5 Response-only SFT step

```python
labels = input_ids.clone()                  # [B,T]
labels[prompt_or_padding_mask] = -100

logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
).logits                                    # [B,T,V]

loss = torch.nn.functional.cross_entropy(
    logits[:, :-1, :].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

---

## 17.6 LoRA training mental map

```text
load base checkpoint
freeze every base parameter
insert A/B low-rank matrices in selected linear layers
optimize only A/B on SFT data
save only adapter state
optionally merge adapter into base for inference
```

---

## 17.7 QLoRA training mental map

```text
load base model in 4-bit NF4
freeze quantized base weights
insert BF16 LoRA matrices
forward:
    dequantize needed base tiles for matmul
    add LoRA branch
backward:
    update only LoRA matrices
save:
    adapter checkpoint + base-model identifier
```

---

# 18. Interview questions to practise aloud

1. Why is transfer learning the foundation of the LLM training pipeline?
2. What is the difference between a pre-trained base model and an instruction-tuned assistant?
3. Write the causal sequence factorization of a language model.
4. Show how one token sequence becomes shifted inputs and labels.
5. What are the shapes of input IDs, hidden states, and vocabulary logits?
6. Why can all target positions be trained in parallel despite causal generation?
7. Why is next-token prediction a strong proxy objective but not an alignment objective?
8. What determines a model's knowledge cutoff?
9. Why is editing one fact in the weights difficult?
10. Distinguish model generalization from memorization.
11. What is the difference between FLOP count and FLOP/s?
12. Explain the approximation `C approximately 6 P Ntok`.
13. Why is training time not simply total FLOPs divided by peak hardware throughput?
14. What does sample efficiency mean in scaling-law discussions?
15. Under a fixed compute budget, why is the largest model not necessarily optimal?
16. State and qualify the Chinchilla `20 tokens per parameter` heuristic.
17. How do teams use small scaling experiments to plan large runs?
18. What tensors must be stored during full Adam training?
19. Give a rough bytes-per-parameter budget for mixed-precision Adam.
20. Why does long context greatly increase activation memory?
21. What does ordinary data parallelism partition?
22. Derive the global gradient from local worker gradients.
23. What memory does data parallelism reduce, and what does it replicate?
24. Compare ZeRO-1, ZeRO-2, and ZeRO-3.
25. Why does ZeRO-3 save memory but increase communication?
26. Compare data, tensor, pipeline, and expert parallelism.
27. Draw a column-sharded linear layer for tensor parallelism.
28. What is the communication bottleneck in expert parallelism?
29. What is the GPU-memory problem solved by FlashAttention?
30. Why is FlashAttention exact rather than approximate?
31. What are HBM and SRAM, and why does the distinction matter?
32. Walk through naive attention's repeated HBM reads/writes.
33. Explain how tiling and online softmax avoid materializing `[B,H,T,T]`.
34. Why can recomputation add FLOPs but reduce runtime?
35. FlashAttention versus sliding-window attention?
36. Explain sign, exponent, and mantissa in a floating-point number.
37. Compare FP16 and BF16.
38. Why can lower precision increase both throughput and memory capacity?
39. What is mixed-precision training?
40. Why keep FP32 master weights?
41. What is loss scaling, and why is it associated with FP16?
42. Distinguish quantization from mixed-precision training.
43. Why is SFT needed if pre-training already learned broad language knowledge?
44. Is SFT still next-token prediction?
45. Write the response-only SFT loss.
46. How do `-100` labels implement prompt masking?
47. What is instruction tuning?
48. Why can a small but high-quality SFT dataset change model behavior broadly?
49. What roles do human and synthetic data play in SFT curation?
50. How are safe refusal and hedging represented in supervised data?
51. Why does SFT prompt-distribution coverage matter?
52. What makes LLM helpfulness difficult to evaluate numerically?
53. Training on the test task versus training on the test set?
54. Why can benchmark improvements overstate real user improvements?
55. What does a pairwise model arena measure?
56. Why can human preference disagree with factuality and safety?
57. Where does optional mid-training fit, and what objective does it use?
58. What does the lecture include under alignment?
59. Write the LoRA update equation and all matrix shapes.
60. Derive the LoRA parameter count.
61. How does rank affect LoRA capacity and cost?
62. Why do frozen base weights still participate in backpropagation?
63. Where can LoRA modules be applied in a Transformer?
64. Why are task-specific LoRA adapters operationally convenient?
65. How can LoRA be merged for inference?
66. Full fine-tuning versus LoRA?
67. What is QLoRA?
68. What is NF4, and why use a normal-distribution-aware 4-bit format?
69. What is double quantization?
70. Where do gradients flow in QLoRA?
71. LoRA versus QLoRA versus mixed-precision full training?
72. Which topics are deferred to the preference-tuning lecture?

---

# 19. Thirty memory liners for interview day

```text
1. Transfer learning:
   Pre-train once, then adapt rather than relearning language per task.

2. Training stages:
   Pre-training builds capability; post-training shapes behavior.

3. Base model:
   Likely continuation is not automatically helpful assistance.

4. Mid-training:
   Same LM objective, more targeted data mixture.

5. Causal LM:
   Each token is predicted from its prefix.

6. Pre-training tensors:
   [B,T] tokens become [B,T,V] vocabulary logits.

7. Training parallelism:
   Full targets are known, while a causal mask blocks future information.

8. Data scale:
   Token count says how many prediction positions the model consumes.

9. Data mixture:
   Scale says how much; mixture says what is learned.

10. Knowledge cutoff:
    Frozen weights cannot learn events after the training-data cutoff.

11. FLOPs:
    FLOP count is work; FLOP/s is speed.

12. Compute estimate:
    Dense pre-training is roughly 6 x parameters x tokens.

13. Chinchilla:
    About twenty tokens per parameter is a study-specific rule of thumb.

14. Sample efficiency:
    Better performance per token does not mean lower compute per token.

15. Training memory:
    Parameters + gradients + optimizer state + activations + buffers.

16. Adam:
    Adaptive optimization stores two extra parameter-sized moments.

17. Data parallelism:
    Split examples, replicate the model, aggregate gradients.

18. ZeRO:
    Replace redundant model-state storage with communication.

19. ZeRO stages:
    Optimizer; then gradients; then parameters are successively sharded.

20. Model parallelism:
    Split one model computation across devices.

21. FlashAttention:
    Exact attention, tiled to minimize HBM traffic.

22. Recomputation:
    Sometimes recalculating is cheaper than reloading.

23. Mixed precision:
    Compute cheaply but accumulate important state accurately.

24. SFT:
    Same causal machinery, curated demonstrations, response-only loss.

25. Instruction tuning:
    Teach the model that natural-language requests should trigger tasks.

26. Evaluation:
    Capability, factuality, preference, safety, and efficiency need separate checks.

27. LoRA:
    Freeze the base and learn a low-rank weight delta.

28. LoRA rank:
    Rank is the capacity-cost knob of the adapter.

29. QLoRA:
    Quantize what is frozen; keep the small learned adapters precise.

30. Alignment boundary:
    Preference-tuning mechanics belong to the next lecture.
```

---

# 20. Lecture boundaries - do not silently add these to this note

The transcript mentions, motivates, or naturally leads to the following topics but does **not** teach them in depth:

- Exact Kaplan/Chinchilla fitted exponents and irreducible-loss equations.
- Full data filtering, deduplication, token packing, and curriculum pipelines.
- Gradient accumulation, gradient clipping, and learning-rate schedules.
- Detailed all-reduce, reduce-scatter, and all-gather algorithms.
- Tensor-parallel row/column algorithms and pipeline scheduling.
- Distributed checkpointing and fault tolerance.
- Exact FlashAttention kernel code and IO-complexity proof.
- Detailed FP8 training, stochastic rounding, and hardware-specific tensor cores.
- Complete zero-point, absmax, GPTQ, AWQ, or quantization-aware-training derivations.
- Preference tuning, reward models, PPO, DPO, GRPO, or RLHF.
- Exact LoRA initialization ablations and adapter-composition algorithms.
- Prefix-tuning and classic adapter-layer equations.
- NF4 codebook values, QLoRA paged optimizers, and kernel implementation.

Keep these for their own lecture or dedicated interview note.

---

# 21. Final previous-day checklist

You should be able to answer or derive every item below without opening the note.

## Training pipeline

- [ ] Explain transfer learning and why LLMs use it.
- [ ] Draw pre-training -> optional mid-training -> SFT -> preference tuning.
- [ ] Explain why a base LM is not automatically a helpful assistant.
- [ ] Define capability versus alignment.

## Pre-training

- [ ] Write `p(x_1:T)=product_t p(x_t|x_<t)`.
- [ ] Construct shifted causal inputs and labels.
- [ ] Derive `[B,T,D] -> [B,T,V]` logits and token cross-entropy.
- [ ] Explain parallel teacher-forced training versus autoregressive inference.
- [ ] Explain knowledge cutoff, memorization, and data-mixture effects.

## Scaling

- [ ] Distinguish FLOP count from FLOP/s.
- [ ] Explain and qualify `C approximately 6 P Ntok`.
- [ ] Define sample efficiency.
- [ ] Explain the fixed-compute model/data trade-off.
- [ ] State and qualify `Ntok approximately 20P`.

## Training memory

- [ ] List parameters, gradients, optimizer moments, activations, and buffers.
- [ ] Explain the rough 16-byte/parameter mixed-precision Adam budget.
- [ ] Explain why sequence length affects activation memory quadratically in naive attention.

## Distributed training

- [ ] Explain data parallelism and gradient averaging.
- [ ] State exactly what ZeRO-1, ZeRO-2, and ZeRO-3 shard.
- [ ] Compare tensor, pipeline, and expert parallelism.
- [ ] Explain the memory-versus-communication trade-off.

## FlashAttention

- [ ] Contrast HBM and SRAM.
- [ ] Explain why naive attention is IO-heavy.
- [ ] Explain tiling and online softmax.
- [ ] Explain why FlashAttention is exact.
- [ ] Explain why extra recomputation can reduce runtime.
- [ ] Distinguish FlashAttention from sparse attention.

## Precision

- [ ] Explain sign, exponent, and mantissa.
- [ ] Compare FP32, FP16, and BF16.
- [ ] Explain mixed precision and FP32 master weights.
- [ ] Explain loss scaling as a follow-up concept.
- [ ] Distinguish quantization from mixed-precision training.

## SFT and evaluation

- [ ] Define SFT and instruction tuning.
- [ ] Write the response-only loss mask equation.
- [ ] Implement prompt masking with `-100` labels.
- [ ] Explain human versus synthetic SFT data.
- [ ] Explain data quality, prompt coverage, helpfulness, and safety.
- [ ] Distinguish training on the test task from test-set contamination.
- [ ] Explain strengths and weaknesses of pairwise human arenas.

## LoRA and QLoRA

- [ ] Write `W=W0+B A` and derive every shape.
- [ ] Derive `r(d_in+d_out)` trainable parameters.
- [ ] Explain rank, target modules, task adapters, and optional weight merging.
- [ ] Compare full fine-tuning, LoRA, and QLoRA.
- [ ] Explain NF4 and double quantization.
- [ ] Explain exactly which QLoRA tensors receive gradients.

---

# 22. Final one-minute mental map

```text
A modern LLM is not trained in one monolithic step.

PRE-TRAINING
Broad text/code + causal next-token loss
    -> learn language, knowledge, and reusable capability
    -> enormous parameter, token, compute, and memory scale

SCALING
Compute roughly grows with parameters x tokens.
At fixed compute, model size and token count must be balanced.

SYSTEMS
Training stores weights, gradients, Adam moments, and activations.
Data parallelism splits examples.
ZeRO shards redundant model states.
Model parallelism splits layers, tensors, or experts.
FlashAttention computes exact attention with tiled, IO-aware kernels.
Mixed precision reduces bytes and increases throughput.

POST-TRAINING
SFT uses curated prompt-response pairs.
The prompt conditions the model; loss is paid on assistant tokens.
Data quality and deployment-distribution coverage matter greatly.
No single benchmark fully measures usefulness, truth, safety, and cost.

PARAMETER-EFFICIENT ADAPTATION
LoRA freezes W0 and learns a low-rank delta BA.
QLoRA additionally stores the frozen base in 4-bit NF4
while training the small LoRA adapters at higher precision.

Core story:
pre-training builds a capable model;
systems techniques make training feasible;
SFT shapes useful behavior;
LoRA/QLoRA make adaptation affordable.
```
